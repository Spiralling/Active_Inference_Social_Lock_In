#!/usr/bin/env python3
"""Independent raw-state and analysis audit; no producer/analysis helpers imported."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from independent_tests import numpy_step

HERE=Path(__file__).resolve().parent
NEW=['G_plain_info','G_gated_info','G_plain_moment','G_gated_moment']
OLD=['C10','C11','C10_gasvar_matched']
WINDOWS={1500:(1126,1500),5000:(3751,5000)}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def audit(raw,candidate_raw,analysis,out):
    start=time.monotonic()
    manifest=json.loads((raw/'manifest.json').read_text());archive=json.loads((candidate_raw/'manifest.json').read_text())
    if len(manifest)!=800 or {r['condition'] for r in manifest}!=set(NEW):raise AssertionError('Wrong new raw records')
    protocol=json.loads((raw/'protocol.json').read_text());seedlist=protocol['seeds']
    if seedlist!=list(range(10000,10050)):raise AssertionError('Seed registry mismatch')
    expected_old={(tuple(r['seeds']),r['start_step']):r for r in archive if r['condition']=='C10'}
    errors={};checks=dict(new_raw_chunks=0,candidate_raw_chunks=0,paired_observation_hashes=0,paired_evaluation_hashes=0,checkpoint_equalities=0,positive_definite_matrices=0,full_gaussian_update_states=0,predeposit_scores=0,sequential_numpy_updates=0,predictive_NLL_values=0)
    independent_rows=[];coverage={};sample_plan=[];vector_means={}
    def close(label,x,y,atol=3e-10,rtol=3e-11):
        difference=float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
        errors[label]=max(errors.get(label,0.),difference)
        np.testing.assert_allclose(x,y,atol=atol,rtol=rtol,err_msg=label)
    for condition in NEW+OLD:
        root,records=(raw,manifest) if condition in NEW else (candidate_raw,archive)
        selected=sorted((r for r in records if r['condition']==condition),key=lambda r:(r['seeds'][0],r['start_step']))
        meta=json.loads((root/condition/'setup.json').read_text())
        if meta['seeds']!=seedlist or meta['T']!=5000 or meta['N']!=16:raise AssertionError('Unexpected dimensions')
        if meta['group_indices']!={'gas':list(range(8)),'mass':list(range(8,16))}:raise AssertionError('Wrong groups')
        if condition in OLD and 'oxygen' not in meta['candidate_names'][1]:raise AssertionError('Candidate label order')
        count=np.zeros((50,5000),np.uint8);last={};acc={}
        for s in seedlist:
            for horizon,(lo,hi) in WINDOWS.items():acc[s,horizon]={}
        W=np.asarray(meta['W']);P0=np.asarray(meta['Pi0']);h0=np.asarray(meta['h0']);H=np.asarray(meta['H']);att=np.asarray(meta['attention'])
        omega=meta['omega'];sigma=meta['sigma_o'];nu=meta.get('gate_nu')
        J=np.einsum('ia,ni,ib->nab',H,att,H)/sigma**2
        for row in selected:
            file=root/row['file']
            if sha(file)!=row['sha256']:raise AssertionError('Raw hash mismatch '+str(file))
            with np.load(file,allow_pickle=False) as npz:data={k:npz[k] for k in npz.files}
            B=len(row['seeds']);T=row['steps'];t0=row['start_step'];ix=np.array(row['seeds'])-10000
            count[ix,t0:t0+T]+=1
            checks['new_raw_chunks' if condition in NEW else 'candidate_raw_chunks']+=1
            if any(not np.isfinite(v).all() for v in data.values()):raise AssertionError('Nonfinite data')
            if data['mu_mixture'].shape!=(B,T,16,3):raise AssertionError('Observable means shape')
            for key in ('Pi','h','L','key'):
                if tuple(row['seeds']) in last:
                    np.testing.assert_array_equal(data['initial_'+key],last[tuple(row['seeds'])][key]);checks['checkpoint_equalities']+=1
            last[tuple(row['seeds'])]={key:data['final_'+key] for key in ('Pi','h','L','key')}
            if t0==0:
                np.testing.assert_array_equal(data['initial_L'],0.)
                np.testing.assert_array_equal(data['initial_Pi'],np.broadcast_to(P0,data['initial_Pi'].shape))
                np.testing.assert_array_equal(data['initial_h'],np.broadcast_to(h0,data['initial_h'].shape))
            close('q_from_L',data['q'],softmax(data['L'],axis=-1),atol=2e-13,rtol=2e-13)
            close('MSE',data['mse_full'],np.mean((data['mu_mixture']-1)**2,axis=-1),atol=2e-14,rtol=2e-14)
            if condition in NEW:
                original=expected_old[tuple(row['seeds']),t0]
                if arrsha(data['obs'])!=row['observation_sha256'] or row['observation_sha256']!=original['observation_sha256']:raise AssertionError('Observation pairing')
                if row['evaluation_observation_sha256']!=original['evaluation_observation_sha256']:raise AssertionError('Evaluation pairing')
                checks['paired_observation_hashes']+=1;checks['paired_evaluation_hashes']+=1
                P=data['Pi_history'];np.linalg.cholesky(P);checks['positive_definite_matrices']+=int(np.prod(P.shape[:-2]))
                np.testing.assert_array_equal(data['q'],1.)
                np.testing.assert_array_equal(data['mu_conditional'][...,0,:],data['mu_mixture'])
                np.testing.assert_array_equal(data['mu_map'],data['mu_mixture'])
                close('precision_dummy_diagonal',P[...,0,0],4.,atol=3e-14,rtol=0.)
                np.testing.assert_array_equal(P[...,0,1:],0.);np.testing.assert_array_equal(P[...,1:,0],0.)
                current_mu=np.concatenate([np.zeros_like(data['mu_conditional'][...,:1]),data['mu_conditional']],axis=-1)
                current_h=np.einsum('btnkij,btnkj->btnki',P,current_mu)
                previous_P=np.concatenate([data['initial_Pi'][:,None],P[:,:-1]],axis=1)
                previous_h=np.concatenate([data['initial_h'][:,None],current_h[:,:-1]],axis=1)
                previous_mu=np.concatenate([np.linalg.solve(data['initial_Pi'],data['initial_h'][...,None])[...,0][:,None],current_mu[:,:-1]],axis=1)
                if nu is None:weights=np.broadcast_to(W,(B,T,16,16))
                else:
                    mu=previous_mu[...,0,1:]
                    var=1/np.diagonal(previous_P[...,0,:,:],axis1=-2,axis2=-1)[...,1:]
                    z=np.sum((mu[:,:,:,None]-mu[:,:,None,:])**2/(var[:,:,:,None]+var[:,:,None,:]),axis=-1)
                    weights=W[None,None]*((nu+1)/(nu+z));weights/=weights.sum(-1,keepdims=True)
                close('weight_row_normalization',weights.sum(-1),1.,atol=16*np.finfo(float).eps/(1-16*np.finfo(float).eps),rtol=0.)
                if np.any(weights<=0):raise AssertionError('Disconnected or negative effective weights')
                cross=np.concatenate([weights[:,:,:8,8:].sum(-1),weights[:,:,8:,:8].sum(-1)],axis=-1)
                close('crossweight',data['gate_crossweight'],cross,atol=3e-12,rtol=3e-11)
                Pbar=np.einsum('btij,btjkab->btikab',weights,previous_P);hbar=np.einsum('btij,btjkd->btikd',weights,previous_h)
                deposit=np.einsum('ia,ni,btni->btna',H,att,data['obs'])/sigma**2
                close('precision_update',P,(1-omega)*P0+omega*(Pbar+J[None,None,:,None]))
                close('information_update',current_h,(1-omega)*h0+omega*(hbar+deposit[...,None,:]))
                close('final_precision',data['final_Pi'],P[:,-1],atol=0,rtol=0)
                close('final_information',data['final_h'],current_h[:,-1],atol=3e-12,rtol=3e-12)
                # Direct multivariate Gaussian pre-deposit predictive density,
                # using independent matrix inverse/solve instead of engine score.
                covariance=np.linalg.inv(Pbar);means=np.linalg.solve(Pbar,hbar[...,None])[...,0]
                Hs=np.sqrt(att)[:,:,None]*H
                V=np.einsum('nia,btnkab,njb->btnkij',Hs,covariance,Hs)+sigma**2*np.eye(3)
                V=(V+V.swapaxes(-1,-2))/2
                residual=np.sqrt(att)[None,None,:,None,:]*data['obs'][...,None,:]-np.einsum('nia,btnka->btnki',Hs,means)
                ell=-.5*(3*np.log(2*np.pi)+np.linalg.slogdet(V)[1]+np.einsum('btnki,btnki->btnk',residual,np.linalg.solve(V,residual[...,None])[...,0]))
                close('predeposit_score',data['ell'],ell,atol=3e-11,rtol=3e-11)
                prevL=np.concatenate([data['initial_L'][:,None],data['L'][:,:-1]],axis=1)
                close('logscore_recurrence',data['L'],prevL+ell,atol=3e-10,rtol=3e-11)
                checks['full_gaussian_update_states']+=B*T;checks['predeposit_scores']+=B*T*16
                # Prespecified first/last chunks of first seed batch: sequential
                # free-running NumPy implementation (rather than raw-state local
                # equation checks), plus all-batch fresh held-out observations.
                if row['seeds'][0]==10000 and t0 in (0,4750):
                    from types import SimpleNamespace
                    cfg=SimpleNamespace(H=H,sigma_o=sigma,omega=omega,Pi0=P0,h0=h0,jitter=0.)
                    pp,hh,ll=(np.array(data['initial_'+key][0]) for key in ('Pi','h','L'))
                    for ti in range(T):
                        ref=numpy_step(pp,hh,ll,data['obs'][0,ti],cfg,W,att,nu)
                        close('sequential_numpy_mean',ref['mu'],data['mu_conditional'][0,ti],atol=2e-10,rtol=2e-11)
                        close('sequential_numpy_score',ref['ell'],data['ell'][0,ti],atol=2e-10,rtol=2e-11)
                        pp,hh,ll=ref['Pi'],ref['h'],ref['L']
                    close('sequential_numpy_final_P',pp,data['final_Pi'][0],atol=2e-10,rtol=2e-11)
                    close('sequential_numpy_final_h',hh,data['final_h'][0],atol=2e-10,rtol=2e-11)
                    close('sequential_numpy_final_L',ll,data['final_L'][0],atol=2e-8,rtol=2e-11)
                    checks['sequential_numpy_updates']+=T;sample_plan.append(row['file'])
                    import jax
                    jax.config.update('jax_enable_x64',True)
                    import jax.numpy as jnp
                    def draw_one(key):
                        def step(k,_):
                            ev=jax.random.fold_in(k,314159)
                            result=jnp.ones(3)+sigma*jax.random.normal(ev,(16,4,3),dtype=jnp.float64)
                            nxt,_=jax.random.split(k)
                            return nxt,result
                        return jax.lax.scan(step,key,None,length=T)
                    finalkey,ev=jax.jit(jax.vmap(draw_one))(jnp.asarray(data['initial_key']))
                    ev=np.asarray(ev);np.testing.assert_array_equal(finalkey,data['final_key'])
                    if arrsha(ev)!=row['evaluation_observation_sha256']:raise AssertionError('Fresh evaluation RNG mismatch')
                    C=np.linalg.inv(P[...,0,:,:])[...,1:,1:]+sigma**2*np.eye(3)
                    resid=ev-data['mu_mixture'][...,None,:]
                    score=-.5*(3*np.log(2*np.pi)+np.linalg.slogdet(C)[1][...,None]+np.einsum('btnri,btnij,btnrj->btnr',resid,np.linalg.inv(C),resid))
                    close('independent_heldout_NLL',-score.mean(-1),data['nll_full'],atol=2e-11,rtol=2e-11)
                    checks['predictive_NLL_values']+=B*T*16
            # Separately reduce every raw state, then seed-level windows. The
            # package analyzer is never imported to construct expected metrics.
            metrics={}
            for src,label in [('mu_mixture','mixture'),('mu_map','map')]:
                signed=data[src][:,:,:8].sum(2)/8-data[src][:,:,8:].sum(2)/8
                metrics[label+'_distance']=np.sqrt(np.sum(signed*signed,axis=-1))
                for d,coordinate in enumerate('CGM'):metrics[label+'_difference_'+coordinate]=signed[...,d]
            for letter,sl in [('A',slice(0,8)),('B',slice(8,16))]:
                for k,source in [('mse','mse_full'),('nll','nll_full')]:metrics[k+'_'+letter]=data[source][:,:,sl].sum(-1)/8
                if condition in OLD:metrics['q_'+letter]=data['q'][:,:,sl,1].sum(-1)/8
                metrics['cross_'+letter]=data['gate_crossweight'][:,:,sl].sum(-1)/8 if condition in NEW else np.full((B,T),.05)
            states=np.arange(t0+1,t0+T+1)
            for horizon,(lo,hi) in WINDOWS.items():
                mask=(states>=lo)&(states<=hi)
                for b,seed in enumerate(row['seeds']):
                    dest=acc[seed,horizon]
                    for key,x in metrics.items():
                        if mask.any():dest[key+'_tail']=dest.get(key+'_tail',0.)+float(x[b,mask].sum())/(hi-lo+1)
                        if hi in states:dest[key+'_endpoint']=float(x[b,hi-t0-1])
            del data
        if not np.all(count==1):raise AssertionError('Missing/duplicated trajectory states')
        coverage[condition]=dict(seeds=50,states_per_seed=5000,coverage_min=int(count.min()),coverage_max=int(count.max()))
        for (seed,horizon),values in acc.items():
            for kind in ('mixture','map'):
                signed=np.array([values[kind+'_difference_'+c+'_tail'] for c in 'CGM'])
                values[kind+'_distance_of_time_means_tail']=float(np.linalg.norm(signed))
            lo,hi=WINDOWS[horizon]
            independent_rows.append(dict(condition=condition,seed=seed,horizon=horizon,window_first_state=lo,window_last_state=hi,**values))
        print('Independently audited',condition,flush=True)
    # Compare every independently reduced CSV metric and all summary intervals.
    with (analysis/'seed_metrics.csv').open() as f:target={(r['condition'],int(r['seed']),int(r['horizon'])):r for r in csv.DictReader(f)}
    if len(target)!=700:raise AssertionError('Wrong analysis row count')
    compared=0
    for row in independent_rows:
        match=target[row['condition'],row['seed'],row['horizon']]
        for key,value in row.items():
            if key in ('condition','seed','horizon'):continue
            close('seed_metric:'+key,value,float(match[key]),atol=2e-12,rtol=2e-12);compared+=1
        if row['condition'] in NEW and any(match.get(k) for k in ('q_A_tail','q_B_tail')):raise AssertionError('Invented K1 oxygen probability')
    idx=np.random.default_rng(20260920).integers(0,50,(2000,50))
    summary=json.loads((analysis/'summary.json').read_text());summary_compares=0
    by={}
    for condition in NEW+OLD:
        for horizon in WINDOWS:
            rows=sorted([r for r in independent_rows if r['condition']==condition and r['horizon']==horizon],key=lambda r:r['seed'])
            block={k:np.array([r[k] for r in rows]) for k in rows[0] if k not in ('condition','seed','horizon','window_first_state','window_last_state')}
            by[condition,horizon]=block
            for k,x in block.items():
                expected=np.r_[x.mean(),np.quantile(x[idx].mean(1),[.025,.975])]
                actual=summary[condition]['windows'][str(horizon)]['metrics'][k]
                close('summary_intervals',[actual['mean'],actual['lo'],actual['hi']],expected,atol=3e-12,rtol=3e-12);summary_compares+=3
            for kind in ('mixture','map'):
                x=np.stack([block[kind+'_difference_'+c+'_tail'] for c in 'CGM'],axis=-1)
                expected=float(np.linalg.norm(x.mean(0)))
                actual=summary[condition]['windows'][str(horizon)]['metrics'][kind+'_norm_of_across_seed_time_mean_difference']
                close('norm_after_seed_averaging',actual['mean'],expected,atol=3e-12,rtol=3e-12);summary_compares+=1
                if actual.get('lo') is not None or actual.get('hi') is not None:raise AssertionError('Invalid percentile interval for nonregular norm at zero')
    contrasts=json.loads((analysis/'paired_contrasts.json').read_text());contrast_compares=0
    for name,contrast in contrasts.items():
        for horizon in WINDOWS:
            plus=by[contrast['plus'],horizon];minus=by[contrast['minus'],horizon]
            for k,actual in contrast['windows'][str(horizon)].items():
                x=plus[k]-minus[k];expected=np.r_[x.mean(),np.quantile(x[idx].mean(1),[.025,.975])]
                close('paired_contrast_intervals',[actual['mean'],actual['lo'],actual['hi']],expected,atol=3e-12,rtol=3e-12);contrast_compares+=3
    forecast=json.loads((HERE/'STATIONARY_PREDICTIONS.json').read_text());forecast_comparisons=[]
    for label,condition in [('information','G_plain_info'),('moment','G_plain_moment')]:
        f=forecast['conditions'][label]
        for horizon in WINDOWS:
            observed=summary[condition]['windows'][str(horizon)]['metrics']
            for key,prediction in [('mixture_difference_'+c+'_tail',f['expected_mean_gap'][i]) for i,c in enumerate('CGM')]+[('mse_A_tail',f['gas_expected_mse']),('mse_B_tail',f['mass_expected_mse'])]:
                m=observed[key];forecast_comparisons.append(dict(condition=condition,horizon=horizon,metric=key,prediction=prediction,mean=m['mean'],lo=m['lo'],hi=m['hi'],inside_pointwise_interval=bool(m['lo']<=prediction<=m['hi'])))
    report=dict(status='passed',elapsed_seconds=time.monotonic()-start,checks=checks,coverage=coverage,max_absolute_discrepancies=errors,seed_scalar_comparisons=compared,summary_scalar_comparisons=summary_compares,paired_contrast_scalar_comparisons=contrast_compares,sequential_sample_plan='Predeclared before production: first seed in first/last chunk of first seed batch of each arm; held-out scores for all five seeds of those chunks.',sequential_sample_files=sample_plan,forecast_comparisons=forecast_comparisons,forecast_qualification='Pointwise intervals are correlated; forecast agreement is diagnostic, not an acceptance requirement. No expected curve ordering asserted.',raw_manifest_sha256=sha(raw/'manifest.json'),candidate_manifest_sha256=sha(candidate_raw/'manifest.json'),independence='All expected equations and reductions constructed here in NumPy/SciPy; no producer gate/score/deposit or analysis helper used. Production setup JSON supplies frozen parameters. Archived candidates independently re-reduced; their engine was audited in the preceding package.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    columns=list(dict.fromkeys(k for r in independent_rows for k in r))
    with out.with_name('INDEPENDENT_SEED_METRICS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(independent_rows)
    print(json.dumps({k:report[k] for k in ('status','elapsed_seconds','checks','seed_scalar_comparisons','summary_scalar_comparisons','paired_contrast_scalar_comparisons')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--candidate-raw',type=Path,required=True)
    parser.add_argument('--analysis',type=Path,default=HERE/'analysis');parser.add_argument('--output',type=Path,default=HERE/'INDEPENDENT_RAW_AUDIT.json')
    a=parser.parse_args();audit(a.raw,a.candidate_raw,a.analysis,a.output)
