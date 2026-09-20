#!/usr/bin/env python3
"""Independent full-data integrity, recurrence, pairing and archive audit.

Every stored chunk is checked. Independent NumPy equation replay checks selected
chunks, declared by position before reading outcomes. JAX is used only to
reconstruct the recorded PRNG samples for those selected chunks; no production
step, score, analysis or plotting function is called by this audit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from scipy.special import logsumexp, softmax
from independent_tests import numpy_step


def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def config(meta,prehistory=False):
    return SimpleNamespace(Pi0=np.asarray(meta['Pi0']),h0=np.asarray(meta['h0']),H=np.asarray(meta['H']),sigma_o=meta['sigma_o'],omega=meta['omega'],alpha_m=.02 if prehistory else meta['alpha_m'],fuse_nets=True if prehistory else meta['fuse_nets'],jitter=0.,rho_L=0.,Pi0_agent=None,h0_agent=None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--archive',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();base=args.run; started=time.monotonic()
    records=json.loads((base/'manifest.json').read_text())
    protocol=json.loads((base/'protocol.json').read_text())
    metas={name:json.loads((base/name/'setup.json').read_text()) for name in sorted({r['condition'] for r in records})}
    maxima={};checks={};filehashes={};precisions={};unique_raw=set();last={};index={};coverage={};selected=[];rngchecks=[]
    def close(label,a,b,atol=2e-9,rtol=2e-11):
        a=np.asarray(a);b=np.asarray(b);err=float(np.max(np.abs(a-b),initial=0))
        maxima[label]=max(maxima.get(label,0.),err);checks[label]=checks.get(label,0)+1
        if not np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=False):raise AssertionError(f'{label}: {err}')
    def exact(label,a,b):
        checks[label]=checks.get(label,0)+1
        if not np.array_equal(a,b):raise AssertionError(label)
    def verify_file(relative,expected):
        if relative not in filehashes:filehashes[relative]=sha(base/relative)
        if filehashes[relative]!=expected:raise AssertionError('Hash '+relative)
    def precision(r):
        p=r['precision_file']
        if p not in precisions:
            verify_file(p,r['precision_sha256'])
            with np.load(base/p) as z:P=z['Pi_history']
            if P.dtype!=np.float64 or not np.isfinite(P).all():raise AssertionError('Precision dtype/finite')
            smallest=float(np.linalg.eigvalsh(P).min())
            if smallest<=0:raise AssertionError('Nonpositive precision')
            precisions[p]=P
        return precisions[p]
    archive={}
    mapping={'N00':'network_all_social_off','N10':'network_hypothesis_off','N11':'network_complete'}
    if args.archive:
        for name,old in mapping.items():archive[name]=np.load(args.archive/(old+'.npz'))
    ordered=sorted(records,key=lambda r:(r['condition'],r['seeds'][0],r['start_step']))
    selections={}
    for name in metas:
        rr=[r for r in ordered if r['condition']==name and r['seeds'][0]==min(metas[name]['seeds'])]
        starts=sorted({r['start_step'] for r in rr})
        wanted={starts[0],starts[-1]}
        if name.startswith('N'):wanted.add(80)
        if name=='S_enable_weights':wanted.add(1500)
        selections[name]=wanted
    for count,r in enumerate(ordered):
        name=r['condition'];meta=metas[name];t=r['start_step'];T=r['steps'];ss=r['seeds'];n=meta['N']
        pre=name.startswith('N') and t<80
        cfg=config(meta,pre)
        if name=='S_enable_weights' and t<1500:cfg.alpha_m=0.
        W=np.asarray(meta['W']);att=np.ones((n,3)) if pre else np.asarray(meta['attention'])
        phi=np.array([1.,1.,-1.]) if pre else np.array([1.,1.,1.])
        verify_file(r['file'],r['sha256']);unique_raw.add(r['file'])
        with np.load(base/r['file']) as z:raw={k:z[k] for k in z.files}
        index[name,tuple(ss),t]=r
        if arrsha(raw['obs'])!=r['observation_sha256']:raise AssertionError('Observation hash')
        for k,v in raw.items():
            if not np.isfinite(v).all():raise AssertionError('Nonfinite '+k)
            if 'key' not in k and v.dtype!=np.float64:raise AssertionError('Notfloat64 '+k)
        P=precision(r)
        close('q_softmax',raw['q'],softmax(raw['L'],axis=-1),atol=3e-14,rtol=3e-14)
        close('q_sum',raw['q'].sum(-1),1,atol=3e-15,rtol=0)
        previous=np.concatenate((raw['initial_L'][:,None],raw['L'][:,:-1]),axis=1)
        pooled=(1-cfg.alpha_m)*previous+cfg.alpha_m*np.einsum('ij,btjk->btik',W,previous)
        close('L_score_recurrence',raw['L'],pooled+raw['ell'],atol=5e-9,rtol=1e-13)
        dz=raw['L'][...,1]-raw['L'][...,0];oldz=previous[...,1]-previous[...,0]
        recovered=dz-(1-cfg.alpha_m)*oldz-cfg.alpha_m*np.einsum('ij,btj->bti',W,oldz)
        close('log_odds_source',recovered,raw['ell'][...,1]-raw['ell'][...,0],atol=5e-9,rtol=1e-11)
        mix=np.einsum('btnk,btnkd->btnd',raw['q'],raw['mu_conditional'])
        close('observable_mixture',raw['mu_mixture'],mix,atol=2e-13,rtol=2e-13)
        close('MSE_noiseless_full',raw['mse_full'],np.mean((mix-phi)**2,axis=-1),atol=2e-13,rtol=2e-13)
        selected_mean=np.take_along_axis(raw['mu_conditional'],raw['L'].argmax(-1)[...,None,None],axis=-2)[...,0,:]
        close('MAP_observable',raw['mu_map'],selected_mean,atol=2e-13,rtol=2e-13)
        exact('initial_precision_deterministic',raw['initial_Pi'],np.broadcast_to(raw['initial_Pi'][:1],raw['initial_Pi'].shape))
        close('final_precision_history',raw['final_Pi'],np.broadcast_to(P[-1],raw['final_Pi'].shape),atol=0,rtol=0)
        Pprev=np.concatenate([raw['initial_Pi'][:1],P[:-1]],axis=0)
        Pbar=np.einsum('ij,tjkab->tikab',W,Pprev) if cfg.fuse_nets else Pprev
        J=np.einsum('ia,ni,ib->nab',cfg.H,att,cfg.H)/cfg.sigma_o**2
        close('precision_update',P,(1-cfg.omega)*cfg.Pi0+cfg.omega*(Pbar+J[None,:,None]),atol=8e-13,rtol=8e-14)
        exact('last_L',raw['L'][:,-1],raw['final_L'])
        group=(name,tuple(ss))
        if group in last:
            old=last[group]
            if t!=old['end']:raise AssertionError('Missingtime')
            for k in ('Pi','h','L'):exact('checkpoint_'+k,raw['initial_'+k],old[k])
            if name.startswith('N') and t==80:
                expected=np.array([[0,s+1000] for s in ss],dtype=np.uint32)
                exact('network_phase2_key_reset',raw['initial_key'],expected)
            else:exact('checkpoint_key',raw['initial_key'],old['key'])
        elif t!=0:raise AssertionError('Missingstart')
        else:
            close('initial_reference_precision',raw['initial_Pi'],np.broadcast_to(cfg.Pi0,raw['initial_Pi'].shape),atol=0,rtol=0)
            close('initial_reference_information',raw['initial_h'],np.broadcast_to(cfg.h0,raw['initial_h'].shape),atol=0,rtol=0)
            exact('initial_log_weights_zero',raw['initial_L'],np.zeros_like(raw['initial_L']))
            exact('initial_seed_keys',raw['initial_key'],np.array([[0,s] for s in ss],dtype=np.uint32))
        last[group]=dict(end=t+T,**{k:raw['final_'+k] for k in ('Pi','h','L','key')})
        coverage[group]=coverage.get(group,0)+T
        if name in archive:
            old=archive[name]
            for new,previous_name in [('L','L'),('q','q'),('mu_map','mu_map'),('mu_mixture','mu_mixture'),('mse_full','mse_obs'),('nll_full','heldout_nll')]:
                reference=old[previous_name][ss,t:t+T]
                if new in ('mu_map','mu_mixture'):reference=reference[...,1:]
                close('archive_'+name+'_'+new,raw[new],reference,atol=3e-10,rtol=3e-10)
            if t+T==600:
                for k in ('Pi','h','L'):close('archive_'+name+'_final_'+k,raw['final_'+k],old['final_'+k][ss],atol=3e-10,rtol=3e-10)
        if ss[0]==min(meta['seeds']) and t in selections[name]:selected.append((r,raw,cfg,W,att,phi,P))
        if count%100==0:print('integrity',count+1,'/',len(ordered),flush=True)
    for group,total in coverage.items():
        expected=600 if group[0].startswith('N') else 5000
        if total!=expected:raise AssertionError('Coverage '+str(group))
    for name in metas:
        actual=sorted(s for (condition,ss) in coverage if condition==name for s in ss)
        if actual!=sorted(metas[name]['seeds']):raise AssertionError('Seedcoverage '+name)
    # Exact channel invariants over every matching chunk, not only examples.
    pairs=[('N00','N01'),('N10','N11'),('C00','C01'),('C10','C11'),('S_keep_private','S_enable_weights')]
    for a,b in pairs:
        for (name,ss,t),ra in index.items():
            if name!=a:continue
            rb=index[b,ss,t]
            if ra['observation_sha256']!=rb['observation_sha256'] or ra['evaluation_observation_sha256']!=rb['evaluation_observation_sha256']:raise AssertionError('Samplepair')
            with np.load(base/ra['file']) as za,np.load(base/rb['file']) as zb:
                for k in ('obs','ell','mu_conditional','initial_Pi','initial_h','initial_key','final_Pi','final_h','final_key'):
                    exact('channel_pair_'+k,za[k],zb[k])
            exact('channel_pair_precision',precision(ra),precision(rb))
    # All camp arms share samples, including diagnostics. Comparing complete sample
    # hashes also checks varying omega/reference does not alter RNG consumption.
    for (name,ss,t),r in index.items():
        if name.startswith('C') or name.startswith('S'):
            anchor=index['C10',ss,t]
            if any(r[k]!=anchor[k] for k in ('observation_sha256','evaluation_observation_sha256')):raise AssertionError('Allcamp samplepair')
        if name.startswith('N'):
            anchor=index['N11',ss,t]
            if any(r[k]!=anchor[k] for k in ('observation_sha256','evaluation_observation_sha256')):raise AssertionError('Allnetwork samplepair')
    # Explicitly independent equation replay: first seed of predeclared first/last
    # chunks and intervention boundary, with separately reconstructed sample keys.
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    def reconstruct(keys,phis,sigma):
        def one(key):
            def step(key,phi):
                evaluation=phi+sigma*jax.random.normal(jax.random.fold_in(key,314159),(N,4,3),dtype=jnp.float64)
                nextkey,sub=jax.random.split(key)
                obs=jax.vmap(lambda k:phi+sigma*jax.random.normal(k,(3,)))(jax.random.split(sub,N))
                return nextkey,(obs,evaluation)
            return jax.lax.scan(step,key,phis)
        return jax.jit(jax.vmap(one))(keys)
    replay_updates=0
    for r,raw,cfg,W,att,phi,Pstored in selected:
        N=raw['obs'].shape[2];T=r['steps']
        keys,(observations,evaluations)=reconstruct(jnp.asarray(raw['initial_key']),jnp.asarray(np.broadcast_to(phi,(T,3))),cfg.sigma_o)
        observations=np.asarray(observations);evaluations=np.asarray(evaluations)
        exact('reconstructed_training',observations,raw['obs'])
        exact('reconstructed_final_key',keys,raw['final_key'])
        if arrsha(evaluations)!=r['evaluation_observation_sha256']:raise AssertionError('Evaluation samplehash')
        P,h,L=[raw['initial_'+k][0].copy() for k in ('Pi','h','L')]
        for ti in range(T):
            ref=numpy_step(P,h,L,raw['obs'][0,ti],cfg,W,att)
            for k in ('ell','L','q','mu_conditional','mu_mixture'):
                close('numpy_replay_'+k,ref[k],raw[k][0,ti],atol=5e-9,rtol=5e-11)
            close('numpy_replay_precision',ref['Pi'],Pstored[ti],atol=8e-12,rtol=8e-12)
            P,h,L=ref['Pi'],ref['h'],ref['L'];replay_updates+=1
            if ti in (0,T-1):
                C=np.linalg.solve(P,np.broadcast_to(np.eye(4),P.shape))[...,1:,1:]+cfg.sigma_o**2*np.eye(3)
                residual=evaluations[0,ti,:, :,None,:]-ref['mu_conditional'][:,None]
                quadratic=np.einsum('nrki,nrkij,nrkj->nrk',residual,np.broadcast_to(np.linalg.inv(C)[:,None],(N,4,2,3,3)),residual)
                logp=-.5*(3*np.log(2*np.pi)+np.linalg.slogdet(C)[1][:,None]+quadratic)
                nll=-logsumexp((L-logsumexp(L,axis=-1,keepdims=True))[:,None]+logp,axis=-1).mean(-1)
                close('numpy_replay_mixture_NLL',nll,raw['nll_full'][0,ti],atol=5e-10,rtol=5e-10)
        for k,v in [('Pi',P),('h',h),('L',L)]:close('numpy_replay_final_'+k,v,raw['final_'+k][0],atol=5e-9,rtol=5e-11)
        rngchecks.append(dict(condition=r['condition'],start_step=r['start_step'],steps=T,seeds=r['seeds']))
        print('independent replay',r['condition'],r['start_step'],flush=True)
    result=dict(status='passed',elapsed_seconds=time.monotonic()-started,manifest_records=len(records),unique_raw_files=len(unique_raw),hashed_files=len(filehashes),logical_conditions=len(metas),seed_condition_runs=sum(len(m['seeds']) for m in metas.values()),independent_replay_updates=replay_updates,independent_replay_chunks=rngchecks,independent_replay_selection='First and last physical chunks of each condition, first seed; plus network round80 and enable-weights round1500. RNG hashes checked all5 seeds in each selected chunk.',checks=checks,maximum_absolute_errors=maxima,archive_regression='all three archived network arms, all50seeds, all600updates and finalstates' if archive else 'not requested',scope='All raw hashes and perround recurrence/probability/MSE/channel-pair checks; independent NumPy replay subset as declared; no production numerical helpers.')
    output=args.output or base/'INDEPENDENT_RAW_AUDIT.json';output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('independent_replay_chunks','checks','maximum_absolute_errors')},indent=2))


if __name__=='__main__':main()
