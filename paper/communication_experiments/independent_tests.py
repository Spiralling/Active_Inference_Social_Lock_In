#!/usr/bin/env python3
"""Adversarial validation independent of the production wrapper's readouts.

The NumPy step below implements equations directly, without calling production
scoring/fusion/update helpers. Test-only seeds 900000..900009 are separate from
confirmation seeds. These deterministic validation runs are not scientific arms.
"""
from __future__ import annotations
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest
import numpy as np
from scipy.special import logsumexp, softmax

HERE=Path(__file__).resolve().parent


def numpy_step(Pi,h,L,obs,cfg,W,attention):
    """Reference implementation for one seed. Returns all update stages."""
    P=np.asarray(Pi); h=np.asarray(h); L=np.asarray(L); W=np.asarray(W)
    H=np.asarray(cfg.H); attention=np.asarray(attention); obs=np.asarray(obs)
    Pbar=np.einsum('ij,jkab->ikab',W,P) if cfg.fuse_nets else P.copy()
    hbar=np.einsum('ij,jkd->ikd',W,h) if cfg.fuse_nets else h.copy()
    Lbar=(1-cfg.alpha_m)*L+cfg.alpha_m*W@L
    cov=np.linalg.solve(Pbar,np.broadcast_to(np.eye(P.shape[-1]),Pbar.shape))
    mean=np.linalg.solve(Pbar,hbar[...,None])[...,0]
    Hs=np.sqrt(attention)[:,:,None]*H
    yy=np.sqrt(attention)*obs
    V=np.einsum('nia,nkab,njb->nkij',Hs,cov,Hs)+cfg.sigma_o**2*np.eye(H.shape[0])
    V=(V+V.swapaxes(-1,-2))/2+cfg.jitter*np.eye(H.shape[0])
    residual=yy[:,None]-np.einsum('nia,nka->nki',Hs,mean)
    quadratic=np.einsum('nki,nki->nk',residual,np.linalg.solve(V,residual[...,None])[...,0])
    ell=-.5*(H.shape[0]*np.log(2*np.pi)+np.linalg.slogdet(V)[1]+quadratic)
    J=np.einsum('ia,ni,ib->nab',H,attention,H)/cfg.sigma_o**2
    j=np.einsum('ia,ni->na',H,attention*obs)/cfg.sigma_o**2
    Pplus=Pbar+J[:,None]; hplus=hbar+j[:,None]
    P0=np.asarray(cfg.Pi0 if cfg.Pi0_agent is None else cfg.Pi0_agent)
    h0=np.asarray(cfg.h0 if cfg.h0_agent is None else cfg.h0_agent)
    Pnext=(1-cfg.omega)*P0+cfg.omega*Pplus
    hnext=(1-cfg.omega)*h0+cfg.omega*hplus
    Lnext=(1-cfg.rho_L)*(Lbar+ell)
    means=np.linalg.solve(Pnext,hnext[...,None])[...,0]
    q=softmax(Lnext,axis=-1)
    return dict(Pi=Pnext,h=hnext,L=Lnext,ell=ell,q=q,
                mu_conditional=means[...,1:],
                mu_mixture=np.einsum('nk,nkd->nd',q,means[...,1:]),
                stage_pre=mean,stage_postlearning=np.linalg.solve(Pplus,hplus[...,None])[...,0],
                stage_postmemory=means,J=J,j=j,Pi_bar=Pbar,h_bar=hbar)


def import_runner():
    sys.path.insert(0,str(HERE))
    import run
    return run


def camp(name='test',**updates):
    result=dict(name=name,family='camp',fuse_nets=True,alpha_m=0.,omega=.3,sigma_o=.5,beta=.05,gasvar_matched=False)
    result.update(updates)
    return result


class TestIndependentImplementation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner=import_runner()
        import jax
        import jax.numpy as jnp
        from src.structural import rival
        cls.jax=jax;cls.jnp=jnp;cls.rv=rival
        cls.stash={}

    def setup_run(self,condition=None,T=11,seed=900000,attention_override=None):
        condition=camp() if condition is None else condition
        cfg,W,A,attention,phi,metadata=self.runner.make_setup(condition)
        if attention_override is not None: attention=attention_override
        phi=np.broadcast_to(np.array([0.,1.,1.,1.]),(T,4))
        f=self.runner.make_batch(cfg,W,A,attention,draws=4)
        result=f(phi,[seed])
        return cfg,np.asarray(W),np.asarray(A),np.asarray(attention),phi,result

    def test_references_against_independent_compilation(self):
        from stationary_predictions import references
        for matched in (False,True):
            cfg,*_=self.runner.make_setup(camp(gasvar_matched=matched))
            P,h,C=references(matched)
            np.testing.assert_allclose(cfg.Pi0,P,atol=2e-14,rtol=2e-14)
            np.testing.assert_allclose(cfg.h0,h,atol=2e-14,rtol=2e-14)
            np.testing.assert_allclose(np.linalg.solve(P,h[...,None])[...,0],[[0,1,1,-1],[0,1,1,1]],atol=3e-14)
            np.testing.assert_allclose(C[:,2,2],[.8648 if matched else .82,.8648],atol=2e-15)
            self.assertGreater(np.linalg.eigvalsh(P).min(),0)

    def test_weight_matrix_semantics(self):
        for beta in (0.,.05,.5):
            cfg,W,*_=self.runner.make_setup(camp(beta=beta));W=np.asarray(W)
            self.assertTrue(np.all(W>=0));np.testing.assert_allclose(W.sum(0),1,atol=1e-15)
            np.testing.assert_allclose(W.sum(1),1,atol=1e-15);np.testing.assert_array_equal(W,W.T)
            np.testing.assert_allclose(W[:8,8:].sum(1),beta,atol=1e-15)
            np.testing.assert_allclose(W[:8,:8].sum(1),1-beta,atol=1e-15)
            if beta==.5:np.testing.assert_array_equal(W,np.ones((16,16))/16)
            if beta==0:self.assertEqual(np.count_nonzero(W),128)

    def test_instrumentation_equals_original_engine(self):
        cfg,W,A,attention,phis,result=self.setup_run()
        jax,jnp,rv=self.jax,self.jnp,self.rv
        P,h,L=rv.init_uniform(cfg,len(W))
        def step(carry,phi):
            nxt,_=rv._step(carry,phi,cfg,jnp.asarray(W),jnp.asarray(A),jnp.asarray(attention),track_means=False)
            return nxt,nxt[:3]
        final,history=jax.jit(lambda:jax.lax.scan(step,(P,h,L,jax.random.PRNGKey(900000)),jnp.asarray(phis)))()
        for name,value in zip(('final_Pi','final_h','final_L','final_key'),final):
            np.testing.assert_array_equal(result[name][0],np.asarray(value),err_msg=name)
        np.testing.assert_array_equal(result['L'][0],np.asarray(history[2]))

    def test_one_step_equations_and_prequential_order(self):
        cfg,W,A,att,phi,r=self.setup_run(T=7)
        P,h,L=(np.asarray(x) for x in self.rv.init_uniform(cfg,len(W)))
        worst=0.
        for t in range(7):
            ref=numpy_step(P,h,L,r['obs'][0,t],cfg,W,att)
            if t==3:
                np.savez(HERE/'VALIDATION_UPDATE_STAGES.npz',
                         entering_means=np.linalg.solve(P,h[...,None])[...,0],
                         postsharing_means=ref['stage_pre'],
                         postlearning_means=ref['stage_postlearning'],
                         postmemory_means=ref['stage_postmemory'],
                         Pi_entering=P,h_entering=h,L_entering=L,
                         Pi_postsharing=ref['Pi_bar'],h_postsharing=ref['h_bar'],
                         Pi_postmemory=ref['Pi'],h_postmemory=ref['h'],
                         observation=r['obs'][0,t],ell=ref['ell'],
                         round_index=np.array(t),validation_seed=np.array(900000))
            for name in ('L','q','ell','mu_conditional','mu_mixture'):
                np.testing.assert_allclose(r[name][0,t],ref[name],rtol=3e-12,atol=3e-12,err_msg=name)
            P,h,L=ref['Pi'],ref['h'],ref['L']
        np.testing.assert_allclose(r['final_Pi'][0],P,atol=5e-13,rtol=5e-13)
        np.testing.assert_allclose(r['final_h'][0],h,atol=5e-13,rtol=5e-13)
        # A learning-then-scoring bug would score the same datum under Pplus/hplus;
        # seven heterogeneous noisy updates above are compared to prelearning ell.

    def test_channel_pairing(self):
        runs={}
        for fuse in (False,True):
            for alpha in (0.,.02):
                runs[fuse,alpha]=self.setup_run(camp(fuse_nets=fuse,alpha_m=alpha),T=15)[-1]
        for fuse in (False,True):
            a,b=runs[fuse,0.],runs[fuse,.02]
            for name in ('obs','ell','mu_conditional','final_Pi','final_h','final_key','Pi_history'):
                np.testing.assert_array_equal(a[name],b[name],err_msg=name)
            self.assertGreater(np.max(np.abs(a['L']-b['L'])),1e-5)
        for value in runs.values():np.testing.assert_array_equal(value['obs'],runs[False,0.]['obs'])

    def test_exact_restart_and_branch(self):
        cfg,W,A,att,phi,r=self.setup_run(T=23)
        f=self.runner.make_batch(cfg,W,A,att,draws=4)
        first=f(phi[:9],[900000]);state=self.runner.final_state(first)
        resumed=f(phi[9:],[900000],initial=state)
        for name in ('L','q','ell','obs','mu_conditional','mu_mixture','mse_full','nll_full'):
            np.testing.assert_array_equal(resumed[name],r[name][:,9:],err_msg=name)
        for name in ('final_Pi','final_h','final_L','final_key'):np.testing.assert_array_equal(resumed[name],r[name])
        enabled=dataclasses.replace(cfg,alpha_m=.02)
        other=self.runner.make_batch(enabled,W,A,att,draws=4)(phi[9:],[900000],initial=state)
        for name in ('ell','obs','mu_conditional','final_Pi','final_h','final_key','Pi_history'):
            np.testing.assert_array_equal(other[name],resumed[name],err_msg=name)
        self.assertGreater(np.max(np.abs(other['L']-resumed['L'])),1e-5)

    def test_fractional_attention_and_zero_attention(self):
        cfg,W,A,att,phi,r=self.setup_run(T=5,attention_override=np.tile([1.,.01,.01],(16,1)))
        P,h,L=(np.asarray(x) for x in self.rv.init_uniform(cfg,len(W)))
        for t in range(5):
            ref=numpy_step(P,h,L,r['obs'][0,t],cfg,W,att)
            np.testing.assert_allclose(r['ell'][0,t],ref['ell'],atol=2e-12,rtol=2e-12)
            np.testing.assert_allclose(r['mu_conditional'][0,t],ref['mu_conditional'],atol=2e-12,rtol=2e-12)
            P,h,L=ref['Pi'],ref['h'],ref['L']
        raw_noise_cov=cfg.sigma_o**2*np.eye(3)
        D=np.diag(att[0]);H=np.asarray(cfg.H)
        innovation=H.T@D@raw_noise_cov@D@H/cfg.sigma_o**4
        np.testing.assert_allclose(innovation,H.T@np.diag(att[0]**2)@H/cfg.sigma_o**2,atol=1e-15)
        self.assertGreater(np.max(np.abs(innovation-ref['J'][0])),1e-3)
        cfg,W,A,att,phi,r=self.setup_run(T=5,attention_override=np.zeros((16,3)))
        np.testing.assert_allclose(r['q'],.5,atol=2e-15)
        np.testing.assert_array_equal(r['ell'][...,0],r['ell'][...,1])
        np.testing.assert_allclose(r['final_Pi'][0],np.broadcast_to(cfg.Pi0,(16,2,4,4)),atol=1e-14)
        np.testing.assert_allclose(r['final_h'][0],np.broadcast_to(cfg.h0,(16,2,4)),atol=1e-14)

    def test_rng_and_independent_replicate_prediction(self):
        from scipy.stats import multivariate_normal
        cfg,W,A,att,phis,r=self.setup_run(T=4)
        P,h,L=(np.asarray(x) for x in self.rv.init_uniform(cfg,len(W)))
        key=self.jax.random.PRNGKey(900000)
        for t,phi in enumerate(phis):
            evalkey=self.jax.random.fold_in(key,314159)
            test_obs=np.asarray(self.jax.jit(lambda k:self.jnp.asarray(phi[1:])+cfg.sigma_o*self.jax.random.normal(k,(16,4,3),dtype=self.jnp.float64))(evalkey))
            key,sk=self.jax.random.split(key)
            keys=self.jax.random.split(sk,16)
            train=np.asarray(self.jax.jit(self.jax.vmap(lambda k:self.jnp.asarray(phi[1:])+cfg.sigma_o*self.jax.random.normal(k,(3,))))(keys))
            np.testing.assert_array_equal(train,r['obs'][0,t])
            ref=numpy_step(P,h,L,train,cfg,W,att)
            P,h,L=ref['Pi'],ref['h'],ref['L']
            means=ref['mu_conditional'];pred=ref['mu_mixture']
            mse=np.mean((pred-phi[1:])**2,axis=-1)
            nll=[]
            cov=np.linalg.solve(P,np.broadcast_to(np.eye(4),P.shape))[...,1:,1:]+cfg.sigma_o**2*np.eye(3)
            for i in range(16):
                logp=np.stack([multivariate_normal.logpdf(test_obs[i],mean=means[i,k],cov=cov[i,k]) for k in range(2)],axis=-1)
                nll.append(-logsumexp(L[i]-logsumexp(L[i])+logp,axis=-1).mean())
            np.testing.assert_allclose(r['mse_full'][0,t],mse,atol=3e-12,rtol=3e-12)
            np.testing.assert_allclose(r['nll_full'][0,t],nll,atol=3e-12,rtol=3e-12)
        np.testing.assert_array_equal(r['final_key'][0],key)

    def test_candidate_permutation(self):
        cfg,W,A,att,phi,r=self.setup_run(T=9)
        perm=dataclasses.replace(cfg,Pi0=cfg.Pi0[::-1],h0=cfg.h0[::-1])
        other=self.runner.make_batch(perm,W,A,att,draws=4)(phi,[900000])
        for name in ('L','q','ell'):np.testing.assert_allclose(other[name][...,::-1],r[name],atol=3e-13,rtol=3e-13)
        for name in ('mu_mixture','mse_full','nll_full'):np.testing.assert_allclose(other[name],r[name],atol=3e-13,rtol=3e-13)

    def test_matched_gas_beta_zero(self):
        cfg,W,A,att,phi,r=self.setup_run(camp(beta=0.,gasvar_matched=True),T=15)
        np.testing.assert_allclose(r['ell'][:,:,:8,0],r['ell'][:,:,:8,1],atol=5e-13,rtol=5e-13)

    def test_exact_log_odds_recurrence(self):
        cfg,W,A,att,phi,r=self.setup_run(camp(alpha_m=.02),T=25)
        z=r['L'][...,1]-r['L'][...,0]; previous=np.concatenate([np.zeros_like(z[:,:1]),z[:,:-1]],axis=1)
        source=z-(1-cfg.alpha_m)*previous-cfg.alpha_m*np.einsum('ij,btj->bti',W,previous)
        np.testing.assert_allclose(source,r['ell'][...,1]-r['ell'][...,0],atol=8e-14,rtol=3e-12)


if __name__=='__main__':
    started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestIndependentImplementation))
    output=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.monotonic()-started,test_seeds='900000..900009; never confirmation seeds',numpy_reference='Independent affine update and full multivariate Gaussian pre-learning density; no production numerical helpers')
    (HERE/'INDEPENDENT_TESTS.json').write_text(json.dumps(output,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
