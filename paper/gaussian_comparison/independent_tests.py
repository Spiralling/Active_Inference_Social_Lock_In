#!/usr/bin/env python3
"""Independent mathematical tests for matched K=1 Gaussian comparison.

No production gate/score/deposit helpers are used to construct expected results.
Validation seeds 900100..900109 are disjoint from the paired scientific streams.
"""
from __future__ import annotations
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest
import numpy as np
from scipy.stats import multivariate_normal

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from stationary_predictions import common_reference,marginal_candidates,stationary


def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def numpy_gate(P,h,W,nu):
    P=np.asarray(P);h=np.asarray(h);W=np.asarray(W)
    if nu is None:return W.copy()
    mu=np.linalg.solve(P[:,0],h[:,0,:,None])[...,0][:,1:]
    variance=1/np.diagonal(P[:,0],axis1=-2,axis2=-1)[:,1:]
    z2=np.sum((mu[:,None]-mu[None,:])**2/(variance[:,None]+variance[None,:]),axis=-1)
    gate=(nu+1)/(nu+z2)
    weights=W*gate
    return weights/weights.sum(1,keepdims=True)


def numpy_step(P,h,L,obs,cfg,W,attention,nu):
    Wn=numpy_gate(P,h,W,nu)
    pp=np.einsum('ij,jkab->ikab',Wn,P);hh=np.einsum('ij,jkd->ikd',Wn,h)
    cov=np.linalg.inv(pp);mu=np.linalg.solve(pp,hh[...,None])[...,0]
    H=np.asarray(cfg.H);att=np.asarray(attention);obs=np.asarray(obs)
    Hs=np.sqrt(att)[:,:,None]*H;yy=np.sqrt(att)*obs
    V=np.einsum('nia,nkab,njb->nkij',Hs,cov,Hs)+cfg.sigma_o**2*np.eye(3)
    V=(V+V.swapaxes(-1,-2))/2+cfg.jitter*np.eye(3)
    residual=yy[:,None]-np.einsum('nia,nka->nki',Hs,mu)
    ell=-.5*(3*np.log(2*np.pi)+np.linalg.slogdet(V)[1]+np.einsum('nki,nki->nk',residual,np.linalg.solve(V,residual[...,None])[...,0]))
    J=np.einsum('ia,ni,ib->nab',H,att,H)/cfg.sigma_o**2
    j=np.einsum('ia,ni->na',H,att*obs)/cfg.sigma_o**2
    Pnext=(1-cfg.omega)*np.asarray(cfg.Pi0)+cfg.omega*(pp+J[:,None])
    hnext=(1-cfg.omega)*np.asarray(cfg.h0)+cfg.omega*(hh+j[:,None])
    Lnext=L+ell
    means=np.linalg.solve(Pnext,hnext[...,None])[...,0][...,1:]
    return dict(Pi=Pnext,h=hnext,L=Lnext,ell=ell,mu=means,weights=Wn,
                pre_mu=mu,pre_cov=cov,
                gate_crossweight=np.r_[Wn[:8,8:].sum(1),Wn[8:,:8].sum(1)])


class IndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner=import_file('gaussian_production_runner',HERE/'run.py')
        import jax
        import jax.numpy as jnp
        from src.structural import rival
        cls.jax=jax;cls.jnp=jnp;cls.rv=rival
        cls.cache={}

    def setup(self,name='G_gated_info',T=13,seed=900100):
        cfg,W,A,att,phi,meta=self.runner.make_setup(name)
        nu=.01 if '_gated_' in name else None
        phis=np.broadcast_to(np.asarray(phi),(T,4))
        key=(name,T,seed)
        if key not in self.cache:
            self.cache[key]=self.runner.make_batch(cfg,W,A,att,gate_nu=nu)(phis,[seed])
        return cfg,np.asarray(W),np.asarray(A),np.asarray(att),phis,self.cache[key],nu

    def test_marginalization_and_reference_compilation(self):
        means,cov=marginal_candidates()
        for name,kind in [('G_plain_info','information'),('G_gated_info','information'),('G_plain_moment','moment'),('G_gated_moment','moment')]:
            cfg,W,A,att,phi,meta=self.runner.make_setup(name)
            P,h=common_reference(kind)
            np.testing.assert_allclose(cfg.Pi0[0,1:,1:],P,atol=3e-14,rtol=3e-14)
            np.testing.assert_allclose(cfg.h0[0,1:],h,atol=3e-14,rtol=3e-14)
            np.testing.assert_array_equal(np.asarray(cfg.Pi0)[0,0,1:],0.)
            np.testing.assert_array_equal(np.asarray(cfg.Pi0)[0,1:,0],0.)
            self.assertEqual(float(cfg.Pi0[0,0,0]),4.)
            self.assertEqual(float(cfg.h0[0,0]),0.)
            self.assertIsNone(cfg.Pi0_agent);self.assertIsNone(cfg.h0_agent)
            self.assertIsNone(cfg.social_nu);self.assertIsNone(cfg.nu_m)
            self.assertEqual(cfg.n_candidates,1);self.assertEqual(cfg.alpha_m,0.)
            self.assertEqual(cfg.omega,.3);self.assertEqual(cfg.sigma_o,.5)
            self.assertGreater(np.linalg.eigvalsh(cfg.Pi0).min(),0.)
        # Information pooling must marginalize before pooling; a precision block
        # would condition on the hidden explanatory coordinate and be incorrect.
        from src.structural.bayesnet import LinearGaussianBN
        world=self.runner.wm.build_world() if hasattr(self.runner,'wm') else self.runner.cr.wm.build_world()
        fullP=np.asarray(world['Pi']);fullh=np.asarray(world['h'])
        schur=fullP[:,1:,1:]-fullP[:,1:,:1]@np.linalg.inv(fullP[:,:1,:1])@fullP[:,:1,1:]
        schurh=fullh[:,1:]-np.einsum('kia,kab,kb->ki',fullP[:,1:,:1],np.linalg.inv(fullP[:,:1,:1]),fullh[:,:1])
        np.testing.assert_allclose(schur,np.linalg.inv(cov),atol=3e-14)
        np.testing.assert_allclose(schurh,np.einsum('kij,kj->ki',schur,means),atol=3e-14)
        self.assertGreater(np.max(abs(schur-fullP[:,1:,1:])),.1)

    def test_moment_between_mean_covariance(self):
        _,cov=marginal_candidates();P,h=common_reference('moment')
        matched=np.linalg.inv(P);expected=cov.mean(0)+np.diag([0,0,1])
        np.testing.assert_allclose(matched,expected,atol=2e-15)
        np.testing.assert_allclose(np.linalg.solve(P,h),[1,1,0],atol=3e-15)
        self.assertGreater(abs(np.linalg.solve(*common_reference('information'))[0]-1),.3)

    def test_base_graph_and_weighted_gate_limit(self):
        cfg,W,A,att,phi,meta=self.runner.make_setup('G_gated_info');W=np.asarray(W)
        expected=np.block([[np.full((8,8),.95/8),np.full((8,8),.05/8)],
                           [np.full((8,8),.05/8),np.full((8,8),.95/8)]])
        np.testing.assert_array_equal(W,expected)
        np.testing.assert_allclose(W.sum(1),1,atol=2e-16,rtol=0)
        P,h,L=self.rv.init_uniform(cfg,16)
        for nu in (None,.01,1e30):
            actual=self.runner.effective_W(P,h,self.jnp.asarray(W),nu)
            np.testing.assert_allclose(actual,W,atol=3e-16,rtol=3e-16)
            np.testing.assert_allclose(np.asarray(actual)[:8,8:].sum(1),.05,atol=2e-16)

    def test_independent_gate_heterogeneity_and_large_nu(self):
        cfg,W,A,att,phi,r,nu=self.setup(T=5)
        P=np.array(r['final_Pi'][0]);h=np.array(r['final_h'][0])
        expected=numpy_gate(P,h,W,.01)
        actual=np.asarray(self.runner.effective_W(self.jnp.asarray(P),self.jnp.asarray(h),self.jnp.asarray(W),.01))
        np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=2e-14)
        np.testing.assert_allclose(actual.sum(1),1,atol=16*np.finfo(float).eps/(1-16*np.finfo(float).eps),rtol=0)
        self.assertTrue(np.all(actual>0))
        self.assertGreater(np.max(abs(actual-W)),.01)
        # Equal means with different covariances restore the original weighted graph.
        common=np.array([0,.7,1.2,-.3]);sameh=np.einsum('nkij,j->nki',P,common)
        equal=np.asarray(self.runner.effective_W(self.jnp.asarray(P),self.jnp.asarray(sameh),self.jnp.asarray(W),.01))
        np.testing.assert_allclose(equal,W,atol=3e-16,rtol=3e-16)
        large=np.asarray(self.runner.effective_W(self.jnp.asarray(P),self.jnp.asarray(h),self.jnp.asarray(W),1e30))
        np.testing.assert_allclose(large,W,atol=3e-16,rtol=3e-16)

    def test_hidden_coordinate_is_excluded_from_gate(self):
        cfg,W,A,att,phi,r,nu=self.setup(T=5)
        P=np.array(r['final_Pi'][0]);h=np.array(r['final_h'][0])
        first=np.asarray(self.runner.effective_W(self.jnp.asarray(P),self.jnp.asarray(h),self.jnp.asarray(W),nu))
        P[:,0,0,0]=np.linspace(.1,100,16);h[:,0,0]=P[:,0,0,0]*np.arange(16)*100
        second=np.asarray(self.runner.effective_W(self.jnp.asarray(P),self.jnp.asarray(h),self.jnp.asarray(W),nu))
        np.testing.assert_allclose(second,first,atol=0,rtol=0)

    def test_numpy_update_and_predeposit_scoring(self):
        for name in ('G_plain_info','G_gated_info','G_plain_moment','G_gated_moment'):
            cfg,W,A,att,phis,r,nu=self.setup(name,T=13)
            P,h,L=(np.asarray(x) for x in self.rv.init_uniform(cfg,16))
            for t in range(13):
                ref=numpy_step(P,h,L,r['obs'][0,t],cfg,W,att,nu)
                for output,reference in [('ell','ell'),('L','L'),('mu_conditional','mu'),('Pi_history','Pi'),('gate_crossweight','gate_crossweight')]:
                    np.testing.assert_allclose(r[output][0,t],ref[reference],atol=3e-11,rtol=3e-12,err_msg=f'{name}:{output}:{t}')
                P,h,L=ref['Pi'],ref['h'],ref['L']
            np.testing.assert_allclose(r['final_h'][0],h,atol=3e-12,rtol=3e-12)
            np.testing.assert_array_equal(r['q'],1.)
            np.testing.assert_array_equal(r['mu_mixture'],r['mu_conditional'][...,0,:])
            np.testing.assert_array_equal(r['mu_mixture'],r['mu_map'])

    def test_original_engine_is_the_state_transition(self):
        cfg,W,A,att,phis,r,nu=self.setup('G_gated_info',T=13)
        jax,jnp=self.jax,self.jnp
        P,h,L=self.rv.init_uniform(cfg,16)
        def reference_step(carry,phi):
            p,h,l,key=carry
            # Independent gate written in JAX solely to match compilation arithmetic.
            mu=jnp.linalg.solve(p[:,0],h[:,0,:,None])[...,0][:,1:]
            v=1/jnp.diagonal(p[:,0],axis1=-2,axis2=-1)[:,1:]
            z=jnp.sum((mu[:,None]-mu[None,:])**2/(v[:,None]+v[None,:]),axis=-1)
            weights=jnp.asarray(W)*((nu+1)/(nu+z));weights=weights/weights.sum(1,keepdims=True)
            nxt,_=self.rv._step(carry,phi,cfg,weights,jnp.asarray(A),jnp.asarray(att),track_means=False)
            return nxt,nxt[2]
        final,history=jax.jit(lambda:jax.lax.scan(reference_step,(P,h,L,jax.random.PRNGKey(900100)),jnp.asarray(phis)))()
        for name,value in zip(('final_Pi','final_h','final_L','final_key'),final):
            np.testing.assert_allclose(r[name][0],value,atol=3e-12,rtol=3e-12)
        np.testing.assert_allclose(r['L'][0],history,atol=3e-12,rtol=3e-12)

    def test_paired_observations_against_archived_candidate(self):
        archived=import_file('gaussian_validation_archived',HERE.parent/'communication_experiments/run.py')
        cfg,W,A,att,phi,meta=archived.make_setup('C10')
        phis=np.broadcast_to(np.asarray(phi),(13,4))
        candidate=archived.make_batch(cfg,W,A,att)(phis,[900100])
        for name in ('G_plain_info','G_gated_info','G_plain_moment','G_gated_moment'):
            result=self.setup(name,T=13)[5]
            for key in ('obs','eval_obs','final_key'):
                np.testing.assert_array_equal(result[key],candidate[key],err_msg=f'{name}:{key}')

    def test_exact_restart_every_history_and_state(self):
        for name in ('G_plain_info','G_gated_moment'):
            cfg,W,A,att,phis,r,nu=self.setup(name,T=37)
            f=self.runner.make_batch(cfg,self.jnp.asarray(W),self.jnp.asarray(A),self.jnp.asarray(att),gate_nu=nu)
            first=f(phis[:14],[900100]);initial={k:first['final_'+k] for k in ('Pi','h','L','key')}
            last=f(phis[14:],[900100],initial=initial)
            for key in ('L','q','ell','obs','mu_conditional','mu_mixture','mu_map','mse_full','nll_full','Pi_history','gate_crossweight','eval_obs'):
                np.testing.assert_array_equal(last[key],r[key][:,14:],err_msg=f'{name}:{key}')
            for key in ('final_Pi','final_h','final_L','final_key'):
                np.testing.assert_array_equal(last[key],r[key],err_msg=f'{name}:{key}')

    def test_rng_and_independent_full_observation_nll(self):
        cfg,W,A,att,phis,r,nu=self.setup('G_gated_moment',T=13)
        key=self.jax.random.PRNGKey(900100)
        for t,phi in enumerate(phis):
            evalkey=self.jax.random.fold_in(key,314159)
            expected=np.asarray(self.jax.jit(lambda k:self.jnp.asarray(phi[1:])+cfg.sigma_o*self.jax.random.normal(k,(16,4,3),dtype=self.jnp.float64))(evalkey))
            np.testing.assert_array_equal(expected,r['eval_obs'][0,t])
            key,sk=self.jax.random.split(key);keys=self.jax.random.split(sk,16)
            train=np.asarray(self.jax.jit(self.jax.vmap(lambda k:self.jnp.asarray(phi[1:])+cfg.sigma_o*self.jax.random.normal(k,(3,))))(keys))
            np.testing.assert_array_equal(train,r['obs'][0,t])
            P=r['Pi_history'][0,t,:,0];mu=r['mu_conditional'][0,t,:,0]
            cov=np.linalg.inv(P)[:,1:,1:]+cfg.sigma_o**2*np.eye(3)
            nll=np.array([-multivariate_normal.logpdf(expected[i],mean=mu[i],cov=cov[i]).mean() for i in range(16)])
            np.testing.assert_allclose(nll,r['nll_full'][0,t],atol=5e-12,rtol=5e-12)
            np.testing.assert_allclose(np.mean((mu-phi[1:])**2,axis=-1),r['mse_full'][0,t],atol=3e-14)
        np.testing.assert_array_equal(key,r['final_key'][0])

    def test_observable_permutation(self):
        cfg,W,A,att,phis,r,nu=self.setup('G_gated_info',T=13)
        perm=np.array([0,3,1,2])
        other=dataclasses.replace(cfg,Pi0=cfg.Pi0[:,perm][:,:,perm],h0=cfg.h0[:,perm],H=cfg.H[:,perm])
        result=self.runner.make_batch(other,self.jnp.asarray(W),self.jnp.asarray(A),self.jnp.asarray(att),gate_nu=nu)(phis[:,perm],[900100])
        # The production readout assumes observable storage order. Compare only
        # state, scoring and gate: changing storage must not change the dynamics.
        inv=np.argsort(perm)
        np.testing.assert_allclose(result['final_Pi'][:,:, :,inv][:,:,:,:,inv],r['final_Pi'],atol=5e-11,rtol=5e-12)
        np.testing.assert_allclose(result['final_h'][...,inv],r['final_h'],atol=5e-11,rtol=5e-12)
        np.testing.assert_allclose(result['ell'],r['ell'],atol=5e-11,rtol=5e-12)
        np.testing.assert_allclose(result['gate_crossweight'],r['gate_crossweight'],atol=5e-11,rtol=5e-12)


if __name__=='__main__':
    start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTests))
    report=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.monotonic()-start,test_seeds='900100..900109; excluded from scientific paired streams',independence='Expected references, weighted gates, Gaussian densities, information updates and predictive NLL derived separately in NumPy/SciPy; production engine called only for actual values and engine-equivalence check.')
    (HERE/'INDEPENDENT_TESTS.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
