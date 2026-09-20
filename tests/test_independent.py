"""Independent scientific checks for the extracted figure model.

These tests use covariance-form conditioning and SciPy probability densities,
not the production information-form update. They do not assert that particular
conditions must separate or choose oxygen. Small examples test mathematical
invariances and the exact random-stream/measurement contract.
"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import multivariate_normal
from iwai_figures import engine, world, simulation, communication, gaussian, camera, structural_check

jax, jnp = engine.jax, engine.jnp


def covariance_reference_step(cfg, W, attention, Pi, h, L, obs):
    """A loop-based NumPy/SciPy oracle using Kalman covariance conditioning."""
    n, k, d = Pi.shape[:3]
    if cfg.fuse_nets:
        pooled_p = np.stack([sum(W[i,j] * Pi[j] for j in range(n)) for i in range(n)])
        pooled_h = np.stack([sum(W[i,j] * h[j] for j in range(n)) for i in range(n)])
    else:
        pooled_p, pooled_h = Pi.copy(), h.copy()
    out_p, out_h, scores = np.empty_like(Pi), np.empty_like(h), np.empty_like(L)
    H = np.asarray(cfg.H)
    ref_p = np.asarray(cfg.Pi0 if cfg.Pi0_agent is None else cfg.Pi0_agent)
    ref_h = np.asarray(cfg.h0 if cfg.h0_agent is None else cfg.h0_agent)
    if ref_p.ndim == 3:
        ref_p = np.broadcast_to(ref_p, Pi.shape)
        ref_h = np.broadcast_to(ref_h, h.shape)
    for i in range(n):
        weighted_H = np.sqrt(attention[i])[:,None] * H
        weighted_o = np.sqrt(attention[i]) * obs[i]
        for c in range(k):
            prior_cov = np.linalg.inv(pooled_p[i,c])
            prior_mean = prior_cov @ pooled_h[i,c]
            innovation_cov = weighted_H @ prior_cov @ weighted_H.T + cfg.sigma_o**2 * np.eye(H.shape[0])
            scores[i,c] = multivariate_normal.logpdf(weighted_o, mean=weighted_H@prior_mean, cov=innovation_cov)
            gain = np.linalg.solve(innovation_cov, weighted_H @ prior_cov).T
            posterior_mean = prior_mean + gain @ (weighted_o - weighted_H @ prior_mean)
            posterior_cov = prior_cov - gain @ weighted_H @ prior_cov
            posterior_p = np.linalg.inv(posterior_cov)
            posterior_h = posterior_p @ posterior_mean
            # Geometric interpolation of conditional densities, not covariance averaging.
            out_p[i,c] = (1-cfg.omega)*ref_p[i,c] + cfg.omega*posterior_p
            out_h[i,c] = (1-cfg.omega)*ref_h[i,c] + cfg.omega*posterior_h
    out_L = ((1-cfg.alpha_m)*L + cfg.alpha_m*W@L if k>1 else L.copy()) + scores
    return out_p, out_h, out_L, scores


def scipy_mixture_nll(Pi, h, L, eval_obs, sigma):
    """Independent marginal predictive mixture, retaining full covariance."""
    n,k = h.shape[:2]
    loss = np.empty(n)
    for i in range(n):
        components = []
        for c in range(k):
            cov = np.linalg.inv(Pi[i,c])
            mean = cov @ h[i,c]
            components.append(multivariate_normal.logpdf(eval_obs[i], mean=mean[1:], cov=cov[1:,1:]+sigma**2*np.eye(3)))
        weights = L[i] - logsumexp(L[i])
        loss[i] = -logsumexp(np.array(components).T+weights,axis=-1).mean()
    return loss


class IndependentMathematics(unittest.TestCase):
    def test_reference_covariance_and_observable_marginal(self):
        w = world.build_world()
        expected_B = np.zeros((2,4,4))
        expected_B[0,0,1] = expected_B[0,2,1] = .8
        expected_B[0,3,0] = -.8
        expected_B[1,0,1] = expected_B[1,2,0] = expected_B[1,3,0] = .8
        np.testing.assert_array_equal(w['B'],expected_B)
        innovations = np.diag([.25,.5,.5,.5])
        for c in range(2):
            transform = np.linalg.inv(np.eye(4)-expected_B[c])
            cov = transform @ innovations @ transform.T
            np.testing.assert_allclose(w['covariance'][c],cov,atol=2e-15,rtol=0)
            np.testing.assert_allclose(np.linalg.inv(w['Pi'][c]),cov,atol=2e-15,rtol=0)
            np.testing.assert_allclose(np.linalg.solve(w['Pi'][c],w['h'][c]),w['mu'][c],atol=2e-15,rtol=0)
        P,h = camera.observable_baselines(w)
        for c in range(2):
            expected_cov = np.asarray(w['covariance'][c])[1:,1:]
            np.testing.assert_allclose(np.linalg.inv(P[c])[1:,1:],expected_cov,atol=2e-15,rtol=0)
            np.testing.assert_allclose(np.linalg.solve(P[c],h[c])[1:],np.asarray(w['mu'][c])[1:],atol=2e-15,rtol=0)
            self.assertGreater(np.max(np.abs(P[c,1:,1:]-np.asarray(w['Pi'][c])[1:,1:])),.1)
        for variant in ('info','moment'):
            pg,hg,meta=gaussian.compress_reference(w,variant)
            self.assertEqual(float(pg[0,0,0]),4.)
            np.testing.assert_array_equal(np.asarray(pg)[0,0,1:],0.)
            np.testing.assert_array_equal(np.asarray(pg)[0,1:,0],0.)
            self.assertEqual(float(hg[0,0]),0.)
            covs = np.asarray(w['covariance'])[:,1:,1:]
            means = np.asarray(w['mu'])[:,1:]
            if variant=='moment':
                mean = means.mean(0)
                cov = np.zeros((3,3))
                for c in range(2): cov += .5*(covs[c]+np.outer(means[c]-mean,means[c]-mean))
                np.testing.assert_allclose(np.linalg.inv(pg[0])[1:,1:],cov,atol=4e-15,rtol=0)
                np.testing.assert_allclose(meta['reference_observable_mean'],[1,1,0],atol=2e-15,rtol=0)
            else:
                # Product of observable Gaussian densities: their Hessians add.
                expected=.5*np.linalg.inv(covs[0])+.5*np.linalg.inv(covs[1])
                np.testing.assert_allclose(pg[0,1:,1:],expected,atol=4e-15,rtol=0)

    def test_independent_hidden_coordinate_changes_leave_observables_unchanged(self):
        w=world.build_world()
        changed=dict(w)
        newP=[];newh=[]
        for c,(scale,offset) in enumerate([(3.1,7.),(-.7,-5.)]):
            # Independently relabel P and O by nonsingular affine maps.
            T=np.diag([scale,1.,1.,1.]);b=np.array([offset,0,0,0])
            cov=T@np.asarray(w['covariance'][c])@T.T
            mu=T@np.asarray(w['mu'][c])+b
            p=np.linalg.inv(cov);newP.append(p);newh.append(p@mu)
        changed['Pi']=np.array(newP);changed['h']=np.array(newh)
        for variant in ('info','moment'):
            a,b,_=gaussian.compress_reference(w,variant)
            c,d,_=gaussian.compress_reference(changed,variant)
            np.testing.assert_allclose(a,c,atol=1e-14,rtol=0)
            np.testing.assert_allclose(b,d,atol=2e-14,rtol=0)

    def test_covariance_form_reference_matches_full_update_and_score_timing(self):
        cfg,W,A,attention,phi,_=communication.make_setup('C11',N=4)
        init=simulation.initial_state(cfg,4,[137])
        P,h,L=(init[k][0].copy() for k in ('Pi','h','L'))
        h += np.arange(h.size).reshape(h.shape)*.012
        L += np.array([[1.,-2.],[-.5,2.],[3.,.1],[.3,-1.]])
        att=np.array([[.2,0,1],[0,1,0],[1,.01,.3],[0,0,0]],float)
        key=jax.random.PRNGKey(137)
        next_key,drawkey=jax.random.split(key)
        keys=jax.random.split(drawkey,4)
        # Draw directly from documented random contract; do not call production sample_o.
        obs=np.stack([np.asarray(cfg.H)@np.asarray(phi)+cfg.sigma_o*np.asarray(jax.random.normal(k,(3,))) for k in keys])
        for omega,fuse,alpha in [(0.,True,.02),(.3,True,.02),(1.,False,0.)]:
            c=replace(cfg,omega=omega,fuse_nets=fuse,alpha_m=alpha)
            expectedP,expectedh,expectedL,_=covariance_reference_step(c,np.asarray(W),att,P,h,L,obs)
            state,_=engine._step((jnp.asarray(P),jnp.asarray(h),jnp.asarray(L),key),phi,c,W,A,jnp.asarray(att))
            for actual,expected in zip(state[:3],(expectedP,expectedh,expectedL)):
                np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=0)
            np.testing.assert_array_equal(state[3],next_key)
            if omega==0:
                np.testing.assert_allclose(state[0],np.broadcast_to(cfg.Pi0,P.shape),atol=2e-15,rtol=0)
                np.testing.assert_allclose(state[1],np.broadcast_to(cfg.h0,h.shape),atol=2e-15,rtol=0)

    def test_zero_attention_ignores_channel_and_all_zero_preserves_model_odds(self):
        w=world.build_world();H=w['H'];p=w['Pi'];h=w['h']
        one=jnp.array([1.,1.,1.]);other=jnp.array([1e8,1.,-1e8]);att=jnp.array([0.,1.,0.])
        a=engine.score_candidates(p,h,H,one,.5,att)
        b=engine.score_candidates(p,h,H,other,.5,att)
        np.testing.assert_array_equal(a,b)
        J,j=engine.fisher_deposit_weighted(H,other,.5,att)
        np.testing.assert_array_equal(J,np.diag([0.,0.,4.,0.]))
        np.testing.assert_array_equal(j,[0.,0.,4.,0.])
        score=engine.score_candidates(p,h,H,other,.5,jnp.zeros(3))
        np.testing.assert_array_equal(score[0],score[1])

    def test_weighted_gate_preserves_base_strengths_and_hidden_invariance(self):
        cfg,W,A,att,phi,_=gaussian.make_setup('G_gated_info',N=4)
        state=simulation.initial_state(cfg,4,[3]);P=state['Pi'][0];h=state['h'][0]
        equal=gaussian.effective_W(jnp.asarray(P),jnp.asarray(h),W,.01)
        np.testing.assert_allclose(equal,W,atol=2e-16,rtol=0)
        np.testing.assert_array_equal(gaussian.effective_W(jnp.asarray(P),jnp.asarray(h),W,None),W)
        mean=np.linalg.solve(P,h[...,None])[...,0]
        mean[:,0,1:]+=np.array([[0,.1,0],[0,.1,0],[.8,-.2,.3],[.8,-.2,.3]])
        h=np.einsum('nkij,nkj->nki',P,mean)
        actual=np.asarray(gaussian.effective_W(jnp.asarray(P),jnp.asarray(h),W,.01))
        expected=np.empty((4,4))
        for i in range(4):
            for j in range(4):
                z=0.
                for v in (1,2,3):z+=(mean[i,0,v]-mean[j,0,v])**2/(1/P[i,0,v,v]+1/P[j,0,v,v])
                expected[i,j]=float(W[i,j])*(1.01)/(.01+z)
            expected[i]/=expected[i].sum()
        np.testing.assert_allclose(actual,expected,atol=2e-15,rtol=0)
        np.testing.assert_allclose(actual.sum(1),1,atol=5e-16,rtol=0)
        self.assertTrue(np.all(actual>0))
        self.assertLess(actual[:2,2:].sum(1).mean(),.05)
        mean[:,0,0]=[0,100,-600,3e4]
        hidden_h=np.einsum('nkij,nkj->nki',P,mean)
        np.testing.assert_array_equal(gaussian.effective_W(jnp.asarray(P),jnp.asarray(hidden_h),W,.01),actual)

    def test_support_sharing_changes_choices_but_not_conditional_learning(self):
        cfg,W,A,att,phi,_=communication.make_setup('C10',N=4)
        sharing=replace(cfg,alpha_m=.02)
        seeds=[10007,10011]
        initial=simulation.initial_state(cfg,4,seeds)
        initial['L'][:,:2]=[2.,-3.];initial['L'][:,2:]=[-9.,7.]
        phis=np.broadcast_to(np.asarray(phi),(9,4))
        private=simulation.make_batch(cfg,W,A,att)(phis,seeds,initial)
        shared=simulation.make_batch(sharing,W,A,att)(phis,seeds,initial)
        for key in ['Pi_history','final_Pi','final_h','ell','obs','eval_obs','mu_conditional','final_key']:
            np.testing.assert_array_equal(private[key],shared[key])
        self.assertGreater(np.max(np.abs(private['q']-shared['q'])),1e-4)
        for arm in (private,shared):
            expected=scipy_mixture_nll(arm['final_Pi'][0],arm['final_h'][0],arm['final_L'][0],arm['eval_obs'][0,-1],cfg.sigma_o)
            np.testing.assert_allclose(arm['nll_full'][0,-1],expected,atol=1e-13,rtol=0)

    def test_checkpoint_and_evaluation_do_not_change_training_stream(self):
        cfg,W,A,att,phi,meta=gaussian.make_setup('G_gated_moment',N=4)
        seeds=[10001,10002];phis=np.broadcast_to(np.asarray(phi),(9,4))
        runner=gaussian.make_batch(cfg,W,A,att,draws=4,gate_nu=meta['gate_nu'])
        whole=runner(phis,seeds)
        first=runner(phis[:4],seeds)
        second=runner(phis[4:],seeds,gaussian.final_state(first))
        for key in gaussian.HISTORY_KEYS+('Pi_history','eval_obs'):
            np.testing.assert_array_equal(whole[key],np.concatenate((first[key],second[key]),axis=1))
        for key in simulation.STATE_KEYS:
            np.testing.assert_array_equal(whole['final_'+key],second['final_'+key])
        different_eval=gaussian.make_batch(cfg,W,A,att,draws=7,gate_nu=meta['gate_nu'])(phis,seeds)
        for key in ['L','q','ell','obs','Pi_history','mu_mixture','final_key','final_h']:
            np.testing.assert_array_equal(whole[key],different_eval[key])
        candidate_cfg,candidate_W,candidate_A,candidate_att,_,_=communication.make_setup('C10',N=4)
        candidate=simulation.make_batch(candidate_cfg,candidate_W,candidate_A,candidate_att)(phis,seeds)
        np.testing.assert_array_equal(whole['obs'],candidate['obs'])
        np.testing.assert_array_equal(whole['eval_obs'],candidate['eval_obs'])
        np.testing.assert_array_equal(whole['final_key'],candidate['final_key'])

    def test_camera_phase_reset_matches_explicit_seed_plus_1000(self):
        cfg,W,A,att,phi,_=communication.make_setup('C11',N=4)
        cfg=replace(cfg,omega=.9,sigma_o=1.)
        seeds=[17,19];pre_phi=np.broadcast_to([0.,1.,1.,-1.],(5,4))
        pre=simulation.make_batch(cfg,W,A,jnp.ones_like(att))(pre_phi,seeds)
        legacy_initial={k:pre['final_'+k] for k in ('Pi','h','L')}
        post_phi=np.broadcast_to(np.asarray(phi),(7,4))
        old=camera.batch(cfg,W,A,att,jnp.asarray(post_phi),[s+1000 for s in seeds],(jax,jnp,engine,world),initial=legacy_initial)
        explicit=simulation.final_state(pre)
        explicit['key']=np.stack([np.asarray(jax.random.PRNGKey(s+1000)) for s in seeds])
        new=simulation.make_batch(cfg,W,A,att)(post_phi,seeds,explicit)
        for left,right in [('q','q'),('L','L'),('final_Pi','final_Pi'),('final_h','final_h'),('mse_obs','mse_full'),('heldout_nll','nll_full')]:
            np.testing.assert_allclose(old[left],new[right],atol=2e-14,rtol=0)
        np.testing.assert_allclose(old['mu_mixture'][...,1:],new['mu_mixture'],atol=2e-14,rtol=0)
        np.testing.assert_array_equal(old['mu_mixture'][...,0],0)

    def test_structural_check_matches_marginals_and_negative_controls(self):
        w=world.build_world();m=structural_check.matched_models(w)
        np.testing.assert_allclose(m['observable_variances'],[[.5,.8648,.8648]]*2,atol=2e-15,rtol=0)
        self.assertGreater(np.max(np.abs(m['observable_covariance'][0]-m['observable_covariance'][1])),.5)
        config={'structural_check':{'seeds':[4,9],'sigmas':[0.,1.],'sample_counts':[1,3,8]}}
        with tempfile.TemporaryDirectory() as directory:
            summary=structural_check.run_check(w,config,directory)
            with np.load(Path(directory)/'structural_check.npz') as a:
                np.testing.assert_allclose(a['q_product_marginals'],.5,atol=1e-14,rtol=0)
                np.testing.assert_allclose(a['q_duplicate'],.5,atol=1e-14,rtol=0)
                # One fixed-model likelihood independently evaluated, with no adaptive update.
                obs=a['observations'][1,0,0]
                logs=np.stack([multivariate_normal.logpdf(obs,mean=np.zeros(3),cov=m['observable_covariance'][k]+np.eye(3)) for k in range(2)],axis=-1)
                np.testing.assert_allclose(a['q'][1,0,0],softmax(logs.cumsum(0),axis=-1),atol=1e-14,rtol=0)
                np.testing.assert_allclose(a['observations'][1]-a['observations'][0],a['measurement_noise_standard'],atol=4e-16,rtol=0)
        self.assertLess(summary['checks']['label_swap_q_max_error'],1e-14)


if __name__=='__main__':
    unittest.main(verbosity=2)
