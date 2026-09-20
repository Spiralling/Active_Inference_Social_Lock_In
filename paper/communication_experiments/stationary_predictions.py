"""Independent NumPy/SciPy stationary-score derivation; no adaptive engine imports."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, block_diag


def references(matched=False):
    B=np.zeros((2,4,4)); B[0,0,1]=.8; B[0,2,1]=.8; B[0,3,0]=-.8
    B[1,0,1]=.8; B[1,2,0]=.8; B[1,3,0]=.8
    means=np.array([[0,1,1,-1],[0,1,1,1]],float)
    residual=np.tile([.25,.5,.5,.5],(2,1))
    if matched: residual[0,2]=.5448
    covariance=[]; precision=[]; information=[]
    for k in range(2):
        T=np.linalg.solve(np.eye(4)-B[k],np.eye(4))
        S=T@np.diag(residual[k])@T.T
        P=np.linalg.solve(S,np.eye(4))
        covariance.append(S); precision.append(P); information.append(P@means[k])
    return np.array(precision),np.array(information),np.array(covariance)


def stationary(beta=.05,omega=.3,matched=False,fuse=True):
    n=16; d=4; sigma=.5
    P0,h0,C0=references(matched)
    W=np.block([[(1-beta)*np.ones((8,8))/8,beta*np.ones((8,8))/8],
                [beta*np.ones((8,8))/8,(1-beta)*np.ones((8,8))/8]]) if fuse else np.eye(n)
    attention=np.zeros((n,3)); attention[:8,1]=1; attention[8:,2]=1
    H=np.eye(4)[1:]; truth=np.array([1.,1.,1.])
    J=np.array([H.T@np.diag(a)@H/sigma**2 for a in attention])
    j=np.array([H.T@(a*truth)/sigma**2 for a in attention])
    innovation=omega**2*block_diag(*[H.T@np.diag(a*a)@H/sigma**2 for a in attention])
    A=omega*np.kron(W,np.eye(d))
    V=solve_discrete_lyapunov(A,innovation)
    residual_cov=np.max(np.abs(V-A@V@A.T-innovation))
    # Independent convergent-series covariance check.
    Vseries=innovation.copy(); inc=innovation.copy()
    for m in range(10000):
        inc=A@inc@A.T; Vseries+=inc
        if np.max(np.abs(inc))<1e-16: break
    variance_series_error=np.max(np.abs(V-Vseries))
    WW=np.kron(W,np.eye(d)); Vbar=WW@V@WW.T
    all_scores=[]; station_residual=[]
    for k in range(2):
        rhsP=(1-omega)*np.broadcast_to(P0[k],(n,d,d))+omega*J
        rhsh=(1-omega)*np.broadcast_to(h0[k],(n,d))+omega*j
        P=np.linalg.solve(np.eye(n)-omega*W,rhsP.reshape(n,-1)).reshape(n,d,d)
        h=np.linalg.solve(np.eye(n)-omega*W,rhsh)
        Pbar=np.einsum('ij,jab->iab',W,P); hbar=W@h
        station_residual.append(max(np.max(np.abs(P-(1-omega)*P0[k]-omega*(Pbar+J))),np.max(np.abs(h-(1-omega)*h0[k]-omega*(hbar+j)))))
        scores=[]
        for i,a in enumerate(attention):
            D=np.diag(np.sqrt(a)); Hs=D@H
            mean=np.linalg.solve(Pbar[i],hbar[i]); cov=np.linalg.solve(Pbar[i],np.eye(d))
            predictive=sigma**2*np.eye(3)+Hs@cov@Hs.T
            mu_noise=cov@Vbar[i*d:(i+1)*d,i*d:(i+1)*d]@cov.T
            residual=D@truth-Hs@mean
            Q=sigma**2*np.diag(a)+Hs@mu_noise@Hs.T
            score=-.5*(3*np.log(2*np.pi)+np.linalg.slogdet(predictive)[1]+residual@np.linalg.solve(predictive,residual)+np.trace(np.linalg.solve(predictive,Q)))
            scores.append(score)
        all_scores.append(scores)
    all_scores=np.array(all_scores).T; delta=all_scores[:,1]-all_scores[:,0]
    return {'gas':float(delta[:8].mean()),'mass':float(delta[8:].mean()),'expected_scores_by_agent_candidate':all_scores.tolist(),'lyapunov_residual':float(residual_cov),'covariance_series_error':float(variance_series_error),'stationary_update_residual':float(max(station_residual)),'gas_reference_variances':C0[:,2,2].tolist()}

if __name__=='__main__':
    expected={'C10':(-.003022079,.582598809),'C10_beta50':(.013773431,.931656831),'C10_omega90':(.001130619,.014425528),'C10_gasvar_matched':(.001078491,.582569828)}
    setups={'C10':{},'C10_beta50':{'beta':.5},'C10_omega90':{'omega':.9},'C10_gasvar_matched':{'matched':True}}
    out={key:stationary(**v) for key,v in setups.items()}
    for key,v in out.items():
        v['brief_rounded_forecast']=list(expected[key]);v['max_discrepancy_from_rounded_forecast']=float(np.max(np.abs(np.array([v['gas'],v['mass']])-expected[key])))
    out['validation_matched_beta_zero']=stationary(beta=0,matched=True)
    out['method']='Independent full 16-agent 4-state NumPy/SciPy stationary affine conditional system and discrete Lyapunov covariance; post-sharing pre-learning scores; real sampled noise, not working predictive noise. No JAX engine imports; no trajectory simulation.'
    print(json.dumps(out,indent=2))
