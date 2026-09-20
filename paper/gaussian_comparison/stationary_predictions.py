from pathlib import Path
import json
import numpy as np
from scipy.linalg import solve_discrete_lyapunov,block_diag


def marginal_candidates():
    B=np.zeros((2,4,4));B[0,0,1]=.8;B[0,2,1]=.8;B[0,3,0]=-.8
    B[1,0,1]=.8;B[1,2,0]=.8;B[1,3,0]=.8
    mu=np.array([[1,1,-1],[1,1,1]],float)
    S=[]
    for k in range(2):
        T=np.linalg.solve(np.eye(4)-B[k],np.eye(4))
        S.append((T@np.diag([.25,.5,.5,.5])@T.T)[1:,1:])
    return mu,np.array(S)


def common_reference(kind):
    mu,S=marginal_candidates()
    if kind=='information':
        P=np.linalg.inv(S).mean(0)
        h=np.einsum('kij,kj->ki',np.linalg.inv(S),mu).mean(0)
        return P,h
    if kind=='moment':
        mean=mu.mean(0);diff=mu-mean
        C=(S+np.einsum('ki,kj->kij',diff,diff)).mean(0)
        P=np.linalg.inv(C)
        return P,P@mean
    raise ValueError(kind)


def stationary(kind,beta=.05,omega=.3):
    P0,h0=common_reference(kind);n=16;d=3;sigma=.5
    J8=np.ones((8,8))/8
    W=np.block([[(1-beta)*J8,beta*J8],[beta*J8,(1-beta)*J8]])
    att=np.zeros((16,3));att[:8,1]=1;att[8:,2]=1
    J=np.array([np.diag(a)/sigma**2 for a in att]);j=att/sigma**2
    rhsP=(1-omega)*P0[None]+omega*J;rhsh=(1-omega)*h0[None]+omega*j
    P=np.linalg.solve(np.eye(n)-omega*W,rhsP.reshape(n,-1)).reshape(n,d,d)
    h=np.linalg.solve(np.eye(n)-omega*W,rhsh)
    C=np.linalg.inv(P);means=np.linalg.solve(P,h[...,None])[...,0]
    A=omega*np.kron(W,np.eye(d));Q=omega**2*block_diag(*J)
    Vh=solve_discrete_lyapunov(A,Q)
    D=block_diag(*C);Vmu=D@Vh@D.T
    R=np.kron(np.r_[np.ones(8)/8,-np.ones(8)/8][None],np.eye(d))
    gap=means[:8].mean(0)-means[8:].mean(0)
    variance_gap=R@Vmu@R.T
    mse=np.mean((means-1)**2,axis=-1)+np.array([np.trace(Vmu[3*i:3*i+3,3*i:3*i+3])/3 for i in range(n)])
    return dict(reference_precision=P0.tolist(),reference_mean=np.linalg.solve(P0,h0).tolist(),reference_covariance=np.linalg.inv(P0).tolist(),gas_expected_mean=means[:8].mean(0).tolist(),mass_expected_mean=means[8:].mean(0).tolist(),expected_mean_gap=gap.tolist(),norm_expected_mean_gap=float(np.linalg.norm(gap)),expected_squared_centroid_gap=float(gap@gap+np.trace(variance_gap)),rms_centroid_gap=float(np.sqrt(gap@gap+np.trace(variance_gap))),gas_expected_mse=float(mse[:8].mean()),mass_expected_mse=float(mse[8:].mean()),stationary_precision_residual=float(np.max(abs(P-(1-omega)*P0-omega*(np.einsum('ij,jab->iab',W,P)+J)))),lyapunov_residual=float(np.max(abs(Vh-A@Vh@A.T-Q))))

if __name__=='__main__':
    result={'status':'pre-production analytic predictions','method':'Independent NumPy/SciPy analytic stationary affine precision/information system and Lyapunov covariance; no trajectory simulation, no producer imports. Euclidean gap is norm of expected group means or RMS centroid gap, not expectation of the norm.','conditions':{k:stationary(k) for k in ('information','moment')}}
    Path(__file__).resolve().with_name('STATIONARY_PREDICTIONS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
