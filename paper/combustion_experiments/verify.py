"""Read-only validation of frozen combustion experiment files and readouts."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.special import softmax

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_structural(run):
    """Recompute dependence-only scores and readouts from archived observations.

    This validates a separate fixed-density classifier, not the adaptive social
    engine and not any desired candidate-recovery rate.
    """
    metadata=json.loads((run/"structural_check.json").read_text())
    assert sha(run/metadata["array_file"])==metadata["array_sha256"]
    max_density_error=0.
    with np.load(run/metadata["array_file"],allow_pickle=False) as z:
        for key,item in metadata["arrays"].items():
            assert list(z[key].shape)==item["shape"] and str(z[key].dtype)==item["dtype"],key
            assert np.isfinite(z[key]).all(),key
        assert np.max(np.abs(z["mu"]))==0
        assert np.max(np.abs(z["observable_variances"][0]-z["observable_variances"][1]))<1e-12
        for k in range(2):
            transform=np.linalg.solve(np.eye(4)-z["B"][k],np.eye(4))
            covariance=transform@np.diag(z["s"][k])@transform.T
            assert np.max(np.abs(covariance-z["covariance"][k]))<1e-12
        expected_obs=(z["latent_states"][None,...,1:]
                      +z["sigmas"][:,None,None,None,None]*z["measurement_noise_standard"][None])
        assert np.max(np.abs(z["observations"]-expected_obs))<1e-12
        for i,sigma in enumerate(z["sigmas"]):
            for k in range(2):
                covariance=z["covariance"][k,1:,1:]+sigma**2*np.eye(3)
                assert np.max(np.abs(z["predictive_covariance"][i,k]-covariance))<1e-12
                x=z["observations"][i]
                density=-.5*(3*np.log(2*np.pi)+np.linalg.slogdet(covariance)[1]
                             +np.einsum("...i,ij,...j->...",x,np.linalg.inv(covariance),x))
                error=float(np.max(np.abs(density-z["log_density"][i,...,k])))
                assert error<1e-11,(i,k,error)
                max_density_error=max(max_density_error,error)
        assert np.max(np.abs(z["L"]-np.cumsum(z["log_density"],axis=-2)))<1e-10
        probability_error=float(np.max(np.abs(z["q"]-softmax(z["L"],axis=-1))))
        assert probability_error<1e-10
        assert np.max(np.abs(z["q"].sum(-1)-1))<1e-10
        for field in ("q_product_marginals","q_duplicate"):
            assert np.max(np.abs(z[field]-.5))<1e-10,field
        selected=z["q"][...,z["sample_counts"]-1,:].transpose(0,3,1,2,4)
        ties=selected==selected.max(-1,keepdims=True)
        counts=(ties/ties.sum(-1,keepdims=True)).sum(-2)
        confusion=counts/len(z["seeds"])
        assert np.max(np.abs(counts-z["confusion_counts"]))<1e-12
        assert np.max(np.abs(confusion-z["confusion_probability"]))<1e-12
        assert np.max(np.abs(selected.mean(-2)-z["mean_q"]))<1e-12
        accuracy=np.diagonal(confusion,axis1=-2,axis2=-1).mean(-1)
        assert np.max(np.abs(accuracy-z["accuracy"]))<1e-12
        checked=len(metadata["arrays"])
    return {"arrays_checked":checked,"max_independent_log_density_error":max_density_error,
            "max_probability_softmax_error":probability_error,
            "classifier_scope":"fixed known joint densities; not adaptive social-engine recovery"}


def verify(run):
    p=json.loads((run/"protocol.json").read_text())
    summary=json.loads((run/"summary.json").read_text())
    assert summary["status"]=="complete" and summary["failed_runs"]==[]
    files=json.loads((run/"files_manifest.json").read_text())
    for name,digest in files.items(): assert sha(run/name)==digest,name
    manifest=json.loads((run/"results_manifest.json").read_text())
    checked_arrays=0; max_prob_error=0.; max_mean_error=0.; min_precision=float('inf')
    for name,item in manifest.items():
        assert sha(run/item["file"])==item["sha256"],name
        with np.load(run/item["file"],allow_pickle=False) as z:
            for key,meta in item["arrays"].items():
                a=z[key]
                assert list(a.shape)==meta["shape"] and str(a.dtype)==meta["dtype"]
                assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==meta["sha256"],(name,key)
                assert np.isfinite(a).all(),(name,key)
                checked_arrays+=1
            q=z["q"]; err=float(np.max(np.abs(q-softmax(z["L"],axis=-1))))
            assert err<1e-12 and np.all((q>=0)&(q<=1)),name
            max_prob_error=max(max_prob_error,err)
            assert np.max(np.abs(q.sum(-1)-1))<1e-12
            Pi=z["final_Pi"]; h=z["final_h"]
            eigen=float(np.linalg.eigvalsh(Pi).min()); assert eigen>0,name
            min_precision=min(min_precision,eigen)
            mu=np.linalg.solve(Pi,h[...,None])[...,0]
            expected=np.einsum("snk,snkd->snd",q[:,-1],mu[...,1:])
            error=float(np.max(np.abs(expected-z["mu_mixture"][:,-1,:,1:])))
            assert error<1e-10,(name,error)
            max_mean_error=max(max_mean_error,error)
            assert np.all(z["mu_mixture"][...,0]==0) and np.all(z["mu_map"][...,0]==0)
            if "D_mu_map" in z:
                n=q.shape[2]
                for field in ("mu_map","mu_mixture"):
                    delta=z[field][:,:,:n//2,1:].mean(2)-z[field][:,:,n//2:,1:].mean(2)
                    assert np.max(np.abs(np.linalg.norm(delta,axis=-1)-z["D_"+field]))<1e-12
    history={}
    for base,others in [
        ("network_complete",["network_hypothesis_off","network_all_social_off","network_zero_direct","network_full_attention"]),
        ("adaptation_social",["adaptation_isolated","adaptation_reference"])]:
        with np.load(run/(base+".npz")) as first:
            for other in others:
                with np.load(run/(other+".npz")) as second:
                    err=max(float(np.max(np.abs(first[k][:,:p["shift"]]-second[k][:,:p["shift"]]))) for k in ("L","q","mu_mixture","mse_obs","heldout_nll"))
                    assert err==0,(base,other,err)
                    history[other]=err
    with np.load(run/"wall_sources_off.npz") as z:
        contraction=float(np.max(np.abs(z["log_odds_gap"]-z["expected_log_odds_gap"])))
        assert contraction<1e-8,contraction
    engine=json.loads((run/"accepted_engine_hashes.json").read_text())
    engine_root=ROOT/p["source_export"]
    for name,digest in engine.items(): assert sha(engine_root/name)==digest,name
    # Compile the archived, non-topologically ordered CPDs independently. The
    # engine's general information-form compiler must not silently discard the
    # C -> P/O edges above the storage matrix diagonal.
    with np.load(run/"world.npz",allow_pickle=False) as world:
        conditional=[]
        for k in range(2):
            A=np.eye(4)-world["B"][k]
            T=np.linalg.solve(A,np.eye(4))
            covariance=T@np.diag(world["s"][k])@T.T
            assert np.max(np.abs(covariance-world["covariance"][k]))<1e-12
            assert np.max(np.abs(np.linalg.inv(world["Pi"][k])-covariance))<1e-12
            assert np.max(np.abs(T@world["b"][k]-world["mu"][k]))<1e-12
            conditional.append(float(covariance[2,3]-covariance[2,1]*covariance[1,3]/covariance[1,1]))
        assert np.max(np.abs(np.asarray(conditional)-[0.,.16]))<1e-12
    structural=verify_structural(run)
    return {"status":"passed","conditions":len(manifest),"files_checked":len(files),"arrays_checked":checked_arrays,
            "engine_files_checked":len(engine),"max_probability_softmax_error":max_prob_error,
            "max_final_observable_mean_error":max_mean_error,"minimum_final_precision_eigenvalue":min_precision,
            "shared_prechange_max_errors":history,"source_off_contraction_max_error":contraction,
            "initial_covariance_G_M_given_C":conditional,
            "structural_check":structural,
            "note":"No assertions require any candidate to win or any intended scientific pattern to occur."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=HERE/"runs/assessment-2026-09-19")
    parser.add_argument("--output",type=Path,default=HERE/"VALIDATION.json")
    args=parser.parse_args(); result=verify(args.run_dir)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
