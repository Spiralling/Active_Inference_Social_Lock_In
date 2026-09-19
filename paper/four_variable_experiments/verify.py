"""Read-only validation of frozen four-variable result files and readouts."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.special import softmax

HERE=Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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
    for name,digest in engine.items(): assert sha(HERE/"engine"/name)==digest,name
    return {"status":"passed","conditions":len(manifest),"files_checked":len(files),"arrays_checked":checked_arrays,
            "engine_files_checked":len(engine),"max_probability_softmax_error":max_prob_error,
            "max_final_observable_mean_error":max_mean_error,"minimum_final_precision_eigenvalue":min_precision,
            "shared_prechange_max_errors":history,"source_off_contraction_max_error":contraction,
            "note":"No assertions require any candidate to win or any intended scientific pattern to occur."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=HERE/"runs/assessment-2026-09-19")
    parser.add_argument("--output",type=Path,default=HERE/"VALIDATION.json")
    args=parser.parse_args(); result=verify(args.run_dir)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
