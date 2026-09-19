"""Paired, seed-level summaries of the frozen assessment; no new simulations."""
from pathlib import Path
import argparse
import json

import numpy as np

HERE=Path(__file__).resolve().parent


def estimate(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20260919)
    boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(-1)
    return {"mean":float(values.mean()),"pointwise_percentile_bootstrap_95":np.quantile(boot,[.025,.975]).tolist(),"seeds":len(values),"per_seed":values.tolist()}


def analyze(run):
    summary=json.loads((run/"summary.json").read_text())
    comparisons={}
    pairs=[("network_complete","network_hypothesis_off"),
           ("network_complete","network_all_social_off"),
           ("adaptation_social","adaptation_isolated"),
           ("adaptation_social","adaptation_reference"),
           ("cycle","cycle_single_gaussian")]
    for first,second in pairs:
        with np.load(run/(first+".npz")) as a, np.load(run/(second+".npz")) as b:
            agents=np.arange(4,24) if first.startswith("network_") else np.arange(a["q"].shape[2])
            tail=slice(-(a["q"].shape[1]//4),None)
            result={"difference":"first minus second; negative predictive loss favours first"}
            for field in ("mse_obs","heldout_nll"):
                av=a[field][:,tail][:,:,agents].mean((1,2));bv=b[field][:,tail][:,:,agents].mean((1,2))
                result[field]=estimate(av-bv)
            if "group_first_crossings" in summary[first] and "group_first_crossings" in summary[second]:
                ta=np.asarray(summary[first]["group_first_crossings"]["times"]);tb=np.asarray(summary[second]["group_first_crossings"]["times"])
                event=(ta>=0)&(tb>=0)
                result["paired_observed_crossings"]=int(event.sum())
                result["pairs_with_a_noncrossing"]=int((~event).sum())
                if event.any():result["first_crossing_step_difference"]=estimate(ta[event]-tb[event])
            comparisons[first+" minus "+second]=result
    with np.load(run/"wall_rivals_original.npz") as z:
        a=z["q"][:,:,:8,1].mean(-1);b=z["q"][:,:,8:,1].mean(-1)
        split=(a<.1)&(b>.9)
        last=np.array([np.flatnonzero(row)[-1] if row.any() else -1 for row in split])
        wall={"split_definition":"camp-average qO below .1 in A and above .9 in B, same stored time",
              "seeds_with_a_split":int(split.any(1).sum()),"seeds_split_at_horizon":int(split[:,-1].sum()),
              "last_split_time_by_seed":last.tolist(),"median_last_split_time":float(np.median(last[last>=0])) if np.any(last>=0) else None}
    return {"paired_tail_differences":comparisons,"wall_transient":wall,
            "interval_definition":"2000 percentile resamples of paired seed-level differences, fixed seed20260919; not simultaneous intervals",
            "crossing_definition":"first group-mean qO>.9, stored post-update index; censored runs remain explicit"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=HERE/"runs/assessment-2026-09-19")
    parser.add_argument("--output",type=Path,default=HERE/"ANALYSIS.json")
    args=parser.parse_args(); result=analyze(args.run_dir)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
