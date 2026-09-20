#!/usr/bin/env python3
"""Compare all independently accumulated seed/horizon metrics to publication tables."""
import csv
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def load(path):
    rows=list(csv.DictReader(Path(path).open()))
    result={(r['condition'],int(r['seed']),int(r['horizon'])):r for r in rows}
    if len(rows)!=len(result):raise AssertionError('Duplicate key '+str(path))
    return result

def main():
    a=load(HERE/'INDEPENDENT_SEED_METRICS.csv');b=load(HERE/'analysis/all_seed_metrics.csv')
    if set(a)!=set(b):raise AssertionError('Mismatched seed/horizon rows')
    worst={};count=0
    for key,row in a.items():
        for name,value in row.items():
            if name in ('condition','seed','horizon','count') or value=='':continue
            if name not in b[key]:raise AssertionError('Missing metric '+name)
            av,bv=float(value),float(b[key][name]);err=abs(av-bv)
            worst[name]=max(worst.get(name,0.),err);count+=1
            if not np.isclose(av,bv,atol=3e-12,rtol=3e-12):raise AssertionError(f'Mismatch {key} {name} {av} {bv}')
    output=dict(status='passed',rows=len(a),scalar_comparisons=count,maximum_absolute_error=max(worst.values()),maximum_absolute_errors_by_metric=worst,method='Independent full-raw chunk summation versus separate publication analysis; all11originalconditions and2branchhistories, bothcampreportinghorizons andnetwork600.')
    (HERE/'INDEPENDENT_ANALYSIS_COMPARISON.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
