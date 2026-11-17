"""Compare each retrained density subset with HA on its identical POIs and events."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from exp.common.audit import load_panel
from exp.common.inputs import density_indices
from exp.common.evaluate import load_events, aligned
from exp.common.summarize import selected_jobs, write_json


def run(args):
    raw,dates,attrs=load_panel(args.bundle)
    baseline_path=args.baselines/'Calendar_HA_T28_H7.npz'
    baseline=load_events(baseline_path)
    expected={'Full'}|{kind+str(level) for kind in ('Dense','Recent-dense','Persist-dense') for level in (10,25,50,75)}
    rows=[]
    for path,meta,_ in selected_jobs(args.queues):
        if meta['stage']=='density':name=meta['id'].split('_MAE_',1)[1]
        elif meta['id']=='GraphGRU_T28_Identity_full_MAE':name='Full'
        else:continue
        if name=='Full':
            nodes=np.arange(len(attrs));definition=dict(training_nonzero_ratio=float(np.nanmean((raw[:50]>0).sum(0)/np.isfinite(raw[:50]).sum(0))))
        else:nodes,definition=density_indices(raw,name)
        data=load_events(path)
        reference={key:(value[...,nodes] if key in ('observed_visits','predicted_visits','valid') else
                        value[nodes] if key=='poi_id' else value) for key,value in baseline.items()}
        aligned(data,reference)
        test=data['dataset_split']=='test';valid=data['valid'][test]
        y=data['observed_visits'][test];p=data['predicted_visits'][test]
        mae=float(np.abs(p-y)[valid].mean());ha=float(np.abs(reference['predicted_visits'][test]-y)[valid].mean())
        rows.append(dict(subset=name,seed=meta['seed'],POIs=len(nodes),training_nonzero_ratio=definition['training_nonzero_ratio'],
                          MAE=mae,HA_MAE=ha,skill=100*(1-mae/ha),prediction_sha256=sha256(path.read_bytes()).hexdigest()))
    frame=pd.DataFrame(rows)
    found=set(frame.subset) if len(frame) else set()
    if found!=expected:raise ValueError('Missing completed subsets: '+str(sorted(expected-found)))
    grouped=frame.groupby('subset',sort=False)
    assert grouped.seed.agg(lambda x:sorted(x)).apply(lambda x:x==[42,43,44,45,46]).all()
    summary=grouped[['POIs','training_nonzero_ratio','MAE','HA_MAE','skill']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    args.out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.out/'density_by_seed.csv',index=False);summary.reset_index().to_csv(args.out/'density_summary.csv',index=False)
    write_json(args.out/'protocol.json',dict(status='complete',subsets=13,seeds=list(range(42,47)),
               baseline_sha256=sha256(baseline_path.read_bytes()).hexdigest(),
               selection='Observed counts within the first 50 calendar days; stable ties; same original POI order',
               source_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--queues',nargs='+',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--baselines',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);run(p.parse_args())
