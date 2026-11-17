"""Summarize translated attribution with the original valid-seed consensus rule."""
from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd


def main(a):
    sys.path.insert(0,str(a.source/'exp/common'))
    from exp.exp5.summarize_stability import profile
    from exp.exp5.method_audit import GROUPS
    reports=[];shares=[[],[]];valids=[[],[]]
    for seed in range(42,47):
        reports.append(json.loads((a.results/f'seed{seed}.json').read_text()))
        with np.load(a.results/'private_attribution'/f'seed{seed}.npz') as z:
            for index,key in enumerate(('original_absolute','absolute')):
                s,_,_,v=profile(z[key],14);shares[index].append(s);valids[index].append(v)
    consensus=[]
    for s,v in zip(shares,valids):
        count=np.sum(v,axis=0)
        consensus.append(np.divide(np.sum(s,axis=0),count[:,None],out=np.zeros_like(s[0]),where=count[:,None]>0))
    eligible=(consensus[0].sum(1)>0)&(consensus[1].sum(1)>0)
    regimes=[]
    for s in consensus:
        centered=(s[eligible]-s[eligible].mean(0))/np.maximum(s[eligible].std(0),1e-12)
        regimes.append(centered.argmax(1))
    transition=np.zeros((5,5),int)
    np.add.at(transition,(regimes[0],regimes[1]),1)
    a.out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(transition,index=GROUPS,columns=GROUPS).to_csv(a.out/'regime_transition.csv')
    rows=[]
    for i,name in enumerate(GROUPS):
        rows.append(dict(group=name,original_share=float(consensus[0][eligible,i].mean()),translated_share=float(consensus[1][eligible,i].mean()),
                         original_seed_mean=float(np.mean([s[v,i].mean() for s,v in zip(shares[0],valids[0])])),
                         translated_seed_mean=float(np.mean([s[v,i].mean() for s,v in zip(shares[1],valids[1])]))))
    pd.DataFrame(rows).to_csv(a.out/'group_shares.csv',index=False)
    report=dict(status='complete',eligible_POIs=int(eligible.sum()),unassigned=int((~eligible).sum()),
                regime_agreement=float((regimes[0]==regimes[1]).mean()),
                parent_regime_original=int((regimes[0]==3).sum()),parent_regime_translated=int((regimes[1]==3).sum()),
                parent_share_original=rows[3]['original_share'],parent_share_translated=rows[3]['translated_share'],
                maximum_prediction_difference=max(r['max_prediction_difference'] for r in reports),
                arithmetic='FP64 paired evaluation of stored model parameters',
                consensus='Normalize each seed separately; average shares across nonzero-attribution seeds per POI; standardize across eligible POIs',
                archived_FP32_replay_checks=[dict(seed=r['seed'],**r['archived_FP32_replay']) for r in reports])
    (a.out/'complete.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=ArgumentParser()
    for f in ('source','results','out'):p.add_argument('--'+f,type=Path,required=True)
    main(p.parse_args())
