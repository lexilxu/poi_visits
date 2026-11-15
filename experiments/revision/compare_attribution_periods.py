"""Compare completed five-seed attribution consensus across evaluation months."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from summarize_queue import write_json


def run(args):
    groups=['history','location','identity','parent','others']
    with np.load(args.december/'private_consensus.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(args.january/'private_consensus.npz',allow_pickle=False) as z:b={k:z[k] for k in z.files}
    np.testing.assert_array_equal(a['poi_id'],b['poi_id']);valid=(a['regime']>=0)&(b['regime']>=0)
    counts=np.zeros((5,5),int);np.add.at(counts,(a['regime'][valid],b['regime'][valid]),1)
    rows=[]
    for i,group in enumerate(groups):
        rows.append(dict(group=group,December_share=float(a['shares'][valid,i].mean()),
             January_share=float(b['shares'][valid,i].mean()),
             Spearman=float(spearmanr(a['shares'][valid,i],b['shares'][valid,i]).statistic)))
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(counts,index=groups,columns=groups).rename_axis('December_regime').to_csv(args.out/'regime_transition.csv')
    pd.DataFrame(rows).to_csv(args.out/'group_stability.csv',index=False)
    write_json(args.out/'protocol.json',dict(status='complete',eligible_POIs=int(valid.sum()),
          regime_agreement=float((a['regime'][valid]==b['regime'][valid]).mean()),
          mean_L1_share_change=float(np.abs(a['shares'][valid]-b['shares'][valid]).sum(1).mean()),
          scope='Same fitted models and POIs, different month-specific sampled forecast inputs; this is temporal stability, not a refitting experiment.',
          December_sha256=sha256((args.december/'private_consensus.npz').read_bytes()).hexdigest(),
          January_sha256=sha256((args.january/'private_consensus.npz').read_bytes()).hexdigest()))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--december',type=Path,required=True)
    p.add_argument('--january',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
