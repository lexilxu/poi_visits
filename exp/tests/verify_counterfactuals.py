"""Verify full-world interventions on a completed real checkpoint."""
from pathlib import Path
from argparse import ArgumentParser
from hashlib import sha256
import json
import numpy as np
from exp.common.evolvegcn_io import load_checkpoint, predict_worlds, targeted_worlds


def main(args):
    loaded=load_checkpoint(args.source,args.run,args.bundle)
    from exp.common.original_protocol import group_codes
    labels=group_codes(loaded['panel'].attrs)[loaded['config']['adjacency']]
    degree=np.bincount(labels)[labels]-1
    targets=np.flatnonzero(loaded['low_history']&(degree>0))[:3]
    origin=np.flatnonzero(loaded['split']=='test')[0]
    own=loaded['own'][origin];empty=[np.empty(0,dtype=int) for _ in targets]
    same,paired=targeted_worlds(loaded,own,targets,empty,batch_size=3,return_baseline=True)
    expected=loaded['prediction'][origin][:,targets]
    np.testing.assert_array_equal(same,paired)
    peers=[np.flatnonzero((labels==labels[j])&(np.arange(len(labels))!=j))[:3] for j in targets]
    donors=[np.flatnonzero(labels!=labels[j])[:len(p)] for j,p in zip(targets,peers)]
    zero=targeted_worlds(loaded,own,targets,peers,batch_size=3)
    replacement=targeted_worlds(loaded,own,targets,peers,donors,batch_size=3)
    manual_zero=[];manual_replacement=[]
    for j,p,d in zip(targets,peers,donors):
        altered=own.copy();altered[:,p]=0
        manual_zero.append(predict_worlds(loaded,altered[None])[0,:,j])
        altered=own.copy();altered[:,p]=own[:,d]
        manual_replacement.append(predict_worlds(loaded,altered[None])[0,:,j])
    manual_zero=np.stack(manual_zero,axis=1);manual_replacement=np.stack(manual_replacement,axis=1)
    np.testing.assert_allclose(zero,manual_zero,atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(replacement,manual_replacement,atol=1e-5,rtol=1e-5)
    external=targeted_worlds(loaded,own,targets,donors,batch_size=3)
    report=dict(status='passed',checkpoint=str(args.run),target_count=len(targets),
                saved_prediction_replay_max_error=loaded['replay_max_absolute_error'],
                saved_prediction_replay_audit=loaded['replay_audit'],
                no_change_max_error=float(np.abs(same-paired).max()),
                same_batch_baseline_max_difference_from_archive=float(np.abs(same-expected).max()),
                zeroing_batch_vs_individual_max_error=float(np.abs(zero-manual_zero).max()),
                replacement_batch_vs_individual_max_error=float(np.abs(replacement-manual_replacement).max()),
                physical_world_batch_size=8,
                disconnected_history_effect_max=float(np.abs(external-expected).max()),
                disconnected_interpretation='Measured sensitivity, not a zero-effect assertion: EvolveGCN-H pools nodes during weight evolution.',
                source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),(Path(__file__).resolve().parents[2] / 'exp/common/evolvegcn_io.py')]})
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=ArgumentParser()
    for field in ['source','run','bundle','out']:p.add_argument('--'+field,type=Path,required=True)
    main(p.parse_args())
