"""Compare reused worlds against the unchanged frozen full-world evaluator."""
from exp.common.provenance import source_file
from pathlib import Path
from argparse import ArgumentParser
from hashlib import sha256
import json,time
import numpy as np
from scipy.spatial import cKDTree
from exp.common.evolvegcn_io import load_checkpoint, targeted_worlds
from exp.exp4.evaluate import selections, NAMES
from exp.exp4.reused_worlds import world_key, targeted_worlds_reused


def main(args):
    assert not args.out.exists();args.out.mkdir()
    reports=[]
    for graph in ['NAICS-4','NAICS-2']:
        run=args.root/'runs'/f'original_masked_T28_H14_{graph}_history_seed42'
        loaded=load_checkpoint(args.source,run,args.bundle)
        from exp.common.original_protocol import group_codes
        panel=loaded['panel'];labels=group_codes(panel.attrs)[graph]
        targets=np.flatnonzero(loaded['low_history']);test=np.flatnonzero(loaded['split']=='test')
        history=loaded['own'][test[0]]
        training=panel.raw[:loaded['train_end']];observed=np.isfinite(training)
        mean=np.divide(np.nansum(training,0),observed.sum(0),out=np.zeros(len(labels)),where=observed.sum(0)>0)
        frequency=np.divide((training>0).sum(0),observed.sum(0),out=np.zeros(len(labels)),where=observed.sum(0)>0)
        features=np.column_stack([np.log1p(mean),frequency,np.log1p(np.nanstd(training,axis=0))])
        features=(features-features.mean(0))/np.maximum(features.std(0),1e-8)
        groups,donors,_=selections(history,labels,targets,np.random.default_rng(42),cKDTree(features),features)
        archived=args.root/'mechanisms/neighbors_NAICS4_seed42/origin_00.npz'
        pick=np.arange(len(targets)) if graph=='NAICS-4' else np.unique(np.linspace(0,len(targets)-1,64,dtype=int))
        chosen=targets[pick]
        original=None
        if graph=='NAICS-4':
            with np.load(archived) as z:
                np.testing.assert_array_equal(z['target_index'],targets)
                original={key:z[key].copy() for key in z.files}
        for name in NAMES:
            chosen_sets=[groups[name][i] for i in pick]
            chosen_donors=[donors[i] for i in pick] if name=='Matched replacement' else None
            keys=[world_key(peers,donors[i] if name=='Matched replacement' else None) for i,peers in enumerate(groups[name])]
            started=time.monotonic()
            if original is None:reference,paired=targeted_worlds(loaded,history,chosen,chosen_sets,chosen_donors,8,True)
            else:reference,paired=original[name],original[name+'__baseline']
            original_seconds=time.monotonic()-started if original is None else None
            started=time.monotonic()
            reused,reused_base=targeted_worlds_reused(loaded,history,chosen,chosen_sets,chosen_donors,8,True)
            reused_seconds=time.monotonic()-started
            np.testing.assert_array_equal(reference,reused);np.testing.assert_array_equal(paired,reused_base)
            report=dict(graph=graph,group=name,targets=len(chosen),horizon=14,max_absolute_error=0.,baseline_max_error=0.,
                total_targets=len(targets),unique_full_worlds=len(set(keys)),original_seconds=original_seconds,reused_seconds=reused_seconds,
                reference_sha256=sha256(archived.read_bytes()).hexdigest() if original is not None else None)
            reports.append(report);print(json.dumps(report),flush=True)
            (args.out/'progress.json').write_text(json.dumps(dict(completed=len(reports),total=20)))
    files=['reused_worlds.py','checkpoint_io.py','export_neighbor_effects.py','verify_reused_worlds.py']
    receipt=dict(status='passed',cases=reports,scope='Exact target predictions and paired baselines for all1942targets inNAICS4 and64fixedtargets inNAICS2,10groups,firstDecemberorigin. Reuse the already completed full NAICS4 origin as reference.',
        source_sha256={name:sha256(source_file(name).read_bytes()).hexdigest() for name in files})
    (args.out/'complete.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(dict(status='passed',cases=len(reports))),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for field in ['source','root','bundle','out']:p.add_argument('--'+field,type=Path,required=True)
    main(p.parse_args())
