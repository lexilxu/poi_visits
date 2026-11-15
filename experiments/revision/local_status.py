"""Read local experiment progress without touching running jobs."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json


def status(root):
    queues=[]
    for folder in sorted((root/'private_local_runs').glob('20261114_v*')):
        execution=json.loads((folder/'execution.json').read_text())
        queue=folder/'queue';manifest_path=queue/'manifest.json'
        manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        fits=[];incomplete=[]
        for recipe in sorted(queue.glob('*')):
            if not recipe.is_dir():continue
            for directory in recipe.glob('seed*'):
                if not directory.is_dir():continue
                if (directory/'fit.json').exists():
                    fits.append(json.loads((directory/'fit.json').read_text()))
                elif (directory/'epoch_log.json').exists():
                    epochs=json.loads((directory/'epoch_log.json').read_text())
                    incomplete.append(dict(recipe=recipe.name,fit=directory.name,
                        completed_epochs=len(epochs),last_validation_MAE=epochs[-1]['validation_MAE'],
                        last_update_utc=datetime.fromtimestamp((directory/'epoch_log.json').stat().st_mtime,timezone.utc).isoformat()))
        selected=[]
        for path in queue.glob('*/selected_runs.json'):
            selection=json.loads(path.read_text())
            assert sorted(r['seed'] for r in selection['runs'])==[42,43,44,45,46]
            selected.append(selection['recipe'])
        queues.append(dict(queue=folder.name,status=execution['status'],
             source_archive_sha256=execution['source_archive_sha256'],
             planned_recipes=len(manifest.get('recipes',[])) or (20 if folder.name.endswith('v2') else 51),
             completed_fits=len(fits),complete_five_seed_recipes=len(selected),
             completed_by_stage=dict(Counter(r['stage'] for r in selected)),
             recorded_fit_seconds=sum(f['seconds'] for f in fits),active=incomplete))
    january=root/'neural_results/january/evaluation_status.json'
    attribution=root/'neural_results/attribution'
    return dict(as_of_utc=datetime.now(timezone.utc).isoformat(),queues=queues,
        January_evaluation=json.loads(january.read_text()) if january.exists() else None,
        completed_attribution_seeds=[int(p.parent.name.removeprefix('seed')) for p in attribution.glob('seed*/attribution_protocol.json')],
        neighbor_cases=len(list((root/'neural_results/neighbors_final').glob('*/*/neighbor_protocol.json'))),
        manuscript_status='Methods revised; final result tables and conclusions await the complete experiments',
        final_main_push_verified=False,final_Overleaf_sync_verified=False)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path)
    p.add_argument('--brief',action='store_true')
    args=p.parse_args();report=status(args.root);text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.out:args.out.write_text(text,encoding='utf8')
    if args.brief:
        print(json.dumps(dict(as_of_utc=report['as_of_utc'],
            completed_fits=sum(q['completed_fits'] for q in report['queues']),
            complete_five_seed_recipes=sum(q['complete_five_seed_recipes'] for q in report['queues']),
            planned_recipes=sum(q['planned_recipes'] for q in report['queues']),
            active=[a for q in report['queues'] for a in q['active']],
            queue_states={q['queue']:q['status'] for q in report['queues']},
            January_evaluated_runs=report['January_evaluation']['evaluated_runs'] if report['January_evaluation'] else 0)))
    else:print(text)
