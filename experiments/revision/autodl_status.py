from pathlib import Path
from datetime import datetime,timezone
import json
root=Path('/root/autodl-tmp/poi_continuation_20261114')
queues=[]
for version in ('v1','v2'):
    queue=root/version/'queue'
    fits=list(queue.glob('*/seed*/fit.json'))
    selections=list(queue.glob('*/selected_runs.json'))
    partial=[]
    for path in queue.glob('*/seed*/epoch_log.json'):
        if not (path.parent/'fit.json').exists():
            logs=json.loads(path.read_text());partial.append(dict(recipe=path.parent.parent.name,fit=path.parent.name,epochs=len(logs),last=logs[-1]))
    queues.append(dict(version=version,complete_fits=len(fits),complete_recipes=len(selections),partial=partial))
report=dict(as_of_utc=datetime.now(timezone.utc).isoformat(),remote_root=str(root),queues=queues,
    completed_fits=sum(q['complete_fits'] for q in queues),complete_recipes=sum(q['complete_recipes'] for q in queues),
    remaining_fits=497-sum(q['complete_fits'] for q in queues),hydration_complete=(root/'hydration_complete.json').exists(),
    newly_evaluated_runs=len(list((root/('new_december_results_v2' if (root/'new_december_results_v2').exists() else 'new_december_results')/'run_records').glob('*.json'))))
report['archive_bytes']=(root/'continuation.tar').stat().st_size
report['archive_expected_bytes']=8732078080
report['new_december_complete']=(root/'new_december_results_v2/complete.json').exists()
report['new_january_runs']=len(list((root/'new_january_results/run_records').glob('*.json')))
report['new_january_diagnostics']=len(list((root/'new_january_results/replay_diagnostics').glob('*.json')))
print(json.dumps(report,indent=2))
