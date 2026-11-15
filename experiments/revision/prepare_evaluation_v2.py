"""Version the density evaluation correction and preserve prior output records."""
from pathlib import Path
import hashlib,json,shutil,datetime
r=Path('/root/autodl-tmp/poi_continuation_20261114')
source=r/'analysis_source_v2'
assert not source.exists()
shutil.copytree(r/'v2/source',source)
target=source/'experiments/revision/summarize_queue.py'
before=hashlib.sha256(target.read_bytes()).hexdigest()
shutil.copy2(r/'summarize_queue_corrected.py',target)
cache=r/'new_december_results_v2/run_records';cache.mkdir(parents=True)
copied=[];recompute=[]
for p in sorted((r/'new_december_results/run_records').glob('*.json')):
    record=json.loads(p.read_text())
    if record['meta']['stage']=='density':
        recompute.append(p.name)
    else:
        shutil.copy2(p,cache/p.name)
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest==hashlib.sha256((cache/p.name).read_bytes()).hexdigest()
        copied.append(dict(name=p.name,sha256=digest))
report=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_before_sha256=before,source_after_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    preserved_records=copied,density_records_recomputed=recompute,
    reason='Density subset evaluation reports the full retained subset and its forecast leads. The low-history definition remains tied to the original full panel.',
    training_artifacts='unchanged')
(r/'density_evaluation_correction.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(preserved=len(copied),density_recomputed=len(recompute))))
