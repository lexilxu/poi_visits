"""Analyze verified complete January seed groups on AutoDL4."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

root = Path('/root/autodl-tmp/poi_continuation_20261114')
scripts = root / 'dependent_20261114/source/experiments/revision'
sys.path.insert(0, str(scripts))
from paired_seed_analysis import collected, pairs
from types import SimpleNamespace

out = root / 'january_uncertainty_verified_2039'
out.mkdir(exist_ok=False)
args = SimpleNamespace(queues=[root/'v1/queue', root/'v2/queue'],
                       baselines=root/'dependent_20261114/baselines',
                       january=root/'consolidated_20261114/january')
groups = collected(args)
plan = dict(scope='Complete five-seed groups with accepted January replay records',
            groups=sorted(groups), comparisons=pairs(groups),
            unresolved_replay_cases=19, overall_january_status='partial',
            started_at=time.time())
(out/'scope.json').write_text(json.dumps(plan, indent=2))
command = [sys.executable, '-u', str(scripts/'paired_seed_analysis.py'),
           '--queues', *map(str, args.queues), '--bundle',
           '/root/autodl-tmp/poi_revision_20261114a/approved_panel.zip',
           '--baselines', str(args.baselines), '--january', str(args.january), '--out', str(out)]
with (out/'analysis.log').open('x') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
completed = list((out/'comparisons').glob('*.json'))
assert len(completed) == len(plan['comparisons'])
(out/'complete.json').write_text(json.dumps(dict(
    status='verified_subset_complete', comparisons=len(completed),
    overall_january_status='partial', completed_at=time.time(),
    summary_sha256=hashlib.sha256((out/'paired_intervals.csv').read_bytes()).hexdigest()), indent=2))
print(json.dumps(dict(status='verified_subset_complete', comparisons=len(completed))), flush=True)
