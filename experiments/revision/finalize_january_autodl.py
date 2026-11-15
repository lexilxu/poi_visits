"""Apply the user-approved, audited January numerical verification revision."""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace
import json
import shutil
import subprocess
import sys
import time
import numpy as np

root=Path('/root/autodl-tmp/poi_continuation_20261114')
scripts=root/'dependent_20261114/source/experiments/revision'
sys.path.insert(0,str(scripts))
from evaluate_january import extended_panel,save_and_evaluate
from summarize_queue import merge
from paired_seed_analysis import collected,pairs

base=root/'replay_deterministic_v4_audit'
audit=json.loads((base/'audit.json').read_text())
report=json.loads((base/'reporting_error_audit.json').read_text())
repeat=json.loads((root/'replay_deterministic_v4_repeat/repeat_audit.json').read_text())
assert len(audit['rows'])==19 and repeat['status']=='complete' and repeat['all_predictions_bit_identical']
assert report['summary']['max_absolute_cohort_MAE_change']<.00005
assert report['summary']['max_absolute_lead_cohort_MAE_change']<.00005
assert report['summary']['max_absolute_rate_change_pp']<.01
out=root/'january_numerical_revision_20261114';out.mkdir(exist_ok=False)
(out/'run_records').mkdir()
decision=dict(status='approved_by_user',approved_choice='Adopt audited numerical verification revision and finish January analysis and manuscript',
    scope='The 19 individually audited checkpoints only',
    original_strict_rtol=.0002,original_strict_atol=.0002,
    accepted_cohort_and_lead_MAE_change=.00005,accepted_rate_change_percentage_points=.01,
    required_independent_prediction_repeats='bit identical over December and January',
    training_protocol='Original weights, splits, seed selection and validation predictions retained',
    audit_sha256=sha256((base/'reporting_error_audit.json').read_bytes()).hexdigest(),
    repeat_audit_sha256=sha256((root/'replay_deterministic_v4_repeat/repeat_audit.json').read_bytes()).hexdigest(),
    created_at=time.time())
(out/'protocol_revision.json').write_text(json.dumps(decision,indent=2))
raw,dates,attrs=extended_panel(Path('/root/autodl-tmp/poi_revision_20261114a/approved_panel.zip'),root/'private_validation_202101/extended_panel.npz')
for row in audit['rows']:
    recipe,seed=row['recipe'],row['seed'];name=f'{recipe}_seed{seed}'
    selection=next(root.glob(f'v*/queue/{recipe}/selected_runs.json'))
    s=json.loads(selection.read_text());r=s['recipe']
    directory=selection.parent/f"seed{seed}_lr{s['selected_lr']:g}"
    assert sha256((directory/'checkpoint.pt').read_bytes()).hexdigest()==row['checkpoint_sha256']
    assert sha256((directory/'predictions.npz').read_bytes()).hexdigest()==row['original_predictions_sha256']
    first=base/(name+'_differences_unverified.npz')
    second=root/'replay_deterministic_v4_repeat'/first.name
    with np.load(first) as a,np.load(second) as b,np.load(directory/'predictions.npz') as old:
        np.testing.assert_array_equal(a['origins'],b['origins'])
        np.testing.assert_array_equal(a['predictions'],b['predictions'])
        origins=a['origins'];pred=a['predictions'].copy()
        np.testing.assert_array_equal(dates[origins[origins<92]],old['forecast_origin'])
        pred[origins<92]=old['predicted_visits']
    keep=(origins<64)|(origins>=92)
    fit=json.loads((directory/'fit.json').read_text())
    meta={k:r[k] for k in ('id','stage','model','T','H','adjacency','variant','loss','graph','k')}
    meta.update(seed=seed,selected_lr=s['selected_lr'],validation_MAE=fit['validation_MAE'],best_epoch=fit['best_epoch'])
    extra=dict(strict_replay_passed=False,numerical_replay_verified=True,
        numerical_protocol_revision='2026-11-14-user-approved-deterministic-graph',
        checkpoint_sha256=row['checkpoint_sha256'],original_predictions_sha256=row['original_predictions_sha256'],
        diagnostic_candidate_sha256=sha256(first.read_bytes()).hexdigest(),
        independent_repeat_sha256=sha256(second.read_bytes()).hexdigest(),
        numerical_replay_check=row,validation_predictions='Archived values restored bit-for-bit')
    save_and_evaluate(out,origins[keep],pred[keep],meta,raw,dates,attrs,extra)
assert len(list((out/'run_records').glob('*.json')))==19
merge(out/'run_records',out)
final=root/'consolidated_final_20261114';final.mkdir(exist_ok=False)
january=final/'january';records=january/'run_records';records.mkdir(parents=True)
predictions=january/'private_predictions';predictions.mkdir()
for source in (root/'consolidated_20261114/january',out):
    for p in (source/'run_records').glob('*.json'):
        assert not (records/p.name).exists()
        shutil.copy2(p,records/p.name)
        prediction=source/'private_predictions'/(p.stem+'.npz')
        assert prediction.is_file()
        (predictions/prediction.name).symlink_to(prediction.resolve())
assert len(list(records.glob('*.json')))==315
merge(records,january)
(final/'january_complete.json').write_text(json.dumps(dict(status='complete',records=315,baseline_records=20,
    neural_records=295,numerical_revision_cases=19,completed_at=time.time()),indent=2))
print(json.dumps(dict(completed='January evaluations',records=315)),flush=True)

args=SimpleNamespace(queues=[root/'v1/queue',root/'v2/queue'],baselines=root/'dependent_20261114/baselines',january=january)
groups=collected(args)
cache=final/'uncertainty_january/comparisons';cache.mkdir(parents=True)
reused=0
for model,reference in pairs(groups):
    name=model.replace(' ','_')+'__vs__'+reference.replace(' ','_')+'.json'
    previous=root/'january_uncertainty_verified_2039/comparisons'/name
    if not previous.exists():continue
    old=json.loads(previous.read_text())
    for key,field in [(model,'model_predictions'),(reference,'reference_predictions')]:
        assert [sha256(p.read_bytes()).hexdigest() for p in groups[key]['paths']]==old[field]
    shutil.copy2(previous,cache/name);reused+=1
print(json.dumps(dict(reused_verified_comparisons=reused)),flush=True)
common=['--queues',*map(str,args.queues),'--bundle','/root/autodl-tmp/poi_revision_20261114a/approved_panel.zip',
        '--baselines',str(args.baselines),'--january',str(january)]
for name,script in [('uncertainty_january','paired_seed_analysis.py'),('coverage_january','spatial_priority_analysis.py')]:
    with (final/(name+'.log')).open('x') as log:
        subprocess.run([sys.executable,'-u',str(scripts/script),*common,'--out',str(final/name)],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
    (final/(name+'_complete.json')).write_text(json.dumps(dict(status='complete',finished_at=time.time())))
    print(json.dumps(dict(completed=name)),flush=True)
