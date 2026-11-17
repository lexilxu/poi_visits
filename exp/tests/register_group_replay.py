"""Register individually verified replays under the reporting-error audit rule."""
from argparse import ArgumentParser
from exp.common.provenance import source_hash
from pathlib import Path
from hashlib import sha256
import json,socket,time
import numpy as np


def digest(p):return sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    first=a.audit/'repeat1';second=a.audit/'repeat2'
    left=json.loads((first/'audit.json').read_text());right=json.loads((second/'audit.json').read_text())
    assert left['status']==right['status']=='diagnostic_only'
    assert left['source_sha256']==right['source_sha256'] and left['cases_sha256']==right['cases_sha256']
    assert left['january_panel_sha256']==right['january_panel_sha256']
    other={row['run']:row for row in right['rows']}
    assert len(other)==len(left['rows'])==len(right['rows'])
    destination=a.root/'replay_acceptance';destination.mkdir(exist_ok=True)
    accepted=[]
    for row in left['rows']:
        name=row['run'];repeat=other[name]
        assert row['parameters_unchanged'] and repeat['parameters_unchanged']
        original=a.root/'runs'/name
        fit=json.loads((original/'complete.json').read_text())
        assert row['checkpoint_sha256']==repeat['checkpoint_sha256']==fit['checkpoint_sha256']==digest(original/'best_weights.pt')
        assert row['original_prediction_sha256']==repeat['original_prediction_sha256']==fit['predictions_sha256']==digest(original/'predictions.npz')
        canonical=first/(name+'.npz')
        with np.load(canonical) as p,np.load(second/(name+'.npz')) as q:
            assert set(p.files)==set(q.files)=={'december','january','december_origin','january_origin'}
            for k in p.files:np.testing.assert_array_equal(p[k],q[k])
            for k in ['december','january']:
                assert np.isfinite(p[k]).all()
                assert sha256(p[k].tobytes()).hexdigest()==row[k+'_prediction_sha256']==repeat[k+'_prediction_sha256']
        cohorts=row['changes']['cohorts']
        bounds=dict(max_cohort_MAE_change=max(abs(c['delta_MAE']) for c in cohorts),
            max_lead_cohort_MAE_change=max(abs(v) for c in cohorts for v in c['lead_delta_MAE']),
            max_day_cohort_MAE_change=max(abs(v) for c in cohorts for v in c['day_delta_MAE'].values()),
            max_rate_change_pp=max(abs(v[k]) for c in cohorts for v in c['thresholds'] for k in ['FAR_pp','recall_pp','precision_pp']))
        assert all(bounds[k]<5e-5 for k in ['max_cohort_MAE_change','max_lead_cohort_MAE_change','max_day_cohort_MAE_change'])
        assert bounds['max_rate_change_pp']<.01
        entry=dict(status='accepted_numerical_replay',run=name,
            checkpoint_sha256=fit['checkpoint_sha256'],original_predictions_sha256=fit['predictions_sha256'],
            source_model_sha256=source_hash(fit, 'exp/common/evolvegcn.py'),
            operator_sha256=left['source_sha256']['deterministic_group.py'],
            canonical_predictions=str(canonical),canonical_file_sha256=digest(canonical),
            december_prediction_sha256=row['december_prediction_sha256'],january_prediction_sha256=row['january_prediction_sha256'],
            january_panel_sha256=left['january_panel_sha256'],independent_process_predictions_bitwise_equal=True,
            strict_December_replay_passed=row['changes']['strict_allclose_1e_5'],
            strict_validation_replay_passed=row['validation_strict_allclose_1e_5'],bounds=bounds,
            audit_sha256=[digest(first/'audit.json'),digest(second/'audit.json')],
            criteria=dict(cohort_lead_day_MAE_change_below=5e-5,FAR_recall_precision_change_pp_below=.01,
                thresholds=[.1,.25,.5,1.,2.],cohorts=['All','Low-history'],independent_process_repeats=2),
            scope='Deterministic numerical replay of this fixed checkpoint only. Original predictions, training, validation selection and parameters remain unchanged.',
            method_basis='Reuse the previously authorized reporting-error audit method; this entry records the separate verification of this corrected checkpoint.',
            registered_utc=time.time())
        file=destination/(name+'.json')
        if file.exists():
            saved=json.loads(file.read_text())
            for k in ['checkpoint_sha256','canonical_file_sha256','operator_sha256','january_prediction_sha256']:assert saved[k]==entry[k]
        else:
            with file.open('x') as f:json.dump(entry,f,indent=2,allow_nan=False)
        accepted.append(name)
    print(json.dumps(dict(registered=len(accepted),runs=accepted)),flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    main(p.parse_args())
