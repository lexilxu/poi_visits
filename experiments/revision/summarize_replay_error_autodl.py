"""Quantify reporting effects of the separate deterministic replay candidates."""
from pathlib import Path
import json
import sys
import numpy as np
root=Path('/root/autodl-tmp/poi_continuation_20261114')
sys.path.insert(0,str(root/'analysis_source_v2/experiments/revision'))
from audit_protocol import load_panel,low_history
raw,dates,attrs=load_panel(Path('/root/autodl-tmp/poi_revision_20261114a/approved_panel.zip'))
low,_=low_history(raw[:50]);assert int(low.sum())==1944
folder=root/'replay_deterministic_v4_audit'
audit=json.loads((folder/'audit.json').read_text());assert audit['status']=='diagnostics_complete'
rows=[]
for row in audit['rows']:
    name=f"{row['recipe']}_seed{row['seed']}"
    selection=next(root.glob(f"v*/queue/{row['recipe']}/selected_runs.json"))
    s=json.loads(selection.read_text());fit=selection.parent/f"seed{row['seed']}_lr{s['selected_lr']:g}"
    with np.load(folder/(name+'_differences_unverified.npz')) as new,np.load(fit/'predictions.npz') as old:
        before=new['origins']<92;test=old['dataset_split']=='test'
        p=new['predictions'][before][test].astype(float);q=old['predicted_visits'][test].astype(float)
        y=old['observed_visits'][test].astype(float);valid=old['valid'][test]
        details=[]
        for cohort,nodes in [('All',np.ones(len(attrs),bool)),('Low-history',low)]:
            mask=valid&nodes[None,None,:]
            loss_delta=np.abs(p-y)-np.abs(q-y)
            delta_mae=float(loss_delta[mask].mean())
            lead_deltas=[float(loss_delta[:,lead,:][mask[:,lead,:]].mean()) for lead in range(7)]
            rate_deltas=[]
            for threshold in (.1,.25,.5,1.,2.):
                rates=[]
                for prediction in (p,q):
                    active=(prediction>=threshold)&mask;positive=(y>0)&mask;zero=(y==0)&mask
                    tp=int((active&positive).sum());fp=int((active&zero).sum())
                    rates.append(np.array([fp/max(1,int(zero.sum())),tp/max(1,int(positive.sum())),tp/max(1,tp+fp)]))
                rate_deltas.append(dict(threshold=threshold,FAR_pp=100*float(rates[0][0]-rates[1][0]),
                    recall_pp=100*float(rates[0][1]-rates[1][1]),precision_pp=100*float(rates[0][2]-rates[1][2])))
            details.append(dict(cohort=cohort,delta_MAE=delta_mae,lead_delta_MAE=lead_deltas,rate_deltas=rate_deltas))
        rows.append(dict(recipe=row['recipe'],seed=row['seed'],details=details,
                         max_absolute_difference=float(np.abs(p-q).max()),
                         mean_absolute_difference=float(np.abs(p-q)[valid].mean())))
summary=dict(cases=len(rows),
    max_absolute_prediction_difference=max(r['max_absolute_difference'] for r in rows),
    max_mean_absolute_prediction_difference=max(r['mean_absolute_difference'] for r in rows),
    max_absolute_cohort_MAE_change=max(abs(d['delta_MAE']) for r in rows for d in r['details']),
    max_absolute_lead_cohort_MAE_change=max(abs(x) for r in rows for d in r['details'] for x in d['lead_delta_MAE']),
    max_absolute_rate_change_pp=max(abs(d[k]) for r in rows for c in r['details'] for d in c['rate_deltas'] for k in ('FAR_pp','recall_pp','precision_pp')))
(folder/'reporting_error_audit.json').write_text(json.dumps(dict(summary=summary,rows=rows,
    status='diagnostic_evidence_only',accepted_january_records=296),indent=2))
print(json.dumps(summary),flush=True)
