"""Render source-backed AutoDL aggregate values without fitting or rescoring."""
from pathlib import Path
from hashlib import sha256
import csv
import json

root=Path(__file__).resolve().parents[2]
source=root/'artifacts/revision_20261114/final_results/extension_20261114'
hashes={}
def read(path):
    hashes[str(path.relative_to(root))]=sha256(path.read_bytes()).hexdigest()
    return list(csv.DictReader(path.open(encoding='utf8',newline='')))
def save(name,lines):
    path=root/'paper/revision'/name;path.write_text('\n'.join(lines)+'\n',encoding='utf8')
    hashes[str(path.relative_to(root))]=sha256(path.read_bytes()).hexdigest()
parent={}
for period in ('december','january'):
    for row in read(source/'parent'/period/'metrics_summary.csv'):
        if row['calibration']=='raw':parent[(period,row['variant'],row['cohort'])]=row
lines=[r'\begin{table}[!htbp]',r'\centering\small',r'\caption{Resolved parent encoding: five-seed mean MAE and sample standard deviation. Recorded-parent results cover 581 POIs.}',
       r'\label{tab:resolved_parent}',r'\begin{tabular}{lrrrr}',r'\toprule',r' & \multicolumn{2}{c}{December} & \multicolumn{2}{c}{January} \\',
       r'Parent input retained & All POIs & Recorded parent & All POIs & Recorded parent \\',r'\midrule']
for key,label in [('full','Presence and identity'),('without_parent_identity','Presence'),('without_parent_presence','Identity'),('without_parent','Neither')]:
    cells=[]
    for period in ('december','january'):
        for cohort in ('All','Known parent'):
            row=parent[(period,key,cohort)];cells.append(f"${float(row['MAE_mean']):.3f}\\pm {float(row['MAE_std']):.3f}$")
    lines.append(label+' & '+' & '.join(cells)+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
save('revision_resolved_parent.tex',lines)
parking={}
for row in read(source/'parking/metrics_summary.csv'):
    if row['lead']=='7' and row['radius_m'] in ('0','500'):parking[(row['period'],row['model'])]=row
lines=[r'\begin{table}[!htbp]',r'\centering\small',r'\caption{Seven-day-ahead paid-parking validation with a 500-m POI catchment. Recall is the percentage of observed high-pressure events selected under a 10\% monitoring budget. Neural-input rows report five-seed means.}',
       r'\label{tab:parking_validation}',r'\begin{tabular}{lrrrr}',r'\toprule',r' & \multicolumn{2}{c}{December} & \multicolumn{2}{c}{January} \\',r'Forecast input & Ratio MAE & Recall (\%) & Ratio MAE & Recall (\%) \\',r'\midrule']
models=[('Parking calendar','Parking calendar'),('Parking lag7','Parking seven-day lag'),('Parking ridge','Parking ridge'),
        ('NLinear-ridge_T14','+ NLinear-ridge'),('Shared-LSTM_T14_Identity_full_MAE','+ Shared-LSTM'),('Matched-LSTM_T14_Identity_full_MAE','+ Matched-LSTM'),
        ('GraphGRU_T14_Identity_full_MAE','+ GraphGRU, Identity'),('GraphGRU_T14_NAICS-2_full_MAE','+ GraphGRU, NAICS-2'),
        ('Fixed-MTGNN_T14_Identity_full_MAE','+ Fixed-MTGNN, Identity'),('Fixed-MTGNN_T14_NAICS-2_full_MAE','+ Fixed-MTGNN, NAICS-2')]
for key,label in models:
    cells=[]
    for period in ('december','january'):
        row=parking[(period,key)];cells += [f"{float(row['ratio_MAE_mean']):.4f}",f"{100*float(row['budget10_recall_mean']):.2f}"]
    lines.append(label+' & '+' & '.join(cells)+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
save('revision_parking_table.tex',lines)
audit=root/'artifacts/revision_20261114/private_substantive_20261114/render_audits'
audit.mkdir(parents=True,exist_ok=True)
(audit/'parent_parking_tables.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
