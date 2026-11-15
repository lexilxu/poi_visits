"""Render the completed earlier-period city summaries."""
from pathlib import Path
from hashlib import sha256
import csv,json
root=Path(__file__).resolve().parents[2]
source=root/'artifacts/revision_20261114/final_results/extension_20261114/cities'
data={}
for row in csv.DictReader((source/'metrics_summary.csv').open(encoding='utf8')):data[(row['city'],row['model'],row['cohort'])]=row
lines=[r'\begin{table}[!htbp]',r'\centering\small',r'\caption{September temporal controls in July-defined city cohorts. Neural entries give five-seed mean MAE and sample standard deviation; all neural models use identity adjacency, $T=14$, and MAE training.}',
       r'\label{tab:city_controls}',r'\begin{tabular}{lrrrr}',r'\toprule',r' & \multicolumn{2}{c}{Seattle} & \multicolumn{2}{c}{Portland} \\',r'Model & All POIs & Low-history & All POIs & Low-history \\',r'\midrule']
for model in ('Zero','Calendar HA','Weekday HA','Seasonal naive','Recent mean','NLinear-ridge','GraphGRU','Matched-LSTM'):
    cells=[]
    for city in ('Seattle','Portland'):
        for cohort in ('All','Low-history'):
            row=data[(city,model,cohort)];value=f"{float(row['MAE_mean']):.3f}"
            if row['MAE_std']:value+=f"\\pm {float(row['MAE_std']):.3f}"
            cells.append('$'+value+'$')
    lines.append(model+' & '+' & '.join(cells)+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
p=root/'paper/revision/revision_city_controls.tex';p.write_text('\n'.join(lines)+'\n',encoding='utf8')
audit=root/'artifacts/revision_20261114/private_substantive_20261114/render_audits'
audit.mkdir(parents=True,exist_ok=True)
(audit/'city_table.json').write_text(json.dumps(dict(input_sha256=sha256((source/'metrics_summary.csv').read_bytes()).hexdigest(),table_sha256=sha256(p.read_bytes()).hexdigest()),indent=2))
