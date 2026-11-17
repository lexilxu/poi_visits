"""Render source-backed AutoDL aggregate values without fitting or rescoring."""
from pathlib import Path
from hashlib import sha256
import csv
import json

from argparse import ArgumentParser
from exp.common.paths import DATA_ROOT, RESULTS_ROOT, REPO_ROOT


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Extension result root containing parent/ or parking/')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source=args.source
    hashes={}
    def read(path):
        hashes[str(path)]=sha256(path.read_bytes()).hexdigest()
        return list(csv.DictReader(path.open(encoding='utf8',newline='')))
    def save(name,lines):
        path=args.out/name;path.write_text('\n'.join(lines)+'\n',encoding='utf8')
        hashes[str(path)]=sha256(path.read_bytes()).hexdigest()
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
    (args.out/'table_audit.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')


if __name__ == '__main__':
    main()
