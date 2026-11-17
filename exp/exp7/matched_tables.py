"""Render already computed AutoDL summaries into a compact manuscript table."""
from pathlib import Path
from hashlib import sha256
import csv
import json

from argparse import ArgumentParser
from exp.common.paths import DATA_ROOT, RESULTS_ROOT, REPO_ROOT


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    source=args.source
    lookup={}
    hashes={}
    for period in ('december','january'):
        p=source/period/'metrics_summary.csv';hashes[str(p)]=sha256(p.read_bytes()).hexdigest()
        for row in csv.DictReader(p.open(encoding='utf8',newline='')):
            if row['calibration']=='raw':lookup[(period,int(row['T']),row['loss'],row['cohort'])]=row
    lines=[r'\begin{table}[!htbp]',r'\centering\small',
           r'\caption{Matched-LSTM magnitude errors. The input projection and last-observation readout match identity GraphGRU. Entries report five-seed mean MAE and sample standard deviation.}',
           r'\label{tab:matched_lstm}',r'\begin{tabular}{llrrrr}',r'\toprule',
           r' & & \multicolumn{2}{c}{December} & \multicolumn{2}{c}{January} \\',
           r'$T$ & Loss & All POIs & Low-history & All POIs & Low-history \\',r'\midrule']
    for t in (14,28):
        for loss in ('MAE','MSE'):
            cells=[]
            for period in ('december','january'):
                for cohort in ('All','Low-history'):
                    row=lookup[(period,t,loss,cohort)]
                    cells.append(f"${float(row['MAE_mean']):.3f}\\pm {float(row['MAE_std']):.3f}$")
            lines.append(f'{t} & {loss} & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    destination=args.out/'revision_matched_lstm.tex'
    destination.write_text('\n'.join(lines),encoding='utf8')
    hashes[str(destination)]=sha256(destination.read_bytes()).hexdigest()
    audit=args.out/'audit'
    audit.mkdir(parents=True,exist_ok=True)
    (audit/'matched_table.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')


if __name__ == '__main__':
    main()
