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
    (args.out/'table_audit.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')


if __name__ == '__main__':
    main()
