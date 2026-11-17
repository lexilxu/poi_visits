"""Prepare exact EvolveGCN row replacements while retaining original comparators."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json,re
import pandas as pd


def main(args):
    original=args.manuscript.read_text(encoding='utf8');candidate=original;edits=[]
    manifest=json.loads(args.input_manifest.read_text())
    for name,digest in manifest.items():assert sha256((args.data/name).read_bytes()).hexdigest()==digest
    meta=json.loads((args.data/'complete.json').read_text());assert meta['status']=='complete_primary_figure_inputs' and meta['seed']==42
    horizon=pd.read_csv(args.data/'07_horizon_skill.csv',keep_default_na=False)
    recovery=pd.read_csv(args.data/'01_recovery_activation.csv',keep_default_na=False)
    choices={(x['T'],x['H']):x['EvolveGCN_adjacency'] for x in meta['selection']}

    def replace(before,after,label):
        nonlocal candidate
        assert candidate.count(before)==1
        candidate=candidate.replace(before,after,1)
        edits.append(dict(table=label,before=before,after=after))

    def body(label):
        start=original.index('\\label{'+label+'}')
        start=original.index('\\midrule',start)+len('\\midrule')
        stop=original.index('\\bottomrule',start)
        return original[start:stop]

    def find(frame,**criteria):
        selected=frame
        for key,value in criteria.items():selected=selected[selected[key].eq(value)]
        assert len(selected)==1,criteria
        return selected.iloc[0]

    for line in body('tab:multihorizon').splitlines():
        if '& EvolveGCN &' not in line:continue
        columns=[x.strip() for x in line.removesuffix('\\\\').split('&')]
        t,h=map(int,re.fullmatch(r'\((\d+),(\d+)\)',columns[0]).groups())
        graph='Identity' if columns[2]=='Identity' else choices[t,h]
        row=find(recovery,model='EvolveGCN',adjacency=graph) if t==14 else find(horizon,H=h,model='EvolveGCN',adjacency=graph)
        values=[row[k] for k in ['mae','rse','corr','skill_percent','anomaly_corr','delta_dmr','far_0_5']]
        after=' & '.join([columns[0],'EvolveGCN',graph]+[f'{float(v):.2f}' for v in values])+r' \\'
        replace(line,after,'tab:multihorizon')
    for line in body('tab:zero_collapse_diagnostic').splitlines():
        if not line.startswith('EvolveGCN &'):continue
        graph=line.split('&')[1].strip();row=find(recovery,model='EvolveGCN',adjacency=graph)
        values=[row[k] for k in ['mae','rmse','lift','dmr_model','dmr_ha','delta_dmr','far_0_5','mean_prediction']]
        after=' & '.join(['EvolveGCN',graph]+[f'{float(v):.2f}' for v in values])+r' \\'
        replace(line,after,'tab:zero_collapse_diagnostic')
    assert len(edits)==14
    for label in ['tab:multihorizon','tab:zero_collapse_diagnostic']:
        for line in body(label).splitlines():
            if '&' in line and 'EvolveGCN &' not in line:assert candidate.count(line)==original.count(line)
    assert args.manuscript.read_text(encoding='utf8')==original
    record=dict(status='candidate_only_pending_complete_paper_integration',source=str(args.manuscript),
        source_sha256=sha256(args.manuscript.read_bytes()).hexdigest(),seed=42,input_sha256=manifest,
        edits=edits,unchanged='All comparator rows, table formatting and all other manuscript text.',
        remaining='Integrate affected prose, remaining corrected tables, five-seed confirmations and figures together.')
    with args.out.open('x',encoding='utf8') as f:json.dump(record,f,indent=2,ensure_ascii=False)
    print(json.dumps(dict(status=record['status'],replaced_rows=len(edits),live_manuscript_unchanged=True)))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for name in ['manuscript','data','input-manifest','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
