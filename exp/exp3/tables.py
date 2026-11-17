"""Format the two original graph-contrast tables from audited primary-seed data."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json,re
import pandas as pd
from exp.exp5.tables import fmt


def main(args):
    manifest=json.loads(args.manifest.read_text())
    for name,digest in manifest.items():assert sha256((args.data/name).read_bytes()).hexdigest()==digest
    report=json.loads((args.data/'complete.json').read_text())
    assert report['status']=='partial_seed_export' and report['seeds']==[42]
    assert report['alias']=={'Top category':'NAICS-4'}
    frame=pd.read_csv(args.data/'transfer_regimes_by_seed.csv');assert set(frame.seed)=={42}
    original=args.manuscript.read_text(encoding='utf8');edits=[];tables=[]

    def table(label,rows,caption=None):
        pos=original.index('\\label{'+label+'}');start=original.rfind('\\begin{table}',0,pos)
        stop=original.index('\\end{table}',pos)+len('\\end{table}');before=original[start:stop]
        a=before.index('\\midrule')+len('\\midrule');b=before.index('\\bottomrule',a)
        after=before[:a]+'\n'+'\n'.join(rows)+'\n'+before[b:]
        if caption:
            old=re.search(r'\\caption\{([^\n]*)\}\n',after).group(0)
            after=after.replace(old,'\\caption{'+caption+'}\n',1)
        edits.append(dict(table=label,before=before,after=after));tables.append(after)

    rows=[]
    for row in frame[frame.adjacency.eq('NAICS-2')].itertuples():
        share=fmt(row.positive_event_share,2)+' / '+fmt(row.false_activation_share,2) if row.n_events else '--'
        cells=[row.regime,f'{row.n_events:,}']+[fmt(getattr(row,k),2) for k in ['graph_lift','target_evidence','neighbor_evidence','mean_mae_gain']]+[share]
        rows.append(' & '.join(cells)+r' \\')
    assert len(rows)==6
    table('tab:graph_transfer_summary',rows,
        r'Graph-transfer regimes under NAICS-2 at \(T=28\), \(H=14\). Shares of positive observed visits and false activations use all events in each regime. Dashes indicate an empty regime.')
    rows=[]
    for display in ['NAICS-2','NAICS-4','Top category']:
        graph='NAICS-4' if display=='Top category' else display
        for regime in ['Spurious transfer','Substitutive transfer']:
            selected=frame[frame.adjacency.eq(graph)&frame.regime.eq(regime)];assert len(selected)==1
            row=selected.iloc[0]
            share=fmt(row.positive_event_share,2)+' / '+fmt(row.false_activation_share,2) if row.n_events else '--'
            rows.append(' & '.join([display,regime.removesuffix(' transfer'),f'{row.n_events:,}',fmt(row.graph_lift,2),fmt(row.mean_mae_gain,2),share])+r' \\')
    table('tab:semantic_granularity_transfer',rows)
    args.out.mkdir(parents=True,exist_ok=False)
    payload=dict(status='candidate_only_pending_complete_paper_integration',source=str(args.manuscript),
        source_sha256=sha256(args.manuscript.read_bytes()).hexdigest(),inputs_sha256=manifest,seed=42,edits=edits,
        remaining='Integrate corrected descriptive prose and all-five-seed results; graph contrasts retain separately fitted parameters.')
    (args.out/'transfer_table_corrections_candidate_v1.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
    preamble=original.split(r'\begin{document}',1)[0]
    preview=preamble+r'\begin{document}'+'\n'+r'\setcounter{table}{7}'+'\n'+('\n'+r'\clearpage'+'\n').join(tables)+'\n'+r'\end{document}'+'\n'
    (args.out/'transfer_preview.tex').write_text(preview,encoding='utf8')
    assert args.manuscript.read_text(encoding='utf8')==original
    print(json.dumps(dict(status=payload['status'],tables=2,live_manuscript_unchanged=True)))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for field in ['manuscript','data','manifest','out']:p.add_argument('--'+field,type=Path,required=True)
    main(p.parse_args())
