"""Format four original tables from verified full primary-seed attribution outputs."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json,re,zipfile,math
from collections import Counter
import pandas as pd


def tex(value):
    return str(value).translate(str.maketrans({'&':r'\&','%':r'\%','#':r'\#','_':r'\_',
        '{':r'\{','}':r'\}','\\':r'\textbackslash{}'}))


def fmt(value,decimals=3,scale=1):
    value=float(value)
    return f'{scale*value:.{decimals}f}' if math.isfinite(value) else '--'


def main(args):
    original=args.manuscript.read_text(encoding='utf8');edits=[];tables=[];checks={}
    with zipfile.ZipFile(args.evidence) as z:
        manifest=json.loads(z.read('manifest.json'))
        for folder in [args.tables,args.profiles]:
            for file in folder.iterdir():
                key='mechanisms/'+folder.name+'/'+file.name
                assert sha256(file.read_bytes()).hexdigest()==manifest[key],key
                checks[str(file)]=manifest[key]
    metadata=json.loads((args.profiles/'complete.json').read_text())
    assert metadata['status']=='complete' and metadata['seed']==42 and metadata['forecast_origins']==22

    def table(label,rows,caption,changes=()):
        pos=original.index('\\label{'+label+'}')
        start=original.rfind('\\begin{table}',0,pos);stop=original.index('\\end{table}',pos)+len('\\end{table}')
        before=original[start:stop];after=before
        a=after.index('\\midrule')+len('\\midrule');b=after.index('\\bottomrule',a)
        after=after[:a]+'\n'+'\n'.join(rows)+'\n'+after[b:]
        old_caption=re.search(r'\\caption\{([^\n]*)\}\n',after).group(0)
        after=after.replace(old_caption,'\\caption{'+caption+'}\n',1)
        for old,new in changes:
            assert after.count(old)==1,(label,old)
            after=after.replace(old,new,1)
        assert original.count(before)==1
        edits.append(dict(table=label,before=before,after=after));tables.append(after)

    clusters=pd.read_csv(args.tables/'04_attribution_clusters.csv')
    rows=[]
    for row in clusters.itertuples():
        names=Counter(str(row.representative_pois).split('; '))
        display='; '.join(name if count==1 else f'{name} ({count} POIs)' for name,count in names.items())
        representatives='--' if row.display_label=='Unassigned' else tex(display)
        rows.append(f'{tex(row.display_label)} & {row.n_pois:,} & {representatives} & {fmt(row.mae)}'+r' \\')
    table('tab:attri_regime',rows,
        r'Attribution regimes for the GNN-static fit at \(T=14\), \(H=7\). Representative POIs have the smallest Euclidean distances to the mean attribution-share profile within each assigned regime. Unassigned POIs have zero total own-input attribution.')

    categories=pd.read_csv(args.tables/'category_cluster_distribution.csv')
    rows=[]
    for _,row in categories.iterrows():
        cells=[tex(row.category9),f'{int(row.n_pois):,}']
        cells += [fmt(row[g],1,100) for g in ['History-driven','Location-driven','Identity-driven','Parent-driven','Others-driven','Unassigned']]
        rows.append(' & '.join(cells+[fmt(row.mae)])+r' \\')
    table('tab:category_regime_mae',rows,
        r'Attribution-regime shares within broad POI categories. H, L, I, P, O, and U denote history-driven, location-driven, identity-driven, parent-driven, others-driven, and unassigned, respectively. Shares and MAE use all POIs in each category.',
        [(r'Xrrrrrrr@{}',r'Xrrrrrrrr@{}'),(r'& O (\%) & MAE',r'& O (\%) & U (\%) & MAE')])

    parent=pd.read_csv(args.tables/'parent_attribution.csv');assert len(parent)==6
    labels=['Cafes and restaurants','Pharmacies and health retail','Retail and service shops',
        'Education-related POIs','Medical and clinic-related POIs','All embedded POIs']
    rows=[]
    for i,(row,label) in enumerate(zip(parent.itertuples(),labels)):
        if i==5:rows.append(r'\midrule')
        rows.append(' & '.join([label,f'{row.n_pois:,}',f'{row.assigned_pois:,}',
            fmt(row.parent_share,1,100),fmt(row.own_share,1,100),fmt(row.gap,1,100),fmt(row.mae)])+r' \\')
    table('tab_parent',rows,
        r'Attribution for POIs with a recorded parent place. Shares condition on assigned POIs with positive total own-input attribution; POI counts and MAE include all group members. Subgroups use the embedded POI top category. Gap is the parent share minus the combined share of the other sources. Dashes indicate an empty assigned group.',
        [(r'lccccc@{}',r'lcccccc@{}'),('& POIs\n','& POIs\n& Assigned\n')])

    semantic=pd.read_csv(args.profiles/'semantic_summary.csv')
    rows=[]
    for row in semantic.itertuples():
        rows.append(' & '.join([tex(row.descriptor),fmt(row.mean_share,2,100),fmt(row.dominance_rate,2,100),
            fmt(row.dominant_mae),f'{row.n_dominant:,}'])+r' \\')
    table('tab:semantic_granularity',rows,
        r'Attribution of four semantic descriptors in the GNN-static fit. Mean shares average over events with positive semantic attribution; each event is normalized by total own-input attribution. Dominance compares the four descriptors within the same eligible events.')

    assert args.manuscript.read_text(encoding='utf8')==original
    args.out.mkdir(parents=True,exist_ok=False)
    record=dict(status='candidate_only_pending_complete_paper_integration',source=str(args.manuscript),
        source_sha256=sha256(args.manuscript.read_bytes()).hexdigest(),source_files=checks,
        checkpoint_sha256=metadata['checkpoint_sha256'],seed=42,edits=edits,
        changes='Four original table bodies and necessary denominator/empty-group captions; one extra category column and one extra parent eligibility column.',
        remaining='Integrate affected result prose, remaining corrected experiments, five-seed confirmations and figures together.')
    (args.out/'attribution_table_corrections_candidate_v1.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    preamble=original.split(r'\begin{document}',1)[0]
    preview=preamble+r'\begin{document}'+'\n'+r'\setcounter{table}{10}'+'\n'+('\n'+r'\clearpage'+'\n').join(tables)+'\n'+r'\end{document}'+'\n'
    (args.out/'table_preview.tex').write_text(preview,encoding='utf8')
    print(json.dumps(dict(status=record['status'],tables=len(tables),live_manuscript_unchanged=True)))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for field in ['manuscript','tables','profiles','evidence','out']:p.add_argument('--'+field,type=Path,required=True)
    main(p.parse_args())
