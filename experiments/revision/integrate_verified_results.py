"""Integrate completed density and feature-removal evidence into the revision."""
from pathlib import Path
import json,hashlib
import pandas as pd
import render_revision_results as render
root=Path(__file__).resolve().parents[2];paper=root/'paper/revision';evidence=root/'artifacts/e2031'
p=paper/'outline.tex';text=p.read_text(encoding='utf8')
old='% Insert the twelve completed H7 density experiments after the frozen extension queue finishes.'
new=r'''\input{revision_density.tex}
Table~\ref{tab:sublevel} compares each refitted model with the historical average on the same retained POIs and forecast events. MAE skill is $100(1-\mathrm{MAE}_{\mathrm{model}}/\mathrm{MAE}_{\mathrm{HA}})$. It is $12.1\%$ on the full panel and ranges from $11.4\%$ to $14.3\%$ across the subsets. The densest overall and persistence subsets attain $14.3\%$ skill, while their larger visit counts raise absolute MAE to approximately $2.77$.'''
assert old in text;text=text.replace(old,new)
old='These shares summarize the pooled parent encoding among locations with a recorded containing place.'
new=r'''The feature-removal comparison in Table~\ref{tab:parent_ablation} gives December MAE $1.454$ for the full model and $1.451$ after removing parent information from the harmonized metadata. The paired difference, defined as removed-feature MAE minus full-model MAE, is $-0.0023$, with a 95\% joint temporal and 1-km spatial interval of $[-0.0068,0.0014]$. January MAE changes from $1.469$ to $1.467$. With earlier metadata, the corresponding full and removed-feature values are $1.436$ and $1.436$ in December, and $1.453$ and $1.452$ in January. Parent-feature removal therefore has a small average effect in both metadata settings.

\input{revision_parent_ablation.tex}

Across recorded-parent POIs, the mean within-seed Spearman correlation between parent attribution share and the MAE increase after removal is $-0.048$ in December and $-0.072$ in January. These weak associations describe limited correspondence between attribution magnitude and predictive benefit.'''
assert old in text;text=text.replace(old,new)
old='Broad industry codes therefore receive the largest semantic attribution most often in these fitted representations; the magnitude depends on the fitted seed and attribution reference.'
new=old+r'''

\input{revision_feature_ablations.tex}

The retrained comparisons in Table~\ref{tab:feature_ablations} quantify predictive utility. Removing NAICS-2 increases December MAE by $0.0062$, with a 95\% joint temporal and 1-km spatial interval of $[0.0032,0.0090]$; removing NAICS-4 increases MAE by $0.0031$, with an interval of $[0.0016,0.0046]$. In January, the corresponding MAEs are $1.313$ and $1.312$, compared with $1.310$ for the full model. The history-only variant attains MAE $1.298$ in December and $1.306$ in January. Its December difference from the full model is $-0.0030$, with an interval of $[-0.0060,0.0002]$. Feature effects thus depend on which other inputs remain available.'''
assert old in text;text=text.replace(old,new)
old='This finding motivates evaluating feature utility through refitting alongside explanations of individual fitted models.'
new='Retraining shows small predictive contributions from NAICS-2 and NAICS-4 when the other inputs are retained, while parent-feature removal has little average effect. The history-only variant remains competitive with the full static-feature model.'
assert old in text;text=text.replace(old,new)
p.write_text(text,encoding='utf8')
data=pd.read_csv(evidence/'consolidated_20261114/parent_december/parent_by_seed.csv')
categories=data[data.seed==42].category.tolist();rows=[]
for category in categories:
    group=data[data.category==category]
    assert sorted(group.seed)==[42,43,44,45,46]
    def pm(name,digits=3,scale=1):
        return f'${group[name].mean()*scale:.{digits}f}\\pm {group[name].std()*scale:.{digits}f}$'
    rows.append([category,str(int(group.POIs.iloc[0])),pm('parent_share',1,100),pm('full_MAE'),pm('removal_MAE_increase')])
render.tex_table(paper/'revision_parent_categories.tex',
    'December attribution and feature-removal effects on POIs with a recorded parent. Shares exclude zero-attribution profiles; errors use all POIs in each group. The last column is removed-feature MAE minus full-model MAE. Entries report five-seed means and standard deviations.',
    'tab_parent',['Embedded POI group','POIs',r'Parent share (\%)','Full MAE',r'$\Delta\mathrm{MAE}$'],rows,'lrrrr')
sources=list((evidence/'results').rglob('*_summary.csv'))+[evidence/'dependent_20261114/uncertainty_december/paired_intervals.csv',evidence/'consolidated_20261114/parent_december/parent_by_seed.csv']
report=dict(sources={str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},
            outputs={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in [p,paper/'revision_density.tex',paper/'revision_feature_ablations.tex',paper/'revision_parent_ablation.tex',paper/'revision_parent_categories.tex']})
(evidence/'paper_integration.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print('Integrated verified density, feature-removal, and parent results.')
