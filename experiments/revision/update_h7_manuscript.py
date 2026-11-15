"""Render the H7 checkpoint table from explicit consolidated summaries."""
from argparse import ArgumentParser
from pathlib import Path
import ast,csv,hashlib,json
ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'paper/revision'
def run(results):
    sources=[results/'metrics_summary.csv',results/'january/metrics_summary.csv']
    data=[]
    for source in sources:
        rows={}
        for row in csv.DictReader(source.open()):
            if row['cohort']!='All' or row['calibration']!='raw' or row['stage'] not in ('baseline','core','adaptive'):continue
            if row['stage']!='baseline' and tuple(ast.literal_eval(row['seeds']))!=(42,43,44,45,46):continue
            assert int(row['H'])==7
            key=(row['model'],row['adjacency'],int(row['T']));assert key not in rows,key
            rows[key]=row
        data.append(rows)
    models=[(m,'None') for m in ('Zero','Calendar HA','Weekday HA','Last value','Seasonal naive','Recent mean','AR-ridge','NLinear ridge','TSB')]
    models += [(m,a) for m in ('GraphGRU','Fixed-MTGNN') for a in ('Identity','NAICS-2','NAICS-4')]
    models += [('Adaptive-MTGNN','Adaptive'),('Shared-LSTM','Identity'),('GNN-static','Identity')]
    lines=[r'\begin{table}[!htbp]',r'\centering\small',
        r'\caption{Mean absolute error at $H=7$. Neural results report the mean and sample standard deviation across seeds 42--46. A dash marks a configuration awaiting a complete five-seed evaluation. January uses frozen fitted parameters.}',
        r'\label{tab:multihorizon}',r'\begin{adjustbox}{max width=\linewidth}',r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Model & Adjacency & $T=14$, Dec. & $T=14$, Jan. & $T=28$, Dec. & $T=28$, Jan. \\',r'\midrule']
    missing=[]
    for model,adj in models:
        cells=[]
        for t in (14,28):
            for period,rows in zip(('december','january'),data):
                row=rows.get((model,adj,t))
                if row is None:
                    cells.append('--');missing.append(dict(model=model,adjacency=adj,T=t,period=period));continue
                value=f"{float(row['MAE_mean']):.3f}"
                if row['stage']!='baseline':value+=r' \pm '+f"{float(row['MAE_std']):.3f}"
                cells.append('$'+value+'$')
        label={'NLinear ridge':'NLinear-ridge','Last value':'Last observation'}.get(model,model)
        lines.append(' & '.join([label,'--' if adj=='None' else adj]+cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table}']
    output=PAPER/'revision_h7_forecasting.tex';output.write_text('\n'.join(lines)+'\n',encoding='utf8')
    report=dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),missing_configurations=missing)
    (results/'forecast_table_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    run(p.parse_args().results.resolve())
