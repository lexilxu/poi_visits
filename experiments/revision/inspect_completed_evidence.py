"""Print completed evidence without filling missing research results."""
from pathlib import Path
import json
from itertools import combinations
import numpy as np
import pandas as pd
from summarize_attribution import profile,agreement

ROOT=Path(__file__).resolve().parents[2]/'artifacts/revision_20261114/neural_results'


def run():
    for period,folder in [('December',ROOT),('January',ROOT/'january')]:
        table=pd.read_csv(folder/'metrics_summary.csv',keep_default_na=False,na_values=[''])
        block=table[(table['T']==14)&(table.stage.isin(['core','loss','baseline']))&(table.cohort=='Low-history')]
        columns=['model','adjacency','loss','calibration','MAE_mean','DMR_mean','FAR_0.5_mean','recall_0.5_mean','average_precision_mean']
        print(period+' low-history:')
        print(block[columns].to_csv(index=False))
    for period,folder in [('December',ROOT/'attribution'),('January',ROOT/'attribution_january')]:
        if not all((folder/f'seed{s}/attribution_protocol.json').exists() for s in range(42,47)):
            print(period+' attribution: not all seeds complete');continue
        full={};rows=[];methods=[]
        for seed in range(42,47):
            datasets={}
            for name in ('Gradient_x_Input','IG_reference_1','IG_reference_2'):
                with np.load(folder/f'seed{seed}/private_attribution'/(name+'.npz'),allow_pickle=False) as z:
                    datasets[name]=(z['poi_index'],z['absolute'])
                _,values=datasets[name];shares,regime,_,valid=profile(values,14)
                rows.append(dict(seed=seed,method=name,unassigned=int((~valid).sum()),
                       **dict(zip(['history','location','identity','parent','others'],shares[valid].mean(0).tolist()))))
            full[seed]=datasets['Gradient_x_Input'][1]
            for name in ('IG_reference_1','IG_reference_2'):
                sample,values=datasets[name]
                methods.append(dict(seed=seed,method=name,**agreement(full[seed][sample],values,14)))
        frame=pd.DataFrame(rows)
        print(period+' attribution shares (eligible POIs):')
        print(frame.groupby('method')[['history','location','identity','parent','others','unassigned']].mean().to_csv())
        print(period+' method agreement:')
        print(pd.DataFrame(methods).groupby('method').mean().to_csv())
        pairs=[agreement(full[a],full[b],14)['regime_agreement'] for a,b in combinations(range(42,47),2)]
        print(json.dumps(dict(period=period,mean_seed_regime_agreement=float(np.mean(pairs)),
                              minimum_seed_regime_agreement=float(np.min(pairs)),maximum_seed_regime_agreement=float(np.max(pairs)))))


if __name__=='__main__':run()
