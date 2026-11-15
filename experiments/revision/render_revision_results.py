"""Render complete, five-seed revision evidence into paper tables and figures."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'paper/revision'
RESULTS=ROOT/'artifacts/revision_20261114/neural_results'
COLORS=['#3B7C9C','#D98253','#69A896','#A383A7','#606D7A']


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,
        'axes.labelweight':'bold','axes.titleweight':'bold',
        'axes.titlesize':11,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
        'xtick.major.width':.6,'ytick.major.width':.6,'pdf.fonttype':42,'ps.fonttype':42,
        'savefig.bbox':'tight','savefig.pad_inches':.06})


def pm(row,key,digits=3,scale=1):
    mean=row[key+'_mean']*scale;sd=row[key+'_std']*scale
    if not np.isfinite(mean):return '--'
    if not np.isfinite(sd):return f'{mean:.{digits}f}'
    return f'${mean:.{digits}f}\\pm {sd:.{digits}f}$'


def tex_table(path,caption,label,columns,rows,alignment):
    content=[r'\begin{table}[!htbp]',r'\centering',r'\begingroup',r'\singlespacing',
        r'\fontsize{9pt}{10.5pt}\selectfont',r'\setlength{\tabcolsep}{4pt}',r'\renewcommand{\arraystretch}{1.08}',
        '\\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{adjustbox}{max width=\linewidth}',
        r'\begin{tabular}{'+alignment+'}',r'\toprule',' & '.join(columns)+r' \\',r'\midrule']
    content += [r'\addlinespace[3pt]' if row is None else ' & '.join(row)+r' \\' for row in rows]
    content += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\endgroup',r'\end{table}']
    path.write_text('\n'.join(content)+'\n',encoding='utf8')


def result_frames(name,period='december'):
    folders=[RESULTS,RESULTS/'extension'] if period=='december' else [RESULTS/'january']
    files=[folder/(name+'_summary.csv') for folder in folders]
    frames=[pd.read_csv(path,keep_default_na=False,na_values=['']) for path in files if path.exists()]
    if not frames:raise ValueError(f'No {period} {name} results')
    return pd.concat(frames,ignore_index=True)


def one(table,model,t,adjacency='None',variant='full',loss='MAE',cohort='All',calibration='raw',stage=None):
    keep=(table.model==model)&(table['T']==t)&(table.adjacency==adjacency)&(table.variant==variant)
    if 'cohort' in table:keep &= table.cohort==cohort
    if 'calibration' in table:keep &= table.calibration==calibration
    keep &= table.loss==('deterministic' if adjacency=='None' else loss)
    if stage is not None:keep &= table.stage==stage
    else:keep &= ~table.stage.isin(['metadata','density'])
    found=table.loc[keep]
    if len(found)!=1:raise ValueError(f'Expected one complete row for {model}/{t}/{adjacency}/{variant}/{loss}/{cohort}/{calibration}; found {len(found)}')
    return found.iloc[0]


def forecasting():
    tables={period:result_frames('metrics',period) for period in ('december','january')}
    configurations=[(name,'None','full',label) for name,label in [
        ('Calendar HA','Historical average'),('Weekday HA','Weekday historical average'),
        ('Seasonal naive','Seasonal naive'),('Recent mean','Recent mean'),('AR-ridge','AR-ridge'),
        ('NLinear ridge','NLinear-ridge'),('TSB','TSB')]]
    configurations += [('Shared-LSTM','Identity','full','Shared-LSTM')]
    configurations += [(model,adj,'full',f'{"Fixed-MTGNN" if model=="Fixed-MTGNN" else model}, {adj}')
                        for model in ('GraphGRU','Fixed-MTGNN') for adj in ('Identity','NAICS-2','NAICS-4')]
    configurations += [('Adaptive-MTGNN','Adaptive','full','Adaptive-MTGNN'),('GNN-static','Identity','full','GNN-static')]
    rows=[]
    for model,adj,variant,label in configurations:
        rows.append([label]+[pm(one(tables[period],model,t,adj,variant),'MAE')
                              for period in ('december','january') for t in (14,28)])
    # Validate every required row before writing a paper table.
    tex_table(PAPER/'revision_forecasting.tex',
       'MAE for seven-day forecasts. Every December configuration uses 1,522,136 events from 22 origins, and every January configuration uses 1,729,700 events from 25 origins. Neural values are means and standard deviations over all five seeds; deterministic baselines have one value. January retains the original fitted weights and validation choices.',
       'tab:multihorizon',['Model / adjacency','Dec., $T=14$','Dec., $T=28$','Jan., $T=14$','Jan., $T=28$'],rows,'lrrrr')
    selected=[('NLinear ridge','None','NLinear-ridge'),('Shared-LSTM','Identity','Shared-LSTM'),
              ('GraphGRU','Identity','GraphGRU, identity'),('Fixed-MTGNN','Identity','Fixed-MTGNN, identity'),
              ('Adaptive-MTGNN','Adaptive','Adaptive-MTGNN')]
    fig,axes=plt.subplots(2,2,figsize=(7.5,5.0),layout='constrained',sharex=True,sharey=True)
    patterns=['-','--','-.','-',':']
    for i,period in enumerate(('december','january')):
        horizons=result_frames('horizons',period)
        for j,t in enumerate((14,28)):
            ax=axes[i,j]
            base=horizons[(horizons.model=='Calendar HA')&(horizons['T']==t)&(horizons.cohort=='All')].sort_values('lead').MAE_mean.to_numpy()
            assert len(base)==7
            for k,(model,adj,label) in enumerate(selected):
                block=horizons[(horizons.model==model)&(horizons.adjacency==adj)&(horizons['T']==t)&
                    horizons.stage.isin(['baseline','core','adaptive'])&(horizons.variant=='full')&
                    (horizons.loss==('deterministic' if adj=='None' else 'MAE'))&(horizons.cohort=='All')].sort_values('lead')
                assert len(block)==7
                skill=100*(1-block.MAE_mean.to_numpy()/base)
                ax.plot(np.arange(1,8),skill,color=COLORS[k],linestyle=patterns[k],lw=1.4,label=label)
            ax.axhline(0,color='#888888',lw=.6);ax.grid(axis='y',color='#E6E6E6',lw=.5)
            ax.set_title(f'{period.capitalize()} · T = {t}',loc='left',fontweight='bold')
            ax.set_xticks(range(1,8))
            if i==1:ax.set_xlabel('Forecast lead (days)')
            if j==0:ax.set_ylabel('MAE skill over HA (%)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False,fontsize=9)
    fig.savefig(PAPER/'forecasting_performance_across_horizons.pdf')
    fig.savefig(ROOT/'artifacts/revision_20261114/qa/forecasting_performance_across_horizons.png',dpi=180);plt.close(fig)
    rows=[]
    for model,adj,variant,label in [('Zero','None','full','Zero forecast')]+configurations:
        if model in ('Weekday HA','Seasonal naive','Recent mean','AR-ridge'):continue
        full=one(tables['december'],model,14,adj,variant)
        low=one(tables['december'],model,14,adj,variant,cohort='Low-history')
        rows.append([label,pm(full,'MAE'),pm(low,'MAE'),pm(low,'DMR',1,100),pm(low,'FAR_0.5',1,100),
                     pm(low,'recall_0.5',1,100),pm(low,'average_precision',1,100)])
    tex_table(PAPER/'revision_low_history.tex',
      'December forecasting diagnostics at $T=14$, $H=7$. The low-history cohort contains the same 1,944 POIs in every row. DMR, FAR, recall, and AP are percentages; activation uses 0.5 visits.',
      'tab:zero_collapse_diagnostic',['Model / adjacency','All MAE','Low-history MAE','DMR','FAR','Recall','AP'],rows,'lrrrrrr')
    uncertainty()
    print(json.dumps(dict(status='rendered',part='forecasting',all_required_rows_complete=True)))


def uncertainty():
    rows=[]
    for period in ('december','january'):
        table=pd.read_csv(RESULTS/'final_uncertainty'/period/'paired_intervals.csv')
        for t in (14,28):
            for model,adj,label in [('GraphGRU','Identity','GraphGRU, identity'),
                                    ('Fixed-MTGNN','Identity','Fixed-MTGNN, identity'),
                                    ('Adaptive-MTGNN','Adaptive','Adaptive-MTGNN')]:
                found=table[(table.model==f'{model}_T{t}_{adj}_full_MAE')&
                    (table.reference==f'NLinear ridge_T{t}')&(table.cohort=='All')&(table.cell_km==3.)]
                if len(found)!=1:raise ValueError(f'Incomplete paired interval: {period}/{model}/{t}')
                row=found.iloc[0]
                rows.append([period.capitalize(),str(t),label,f'{row.delta_MAE:+.3f}',
                             f'[{row.origin_7day_low:+.3f}, {row.origin_7day_high:+.3f}]',
                             f'[{row.joint_low:+.3f}, {row.joint_high:+.3f}]'])
    tex_table(PAPER/'revision_forecasting_intervals.tex',
        r'Paired MAE differences relative to NLinear-ridge. Negative values favor the neural model. Intervals use 2,000 paired resamples and the 2.5th and 97.5th percentiles, conditional on the five fitted seeds. Temporal blocks retain seven consecutive origins and all POIs and leads; the joint sensitivity also resamples occupied 3-km cells.',
        'tab:forecasting_intervals',['Period','$T$','Model / adjacency',r'$\Delta$MAE','Temporal interval','Joint interval'],rows,'lllrrr')


def low_history_core():
    table=result_frames('metrics')
    configs=[(model,'None',label) for model,label in [('Zero','Zero forecast'),('Calendar HA','Historical average'),
                 ('NLinear ridge','NLinear-ridge'),('TSB','TSB')]]
    configs += [('Shared-LSTM','Identity','Shared-LSTM'),('GNN-static','Identity','GNN-static')]
    configs += [(model,adj,f'{"Fixed-MTGNN" if model=="Fixed-MTGNN" else model}, {adj}')
                 for model in ('GraphGRU','Fixed-MTGNN') for adj in ('Identity','NAICS-2','NAICS-4')]
    rows=[]
    for model,adj,label in configs:
        full=one(table,model,14,adj);low=one(table,model,14,adj,cohort='Low-history')
        rows.append([label,pm(full,'MAE'),pm(low,'MAE'),pm(low,'DMR',1,100),pm(low,'FAR_0.5',1,100),
                     pm(low,'recall_0.5',1,100),pm(low,'average_precision',1,100)])
    tex_table(PAPER/'revision_low_history.tex',
       'December forecasting diagnostics at $T=14$, $H=7$. The low-history cohort contains the same 1,944 POIs in every row. DMR, FAR, recall, and AP are percentages; activation uses 0.5 visits.',
       'tab:zero_collapse_diagnostic',['Model / adjacency','All MAE','Low-history MAE','DMR','FAR','Recall','AP'],rows,'lrrrrrr')
    print(json.dumps(dict(status='core_table_rendered',part='low_history_core',adaptive_comparison_pending=True)))


def controls():
    tables={period:result_frames('metrics',period) for period in ('december','january')}
    rows=[]
    for model in ('GraphGRU','Fixed-MTGNN'):
        if rows:rows.append(None)
        for k in (5,10,20):
            row=['Fixed-MTGNN' if model=='Fixed-MTGNN' else model,str(k)]
            for period in ('december','january'):
                table=tables[period]
                for graph in ('NAICS-2','NAICS-4','Random'):
                    found=table[(table.model==model)&(table.stage=='graph')&(table.k==k)&
                        (table.graph==graph)&(table.cohort=='All')&(table.calibration=='raw')]
                    if len(found)!=1:raise ValueError(f'Incomplete controlled graph: {model}/{k}/{graph}/{period}')
                    row.append(pm(found.iloc[0],'MAE'))
            rows.append(row)
    tex_table(PAPER/'revision_degree_controls.tex',
        'Degree-controlled graph comparison at $T=14$, $H=7$. Each target has the same non-self degree under NAICS-2, NAICS-4, and random graphs, capped by $k$ and its available NAICS-4 peers. Values are five-seed MAE means and standard deviations.',
        'tab:semantic_granularity_transfer',['Model','$k$','Dec. NAICS-2','Dec. NAICS-4','Dec. random',
        'Jan. NAICS-2','Jan. NAICS-4','Jan. random'],rows,'llrrrrrr')
    rows=[]
    for model,adj in [('GraphGRU','Distance KNN'),('GraphGRU','Distance RBF'),('GraphGRU','Visit correlation'),('Fixed-MTGNN','Distance KNN')]:
        rows.append(['Fixed-MTGNN' if model=='Fixed-MTGNN' else model,adj]+[pm(one(tables[p],model,14,adj),'MAE') for p in ('december','january')])
    tex_table(PAPER/'revision_spatial_graphs.tex',
        'Additional fixed spatial and visit-correlation graphs at $T=14$, $H=7$. Values are five-seed MAE means and standard deviations.',
        'tab:spatial_graphs',['Model','Adjacency','December','January'],rows,'llrr')
    print(json.dumps(dict(status='rendered',part='controls')))


def ablations():
    tables={period:result_frames('metrics',period) for period in ('december','january')}
    variants=[('full','All static inputs'),('history_only','History only'),('without_location','Without location'),
       ('without_identity','Without identity'),('without_parent','Without parent'),('without_others','Without others'),
       ('only_category','Only top category'),('only_subcategory','Only sub-category'),('only_naics2','Only NAICS-2'),
       ('only_naics4','Only NAICS-4'),('without_naics2','Without NAICS-2'),('without_naics4','Without NAICS-4')]
    rows=[]
    for variant,label in variants:
        row=[label]
        for period in ('december','january'):
            row.extend(pm(one(tables[period],'GNN-static',14,'Identity',variant,cohort=cohort),'MAE')
                       for cohort in ('All','Low-history'))
        rows.append(row)
    tex_table(PAPER/'revision_feature_ablations.tex',
      'Retrained GNN-static variants at $T=14$, $H=7$. Values are five-seed MAE means and standard deviations. Single-descriptor variants retain all nonsemantic inputs, including brand and parent information.',
      'tab:feature_ablations',['Inputs','Dec., all','Dec., low-history','Jan., all','Jan., low-history'],rows,'lrrrr')
    rows=[]
    for period in ('december','january'):
        for metadata,label in [('core','Harmonized metadata'),('metadata','Earlier metadata')]:
            full=one(tables[period],'GNN-static',14,'Identity','full',cohort='Known parent',stage=metadata)
            removed=one(tables[period],'GNN-static',14,'Identity','without_parent',cohort='Known parent',stage='metadata' if metadata=='metadata' else 'ablation')
            rows.append([period.capitalize(),label,pm(full,'MAE'),pm(removed,'MAE')])
    tex_table(PAPER/'revision_parent_ablation.tex',
      'Parent-feature removal on the same 581 POIs with a recorded parent in the harmonized metadata. Earlier metadata uses the earliest available 2020 record and its parent-identifier crosswalk; POI membership is held fixed for comparison. Paired differences use unrounded seed-level errors.',
      'tab:parent_ablation',['Period','Metadata','Full MAE','Without-parent MAE'],rows,'llrr')
    print(json.dumps(dict(status='rendered',part='ablations')))


def density():
    folder=RESULTS/'density'
    protocol=json.loads((folder/'protocol.json').read_text());assert protocol['status']=='complete'
    table=pd.read_csv(folder/'density_summary.csv').set_index('subset')
    names=['Full']+[kind+str(level) for kind in ('Dense','Recent-dense','Persist-dense') for level in (10,25,50,75)]
    rows=[]
    for name in names:
        row=table.loc[name]
        rows.append([name,f"{int(row.POIs_mean):,}",f"{row.training_nonzero_ratio_mean:.2f}",
                    pm(row,'MAE'),f"{row.HA_MAE_mean:.3f}",pm(row,'skill',1)])
    tex_table(PAPER/'revision_density.tex',
        'Visit-density sensitivity at $T=28$, $H=7$. Each subset is selected from training data and refitted at all five seeds. HA uses the identical POIs and forecast events. Nonzero ratios use observed counts within the first 50 calendar days.',
        'tab:sublevel',['Subset','POIs','Nonzero ratio','GraphGRU MAE','HA MAE',r'Skill (\%)'],rows,'lrrrrr')
    print(json.dumps(dict(status='rendered',part='density')))


def diagnostics():
    tables={p:result_frames('metrics',p) for p in ('december','january')}
    far={p:result_frames('matched_far',p) for p in ('december','january')}
    configurations=[(model,'Identity',loss,label+' ('+loss+')')
                    for model,label in [('GraphGRU','GraphGRU'),('Fixed-MTGNN','Fixed-MTGNN'),('Shared-LSTM','Shared-LSTM')]
                    for loss in ('MAE','MSE')]
    configurations += [('NLinear ridge','None','deterministic','NLinear-ridge'),('TSB','None','deterministic','TSB')]
    rows=[]
    for model,adj,loss,label in configurations:
        raw=one(tables['december'],model,14,adj,loss=loss,cohort='Low-history')
        cal=one(tables['december'],model,14,adj,loss=loss,cohort='Low-history',calibration='calibrated')
        matched=one(far['december'][far['december'].target_validation_FAR==.1],model,14,adj,loss=loss)
        rows.append([label,pm(raw,'MAE'),pm(raw,'DMR',1,100),pm(raw,'FAR_0.5',1,100),
                     pm(cal,'DMR',1,100),pm(cal,'FAR_0.5',1,100),pm(matched,'test_FAR',1,100),pm(matched,'test_recall',1,100)])
    tex_table(PAPER/'revision_calibration_loss.tex',
        r'Loss-function and calibration diagnostics on low-history POIs in December at $T=14$, $H=7$. Neural models use identity adjacency. Calibration minimizes validation MAE. The last two columns use a threshold selected to keep validation FAR at or below 10\%; all rates are percentages.',
        'tab:calibration_loss',['Model (training loss)','Raw MAE','Raw DMR','Raw FAR',
        r'\makecell{Calibrated\\DMR}',r'\makecell{Calibrated\\FAR}',
        r'\makecell{Matched-threshold\\test FAR}',r'\makecell{Matched-threshold\\test recall}'],rows,'lrrrrrrr')
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.4),layout='constrained')
    model_colors={'GraphGRU':COLORS[0],'Fixed-MTGNN':COLORS[1],'Shared-LSTM':COLORS[2],
                  'NLinear ridge':COLORS[3],'TSB':COLORS[4]}
    for model,adj,loss,label in configurations:
        row=one(tables['december'],model,14,adj,loss=loss,cohort='Low-history')
        marker='s' if loss=='MSE' else 'o';color=model_colors[model]
        axes[0].errorbar(row['FAR_0.5_mean']*100,row['DMR_mean']*100,
             xerr=np.nan_to_num(row['FAR_0.5_std'])*100,yerr=np.nan_to_num(row['DMR_std'])*100,
             fmt=marker,color=color,ms=5,capsize=2,lw=.8,label=label)
        block=far['december']
        block=block[(block.model==model)&(block.adjacency==adj)&(block['T']==14)&(block.loss==loss)&(block.calibration=='raw')].sort_values('target_validation_FAR')
        assert len(block)==4
        axes[1].plot(block.test_FAR_mean*100,block.test_recall_mean*100,color=color,
                     marker=marker,markersize=3,lw=1.1,ls='--' if loss=='MSE' else '-')
    axes[0].set_xlabel('False activation rate (%)');axes[0].set_ylabel('Demand mass recovery (%)')
    axes[0].set_title('a  Fixed threshold: 0.5 visits',loc='left')
    axes[1].set_xlabel('Realized test false activation rate (%)');axes[1].set_ylabel('Positive-event recall (%)')
    axes[1].set_title('b  Validation-selected thresholds',loc='left')
    for ax in axes:
        ax.grid(color='#E6E6E6',lw=.5);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='outside upper center',frameon=False,fontsize=9)
    fig.savefig(PAPER/'recovery_activation_tradeoff.pdf')
    fig.savefig(ROOT/'artifacts/revision_20261114/qa/recovery_activation_tradeoff.png',dpi=180);plt.close(fig)
    # Later-period evidence uses the same validation calibration and thresholds.
    rows=[]
    for model,adj,loss,label in configurations:
        row=one(tables['january'],model,14,adj,loss=loss,cohort='Low-history')
        matched=one(far['january'][far['january'].target_validation_FAR==.1],model,14,adj,loss=loss)
        rows.append([label,pm(row,'MAE'),pm(row,'DMR',1,100),pm(row,'FAR_0.5',1,100),
                    pm(row,'average_precision',1,100),pm(matched,'test_FAR',1,100),pm(matched,'test_recall',1,100)])
    tex_table(PAPER/'revision_january_low_history.tex',
        'January diagnostics for the fixed low-history cohort at $T=14$, $H=7$. Models, calibration, and activation thresholds retain their original validation choices. All rates are percentages.',
        'tab:january_low_history',['Model (training loss)','MAE','DMR','FAR','AP','Matched test FAR','Matched test recall'],rows,'lrrrrrr')
    print(json.dumps(dict(status='rendered',part='diagnostics')))


def activity():
    tables={period:result_frames('activity_coverage',period) for period in ('december','january')}
    configurations=[('Calendar HA','None','Historical average'),('Weekday HA','None','Weekday historical average'),
      ('Seasonal naive','None','Seasonal naive'),('NLinear ridge','None','NLinear-ridge'),('TSB','None','TSB'),
      ('Shared-LSTM','Identity','Shared-LSTM'),('GraphGRU','Identity','GraphGRU, identity'),
      ('Fixed-MTGNN','Identity','Fixed-MTGNN, identity'),('Adaptive-MTGNN','Adaptive','Adaptive-MTGNN'),
      ('GNN-static','Identity','GNN-static')]
    rows=[]
    for model,adj,label in configurations:
        row=[label]
        for period in ('december','january'):
            for fraction in (.05,.1,.2):
                found=one(tables[period][tables[period].budget_fraction==fraction],model,14,adj)
                row.append(pm(found,'coverage',2,100))
        rows.append(row)
    oracle=['Observed-ranking upper bound']
    for period in ('december','january'):
        for fraction in (.05,.1,.2):
            row=one(tables[period][tables[period].budget_fraction==fraction],'Calendar HA',14)
            oracle.append(f"{row.oracle_coverage_mean*100:.2f}")
    rows += [None,oracle]
    tex_table(PAPER/'revision_activity_coverage.tex',
        r'Observed-visit coverage (\%) when prioritizing 13, 26, or 52 of the 256 occupied 1-km cells at $T=14$, $H=7$. Selection ranks each fitted model separately; reported values average coverage across seeds, origins, and leads. The last row ranks cells retrospectively by observed visits. Paired gains use unrounded coverage values.',
        'tab:activity_coverage',['Model','Dec., 13','Dec., 26','Dec., 52','Jan., 13','Jan., 26','Jan., 52'],rows,'lrrrrrr')
    print(json.dumps(dict(status='rendered',part='activity')))


def attribution(profiles_only=False):
    subdir='profile_checkpoint' if profiles_only else 'summary'
    folders={'december':RESULTS/'attribution'/subdir,'january':RESULTS/'attribution_january'/subdir}
    expected='profiles_complete_parent_pending' if profiles_only else 'complete'
    for folder in folders.values():assert json.loads((folder/'summary_protocol.json').read_text())['status']==expected
    groups=['history','location','identity','parent','others']
    labels=[g.capitalize()+'-driven' for g in groups]
    profiles={p:pd.read_csv(f/'profiles.csv') for p,f in folders.items()}
    comparisons=[]
    comparison_labels={('Gradient_x_Input','IG_reference_1'):'Gradient $\\times$ Input / IG 1',
                       ('Gradient_x_Input','IG_reference_2'):'Gradient $\\times$ Input / IG 2',
                       ('IG_reference_1','IG_reference_2'):'IG 1 / IG 2'}
    for period,folder in folders.items():
        methods=pd.read_csv(folder/'method_agreement.csv')
        for (a,b),label in comparison_labels.items():
            values=methods[(methods.method_A==a)&(methods.method_B==b)]
            assert len(values)==5
            comparisons.append([period.capitalize(),label,
                f'${100*values.regime_agreement.mean():.1f}\\pm {100*values.regime_agreement.std():.1f}$']+
                [f"{values[g+'_rank_spearman'].mean():.2f}" for g in groups])
        values=pd.read_csv(folder/'seed_agreement.csv');assert len(values)==10
        comparisons.append([period.capitalize(),'Different fitted seeds',
            f'{100*values.regime_agreement.mean():.1f}']+
            [f"{values[g+'_rank_spearman'].mean():.2f}" for g in groups])
    tex_table(PAPER/'revision_attribution_stability.tex',
      r'Attribution stability. Method comparisons use the same 1,024 sampled POIs and forecast events; regime agreement reports the mean and SD across five seeds. The last row in each period averages the ten seed-pair comparisons over all eligible POIs. Remaining columns are mean Spearman rank correlations for feature-group shares. Zero-attribution profiles are excluded.',
      'tab:attribution_stability',['Period','Comparison',r'Agreement (\%)','History','Location','Identity','Parent','Others'],comparisons,'llrrrrrr')
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.5),sharex=True,sharey=True,layout='constrained')
    for ax,(period,folder) in zip(axes,folders.items()):
        methods=pd.read_csv(folder/'method_agreement.csv')
        first=methods[(methods.method_A=='Gradient_x_Input')&(methods.method_B=='IG_reference_1')].sort_values('seed')
        second=methods[(methods.method_A=='Gradient_x_Input')&(methods.method_B=='IG_reference_2')].sort_values('seed')
        a=first[[g+'_share_mean_A' for g in groups]].to_numpy()*100
        np.testing.assert_allclose(a,second[[g+'_share_mean_A' for g in groups]].to_numpy()*100)
        values=[a,first[[g+'_share_mean_B' for g in groups]].to_numpy()*100,
                  second[[g+'_share_mean_B' for g in groups]].to_numpy()*100]
        for k,(label,points) in enumerate(zip(['Gradient × Input','IG reference 1','IG reference 2'],values)):
            assert points.shape==(5,5) and np.isfinite(points).all()
            ax.errorbar(points.mean(0),np.arange(5)+(k-1)*.20,xerr=points.std(0,ddof=1),
                fmt=['o','s','^'][k],color=COLORS[k],ms=4,capsize=2,lw=1,label=label)
        ax.set_yticks(range(5),[g.capitalize() for g in groups]);ax.set_ylim(4.5,-.5)
        ax.set_xlim(0,100);ax.set_xticks([0,20,40,60,80,100]);ax.grid(axis='x',color='#E6E6E6',lw=.5)
        ax.set_xlabel('Mean feature-group share (%)');ax.set_title(period.capitalize(),loc='left',fontweight='bold')
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='outside upper center',ncol=3,frameon=False,fontsize=9)
    fig.savefig(PAPER/'art/attribution_method_comparison.pdf')
    fig.savefig(ROOT/'artifacts/revision_20261114/qa/attribution_method_comparison.png',dpi=180);plt.close(fig)
    # Consensus profiles average the same five fitted seeds; no seed is selected.
    folder=folders['december'];regimes=pd.read_csv(folder/'regimes.csv').set_index('regime')
    fingerprint=pd.read_csv(folder/'fingerprint.csv')
    matrix=fingerprint.pivot(index='regime',columns='source',values='standardized_share').reindex(index=groups,columns=groups)
    counts=regimes.reindex(groups).POIs.fillna(0).to_numpy()
    fig,(ax,bx)=plt.subplots(1,2,figsize=(8.0,3.5),gridspec_kw={'width_ratios':[4,1.4]},layout='constrained')
    extent=max(1.,np.ceil(2*np.nanmax(np.abs(matrix.to_numpy())))/2)
    cmap=LinearSegmentedColormap.from_list('attribution',['#3B7C9C','#F8F8F5','#D98253'])
    im=ax.imshow(matrix,cmap=cmap,vmin=-extent,vmax=extent,aspect='auto')
    for i in range(5):
        for j in range(5):
            value=matrix.iloc[i,j]
            if np.isfinite(value):ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=10,
                fontweight='bold' if i==j else 'normal',color='white' if abs(value)>.75*extent else '#222222')
    ax.set_yticks(range(5),labels);ax.set_xticks(range(5),[g.capitalize() for g in groups])
    ax.xaxis.tick_top();ax.tick_params(length=0,pad=6)
    fig.colorbar(im,ax=ax,location='bottom',shrink=.9,aspect=35,label='Mean standardized attribution share',pad=.07)
    bx.barh(range(5),counts,color=COLORS[0],height=.5)
    for i,n in enumerate(counts):bx.text(n+max(counts)*.04,i,f'{int(n):,}',va='center',fontsize=10)
    bx.set_ylim(4.5,-.5);bx.set_xlim(0,max(counts)*1.5);bx.set_axis_off();bx.set_title('POIs',loc='left')
    fig.savefig(PAPER/'art/attribution_fingerprint.pdf')
    fig.savefig(ROOT/'artifacts/revision_20261114/qa/attribution_fingerprint.png',dpi=180);plt.close(fig)
    def escape(value):
        return str(value).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
    rows=[]
    for group,label in zip(groups,labels):
        if group not in regimes.index:continue
        row=regimes.loc[group]
        rows.append([label,f'{int(row.POIs):,}',f'{int(row.recorded_parent_POIs):,}',f'{row.MAE:.3f}'])
    tex_table(PAPER/'revision_attribution_regimes.tex',
      'December attribution regimes obtained from the five-seed consensus at $T=14$, $H=7$. Recorded parent counts POIs with an observed containing-place identifier. MAE averages each fitted model\'s absolute errors over the same POIs and all test events.',
      'tab:attri_regime',['Regime','POIs','Recorded parent','MAE'],rows,'lrrr')
    category=pd.read_csv(folder/'regime_categories.csv');rows=[]
    for _,row in category.iterrows():
        rows.append([escape(row.category),f'{int(row.POIs):,}']+[f'{row[g]*100:.1f}' for g in groups]+[f'{row.MAE:.3f}'])
    tex_table(PAPER/'revision_regime_categories.tex',
      r'December regime shares (\%) within broad POI categories. H, L, I, P, and O denote history-driven, location-driven, identity-driven, parent-driven, and others-driven, respectively. Shares use assigned POIs; MAE uses all POIs in the category.',
      'tab:category_regime_mae',['POI category','POIs','H','L','I','P','O','MAE'],rows,'lrrrrrrr')
    if not profiles_only:
        parent=pd.read_csv(folder/'parent_by_seed.csv');rows=[]
        for category,block in parent.groupby('category',sort=False):
            assert len(block)==5
            rows.append([escape(category),f'{int(block.POIs.iloc[0]):,}',
                  f'${block.parent_share.mean()*100:.1f}\\pm {block.parent_share.std()*100:.1f}$',
                  f'${block.full_MAE.mean():.3f}\\pm {block.full_MAE.std():.3f}$',
                  f'${block.removal_MAE_increase.mean():+.3f}\\pm {block.removal_MAE_increase.std():.3f}$'])
        tex_table(PAPER/'revision_parent_categories.tex',
           r'Parent attribution and retraining diagnostics in December on POIs with a recorded parent. The MAE change is without-parent minus full-model MAE on the same POIs. All entries average the five fitted seeds.',
           'tab_parent',['Embedded POI group','POIs',r'Parent share (\%)','Full MAE',r'$\Delta$MAE on removal'],rows,'lrrrr')
    else:
        parent=pd.read_csv(folder/'parent_profiles_by_seed.csv');rows=[]
        for category,block in parent.groupby('category',sort=False):
            assert len(block)==5
            rows.append([escape(category),f'{int(block.POIs.iloc[0]):,}',
                  f'${block.parent_share.mean()*100:.1f}\\pm {block.parent_share.std()*100:.1f}$',
                  f'${block.full_MAE.mean():.3f}\\pm {block.full_MAE.std():.3f}$'])
        tex_table(PAPER/'revision_parent_categories.tex',
           r'December attribution and forecasting error on POIs with a recorded parent. Shares exclude zero-attribution profiles; MAE uses all POIs in the group. Entries report five-seed means and SDs.',
           'tab_parent',['Embedded POI group','POIs',r'Parent share (\%)','Full MAE'],rows,'lrrr')
    rows=[]
    for descriptor,label in [('category','Top category'),('subcategory','Sub-category'),('naics2','NAICS-2'),('naics4','NAICS-4')]:
        row=[label]
        for period in ('december','january'):
            block=profiles[period];block=block[(block.method=='Gradient_x_Input')&(block.group==descriptor)]
            assert len(block)==5
            row += [f'${block.mean_share.mean()*100:.1f}\\pm {block.mean_share.std()*100:.1f}$',
                    f'${block.dominant_POI_fraction.mean()*100:.1f}\\pm {block.dominant_POI_fraction.std()*100:.1f}$']
        rows.append(row)
    tex_table(PAPER/'revision_semantic_attribution.tex',
       r'Semantic attribution under jointly encoded descriptors. Mean share (\%) uses total attribution across all inputs. Dominance (\%) is the fraction of POIs for which that descriptor has the largest aggregated semantic attribution; POIs with zero semantic attribution are excluded. Values are five-seed means and SDs.',
       'tab:semantic_granularity',['Descriptor','Dec. share','Dec. dominance','Jan. share','Jan. dominance'],rows,'lrrrr')
    print(json.dumps(dict(status='profiles_rendered_parent_pending' if profiles_only else 'rendered',part='attribution')))


def neighbors():
    source=RESULTS/'neighbors_final/summary'
    protocol=json.loads((source/'summary_protocol.json').read_text())
    assert protocol['status']=='complete' and len(protocol['provenance'])==20
    qa=ROOT/'artifacts/revision_20261114/qa';qa.mkdir(exist_ok=True)
    table=pd.read_csv(source/'graph_transfer_summary.csv')
    strata=['Low target / low neighbor','Low target / high neighbor','High target / low neighbor','High target / high neighbor']
    labels=['Low / low','Low / high','High / low','High / high']
    rows=[]
    for stratum,label in zip(strata,labels):
        row=[label]
        for period in ('december','january'):
            for graph in ('NAICS-2','NAICS-4'):
                found=table[(table.period==period)&(table.adjacency==graph)&(table.grouping=='evidence')&(table.stratum==stratum)]
                assert len(found)==1;row.append(pm(found.iloc[0],'MAE_gain'))
        rows.append(row)
    tex_table(PAPER/'revision_graph_evidence.tex',
        'MAE gain over identity adjacency within validation-defined target/neighbor evidence strata at $T=14$, $H=7$. Values are means and standard deviations across five fitted seeds; positive values indicate lower error under functional adjacency.',
        'tab:graph_transfer_summary',['Target / neighbor evidence','Dec. NAICS-2','Dec. NAICS-4','Jan. NAICS-2','Jan. NAICS-4'],rows,'lrrrr')
    import sys
    sys.path.insert(0,str(ROOT/'experiments/extension_20261114'))
    from render_graph_evidence_grid import render
    render(ROOT/'artifacts/revision_20261114/final_results/extension_20261114/graph_evidence_grid',
           PAPER/'art/pre_outcome_graph_transfer.pdf',qa/'pre_outcome_graph_transfer.png')
    interventions=pd.read_csv(source/'interventions_summary.csv')
    names={'active':'Training-active','recent_active':'Recent-active','correlated':'Correlated',
           'random':'Random','activity_matched_replacement':'Matched replacement'}
    rows=[]
    for period in ('december','january'):
        if rows:rows.append(None)
        for k,(name,label) in enumerate(names.items()):
            row=interventions[(interventions.period==period)&(interventions.adjacency=='NAICS-4')&
                               (interventions.grouping=='all')&(interventions.intervention==name)].iloc[0]
            rows.append([('December' if period=='december' else 'January') if k==0 else '',label,
                pm(row,'forecast_reduction_per_node',4),pm(row,'MAE_change',3),
                pm(row,'fraction_false_activations_removed',1,100),pm(row,'fraction_positive_activations_lost',1,100)])
    tex_table(PAPER/'revision_neighbor_interventions.tex',
        'Matched neighbor interventions under NAICS-4 at $T=14$, $H=7$, averaged over five seeds. Every rule intervenes on the same number of neighbors for each target. Removed false activations and lost positive activations are percentages of their originally activated zero-visit and positive-visit events, respectively.',
        'tab:metric_targeted_perturbation_naics4',['Period','Neighbor group',r'\makecell{Reduction\\per neighbor}',
        r'$\Delta\mathrm{MAE}$',r'\makecell{False activations\\removed (\%)}',r'\makecell{Positive activations\\lost (\%)}'],rows,'llrrrr')
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.2),sharey=True)
    fig.subplots_adjust(left=.24,right=.985,bottom=.19,top=.78,wspace=.13)
    for ax,period in zip(axes,('december','january')):
        block=interventions[(interventions.period==period)&(interventions.adjacency=='NAICS-4')&(interventions.grouping=='all')].set_index('intervention')
        y=np.arange(len(names))
        for offset,key,color,label in [(-.13,'fraction_false_activations_removed',COLORS[0],'False activations removed'),
                                     (.13,'fraction_positive_activations_lost',COLORS[1],'Positive activations lost')]:
            values=block.loc[list(names),key+'_mean']*100;errors=block.loc[list(names),key+'_std']*100
            ax.errorbar(values,y+offset,xerr=errors,fmt='o',ms=4,capsize=2,lw=.8,color=color,label=label)
        ax.set_yticks(y,list(names.values()));ax.set_ylim(len(names)-.5,-.5);ax.set_xlim(0,55)
        ax.set_title(period.capitalize(),loc='left',fontweight='bold')
        ax.grid(axis='x',color='#E5E7E7',lw=.5);ax.set_axisbelow(True)
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.57,.995),ncol=2)
    fig.supxlabel('Originally activated events (%)',x=.61,y=.035,fontsize=10)
    fig.savefig(PAPER/'art/neighbor_perturbation_tradeoff.pdf')
    fig.savefig(qa/'neighbor_perturbation_tradeoff.png',dpi=180);plt.close(fig)
    outputs=['revision_graph_evidence.tex','revision_neighbor_interventions.tex','art/pre_outcome_graph_transfer.pdf','art/neighbor_perturbation_tradeoff.pdf']
    manifest=dict(source_directory=str(source.relative_to(ROOT)),
        source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in source.glob('*.csv')},
        output_sha256={p:sha256((PAPER/p).read_bytes()).hexdigest() for p in outputs},
        renderer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (source/'paper_render_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='rendered',part='neighbors',outputs=outputs)))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--part',choices=['neighbors','forecasting','controls','ablations','density','diagnostics','activity','attribution','low_history_core'],required=True)
    p.add_argument('--profiles-only',action='store_true')
    p.add_argument('--results',type=Path,help='Directory containing the verified result summaries')
    p.add_argument('--paper',type=Path,help='Destination revision directory')
    args=p.parse_args()
    if args.results is not None:RESULTS=args.results.resolve()
    if args.paper is not None:PAPER=args.paper.resolve()
    style()
    if args.part=='attribution':attribution(args.profiles_only);raise SystemExit(0)
    {'neighbors':neighbors,'forecasting':forecasting,'controls':controls,'ablations':ablations,'density':density,
     'diagnostics':diagnostics,'activity':activity,'attribution':attribution,'low_history_core':low_history_core}[args.part]()
