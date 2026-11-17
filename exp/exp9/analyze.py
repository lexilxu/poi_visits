"""Forecast reductions from saved archives; no training framework required."""
from exp.common.paths import run_path
from pathlib import Path, PureWindowsPath
from io import BytesIO
from zipfile import ZipFile
from hashlib import sha256
import json
import time
import argparse
import numpy as np
import pandas as pd

BASE = run_path('poi_continuation_20261114')
EXT = run_path('poi_extension_20261114')
BUNDLE = run_path('poi_formal_2020q4_12d6414990ee/approved_panel.zip')
DIGEST = '12d6414990eeb5703b1179152b5993518dcfa68361e39e09ae621919950695ec'
RECIPES = {
    'Shared LSTM': 'Shared-LSTM_T14_Identity_full_MAE',
    'Matched LSTM': 'Matched-LSTM_T14_Identity_full_MAE',
    'MTGNN (NAICS-2)': 'Fixed-MTGNN_T14_NAICS-2_full_MAE',
    'GraphGRU (Identity)': 'GraphGRU_T14_Identity_full_MAE',
    'GraphGRU (NAICS-2)': 'GraphGRU_T14_NAICS-2_full_MAE',
    'GraphGRU (NAICS-4)': 'GraphGRU_T14_NAICS-4_full_MAE',
}
MODELS = ['Historical average', 'AR-ridge'] + list(RECIPES)
EXPECTED = {
    'December': [(1.483,None),(1.453,None),(1.360,.013),(1.298,.015),(1.222,.008),(1.302,.007),(1.336,.007),(1.350,.004)],
    'January': [(1.590,None),(1.583,None),(1.366,.012),(1.309,.014),(1.244,.011),(1.311,.007),(1.351,.010),(1.361,.007)]}
IDENTITY = 'GraphGRU (Identity)'
GRAPH_MODELS = [IDENTITY, 'GraphGRU (NAICS-2)', 'GraphGRU (NAICS-4)']
CELLS = ['Low/Low','Low/High','High/Low','High/High']

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)

def sources(period):
    result = {}
    jan = BASE/'consolidated_final_20261114/january/private_predictions'
    baselines = BASE/'dependent_20261114/baselines'
    result['Historical average'] = [jan/'Calendar_HA_T14_seed-1.npz' if period=='January' else baselines/'Calendar_HA_T14_H7.npz']
    result['AR-ridge'] = [jan/'AR-ridge_T14_seed-1.npz' if period=='January' else baselines/'AR_ridge_T14_H7.npz']
    selections = []
    for name, recipe in RECIPES.items():
        folder = (EXT/'matched_queue' if name=='Matched LSTM' else BASE/'v1/queue')/recipe
        selection = folder/'selected_runs.json'
        selections.append(selection)
        runs = json.loads(selection.read_text())['runs']
        assert sorted(r['seed'] for r in runs)==list(range(42,47))
        if period=='January':
            parent = EXT/'matched_evaluation/january/private_predictions' if name=='Matched LSTM' else jan
            result[name] = [parent/f'{recipe}_seed{r["seed"]}.npz' for r in runs]
        else:
            result[name] = [folder/PureWindowsPath(r['directory']).name/'predictions.npz' for r in runs]
    return result, selections

def joint_interval(difference, valid, spatial):
    n = len(valid); ng = int(spatial.max()+1)
    sums = np.stack([np.bincount(spatial, weights=np.where(valid[i],difference[i],0).sum(0), minlength=ng) for i in range(n)])
    counts = np.stack([np.bincount(spatial, weights=valid[i].sum(0), minlength=ng) for i in range(n)])
    use = counts.sum(0)>0
    sums,counts=sums[:,use],counts[:,use]; ng=int(use.sum())
    rng=np.random.default_rng(421); draws=[]
    for _ in range(2000):
        origins=((rng.integers(n,size=int(np.ceil(n/7)))[:,None]+np.arange(7))%n).ravel()[:n]
        weights=np.bincount(rng.integers(ng,size=ng),minlength=ng)
        den=counts[origins].sum(0)@weights
        if den: draws.append(float(sums[origins].sum(0)@weights/den))
    assert len(draws)>=1990
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(estimate=float(sums.sum()/counts.sum()),ci_low=float(lo),ci_high=float(hi),spatial_cells=ng,bootstrap_draws=len(draws))

def metrics(y,p,keep):
    a,b=y[keep],p[keep]; pos=a>0; zero=a==0
    return dict(events=len(a),POIs=int(keep.any((0,1)).sum()),positive_events=int(pos.sum()),zero_events=int(zero.sum()),
        MAE=float(np.abs(a-b).mean()),
        recall=float((b[pos]>=.5).mean()) if pos.any() else None,
        FAR=float((b[zero]>=.5).mean()) if zero.any() else None,
        DMR=float(b[pos].sum()/a[pos].sum()) if pos.any() else None)

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    tracked={}; audits=[]; strata_rows=[]; gains=[]; costs=[]; cost_pairs=[]; overall=[]
    def track(p):
        p=Path(p); tracked[str(p)]=digest(p)
    track(BUNDLE); assert tracked[str(BUNDLE)]==DIGEST
    track(Path(__file__)); track((Path(__file__).resolve().parents[2] / 'exp/exp9/PROTOCOL.md'))
    with ZipFile(BUNDLE) as z:
        raw=np.load(BytesIO(z.read('visits.npy')),allow_pickle=False)
        attrs=pd.read_csv(BytesIO(z.read('poi_attributes.csv')))
        original_dates=pd.read_csv(BytesIO(z.read('dates.csv'))).date.to_numpy(str)
    ep=BASE/'private_validation_202101/extended_panel.npz';track(ep)
    with np.load(ep,allow_pickle=False) as z:
        extended=z['visits'].astype(float);dates=z['dates'].astype(str);ids=z['poi_id'].astype(str)
    np.testing.assert_array_equal(extended[:92],raw)
    np.testing.assert_array_equal(ids,attrs.placekey.to_numpy(str))
    np.testing.assert_array_equal(dates[:92],original_dates)
    labels={g:pd.factorize(attrs.naics_code.astype('int64').astype(str).str[:d])[0] for g,d in [('NAICS-2',2),('NAICS-4',4)]}
    eligible=np.ones(len(ids),bool)
    for lab in labels.values():eligible &= np.bincount(lab)[lab]>1
    filled=np.nan_to_num(extended,nan=0.)
    history=np.stack([filled[o-14:o].mean(0) for o in range(14,len(filled)-6)])
    train_origins=np.arange(14,44)
    train_history=history[train_origins-14]
    target_cut=float(np.median(train_history[:,eligible]))
    neighbors={};cuts={}
    for graph,lab in labels.items():
        sums=np.zeros((len(history),lab.max()+1))
        np.add.at(sums.T,lab,history.T)
        degree=np.bincount(lab)[lab]-1
        neighbor=np.divide(sums[:,lab]-history,degree[None,:],out=np.zeros_like(history),where=degree[None,:]>0)
        assert neighbor.min()>-1e-10
        neighbors[graph]=neighbor
        cuts[graph]=float(np.median(neighbor[train_origins-14][:,eligible]))
    count=np.isfinite(raw[:50]).sum(0);means=np.nansum(raw[:50],0)/count
    low_cut=float(np.quantile(means[means>0],.25))
    low=(means>0)&(means<=low_cut)&(((raw[:50]==0).sum(0)/count)>=.7)
    assert low.sum()==1944
    xy=np.column_stack([attrs.longitude.to_numpy()*111.320*np.cos(np.deg2rad(47.6)),attrs.latitude.to_numpy()*110.574])
    _,spatial=np.unique(np.floor(xy/3).astype(int),axis=0,return_inverse=True)
    thresholds=dict(target=target_cut,neighbors=cuts,training_origins=train_origins.tolist(),training_origin_dates=dates[train_origins].tolist(),
        eligible_POIs=int(eligible.sum()),excluded_POIs=int((~eligible).sum()),low_history_POIs=int(low.sum()),low_history_mean_cutoff=low_cut,
        ties='Low',history_missing='zero placeholders',training_threshold_sha256=sha256(train_history[:,eligible].tobytes()).hexdigest())
    write_json(out/'thresholds.json',thresholds)
    log('Training-only thresholds frozen: '+json.dumps(thresholds))
    for period,origins in [('December',np.arange(64,86)),('January',np.arange(92,117))]:
        y=extended[origins[:,None]+np.arange(7)];valid=np.isfinite(y)
        strata={g:((history[origins-14]>target_cut)*2+(neighbors[g][origins-14]>cuts[g])).astype('int8') for g in labels}
        masks={g:{c:valid & eligible[None,None,:] & (s==k)[:,None,:] for k,c in enumerate(CELLS)} for g,s in strata.items()}
        for graph in masks:
            np.testing.assert_array_equal(sum(masks[graph].values()),valid & eligible[None,None,:])
        files,selections=sources(period)
        for p in selections:track(p)
        losses={};model_maes={}
        for name,paths in files.items():
            model_maes[name]=[];under=np.zeros_like(y);over=np.zeros_like(y);seeds=[]
            for path in paths:
                track(path)
                with np.load(path,allow_pickle=False) as z:
                    d=dict(z)
                test=d['dataset_split']=='test';p=d['predicted_visits'][test].astype(float)
                np.testing.assert_array_equal(d['poi_id'].astype(str),ids)
                np.testing.assert_array_equal(d['forecast_origin'][test].astype(str),dates[origins])
                np.testing.assert_array_equal(d['horizon_step'],np.arange(1,8))
                np.testing.assert_array_equal(d['observed_visits'][test],y)
                np.testing.assert_array_equal(d['valid'][test],valid)
                assert int(d['T'])==14 and int(d['H'])==7 and int(d['train_end'])==50
                assert str(d['bundle_sha256'])==DIGEST
                assert np.isfinite(p).all() and p.min()>=0
                seed=int(d['seed']);seeds.append(seed)
                u=np.where(valid,np.maximum(y-p,0),0);o=np.where(valid,np.maximum(p-y,0),0)
                under+=u/len(paths);over+=o/len(paths)
                direct=np.abs(y[valid]-p[valid]).mean();model_maes[name].append(float(direct))
                overall.append(dict(period=period,model=name,seed=seed,**metrics(y,p,valid)))
                for cohort,poi_keep in [('All',np.ones(len(ids),bool)),('Low-history',low)]:
                    keep=valid & poi_keep[None,None,:]
                    um=float(u[keep].mean());om=float(o[keep].mean());mae=float(np.abs(y[keep]-p[keep]).mean())
                    assert abs(um+om-mae)<1e-12
                    for lam in [.5,1,2,3,5]:
                        # Independent direct formula guards the component reduction.
                        direct_loss=float((lam*np.maximum(y[keep]-p[keep],0)+np.maximum(p[keep]-y[keep],0)).mean())
                        assert abs(direct_loss-(lam*um+om))<1e-12
                        costs.append(dict(period=period,cohort=cohort,model=name,seed=seed,weight=lam,events=int(keep.sum()),POIs=int(poi_keep.sum()),under=um,over=om,loss=direct_loss))
                if name in GRAPH_MODELS:
                    for graph,parts in masks.items():
                        cell_sum=0;cell_n=0
                        for cell,keep in parts.items():
                            row=metrics(y,p,keep)
                            strata_rows.append(dict(period=period,neighbor_partition=graph,cell=cell,model=name,seed=seed,**row))
                            cell_sum+=row['MAE']*row['events'];cell_n+=row['events']
                        pooled=float(np.abs(y-p)[valid & eligible[None,None,:]].mean())
                        assert abs(cell_sum/cell_n-pooled)<1e-12
            assert sorted(seeds)==([-1] if len(paths)==1 else list(range(42,47)))
            losses[name]=(under,over)
            log(f'{period}: validated {name}, {len(paths)} archived predictions, MAE={np.mean(model_maes[name]):.9f}')
        for name,(expected_mean,expected_sd) in zip(MODELS,EXPECTED[period]):
            values=model_maes[name]
            assert round(float(np.mean(values)),3)==expected_mean,(period,name,values,expected_mean)
            if expected_sd is not None:assert round(float(np.std(values,ddof=1)),3)==expected_sd,(period,name,values,expected_sd)
        log(f'{period}: all Appendix A.1 means and standard deviations reproduced')
        ref=sum(losses[IDENTITY])
        np.testing.assert_array_equal(ref-ref,np.zeros_like(ref))
        for graph,parts in masks.items():
            for cell,keep in parts.items():
                for name in GRAPH_MODELS[1:]:
                    difference=ref-sum(losses[name])
                    gains.append(dict(period=period,neighbor_partition=graph,cell=cell,model=name,events=int(keep.sum()),**joint_interval(difference,keep,spatial)))
        for cohort,poi_keep in [('All',np.ones(len(ids),bool)),('Low-history',low)]:
            keep=valid & poi_keep[None,None,:]
            for lam in [.5,1,2,3,5]:
                delta=(lam*losses['MTGNN (NAICS-2)'][0]+losses['MTGNN (NAICS-2)'][1])-(lam*losses[IDENTITY][0]+losses[IDENTITY][1])
                cost_pairs.append(dict(period=period,cohort=cohort,weight=lam,contrast='MTGNN minus EvolveGCN Identity',**joint_interval(delta,keep,spatial)))
        audits.append(dict(period=period,origins=len(origins),events=int(valid.sum()),POIs=len(ids),archives=sum(map(len,files.values())),appendix_A1_reproduced=True))
        log(f'{period}: strata and asymmetric loss complete, paired bootstrap complete')
    for filename,rows in [('strata_per_seed.csv',strata_rows),('strata_gain_intervals.csv',gains),('cost_per_seed.csv',costs),('cost_pair_intervals.csv',cost_pairs),('overall_per_seed.csv',overall)]:
        pd.DataFrame(rows).to_csv(out/filename,index=False)
    sf=pd.DataFrame(strata_rows)
    ss=sf.groupby(['period','neighbor_partition','cell','model'],sort=False).agg(
        events=('events','first'),POIs=('POIs','first'),positive_events=('positive_events','first'),zero_events=('zero_events','first'),
        MAE=('MAE','mean'),MAE_sd=('MAE','std'),recall=('recall','mean'),recall_sd=('recall','std'),FAR=('FAR','mean'),FAR_sd=('FAR','std'),DMR=('DMR','mean'),DMR_sd=('DMR','std')).reset_index()
    ss.to_csv(out/'strata_summary.csv',index=False)
    cf=pd.DataFrame(costs)
    cs=cf.groupby(['period','cohort','weight','model'],sort=False).agg(events=('events','first'),POIs=('POIs','first'),under=('under','mean'),over=('over','mean'),loss=('loss','mean'),sd=('loss','std')).reset_index()
    cs['rank']=cs.groupby(['period','cohort','weight'])['loss'].rank(method='min')
    cs.to_csv(out/'cost_summary.csv',index=False)
    for p,h in tracked.items():assert digest(Path(p))==h, f'Input changed: {p}'
    audit=dict(status='complete',protocol=(Path(__file__).resolve().parents[2] / 'exp/exp9/PROTOCOL.md').read_text(),thresholds=thresholds,periods=audits,
        gates=['panel/order/target/mask alignment','complete five seeds','Appendix A1 reproduction','partition exhaustion','pooled MAE reconstruction','zero self contrast','L1 equals MAE','direct asymmetric formula','input hashes unchanged'],input_sha256=tracked)
    write_json(out/'audit.json',audit)
    log('COMPLETE '+str(out))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--base',type=Path,default=BASE)
    ap.add_argument('--extension',type=Path,default=EXT)
    ap.add_argument('--bundle',type=Path,default=BUNDLE)
    args=ap.parse_args()
    BASE,EXT,BUNDLE=args.base,args.extension,args.bundle
    main(args.out)
