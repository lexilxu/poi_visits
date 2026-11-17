"""Paired uncertainty for five-seed losses on fixed corrected forecasts."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json,socket,sys,time
import numpy as np
import pandas as pd


def digest(path):return sha256(path.read_bytes()).hexdigest()


def grid_groups(latitude,longitude,cell_km):
    # Existing uncertainty protocol: fixed Seattle equirectangular grid.
    xy=np.column_stack([np.asarray(longitude)*111.320*np.cos(np.deg2rad(47.6)),np.asarray(latitude)*110.574])
    return np.unique(np.floor(xy/cell_km).astype(np.int64),axis=0,return_inverse=True)[1]


def observed_mean(values):
    count=np.isfinite(values).sum(0)
    return np.divide(np.nansum(values,axis=0),count,out=np.zeros(count.shape,dtype=float),where=count>0)


def key(cfg):
    return (cfg['protocol'],cfg['T'],cfg['H'],cfg['adjacency'],cfg['static'])


def group_name(group):
    protocol,t,h,graph,static=group
    return f"{protocol}_T{t}_H{h}_{graph.replace(' ','_')}_{'static' if static else 'history'}"


def load_group(paths):
    reference=None;losses=[];records=[]
    for path in sorted(paths):
        rec=json.loads((path/'complete.json').read_text());file=path/'predictions.npz'
        assert rec['status']=='complete' and digest(file)==rec['predictions_sha256']
        with np.load(file,allow_pickle=False) as z:
            test=z['dataset_split']=='test'
            data={field:z[field][test] for field in ['observed_visits','predicted_visits','valid','forecast_origin']}
            data['poi_id']=z['poi_id'];data['horizon_step']=z['horizon_step']
        if reference is None:reference=data
        else:
            for field in ['observed_visits','valid','forecast_origin','poi_id','horizon_step']:
                np.testing.assert_array_equal(data[field],reference[field])
        losses.append(np.where(data['valid'],np.abs(data['predicted_visits'].astype(float)-data['observed_visits']),0.))
        records.append(dict(seed=rec['config']['seed'],path=str(file),sha256=rec['predictions_sha256']))
    assert sorted(r['seed'] for r in records)==[42,43,44,45,46]
    return reference,np.stack(losses),records


def intervals(delta,valid,groups,h,repetitions=2000):
    """Moving origin blocks and occupied spatial cells; fixed fitted seeds."""
    n=valid.shape[0];ng=int(groups.max())+1
    sums=np.stack([np.bincount(groups,weights=np.where(valid[i],delta[i],0.).sum(0),minlength=ng) for i in range(n)])
    counts=np.stack([np.bincount(groups,weights=valid[i].sum(0),minlength=ng) for i in range(n)])
    occupied=counts.sum(0)>0;sums=sums[:,occupied];counts=counts[:,occupied];ng=int(occupied.sum())
    if not ng:raise ValueError('Empty cohort')
    block=max(7,h);temporal=n>=2*block
    rng=np.random.default_rng(421);draws=np.full((repetitions,3),np.nan)
    for rep in range(repetitions):
        weights=np.bincount(rng.integers(ng,size=ng),minlength=ng)
        draws[rep,1]=(sums.sum(0)@weights)/(counts.sum(0)@weights)
        if temporal:
            starts=rng.integers(n-block+1,size=int(np.ceil(n/block)))
            origins=(starts[:,None]+np.arange(block)).ravel()[:n]
            draws[rep,0]=sums[origins].sum()/counts[origins].sum()
            draws[rep,2]=(sums[origins].sum(0)@weights)/(counts[origins].sum(0)@weights)
    row=dict(delta_MAE=float(sums.sum()/counts.sum()),origins=n,origin_block_length=block,
        spatial_groups=ng,events=int(counts.sum()),temporal_interval_available=temporal)
    for col,name in enumerate(['origin_block','spatial_group','joint']):
        if col!=1 and not temporal:row[name+'_low']=None;row[name+'_high']=None
        else:
            lo,hi=np.quantile(draws[:,col],[.025,.975]);row[name+'_low']=float(lo);row[name+'_high']=float(hi)
    return row


def main(args):
    sys.path[:0]=[str(args.source/'exp/common'),str(args.source/'exp/common')]
    from exp.common.original_protocol import load_panel, make_windows, model_key
    from exp.common.audit import EXPECTED_BUNDLE_SHA256, low_history
    panel=load_panel(args.bundle,EXPECTED_BUNDLE_SHA256)
    roots=args.root/'runs' if args.period=='december' else args.root/'january/runs'
    configs={}
    for path in roots.glob('*/complete.json'):
        rec=json.loads(path.read_text());configs.setdefault(key(rec['config']),[]).append(path.parent)
    assert len(configs)==32 and all(len(paths)==5 for paths in configs.values()),'All 160 fits are required'
    args.out.mkdir(parents=True,exist_ok=False)
    rows=[];audits=[];loaded={};settings={}
    for group in sorted(configs):
        protocol,t,h,graph,static=group
        base=(protocol,t,h,'Identity',False)
        if base not in loaded:loaded[base]=load_group(configs[base])
        data,loss,provenance=load_group(configs[group])
        if group==base:loaded[base]=(data,loss,provenance)
        np.testing.assert_array_equal(data['poi_id'],panel.attrs.placekey.to_numpy(dtype=str))
        setting=(protocol,t,h)
        if setting not in settings:
            if protocol=='original_masked':
                w=make_windows(panel.raw,t,h);ha=w.history_mean
                low=(ha>0)&(ha<=np.quantile(ha[ha>0],.25))&(w.zero_ratio>=.7)
                legacy=args.legacy/f'T{t}_H{h}'/(model_key('Historical average','None')+'.npy')
                expected=np.load(legacy,allow_pickle=False)
                np.testing.assert_allclose(expected,np.broadcast_to(ha,expected.shape),rtol=0,atol=0)
                ha_source=dict(rule='Original training-window-weighted history mean',legacy_sha256=digest(legacy))
            else:
                ha=observed_mean(panel.raw[:50]);low,_=low_history(panel.raw[:50])
                ha_source=dict(rule='Complete-window confirmation calendar mean over the first 50 days')
            settings[setting]=(ha,low,ha_source)
        ha,low,ha_source=settings[setting]
        valid=data['valid'];truth=data['observed_visits']
        baseline_loss=np.where(valid,np.abs(truth-ha[None,None,:]),0.).astype(float)
        references=[('Historical average',baseline_loss[None],ha_source)]
        if group!=base:
            ref,ref_loss,ref_sources=loaded[base]
            for field in ['forecast_origin','observed_visits','valid','poi_id']:
                np.testing.assert_array_equal(data[field],ref[field])
            references.append(('EvolveGCN Identity history',ref_loss,ref_sources))
        for reference,ref_loss,ref_provenance in references:
            delta=loss.mean(0)-ref_loss.mean(0)
            for cohort,nodes in [('All',np.ones(len(low),bool)),('Low-history',low)]:
                mask=valid&nodes[None,None,:]
                seed_delta=(loss-ref_loss)[:,mask].mean(1)
                for km in (1.,3.,5.):
                    rows.append(dict(model=group_name(group),reference=reference,period=args.period,
                        protocol=protocol,T=t,H=h,adjacency=graph,static=static,cohort=cohort,cell_km=km,
                        seed_delta_mean=float(seed_delta.mean()),seed_delta_sd=float(seed_delta.std(ddof=1)),
                        **intervals(delta,mask,grid_groups(panel.attrs.latitude,panel.attrs.longitude,km),h,args.repetitions)))
            audits.append(dict(model=group_name(group),reference=reference,model_sources=provenance,reference_sources=ref_provenance))
        # Keep only identity losses in memory for later graph and static contrasts.
        print(json.dumps(dict(group=group_name(group),status='complete')),flush=True)
    pd.DataFrame(rows).to_csv(args.out/'paired_intervals.csv',index=False)
    report=dict(status='complete',period=args.period,groups=len(configs),comparisons=len(audits),
        repetitions=args.repetitions,created_utc=time.time(),sources=audits,
        estimand='Model minus reference MAE; average absolute losses across five fixed seeds before resampling.',
        inference='Paired moving forecast-origin blocks of max(7,H) days and 1/3/5-km occupied spatial cells. Intervals condition on fitted seeds and the sampled Seattle panel.',
        temporal_rule='Origin and joint intervals are reported when at least two full blocks fit in the test-origin sequence. Spatial intervals for shorter sequences condition on those few origins.',
        seed_variability='Sample standard deviation of the five paired seed MAE contrasts, reported separately from bootstrap intervals.',
        code_sha256=digest(Path(__file__)))
    (args.out/'complete.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for name in ['source','root','bundle','legacy','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--period',choices=['december','january'],required=True)
    p.add_argument('--repetitions',type=int,default=2000)
    main(p.parse_args())
