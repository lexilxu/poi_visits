"""Extract a later-month holdout without changing the original POI cohort."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd
from audit_protocol import load_panel,EXPECTED_BUNDLE_SHA256

SOURCES={
 'patterns-part1.csv.gz':('1alVI8YOQ2_sI5wYUN_kiGyBIsoONhJIB',513340075),
 'patterns-part2.csv.gz':('1OgwTrNwX0_Xa0Up02Vm27rKJfUs9ytsn',502139006),
 'patterns-part3.csv.gz':('1-2vQGwuiNXNMWUq97YP4dV7WQK_VbhEt',494883884),
 'patterns-part4.csv.gz':('1oPztKf4JA6bMWs8zfG54-9nQ35g7cs4g',484644196),
}


def digest(path):
    value=sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def run(args):
    raw,dates,attrs=load_panel(args.bundle)
    ids=set(attrs.placekey);selected=[];sources=[]
    args.out.mkdir(parents=True,exist_ok=False)
    for name,(file_id,size) in SOURCES.items():
        path=args.source/name
        if path.stat().st_size!=size:raise ValueError(f'Unexpected download size: {name}')
        header=pd.read_csv(path,compression='gzip',nrows=0).columns.tolist()
        fields=['placekey','date_range_start','date_range_end','visits_by_day']
        if not set(fields).issubset(header):raise ValueError(f'Unsupported monthly schema: {header}')
        rows=0;matches=0;months=set()
        for chunk in pd.read_csv(path,compression='gzip',usecols=fields,dtype=str,chunksize=50000):
            rows+=len(chunk);months.update(chunk.date_range_start.str[:10].unique())
            chosen=chunk.loc[chunk.placekey.isin(ids)].copy();matches+=len(chosen)
            if len(chosen):selected.append(chosen)
        if months!={'2021-01-01'}:raise ValueError(f'Dates differ from the intended holdout: {months}')
        record=dict(file=name,drive_file_id=file_id,bytes=size,sha256=digest(path),rows=rows,matched_rows=matches,
                    url=f'https://drive.google.com/file/d/{file_id}/view')
        sources.append(record);print(json.dumps(record),flush=True)
    frame=pd.concat(selected,ignore_index=True)
    if frame.placekey.duplicated().any():raise ValueError('Duplicate POI-month records across shards')
    if not frame.date_range_end.str[:10].eq('2021-02-01').all():raise ValueError('Unexpected month-end boundary')
    mapping=dict(zip(frame.placekey,frame.visits_by_day.map(ast.literal_eval)))
    if any(len(value)!=31 for value in mapping.values()):raise ValueError('Unexpected daily sequence length')
    january=np.full((31,len(attrs)),np.nan,dtype=np.float32)
    for j,key in enumerate(attrs.placekey):
        if key in mapping:january[:,j]=mapping[key]
    observed=np.isfinite(january)
    if np.any(january[observed]<0) or not np.all(january[observed]==np.floor(january[observed])):
        raise ValueError('Visit counts must be nonnegative integers')
    extended=np.concatenate([raw,january])
    extended_dates=np.r_[np.asarray(dates,dtype=str),pd.date_range('2021-01-01','2021-01-31').strftime('%Y-%m-%d')]
    np.savez_compressed(args.out/'extended_panel.npz',visits=extended,dates=extended_dates.astype(str),
                        poi_id=attrs.placekey.to_numpy(dtype=str),original_bundle_sha256=EXPECTED_BUNDLE_SHA256)
    frame.to_csv(args.out/'selected_monthly_rows.csv.gz',index=False,compression='gzip')
    report=dict(status='complete',original_bundle_sha256=EXPECTED_BUNDLE_SHA256,
                original_days=len(raw),holdout_start='2021-01-01',holdout_end='2021-01-31',holdout_days=31,
                original_pois=len(attrs),matched_pois=len(mapping),missing_pois=len(attrs)-len(mapping),
                holdout_observed_poi_days=int(observed.sum()),holdout_missing_poi_days=int((~observed).sum()),
                holdout_mean_observed_visits=float(january[observed].mean()),
                holdout_zero_ratio=float((january[observed]==0).mean()),
                cohort_rule='Original 9884 POIs retained before reading holdout outcomes; missing monthly records stay missing.',
                evaluation_rule='Training/validation, hyperparameters, feature metadata, and calendar low-history cohort remain fixed at the original protocol. No January outcome is used for fitting or selection. Rolling inputs use only observations before each origin.',
                scope='Later-month validation in the same city and cohort, not an independent-city replication.',
                sources=sources,panel_sha256=digest(args.out/'extended_panel.npz'))
    (args.out/'holdout_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    run(p.parse_args())
