"""Construct calendar and explicitly dated lag inputs for the parking audit."""
import numpy as np
import pandas as pd


def features(panel, lags):
    dates=panel['dates'];ratio=panel['occupancy']/panel['capacity']
    weekday=pd.to_datetime(dates).dayofweek.to_numpy()
    train=(dates>='2020-10-01')&(dates<='2020-11-19')
    valid=np.isfinite(ratio);training=valid&train[:,None,None]
    counts=training.sum(0);sums=np.where(training,ratio,0.).sum(0)
    means=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0)
    calendar=np.empty(ratio.shape);weekday_counts=[]
    for day in range(7):
        mask=training&(weekday[:,None,None]==day);ct=mask.sum(0)
        weekday_counts.append(ct)
        calendar[weekday==day]=(np.where(mask,ratio,0.).sum(0)+5*means)/(ct+5)
    arrays=[calendar];lag_arrays=[]
    for lag in lags:
        past=np.full(ratio.shape,np.nan);past[lag:]=ratio[:-lag]
        missing=~np.isfinite(past);filled=np.where(missing,means[None],past)
        arrays.extend([filled,missing.astype(float)]);lag_arrays.append(filled)
    for hour in (1,2):arrays.append(np.broadcast_to((np.arange(3)==hour)[None,:,None],ratio.shape))
    for day in range(1,7):arrays.append(np.broadcast_to((weekday==day)[:,None,None],ratio.shape))
    return dict(base=np.stack(arrays,axis=-1),calendar=calendar,lags=lag_arrays,
                ratio=ratio,training=training,weekday=weekday,candidates=counts>0,
                weekday_counts=np.stack(weekday_counts),dates=dates)


def candidates_for_dates(data, indices, rule):
    if rule=='original':return np.broadcast_to(data['candidates'],(len(indices),*data['candidates'].shape))
    assert rule=='training_calendar'
    mask=data['weekday_counts'][data['weekday'][indices]]>=3
    holiday=np.isin(data['dates'][indices],['2020-12-25','2021-01-01','2021-01-18'])
    mask[(data['weekday'][indices]==6)|holiday]=False
    return mask
