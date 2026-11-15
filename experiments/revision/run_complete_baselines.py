"""Save remaining simple and direct per-POI AR-ridge baselines on complete horizons."""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
from audit_protocol import load_panel, EXPECTED_BUNDLE_SHA256
from training_protocol import Protocol
from run_baseline_checks import predictions


def ar_ridge(raw, history, target, split, valid, penalties=(1.,10.,100.,1000.)):
    """Direct lead-specific regression, unpenalized intercept, one global validation penalty."""
    own=np.nan_to_num(raw[history],nan=0.).transpose(2,0,1).astype('float64')
    train=split=='train'; val=(split=='validation')[:,None,None] & valid
    n,t=raw.shape[1],history.shape[1]
    coefficients={alpha:np.empty((n,t+1,target.shape[1])) for alpha in penalties}
    for lead in range(target.shape[1]):
        masks,groups=np.unique(valid[train,lead].T,axis=0,return_inverse=True)
        for group,keep in enumerate(masks):
            nodes=np.flatnonzero(groups==group)
            if not keep.any():
                raise ValueError('No training observations for an AR-ridge target')
            x=own[nodes][:,train][:,keep]
            y=raw[target[train,lead]][keep][:,nodes].T
            mx=x.mean(1,keepdims=True); my=y.mean(1,keepdims=True)
            xc=x-mx; yc=y-my
            gram=xc.transpose(0,2,1)@xc
            cross=xc.transpose(0,2,1)@yc[...,None]
            for alpha in penalties:
                weights=np.linalg.solve(gram+alpha*np.eye(t),cross)[...,0]
                coefficients[alpha][nodes,:t,lead]=weights
                coefficients[alpha][nodes,t,lead]=my[:,0]-(mx[:,0]*weights).sum(1)
    scores={}; chosen=None
    for alpha,coef in coefficients.items():
        forecast=np.maximum(0,np.einsum('nbt,nth->bhn',own,coef[:,:t])+coef[:,t].T[None])
        score=float(np.abs(forecast-raw[target])[val].mean()); scores[str(alpha)]=score
        if chosen is None or score<chosen[0]:
            chosen=(score,alpha,forecast,coef)
    return chosen[2],dict(selected_penalty=chosen[1],validation_MAE_by_penalty=scores,
                          architecture='Separate direct AR per POI and forecast lead, with intercept',
                          training_loss='MSE with ridge penalty; intercept unpenalized',
                          selection='One penalty for all POIs and leads, selected using validation MAE'),chosen[3]


def run(args):
    raw,dates,attrs=load_panel(args.bundle)
    args.out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for t in (14,28):
        models,origins,history,target,split,valid,selection=predictions(raw,dates,t,7,Protocol(T=t))
        models['AR-ridge'],fit,weights=ar_ridge(raw,history,target,split,valid)
        np.savez_compressed(args.out/f'AR_ridge_T{t}_weights.npz',weights=weights)
        reports.append(dict(T=t,model='AR-ridge',**fit))
        keep=split!='train'
        for name,prediction in models.items():
            path=args.out/f'{name.replace(" ","_").replace("-","_")}_T{t}_H7.npz'
            temporary=path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,model=name,adjacency='None',seed=-1,
                                forecast_origin=np.asarray(dates[origins[keep]],dtype=str),
                                horizon_step=np.arange(1,8),poi_id=attrs.placekey.to_numpy(dtype=str),
                                observed_visits=raw[target[keep]],predicted_visits=prediction[keep].astype('float32'),
                                valid=valid[keep],dataset_split=split[keep],train_end=50,T=t,H=7,
                                protocol='complete_horizon',bundle_sha256=EXPECTED_BUNDLE_SHA256)
            temporary.replace(path)
            test=split=='test'
            print(json.dumps(dict(T=t,model=name,test_MAE=float(np.abs(prediction[test]-raw[target[test]])[valid[test]].mean()))),flush=True)
    (args.out/'baseline_fits.json').write_text(json.dumps(reports,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
