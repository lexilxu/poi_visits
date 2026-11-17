"""Replay locally produced revision checkpoints before mechanism experiments.

Only checkpoints produced by train_controlled.py are supported. Prediction
replay is checked against the separately saved event archive before analysis.
"""
from pathlib import Path, PureWindowsPath
import json
import numpy as np
from exp.common.audit import load_panel, EXPECTED_BUNDLE_SHA256
from exp.common.protocol import Protocol, feature_mask
from exp.common.controlled_training import make_model


def selected_directories(queue, model=None, adjacency=None, variant=None, t=None):
    result = []
    for path in sorted(Path(queue).glob('*/selected_runs.json')):
        selection = json.loads(path.read_text(encoding='utf8'))
        recipe = selection['recipe']
        if any(want is not None and recipe[key] != want for key, want in
               [('model',model),('adjacency',adjacency),('variant',variant),('T',t)]):
            continue
        assert sorted(run['seed'] for run in selection['runs']) == [42,43,44,45,46]
        for run in selection['runs']:
            # Relative paths in an exported queue remain portable.
            directory = path.parent / PureWindowsPath(run['directory']).name
            if not (directory/'fit.json').is_file():
                raise FileNotFoundError(directory/'fit.json')
            result.append(directory)
    return result


def load_gru(directory, bundle, device='cuda:0'):
    import torch
    from exp.common.original_protocol import static_features, group_average
    import pandas as pd
    directory = Path(directory)
    fit = json.loads((directory/'fit.json').read_text(encoding='utf8'))
    if fit['model'] not in ('GraphGRU','GNN-static'):
        raise ValueError('This mechanism adapter is specific to the fitted GraphGRU recurrence')
    if fit['bundle_sha256'] != EXPECTED_BUNDLE_SHA256:
        raise ValueError('Checkpoint dataset differs from the approved input')
    # The weights are produced by this project's own training entry point.
    saved = torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    raw, dates, attrs = load_panel(bundle)
    protocol = Protocol(fit['T'],fit['H'],fit['train_end'],fit['validation_end'])
    origins, history, target, split, valid = protocol.windows(raw)
    keep = split != 'train'
    own = np.nan_to_num(raw[history[keep]],nan=0.).astype('float32')
    graph = own.copy()
    labels = None
    if fit['adjacency'] in ('NAICS-2','NAICS-4'):
        digits = 2 if fit['adjacency'] == 'NAICS-2' else 4
        labels = pd.factorize(pd.to_numeric(attrs.naics_code).astype('int64').astype(str).str[:digits])[0]
        filled = np.nan_to_num(raw,nan=0.)
        graph = ((filled+group_average(filled,labels))*.5)[history[keep]].astype('float32')
    elif fit['adjacency'] != 'Identity':
        raise ValueError('The mechanism adapter requires identity or a semantic clique')
    static = None
    if fit['model'] == 'GNN-static':
        static,_ = static_features(attrs)
        static[:,:4] = (static[:,:4]-saved['numeric_mean'])/saved['numeric_std']
    model = make_model(fit['model'],fit['H'],static,feature_mask(fit['variant']),
                       len(attrs),fit['T'],(lambda x:x,lambda x:x),torch.device(device),fit['dropout'])
    model.load_state_dict(saved['state_dict'],strict=True)
    model = model.to(device).eval()
    mean,std = saved['visit_mean'],saved['visit_std']
    prediction = predict_gru(model,graph,own,mean,std,device)
    with np.load(directory/'predictions.npz',allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['forecast_origin'],dates[origins[keep]])
        np.testing.assert_allclose(prediction,archive['predicted_visits'],rtol=2e-5,atol=2e-5)
    return dict(model=model,fit=fit,raw=raw,dates=dates,attrs=attrs,labels=labels,
                origins=origins[keep],history=history[keep],target=target[keep],split=split[keep],
                valid=valid[keep],own=own,graph=graph,prediction=prediction,mean=mean,std=std)


def predict_gru(model, graph, own, mean, std, device, batch=4):
    import torch
    output=[]
    with torch.no_grad():
        for start in range(0,len(own),batch):
            x=torch.as_tensor((graph[start:start+batch]-mean)/std,device=device,dtype=torch.float32)
            z=torch.as_tensor((own[start:start+batch]-mean)/std,device=device,dtype=torch.float32)
            output.append((model(x,z)*std+mean).clamp_min(0).cpu().numpy())
    return np.concatenate(output)


def encoded_forecast(model,history,static,mean,std):
    """Independent node/event rows; history is (events,T), static is (events,52)."""
    import torch.nn.functional as F
    encoded=F.linear(history.T[None,...,None],model.project.weight[:,:1],model.project.bias)
    encoded=encoded+F.linear(static,model.project.weight[:,1:])[None,None,:,:]
    return (model.forecast_projected(encoded,history[:,-1][None,:])[0].T*std+mean).clamp_min(0)
