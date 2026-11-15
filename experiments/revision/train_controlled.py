"""One reproducible neural fit, gated by environment and complete-horizon checks.

Run the smoke check before launching the manifest. Existing archived results
remain separate from the complete-horizon refits.
"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import platform
import sys
import time
import numpy as np
from training_protocol import Protocol, feature_mask
from audit_protocol import load_panel, EXPECTED_BUNDLE_SHA256
from controlled_checks import propagate_neighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'experiments/archived_2020q4'),str(ROOT/'external/MTGNN')]


def environment():
    report = {'python':sys.version,'platform':platform.platform()}
    try:
        import torch
        report.update(torch=torch.__version__,cuda_available=torch.cuda.is_available())
        if report['cuda_available']:
            report['gpu'] = torch.cuda.get_device_name(0)
    except Exception as error:
        report.update(cuda_available=False,torch_import_error=str(error))
    return report


def model_static_features(attrs, model_name):
    from train_formal import static_features
    static,info=static_features(attrs)
    if model_name=='GNN-parent':
        import pandas as pd
        parents=attrs.parent_placekey.fillna('').astype(str)
        labels=sorted(set(parents)-{''})
        mapping={name:i+1 for i,name in enumerate(labels)}
        static[:,8]=parents.map(mapping).fillna(0).to_numpy()
        info['resolved_parent_vocabulary']=labels
    return static,info


def make_model(model_name, horizon, static, mask, nodes, t, graph_operator, device, dropout):
    import torch
    from torch import nn
    from train_formal import GraphGRU
    from train_mtgnn import WindowMTGNN, replace_graph_operators

    class StaticGRU(GraphGRU):
        def __init__(self):
            super().__init__(horizon,static)
            if model_name=='GNN-parent':
                self.embeddings[4]=nn.Embedding(int(static[:,8].max())+1,7,padding_idx=0)
            self.register_buffer('feature_mask',torch.as_tensor(mask))

        def encode_static(self):
            if model_name=='GNN-parent':
                groups=[self.static[:,:4]]
                for i,layer in enumerate(self.embeddings):
                    value=layer(self.static[:,4+i].long())
                    if i==4:
                        present=(self.static[:,8]>0).to(value.dtype)[:,None]
                        value=torch.cat([present,value],dim=1)
                    groups.append(value)
                return torch.cat(groups,dim=1)*self.feature_mask
            return super().encode_static()*self.feature_mask

    class SharedLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm=nn.LSTM(1,32,batch_first=True)
            self.output=nn.Linear(32,horizon)

        def forward(self,x,own):
            b,steps,n=own.shape
            _,(state,_)=self.lstm(own.transpose(1,2).reshape(b*n,steps,1))
            return self.output(state[-1]).reshape(b,n,horizon).transpose(1,2)

    class FixedMTGNN(nn.Module):
        def __init__(self):
            super().__init__()
            from net import gtnet
            network=gtnet(True,False,2,nodes,device,
                          predefined_A=torch.empty((0,0),device=device),dropout=dropout,
                          seq_length=t,in_dim=1,out_dim=horizon)
            self.network=WindowMTGNN(replace_graph_operators(network,graph_operator))

        def forward(self,x,own):
            return self.network(own)

    class MatchedLSTM(nn.Module):
        """Match GraphGRU input projection and last-observation readout."""
        def __init__(self):
            super().__init__()
            self.project=nn.Linear(1,32)
            self.lstm=nn.LSTM(32,32,batch_first=True)
            self.last=nn.Linear(1,32)
            self.output=nn.Linear(32,horizon)

        def forward(self,x,own):
            b,steps,n=own.shape
            visits=own.transpose(1,2).reshape(b*n,steps,1)
            _,(state,_)=self.lstm(self.project(visits))
            state=state[-1]+self.last(visits[:,-1])
            return self.output(state).reshape(b,n,horizon).transpose(1,2)

    class AdaptiveMTGNN(nn.Module):
        def __init__(self):
            super().__init__()
            from net import gtnet
            from adaptive_mtgnn import adapt
            network=gtnet(True,True,2,nodes,device,dropout=dropout,
                          seq_length=t,in_dim=1,out_dim=horizon)
            self.network=WindowMTGNN(adapt(network))

        def forward(self,x,own):
            return self.network(own)

    if model_name == 'GraphGRU':
        return GraphGRU(horizon)
    if model_name in ('GNN-static','GNN-parent'):
        return StaticGRU()
    if model_name == 'Shared-LSTM':
        return SharedLSTM()
    if model_name == 'Matched-LSTM':
        return MatchedLSTM()
    if model_name == 'Fixed-MTGNN':
        return FixedMTGNN()
    if model_name == 'Adaptive-MTGNN':
        return AdaptiveMTGNN()
    raise ValueError(model_name)


def run(args):
    import torch
    from train_formal import group_average, static_features
    from train_mtgnn import sparse_operator, clique_operator
    from scipy import sparse
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA preflight failed; do not launch the training matrix')
    dataset_sha256=EXPECTED_BUNDLE_SHA256
    if getattr(args,'panel_file',None):
        import pandas as pd
        if args.model not in ('GraphGRU','Matched-LSTM') or args.adjacency!='Identity':
            raise ValueError('Supplementary city panels use the prespecified temporal controls')
        if args.attributes_file or args.density or args.graph_file or args.sparse_graph:
            raise ValueError('City-panel protocol uses its complete July-defined cohort')
        with np.load(args.panel_file,allow_pickle=False) as panel:
            raw=panel['visits'];dates=panel['dates'];attrs=pd.DataFrame({'placekey':panel['poi_id']})
        assert raw.shape==(92,len(attrs)) and len(attrs)==attrs.placekey.nunique()
        assert np.isfinite(raw[:50]).any(0).all() and np.all(raw[np.isfinite(raw)]>=0)
        assert np.all(np.diff(dates.astype('datetime64[D]'))==np.timedelta64(1,'D'))
        dataset_sha256=sha256(args.panel_file.read_bytes()).hexdigest()
    else:
        raw,dates,attrs=load_panel(args.bundle)
    from experiment_inputs import density_indices,replace_attributes
    metadata_sha256=None;density=None
    if args.attributes_file:
        attrs,metadata_sha256=replace_attributes(attrs,args.attributes_file)
    if args.density:
        if args.adjacency!='Identity':raise ValueError('Density refits use identity adjacency')
        nodes,density=density_indices(raw,args.density,args.train_end)
        raw=raw[:,nodes];attrs=attrs.iloc[nodes].reset_index(drop=True)
    protocol=Protocol(args.T,args.H,args.train_end,64)
    origins,history,target,split,valid=protocol.windows(raw)
    args.out.mkdir(parents=True,exist_ok=False)
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (args.out/'running.json').write_text(json.dumps(configuration,indent=2)+'\n',encoding='utf8')
    mean,std=float(np.nanmean(raw[:protocol.train_end])),float(np.nanstd(raw[:protocol.train_end]))
    if std<=0:
        raise ValueError('Training scale must be positive')
    own=np.nan_to_num(raw[history],nan=0.).astype('float32')
    graph_input=own
    graph_operators=(lambda x:x,lambda x:x)
    graph_digest=None
    graph_label=args.adjacency
    if args.sparse_graph:
        adjacency=sparse.load_npz(args.sparse_graph)
        if adjacency.shape!=(len(attrs),len(attrs)):raise ValueError('Sparse graph shape differs from panel')
        graph_digest=sha256(args.sparse_graph.read_bytes()).hexdigest()
        graph_label=args.adjacency
        transition=adjacency+sparse.eye(len(attrs),dtype=np.float32,format='csr')
        transition=sparse.diags(1/np.asarray(transition.sum(1)).ravel())@transition
        filled=np.nan_to_num(raw,nan=0.)
        graph_input=(transition@filled.T).T[history].astype('float32')
        graph_operators=(sparse_operator(adjacency,device),sparse_operator(adjacency.T,device))
    elif args.graph_file:
        with np.load(args.graph_file,allow_pickle=False) as saved:
            peers=saved['neighbors']
            assert np.array_equal(saved['poi_id'],attrs.placekey.to_numpy(dtype=str))
        graph_digest=sha256(peers.astype('<i4').tobytes()).hexdigest()
        graph_label=args.graph_file.stem.rsplit('_seed',1)[0]
        graph_input=propagate_neighbors(own,peers).astype('float32')
        # Store a transition matrix whose non-isolated rows average neighbors.
        degree=(peers>=0).sum(1)
        row=np.repeat(np.arange(len(peers)),degree)
        col=peers[peers>=0]
        adjacency=sparse.csr_matrix((1/degree[row],(row,col)),shape=(len(peers),len(peers)))
        # sparse_operator adds own and row-normalizes; isolates become identity.
        graph_operators=(sparse_operator(adjacency,device),sparse_operator(adjacency.T,device))
    elif args.adjacency in ('NAICS-2','NAICS-4'):
        import pandas as pd
        digits=2 if args.adjacency=='NAICS-2' else 4
        labels=pd.factorize(pd.to_numeric(attrs.naics_code).astype('int64').astype(str).str[:digits])[0]
        graph_input=(np.nan_to_num(raw,nan=0.)+group_average(np.nan_to_num(raw,nan=0.),labels))*.5
        graph_input=graph_input[history]
        operation=clique_operator(labels,device)
        graph_operators=(operation,operation)
    static=None
    numeric_mean,numeric_std=None,None
    if args.model in ('GNN-static','GNN-parent'):
        static,_=model_static_features(attrs,args.model)
        eligible=np.isfinite(raw[:protocol.train_end]).any(0)
        numeric_mean=static[eligible,:4].mean(0)
        numeric_std=static[eligible,:4].std(0)
        numeric_std=np.where(numeric_std>0,numeric_std,1.)
        static[:,:4]=(static[:,:4]-numeric_mean)/numeric_std
    mask=feature_mask(args.variant)
    model=make_model(args.model,args.H,static,mask,len(attrs),args.T,graph_operators,device,args.dropout).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4)
    arrays=[torch.as_tensor((value-mean)/std,device=device,dtype=torch.float32) for value in (graph_input,own)]
    response=torch.as_tensor(np.nan_to_num(raw[target],nan=0.),device=device,dtype=torch.float32)
    observed=torch.as_tensor(valid,device=device)
    train_ids=np.flatnonzero(split=='train'); val_ids=np.flatnonzero(split=='validation')
    rng=np.random.default_rng(args.seed)
    best,best_epoch=float('inf'),0
    logs=[]
    started=time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.train(); sums=[]
        for ids in np.array_split(rng.permutation(train_ids),int(np.ceil(len(train_ids)/args.batch_size))):
            optimizer.zero_grad(set_to_none=True)
            prediction=model(*(value[ids] for value in arrays))*std+mean
            error=(prediction-response[ids])/std
            error=error.square() if args.loss=='MSE' else error.abs()
            loss=error[observed[ids]].mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step(); sums.append(float(loss.detach().cpu()))
        model.eval(); numerator,denominator=0.,0
        with torch.no_grad():
            for ids in np.array_split(val_ids,int(np.ceil(len(val_ids)/args.batch_size))):
                prediction=torch.clamp(model(*(value[ids] for value in arrays))*std+mean,min=0.)
                error=(prediction-response[ids]).abs()[observed[ids]]
                numerator+=float(error.sum().cpu()); denominator+=error.numel()
        score=numerator/denominator
        if not np.isfinite(score):
            raise RuntimeError('Nonfinite validation MAE')
        logs.append({'epoch':epoch,'train_loss':float(np.mean(sums)),'validation_MAE':score})
        if score<best:
            best,best_epoch=score,epoch
            state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            torch.save({'state_dict':state,'best_epoch':best_epoch,'validation_MAE':best},args.out/'best_weights.pt')
        (args.out/'epoch_log.json').write_text(json.dumps(logs,indent=2)+'\n',encoding='utf8')
        if epoch==1 or epoch%5==0:
            print(json.dumps(logs[-1]),flush=True)
        if epoch-best_epoch>=args.patience:
            break
    model.load_state_dict(state); model.eval()
    selected=np.flatnonzero(split!='train'); output=[]
    with torch.no_grad():
        for ids in np.array_split(selected,int(np.ceil(len(selected)/args.batch_size))):
            output.append(torch.clamp(model(*(value[ids] for value in arrays))*std+mean,min=0.).cpu().numpy())
    output=np.concatenate(output)
    if not np.isfinite(output).all():
        raise RuntimeError('Nonfinite saved predictions')
    torch.save({'state_dict':state,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'visit_mean':mean,'visit_std':std,'numeric_mean':numeric_mean,'numeric_std':numeric_std},args.out/'checkpoint.pt')
    np.savez_compressed(args.out/'predictions.npz',model=args.model,adjacency=graph_label,seed=args.seed,
                        forecast_origin=np.asarray(dates[origins[selected]],dtype=str),horizon_step=np.arange(1,args.H+1),
                        poi_id=attrs.placekey.to_numpy(dtype=str),observed_visits=raw[target[selected]],
                        predicted_visits=output,valid=valid[selected],dataset_split=split[selected],
                        train_end=protocol.train_end,T=args.T,H=args.H,protocol='complete_horizon',
                        bundle_sha256=dataset_sha256,density=args.density or '')
    source_files=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'experiments/archived_2020q4').glob('*.py'))
    source_files+=list((ROOT/'external/MTGNN').glob('*.py'))+list((ROOT/'src').glob('*.py'))
    record={'status':'complete','model':args.model,'adjacency':graph_label,'variant':args.variant,
            'T':args.T,'H':args.H,'seed':args.seed,'training_loss':args.loss,'lr':args.lr,
            'dropout':args.dropout,'validation_MAE':best,'best_epoch':best_epoch,'epochs':logs,
            'seconds':time.monotonic()-started,'parameters':sum(p.numel() for p in model.parameters()),
            'train_end':protocol.train_end,'validation_end':protocol.validation_end,
            'origins':{s:int((split==s).sum()) for s in ('train','validation','test')},
            'graph_sha256':graph_digest,'bundle_sha256':dataset_sha256,
            'metadata_sha256':metadata_sha256,'density':density,
            'source_sha256':{str(f.relative_to(ROOT)).replace('\\','/'):sha256(f.read_bytes()).hexdigest() for f in sorted(source_files)},
            'environment':environment()}
    (args.out/'fit.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:record[k] for k in ('status','validation_MAE','best_epoch','seconds')}),flush=True)


def smoke(device_name):
    import torch
    device=torch.device(device_name)
    torch.manual_seed(42)
    x=torch.randn(2,14,12,device=device)
    static=np.zeros((12,10),dtype=np.float32)
    static[:,:4]=np.arange(48,dtype=np.float32).reshape(12,4)/48
    for name in ('GraphGRU','GNN-static','GNN-parent','Shared-LSTM','Matched-LSTM','Fixed-MTGNN'):
        model=make_model(name,7,static,feature_mask('full'),12,14,
                         (lambda z:z,lambda z:z),device,.1).to(device)
        output=model(x,x)
        assert output.shape==(2,7,12) and torch.isfinite(output).all()
        output.square().mean().backward()
        assert any(p.grad is not None for p in model.parameters())
        model.eval()
        with torch.no_grad():
            a=model(x,x)
            changed=x.clone(); changed[:,:,5]+=100
            b=model(changed,changed)
            torch.testing.assert_close(a[:,:,0],b[:,:,0],rtol=1e-5,atol=1e-6)
        print(json.dumps({'model':name,'forward_backward':'passed','identity_isolation':'passed'}),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--check-environment',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--bundle',type=Path); p.add_argument('--out',type=Path)
    p.add_argument('--panel-file',type=Path)
    p.add_argument('--model',choices=('GraphGRU','GNN-static','GNN-parent','Shared-LSTM','Matched-LSTM','Fixed-MTGNN','Adaptive-MTGNN'),default='GraphGRU')
    p.add_argument('--adjacency',choices=('Identity','NAICS-2','NAICS-4','Controlled','Adaptive','Distance KNN','Distance RBF','Visit correlation'),default='Identity')
    p.add_argument('--graph-file',type=Path); p.add_argument('--variant',default='full')
    p.add_argument('--sparse-graph',type=Path);p.add_argument('--attributes-file',type=Path)
    p.add_argument('--density')
    p.add_argument('--T',type=int,default=14); p.add_argument('--H',type=int,default=7)
    p.add_argument('--train-end',type=int,default=50); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=.001); p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--patience',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=4); p.add_argument('--loss',choices=('MAE','MSE'),default='MAE')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if args.check_environment:
        report=environment(); print(json.dumps(report,indent=2)); sys.exit(0 if report['cuda_available'] else 2)
    if args.smoke:
        smoke(args.device); sys.exit(0)
    if args.bundle is None or args.out is None:
        p.error('--bundle and --out are required for training')
    if args.out.exists():
        p.error('Output already exists; preserve it or use a new run directory')
    if args.adjacency=='Controlled' and args.graph_file is None:
        p.error('Controlled adjacency requires --graph-file')
    if args.model in ('GNN-static','Shared-LSTM','Matched-LSTM') and args.adjacency!='Identity':
        p.error('This configuration requires identity adjacency')
    if (args.model=='Adaptive-MTGNN') != (args.adjacency=='Adaptive'):
        p.error('Adaptive-MTGNN requires adjacency Adaptive, and vice versa')
    run(args)
