"""CPU-only reference-centered attribution for the archived original GNN-static fit."""
from pathlib import Path
from argparse import ArgumentParser
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT
from exp.common.paths import DATA_ROOT, RESULTS_ROOT
FORMAL = DATA_ROOT / 'frozen_original'
OUT = RESULTS_ROOT / 'exp5/reference_centered'
parser = ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, default=REPO)
parser.add_argument('--formal-source', type=Path, default=FORMAL,
                    help='Directory containing the original train_formal.py.')
parser.add_argument('--bundle', type=Path, default=FORMAL/'approved_panel.zip')
parser.add_argument('--source', type=Path, default=OUT/'source',
                    help='Original checkpoint and predictions.')
parser.add_argument('--gpu-audit', type=Path, default=OUT.parent/'private_location_origin_20261115',
                    help='Independent original GPU FP64 complete.json and absolute_attribution.npz.')
parser.add_argument('--out', type=Path, default=OUT)
args = parser.parse_args() if __name__ == '__main__' else parser.parse_args([])
REPO, FORMAL, OUT, SRC = args.repo, args.formal_source, args.out, args.source
GROUPS = ['History', 'Location', 'Identity', 'Parent', 'Others']
sys.path[:0] = [str(REPO), str(FORMAL)]
from exp.common.original_protocol import GraphGRU, load_panel, make_windows, static_features


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def profile(raw):
    shares = raw / raw.sum(1, keepdims=True)
    spread = shares.std(0)
    z = np.divide(shares - shares.mean(0), spread, out=np.zeros_like(shares), where=spread > 0)
    return shares, z, z.argmax(1)


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_available(), 'Use the isolated CPU-only environment.'
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'complete.json').exists(), 'Preserve completed calculations.'
    checkpoint = SRC / 'gnn-static__identity.pt'
    assert digest(checkpoint) == '46ed4633a378136366a43787d09e3fe0a57b71c5d66e01cae673877a97c9022c'
    panel = load_panel(args.bundle, '12d6414990eeb5703b1179152b5993518dcfa68361e39e09ae621919950695ec')
    w = make_windows(panel.raw, 14, 7)
    static, _ = static_features(panel.attrs)
    model = GraphGRU(7, static).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.requires_grad_(False)
    hx_all = (panel.filled[w.history_idx[w.test_start:]] - w.mean) / w.std
    train_hx = (panel.filled[w.history_idx[:w.n_train]] - w.mean) / w.std
    history_ref = train_hx.astype(np.float64).mean(axis=(0, 2))
    old_predictions = np.load(SRC / 'gnn-static__identity.npy', allow_pickle=False)
    errors = np.abs(w.y[w.test_start:] - old_predictions)
    missing = panel.attrs.parent_placekey.fillna('').astype(str).str.strip().eq('').to_numpy()
    parent_codes = np.unique(static[missing, 8])
    assert len(parent_codes) == 1
    missing_code = int(parent_codes[0])
    protocol = dict(method='Reference-centered Gradient x Input', device='CPU', threads=4,
                    seed=42, T=14, H=7, POIs=len(static), training_origins=w.n_train,
                    test_origins=[panel.dates[i] for i in w.origins[w.test_start:]],
                    history_reference='Per-lag mean of actual normalized training inputs across training origins and POIs, including original zero placeholders.',
                    static_numeric_reference='Mean of actual static numeric inputs across the training POI cohort.',
                    category_reference='Mean learned embedding over training POIs, frequency weighted by POI occurrence.',
                    parent_reference='Learned embedding used for missing parent records.',
                    missing_parent_code=missing_code, missing_parent_POIs=int(missing.sum()),
                    aggregation='Absolute component attribution before summing over lags, horizons and origins; same five groups and population-standard-deviation assignment as original.',
                    output='Normalized model output before inverse scaling and clipping; last-visit residual gradient included.',
                    arithmetic='FP64 attribution and paired function-preserving tests; FP32 CPU replay compared to saved GPU forecasts.',
                    replay_atol=.002, replay_rtol=.001,
                    paired_prediction_atol=1e-9, paired_prediction_rtol=1e-9,
                    paired_component_atol=1e-9, paired_component_rtol=1e-9,
                    paired_share_atol=1e-10, paired_share_rtol=1e-10,
                    assignment_requirement='Exact agreement for all POIs.',
                    checkpoint_sha256=digest(checkpoint), prediction_sha256=digest(SRC / 'gnn-static__identity.npy'),
                    train_source_sha256=digest(FORMAL / 'train_formal.py'),
                    current_model_source_sha256=digest(Path(__import__('exp.common.original_protocol', fromlist=['__file__']).__file__)), source_sha256=digest(Path(__file__)))
    write('protocol.json', protocol)
    replay_parts = []
    with torch.no_grad():
        for x in hx_all:
            hx = torch.as_tensor(x[None])
            replay_parts.append((model(hx, hx)*w.std+w.mean).clamp_min(0).numpy())
    replay_prediction = np.concatenate(replay_parts)
    delta = np.abs(replay_prediction-old_predictions)
    replay = dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                  atol=.002, rtol=.001, passed=bool(np.allclose(replay_prediction, old_predictions, atol=.002, rtol=.001)))
    write('replay.json', replay)
    # Cross-device FP32 is diagnostic. Acceptance of CPU derivatives uses the
    # independently archived GPU FP64 attribution below, at a strict tolerance.
    gpu_audit_dir = args.gpu_audit
    gpu_audit = json.loads((gpu_audit_dir / 'complete.json').read_text())
    assert gpu_audit['checkpoint_sha256'] == protocol['checkpoint_sha256']
    assert gpu_audit['formal_source_sha256'] == protocol['train_source_sha256']
    assert gpu_audit['forecast_origin_dates'] == protocol['test_origins']
    assert gpu_audit['archived_FP32_replay']['max_abs'] == 0
    gpu_reference = np.load(gpu_audit_dir / 'absolute_attribution.npz', allow_pickle=False)
    np.testing.assert_array_equal(gpu_reference['placekey'], panel.attrs.placekey.to_numpy(str))
    protocol['CPU_derivative_validation'] = dict(reference=str(gpu_audit_dir),
         archive_sha256=digest(gpu_audit_dir / 'absolute_attribution.npz'), atol=1e-9, rtol=1e-9,
         requirement='Original uncentered CPU FP64 group totals agree with independently computed GPU FP64 totals at all 9884 POIs.')
    write('protocol.json', protocol)
    print(json.dumps({'phase':'FP32_replay', **replay}), flush=True)
    model.double()
    variants = {'baseline':model, 'parent_recentered':copy.deepcopy(model), 'location_recentered':copy.deepcopy(model)}
    parent_shift = model.embeddings[4].weight[missing_code].detach().clone()
    location_shift = model.static[:, :2].median(0).values.detach().clone()
    with torch.no_grad():
        pm = variants['parent_recentered']
        pm.embeddings[4].weight.sub_(parent_shift)
        pm.project.bias.add_(pm.project.weight[:, 37:45] @ parent_shift)
        lm = variants['location_recentered']
        lm.static[:, :2].sub_(location_shift)
        lm.project.bias.add_(lm.project.weight[:, 1:3] @ location_shift)
    references = {}
    for name, m in variants.items():
        ref = m.encode_static().detach().mean(0)
        ref[36:44] = m.embeddings[4].weight[missing_code]
        references[name] = ref
    np.savez_compressed(OUT / 'references.npz', history=history_ref,
                        static_reference=references['baseline'].numpy(), parent_shift=parent_shift.numpy(),
                        location_shift=location_shift.numpy())
    totals = {name:np.zeros((len(static),53), np.float64) for name in variants}
    old_totals = np.zeros_like(totals['baseline'])
    checks = {name:dict(max_prediction_difference=0., max_normalized_output_difference=0.,
                       max_signed_component_difference=0.) for name in list(variants)[1:]}
    semantics = ('Top category', 'Sub-category', 'NAICS-2', 'NAICS-4')
    slices = (slice(5,13),slice(45,53),slice(13,21),slice(21,29))
    attr_sums, dominates, dominant_error = [np.zeros(4,np.float64) for _ in range(3)]
    events = 0
    peak_rss = 0
    def components(m, hx, ref):
        es = m.encode_static().detach()
        projected = (F.linear(hx[...,None],m.project.weight[:,:1],m.project.bias)
                     +F.linear(es,m.project.weight[:,1:])).detach().requires_grad_(True)
        last = hx[:,-1,:].detach().requires_grad_(True)
        encoded = torch.cat((hx[...,None],es[None,None].expand(1,14,-1,-1)),dim=-1)
        reference = torch.cat((torch.as_tensor(history_ref)[None,:,None,None].expand(1,14,len(static),1),
                               ref[None,None,None].expand(1,14,len(static),52)),dim=-1)
        logits = m.forecast_projected(projected,last)
        for h in range(7):
            dp,dl = torch.autograd.grad(logits[:,h].sum(),(projected,last),retain_graph=h<6)
            grad = dp @ m.project.weight
            grad[:,-1,:,0] += dl
            yield logits.detach(), (encoded-reference)*grad, encoded*grad
    def grouped(scores):
        return np.stack((scores[:,0],scores[:,1:3].sum(-1),scores[:,5:37].sum(-1)+scores[:,45:53].sum(-1),
                         scores[:,37:45].sum(-1),scores[:,3:5].sum(-1)),axis=-1)
    for i,x in enumerate(hx_all):
        start = time.monotonic()
        hx = torch.as_tensor(x[None],dtype=torch.float64)
        generators = {name:components(m,hx,references[name]) for name,m in variants.items()}
        for h in range(7):
            logits, signed, old_signed = next(generators['baseline'])
            scores = signed.abs().sum(1)[0].numpy()
            totals['baseline'] += scores
            old_totals += old_signed.abs().sum(1)[0].numpy()
            group = grouped(scores)
            descriptors = np.stack([scores[:,s].sum(-1) for s in slices],axis=-1)
            eligible = descriptors.sum(1)>0
            desc = descriptors[eligible]
            attr_sums += (desc/group[eligible].sum(1)[:,None]).sum(0)
            winners = desc.argmax(1)
            dominates += np.bincount(winners,minlength=4)
            dominant_error += np.bincount(winners,weights=errors[i,h,eligible],minlength=4)
            events += int(eligible.sum())
            for name in checks:
                new_logits,new_signed,_ = next(generators[name])
                ck = checks[name]
                pred_delta = ((new_logits*w.std+w.mean).clamp_min(0)-(logits*w.std+w.mean).clamp_min(0)).abs().max().item()
                ck['max_prediction_difference'] = max(ck['max_prediction_difference'],pred_delta)
                ck['max_normalized_output_difference'] = max(ck['max_normalized_output_difference'],(new_logits-logits).abs().max().item())
                ck['max_signed_component_difference'] = max(ck['max_signed_component_difference'],(new_signed-signed).abs().max().item())
                torch.testing.assert_close(new_logits,logits,atol=1e-9,rtol=1e-9)
                torch.testing.assert_close(new_signed,signed,atol=1e-9,rtol=1e-9)
                totals[name] += new_signed.abs().sum(1)[0].numpy()
        peak_rss = max(peak_rss,psutil.Process().memory_info().rss)
        print(json.dumps(dict(phase='centered_attribution_and_two_recenterings',origins_completed=i+1,
                              total_origins=len(hx_all),seconds_this_origin=round(time.monotonic()-start,2),
                              elapsed_seconds=round(time.monotonic()-started,2),rss_MiB=round(peak_rss/2**20,1))),flush=True)
    raw = grouped(totals['baseline'])
    old_raw = grouped(old_totals)
    np.testing.assert_allclose(old_raw,gpu_reference['before'],atol=1e-9,rtol=1e-9)
    cpu_derivative_validation = dict(passed=True,atol=1e-9,rtol=1e-9,
        max_absolute_difference=float(np.abs(old_raw-gpu_reference['before']).max()))
    shares,z,labels = profile(raw)
    old_shares,old_z,old_labels = profile(old_raw)
    np.testing.assert_array_equal(raw[missing,3],0.)
    for name,ck in checks.items():
        test_raw = grouped(totals[name])
        ts,tz,tl = profile(test_raw)
        ck.update(max_absolute_total_difference=float(np.abs(totals[name]-totals['baseline']).max()),
                  max_share_difference=float(np.abs(ts-shares).max()),max_z_difference=float(np.abs(tz-z).max()),
                  changed_regimes=int((tl!=labels).sum()),regime_agreement=float((tl==labels).mean()))
        np.testing.assert_allclose(ts,shares,atol=1e-10,rtol=1e-10)
        np.testing.assert_array_equal(tl,labels)
        ck['passed'] = True
    from exp.exp5.regimes import export_attribution_regimes
    export_attribution_regimes(panel.attrs,shares,errors.mean((0,1)),OUT)
    pd.DataFrame(dict(descriptor=semantics,mean_share=attr_sums/events,dominance_rate=dominates/events,
                      dominant_mae=np.divide(dominant_error,dominates,out=np.full(4,np.nan),where=dominates>0),
                      n_dominant=dominates.astype(int))).to_csv(OUT/'semantic_attribution.csv',index=False)
    from exp.exp6.categories import EMBEDDED_CATEGORY_GROUPS as groups
    rows=[]
    for name,select in [(name,(~missing)&panel.attrs.top_category.isin(cats).to_numpy()) for name,cats in groups.items()]+[('All embedded POIs',~missing)]:
        if select.any():
            s=float(shares[select,3].mean())
            rows.append(dict(group=name,n_pois=int(select.sum()),parent_share=s,own_share=1-s,gap=2*s-1,mae=float(errors.mean((0,1))[select].mean())))
    pd.DataFrame(rows).to_csv(OUT/'parent_attribution.csv',index=False)
    pd.crosstab(pd.Series(old_labels,name='original_GxI'),pd.Series(labels,name='reference_centered')).reindex(index=range(5),columns=range(5),fill_value=0).to_csv(OUT/'regime_transition.csv')
    np.savez_compressed(OUT/'absolute_attribution.npz',**totals,original_GxI=old_totals,placekey=panel.attrs.placekey.to_numpy(str))
    panel.attrs[['placekey','location_name','category9','latitude','longitude']].to_csv(OUT/'poi_metadata.csv',index=False)
    result=dict(status='complete',protocol=protocol,archived_replay=replay,CPU_derivative_validation=cpu_derivative_validation,paired_checks=checks,
                group_order=GROUPS,mean_shares=shares.mean(0).tolist(),raw_dominance_counts=np.bincount(shares.argmax(1),minlength=5).tolist(),
                regime_counts=np.bincount(labels,minlength=5).tolist(),original_GxI_regime_counts=np.bincount(old_labels,minlength=5).tolist(),
                original_GxI_mean_shares=old_shares.mean(0).tolist(),original_GxI_raw_dominance=np.bincount(old_shares.argmax(1),minlength=5).tolist(),
                old_to_centered_regime_agreement=float((old_labels==labels).mean()),eligible_events=events,
                missing_parent_attribution_zero=True,maximum_missing_parent_attribution=float(raw[missing,3].max()),
                elapsed_seconds=time.monotonic()-started,peak_sampled_rss_MiB=peak_rss/2**20,
                checkpoint_unchanged=digest(checkpoint)==protocol['checkpoint_sha256'],remote_GPU_usage=0)
    write('complete.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
