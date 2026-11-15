"""Audit completed fitting and manuscript sources before final publication.

This creates a public, data-free inventory of all fits, including the tuning
fits. It checks validation selection instead of assuming selected_runs is right.
It never publishes the repository or labels an incomplete queue as complete.
"""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json
import re
import shutil
import pandas as pd
from summarize_queue import write_json

ROOT=Path(__file__).resolve().parents[2]


def digest(path):return sha256(path.read_bytes()).hexdigest()


def manuscript_check(folder):
    used=[]
    def read(path,stack=()):
        path=path.resolve()
        if path in stack:raise ValueError('Recursive TeX input')
        used.append(path)
        content=path.read_text(encoding='utf8')
        content=re.sub(r'(?<!\\)%[^\n]*','',content)
        def include(match):
            child=folder/match.group(1)
            if not child.suffix:child=child.with_suffix('.tex')
            return read(child,stack+(path,))
        return re.sub(r'\\input\{([^}]+)\}',include,content)
    content=read(folder/'outline.tex')
    labels=Counter(re.findall(r'\\label\{([^}]+)\}',content))
    refs=set(re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}',content))
    citations={key.strip() for block in re.findall(r'\\cite[a-zA-Z]*\{([^}]+)\}',content) for key in block.split(',')}
    bib=(folder/'reference.bib').read_text(encoding='utf8')
    entries=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib))
    missing_graphics=[]
    for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',content):
        path=folder/name
        candidates=[path] if path.suffix else [path.with_suffix(s) for s in ('.pdf','.png','.jpg')]
        if not any(p.exists() for p in candidates):missing_graphics.append(name)
    sections=re.findall(r'\\section\{([^}]+)\}',content)
    expected=['Introduction','Literature Review','Data and Problem Formulation','Methodology','Empirical Results','Conclusions and future work']
    required={'revision_forecasting.tex','revision_forecasting_intervals.tex','revision_density.tex',
              'revision_degree_controls.tex','revision_spatial_graphs.tex','revision_feature_ablations.tex',
              'revision_parent_ablation.tex','revision_activity_coverage.tex'}
    return dict(duplicate_labels=[k for k,v in labels.items() if v>1],undefined_references=sorted(refs-set(labels)),
                undefined_citations=sorted(citations-entries),missing_graphics=missing_graphics,
                missing_completed_result_inputs=sorted(required-{p.name for p in used}),
                original_six_sections_preserved=sections[:6]==expected,
                tex_sources={str(p.relative_to(folder)):digest(p) for p in used},
                unresolved_result_tokens=re.findall(r'\b(?:TODO|TBD|PLACEHOLDER|XX+|pending results)\b',content))


def run(args):
    artifact=args.root.resolve();original=json.loads((ROOT/'paper/original_manifest.json').read_text())
    for item in original['files']:
        if digest(ROOT/'paper/original'/item['path'])!=item['sha256']:raise ValueError('Original manuscript was changed')
    missing=[];rows=[];sources=[];recipes=[]
    archive_dir=artifact/'fitted_source_archives';archive_dir.mkdir(exist_ok=True)
    for name in ('20261114_v1','20261114_v2'):
        local=artifact/'private_local_runs'/name;queue=local/'queue'
        archive=local/'training_source.zip';execution=json.loads((local/'execution.json').read_text())
        if digest(archive)!=execution['source_archive_sha256']:raise ValueError('Training archive changed')
        for suffix in ('.zip','.json'):
            source=local/('training_source'+suffix);destination=archive_dir/(name+suffix)
            if destination.exists() and digest(source)!=digest(destination):raise ValueError('Published archive differs')
            if not destination.exists():shutil.copyfile(source,destination)
        sources.append(dict(queue=name,archive=f'fitted_source_archives/{name}.zip',sha256=digest(archive)))
        manifest_path=queue/'manifest.json'
        if not manifest_path.exists():missing.append(name+':not_started');continue
        manifest=json.loads(manifest_path.read_text());recipes.extend(manifest['recipes'])
        for recipe in manifest['recipes']:
            folder=queue/recipe['id'];selected=folder/'selected_runs.json'
            if not selected.exists():missing.append(recipe['id']);continue
            selection=json.loads(selected.read_text())
            tuning=[(path,json.loads(path.read_text())) for path in folder.glob('seed42_*/fit.json')]
            if sorted(f['lr'] for _,f in tuning)!=[.0003,.001,.003]:raise ValueError('Missing learning-rate tuning fits')
            winner=min(tuning,key=lambda value:(value[1]['validation_MAE'],value[1]['lr']))[1]
            if selection['selected_lr']!=winner['lr']:raise ValueError('Selection is not validation-minimal')
            if sorted(f['seed'] for f in selection['runs'])!=[42,43,44,45,46]:raise ValueError('Incomplete seed set')
            selected_paths={Path(run['directory']).name for run in selection['runs']}
            for path in sorted(folder.glob('*/fit.json')):
                fit=json.loads(path.read_text());assert fit['status']=='complete'
                directory=path.parent
                if fit['source_sha256']!=manifest['source_sha256']:raise ValueError('Fit code provenance differs from queue')
                rows.append(dict(queue=name,recipe=recipe['id'],stage=recipe['stage'],
                    model=fit['model'],adjacency=fit['adjacency'],T=fit['T'],H=fit['H'],variant=fit['variant'],
                    loss=fit['training_loss'],seed=fit['seed'],learning_rate=fit['lr'],
                    selected=directory.name in selected_paths,validation_MAE=fit['validation_MAE'],
                    best_epoch=fit['best_epoch'],completed_epochs=len(fit['epochs']),parameters=fit['parameters'],
                    seconds=fit['seconds'],fit_sha256=digest(path),checkpoint_sha256=digest(directory/'checkpoint.pt'),
                    predictions_sha256=digest(directory/'predictions.npz'),source_archive_sha256=digest(archive)))
    pd.DataFrame(rows).to_csv(artifact/'fitting_inventory.csv',index=False)
    paper=manuscript_check(ROOT/'paper/revision')
    errors=bool(missing) or any(paper[key] for key in ('duplicate_labels','undefined_references','undefined_citations',
                      'missing_graphics','unresolved_result_tokens','missing_completed_result_inputs')) or not paper['original_six_sections_preserved']
    report=dict(status='incomplete' if errors else 'source_checks_passed',original_files_unchanged=len(original['files']),
        completed_fits_in_complete_recipes=len(rows),missing_completed_recipes=missing,source_archives=sources,
        manuscript=paper,scope='Fit provenance and manuscript source checks; PDF rendering, final result interpretation, remote synchronization, and independent review require separate verification.')
    write_json(artifact/'release_source_audit.json',report)
    print(json.dumps(dict(status=report['status'],fits=len(rows),missing_recipes=len(missing),
                    manuscript_issues={k:v for k,v in paper.items() if k not in ('tex_sources',)})))
    if args.require_complete and errors:raise SystemExit(1)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true');run(p.parse_args())
