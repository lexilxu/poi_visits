"""Evaluate all selected matched controls on AutoDL using the frozen evaluators."""
from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import time


def main(a):
    sys.path.insert(0,str(a.source/'experiments/revision'))
    sys.path.insert(1,str(a.source/'experiments/archived_2020q4'))
    import torch
    from audit_protocol import load_panel
    from summarize_queue import selected_jobs,evaluate_one,merge,write_json,evaluate_arrays
    from evaluate_january import extended_panel,replay_model,save_and_evaluate
    from evaluate_saved import load_events
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    raw,dates,attrs=load_panel(a.bundle)
    extended,extended_dates,extended_attrs=extended_panel(a.bundle,a.panel)
    jobs=selected_jobs([a.queue])
    assert len(jobs)==20,len(jobs)
    for period in ('december','january'):
        (a.out/period/'run_records').mkdir(parents=True,exist_ok=True)
    audit=[]
    for path,meta,_ in jobs:
        name=meta['id'].replace(' ','_')+f"_seed{meta['seed']}"
        december=a.out/'december';january=a.out/'january'
        if not (december/'run_records'/f'{name}.json').exists():
            write_json(december/'run_records'/f'{name}.json',evaluate_one(path,meta,raw,dates,attrs,True))
        if not (january/'run_records'/f'{name}.json').exists():
            origins,pred,extra=replay_model(path.parent,extended,extended_dates,extended_attrs,'cuda:0',a.queue)
            assert extra['strict_replay_passed']
            save_and_evaluate(january,origins,pred,meta,extended,extended_dates,extended_attrs,extra)
            record=january/'run_records'/f'{name}.json'
            report=json.loads(record.read_text())
            report.update(evaluate_arrays(load_events(january/'private_predictions'/f'{name}.npz'),meta,extended,extended_attrs,True))
            write_json(record,report)
        audit.append(dict(id=meta['id'],seed=meta['seed']))
        print(json.dumps(dict(completed=len(audit),total=len(jobs),id=meta['id'],seed=meta['seed'])),flush=True)
    for period in ('december','january'):merge(a.out/period/'run_records',a.out/period)
    write_json(a.out/'complete.json',dict(status='complete',selected_runs=audit,completed_at=time.time()))


if __name__=='__main__':
    p=ArgumentParser()
    for field in ('source','queue','bundle','panel','out'):p.add_argument('--'+field,type=Path,required=True)
    main(p.parse_args())
