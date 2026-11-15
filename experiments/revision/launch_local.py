"""Run the bounded revision queue from an immutable source snapshot on local CUDA."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
import json
import os
import subprocess
import sys
import time
from package_training import package

ROOT = Path(__file__).resolve().parents[2]


def main(args):
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / 'training_source.zip'
    manifest = package(archive)
    source = destination / 'source'
    with ZipFile(archive) as bundle:
        for name in bundle.namelist():
            path = (source / name).resolve()
            if not path.is_relative_to(source.resolve()):
                raise ValueError('Invalid source archive path')
        bundle.extractall(source)
    queue = destination / 'queue'
    cmd = [sys.executable, '-u', str(source/'experiments/revision/training_queue.py'),
           '--stage', args.stage, '--execute', '--bundle', str(args.bundle.resolve()),
           '--graph-dir', str(args.graph_dir.resolve()), '--out', str(queue),
           '--max-hours', str(args.max_hours), '--max-new-fits', str(args.max_new_fits)]
    if args.extra_input_dir:cmd+=['--extra-input-dir',str(args.extra_input_dir.resolve())]
    status = dict(status='running', launcher_pid=os.getpid(),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  executable=sys.executable, source_archive_sha256=manifest['sha256'],
                  source_files=len(manifest['members']), command=cmd,
                  queue=str(queue), source=str(source))
    status_path = destination/'execution.json'
    status_path.write_text(json.dumps(status, indent=2)+'\n', encoding='utf8')
    print(json.dumps(status), flush=True)
    if args.after:
        status['status']='waiting_for_predecessor'
        status['predecessor_queue']=str(args.after.resolve())
        status_path.write_text(json.dumps(status,indent=2)+'\n',encoding='utf8')
        started=time.monotonic()
        while True:
            path=args.after/'queue_status.json'
            if path.exists():
                predecessor=json.loads(path.read_text())
                if predecessor['status']!='complete':
                    raise RuntimeError('Predecessor stopped: '+json.dumps(predecessor))
                break
            if time.monotonic()-started>12*3600:raise TimeoutError('Predecessor wait exceeded 12 hours')
            time.sleep(10)
        status['status']='running'
        status_path.write_text(json.dumps(status,indent=2)+'\n',encoding='utf8')
    try:
        with (destination/'queue_console.log').open('w', encoding='utf8') as log:
            process = subprocess.Popen(cmd, cwd=source, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf8', errors='replace')
            status['queue_pid'] = process.pid
            status_path.write_text(json.dumps(status, indent=2)+'\n', encoding='utf8')
            for line in process.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
            result = process.wait()
        status.update(status='complete' if result==0 else 'stopped', returncode=result,
                      finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(status='interrupted', reason=str(error))
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2)+'\n', encoding='utf8')
    raise SystemExit(result)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--graph-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=('core','loss','graph','ablation','adaptive','spatial','density','metadata','extension','all'), default='all')
    parser.add_argument('--extra-input-dir',type=Path)
    parser.add_argument('--after',type=Path)
    parser.add_argument('--max-hours', type=float, default=12)
    parser.add_argument('--max-new-fits', type=int, default=357)
    main(parser.parse_args())
