"""Discover and run a single experiment action without importing other experiments."""
from pathlib import Path
import json
import runpy
import sys

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    catalog=json.loads(Path(__file__).with_name('catalog.json').read_text(encoding='utf-8'))
    if not args or args[0] in ('--help','-h','--list'):
        print('Usage: python -m exp <experiment> <action> [options]')
        print('       python -m exp <experiment> --help')
        for key,value in catalog.items():print(f"{key:8} {value['title']}")
        return
    group=args.pop(0)
    if group not in catalog:raise SystemExit(f'Unknown experiment: {group}; use --list')
    spec=catalog[group]
    if not args or args[0] in ('--help','-h','--list'):
        print(spec['title'])
        for key,(module,preset) in spec['actions'].items():
            print(f"  {key:26} {module} {' '.join(preset)}")
        return
    action=args.pop(0)
    if action not in spec['actions']:raise SystemExit(f'Unknown action: {group} {action}; use {group} --help')
    module,preset=spec['actions'][action]
    if '--stage' in preset and any(x=='--stage' or x.startswith('--stage=') for x in args):
        raise SystemExit('This action has a fixed stage; use exp1 queue-controls to select another stage')
    sys.argv=[module,*preset,*args]
    runpy.run_module(module,run_name='__main__')

if __name__=='__main__':main()
