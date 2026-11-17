"""Resolve source names in both historical and reorganized run manifests.

Only paths are aliased. The caller must still compare the hash of the code
actually being executed; moving code never waives a numerical verification gate.
"""
import json
from pathlib import Path
from exp.common.paths import REPO_ROOT

def source_hash(record, path):
    hashes = {key.replace('\\', '/'): value for key, value in record['source_sha256'].items()}
    path = str(path).replace('\\', '/')
    if path in hashes:
        return hashes[path]
    migration = json.loads((REPO_ROOT / 'exp/MIGRATION.json').read_text(encoding='utf-8'))['paths']
    for old, new in migration.items():
        if new == path and old in hashes:
            return hashes[old]
    raise KeyError(f'No recorded source hash for {path}')

def source_file(name):
    """Find a uniquely named historical helper in its current location."""
    migration = json.loads((REPO_ROOT / 'exp/MIGRATION.json').read_text(encoding='utf-8'))['paths']
    candidates = {new for old, new in migration.items() if Path(old).name == name}
    if len(candidates) != 1:
        raise ValueError(f'Ambiguous or absent historical helper: {name}')
    return REPO_ROOT / candidates.pop()
