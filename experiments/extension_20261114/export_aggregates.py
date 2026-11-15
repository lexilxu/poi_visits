"""Export the explicit aggregate-only extension evidence release."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil


def main():
    private = Path('artifacts/revision_20261114/private_substantive_20261114')
    out = Path('artifacts/revision_20261114/final_results/extension_20261114')
    assert not out.exists(), 'Preserve the existing release; use a new destination.'
    files = {}

    def add(source, target):
        assert source.is_file(), source
        assert target not in files, target
        files[target] = source

    for kind, base in [('matched', private / 'matched_aggregates'),
                       ('parent', private / 'substantive_aggregates_v1')]:
        for period in ['december', 'january']:
            root = base / f'{kind}_evaluation' / period
            for source in sorted(root.glob('*.csv')):
                add(source, f'{kind}/{period}/{source.name}')
            add(base / f'{kind}_comparison' / period / 'paired_intervals.csv',
                f'{kind}/{period}/paired_intervals.csv')
        add(base / f'{kind}_comparison' / 'selected_fit_resources.csv',
            f'{kind}/selected_fit_resources.csv')
    base = private / 'substantive_aggregates_v1'
    add(base / 'matched_comparison/hardware_resources.json', 'matched/hardware_resources.json')
    for name in ['metrics_by_seed.csv', 'metrics_summary.csv', 'screening_intervals.csv']:
        add(base / 'parking_screening_audit' / name, f'parking/{name}')
    add(base / 'parking_evaluation/paired_intervals.csv', 'parking/paired_intervals.csv')
    for name in ['audit.json', 'parking_daily_coverage.csv']:
        add(base / 'inputs' / name, f'input_audit/{name}')
    add(base / 'city_panels/audit.json', 'cities/panel_audit.json')
    for source in sorted((private / 'city_aggregates').glob('*')):
        if source.suffix in ['.csv', '.json'] and source.name != 'manuscript_source.json':
            add(source, f'cities/{source.name}')
    for name in ['complete.json', 'group_shares.csv', 'regime_transition.csv']:
        add(private / 'attribution_origin_summary' / name, f'attribution_origin/{name}')

    forbidden = {'placekey', 'safegraph_place_id', 'poi_id', 'poi_ids', 'latitude',
                 'longitude', 'location_name', 'street_address', 'weights', 'coefficients'}
    receipts = {}
    for target, source in sorted(files.items()):
        data = source.read_bytes()
        text = data.decode('utf8')
        assert not re.search(r'[a-z0-9]{3}-[a-z0-9]{3}@[a-z0-9]{3}-[a-z0-9]{3}-[a-z0-9]{3}', text, re.I), source
        assert not re.search(r'github_pat_|ghp_[a-zA-Z0-9]{30}|sk-[a-zA-Z0-9]{30}|AIza[a-zA-Z0-9_-]{30}', text), source
        if source.suffix == '.csv':
            columns = next(csv.reader(text.splitlines()))
            assert not forbidden.intersection(c.lower() for c in columns), source
        else:
            def inspect(value):
                if isinstance(value, dict):
                    assert not forbidden.intersection(k.lower() for k in value), source
                    for child in value.values(): inspect(child)
                elif isinstance(value, list):
                    for child in value: inspect(child)
            inspect(json.loads(text))
        destination = out / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        receipts[target] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    manifest = dict(status='aggregate_only_export_verified', files=receipts,
                    new_neural_fits=84, new_configurations=12, new_selected_runs=60,
                    cumulative_neural_fits=581, cumulative_configurations=83,
                    cumulative_selected_runs=415,
                    parking_input_seed_radius_combinations=93,
                    parking_distinct_ridge_candidate_fits=16,
                    parking_screening='Training-defined candidates ranked before observation masking',
                    attribution_origin='Paired FP64 arithmetic; archived FP32 replay discrepancies retained')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'files': len(files), 'destination': str(out)}))


if __name__ == '__main__':
    main()
