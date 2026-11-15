"""Copy reviewed aggregate result tables into a data-free release directory."""
from pathlib import Path
from hashlib import sha256
import csv
import io
import json
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]
private=ROOT/'artifacts/e2031'
source=private/'results_final'
out=ROOT/'artifacts/revision_20261114/final_results'
out.mkdir(parents=True,exist_ok=True)
manifest=[]
forbidden={'poi_id','placekey','safegraph_place_id','latitude','longitude','forecast_origin',
           'observed_visits','predicted_visits','address','street_address'}
placekey=re.compile(r'\b[a-z0-9]{3}-[a-z0-9]{3}@[a-z0-9]{3}-[a-z0-9]{3}-[a-z0-9]{3}\b',re.I)
def copy(p,relative):
    payload=p.read_bytes()
    original_payload=payload
    omitted=[]
    if p.suffix=='.csv':
        text=payload.decode('utf-8-sig')
        rows=csv.DictReader(text.splitlines())
        assert not forbidden.intersection(rows.fieldnames),p
        assert not placekey.search(text),p
        fields=list(rows.fieldnames);records=list(rows)
        omitted=[field for field in fields if field=='representative_POIs']
        if omitted:
            buffer=io.StringIO(newline='')
            writer=csv.DictWriter(buffer,fieldnames=[f for f in fields if f not in omitted],lineterminator='\n',extrasaction='ignore')
            writer.writeheader();writer.writerows(records)
            payload=buffer.getvalue().encode('utf8')
    destination=out/relative;destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        existing=destination.read_bytes()
        assert existing in (payload,original_payload),destination
    destination.write_bytes(payload)
    manifest.append(dict(path=Path(relative).as_posix(),sha256=sha256(payload).hexdigest(),bytes=len(payload),omitted_columns=omitted))
tables=['metrics_summary.csv','horizons_summary.csv','categories_summary.csv',
        'matched_far_summary.csv','activity_coverage_summary.csv']
for period in ('december','january'):
    folder=source if period=='december' else source/'january'
    for name in tables:copy(folder/name,Path(name) if period=='december' else Path('january')/name)
    for name in ('paired_intervals.csv',):copy(source/'final_uncertainty'/period/name,Path('final_uncertainty')/period/name)
    for name in ('paired_coverage.csv','coverage_by_seed.csv','protocol.json'):
        copy(source/'final_coverage'/period/name,Path('final_coverage')/period/name)
    parent=private/'consolidated_20261114'/('parent_'+period)
    destination=Path('attribution' if period=='december' else 'attribution_january')/'summary'
    for p in parent.iterdir():
        if p.suffix=='.csv' or p.name=='summary_protocol.json':copy(p,destination/p.name)
for name in ('density_summary.csv','density_by_seed.csv','protocol.json'):
    p=source/'density'/name
    if p.exists():copy(p,Path('density')/name)
neighbors=ROOT/'artifacts/revision_20261114/neural_results/neighbors_final/summary'
for p in neighbors.iterdir():
    if p.suffix in ('.csv','.json') and p.name!='paper_render_manifest.json':copy(p,Path('neighbors_final/summary')/p.name)
copy(private/'bysgnn_source/resource_probe.json','bysgnn_resource_probe.json')
copy(private/'replay_evidence/replay_deterministic_v4_audit/reporting_error_audit.json','numerical_reporting_audit.json')
copy(private/'replay_evidence/replay_deterministic_v4_repeat/repeat_audit.json','numerical_repeat_audit.json')
copy(private/'final_evidence/january_numerical_revision_20261114/protocol_revision.json','numerical_protocol_revision.json')
for label in ('20261114_v1','20261114_v2'):
    p=ROOT/'artifacts/revision_20261114/private_local_runs'/label/'training_source.zip'
    with zipfile.ZipFile(p) as z:
        assert all(Path(name).suffix=='.py' for name in z.namelist())
        for name in z.namelist():
            text=z.read(name).decode('utf8')
            assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|BEGIN [A-Z ]*PRIVATE KEY)',text)
    copy(p,Path('fitted_source_archives')/(label+'.zip'))
for name in ('fitting_inventory.csv','fit_audit.json'):
    copy(out/name,name)
(out/'aggregate_export_manifest.json').write_text(json.dumps(dict(
    status='aggregate_content_checks_passed',files=manifest,
    policy='Aggregate metrics, hashes and Python sources; licensed event arrays and identifiers stay in private storage'),indent=2))
print(json.dumps(dict(status='exported',files=len(manifest),bytes=sum(r['bytes'] for r in manifest))))
