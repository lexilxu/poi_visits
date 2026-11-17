# exp7: Temporal validation and matched architecture controls

Validate forecasts in January, earlier city cohorts and matched architecture controls. `extract-monthly` validates streamed source size/checksum before extraction; `prepare-january` and `prepare-cities` construct aligned holdout panels. [PROTOCOL.md](PROTOCOL.md) records cohort and selection definitions.

```bash
python -m exp exp7 prepare-january --help
python -m exp exp7 evaluate-evolvegcn --help
python -m exp exp7 evaluate-controls --help
python -m exp exp7 train-matched --plan-out results/exp7/matched_plan.json
python -m exp exp7 train-cities --plan-out results/exp7/city_plan.json
```

`evaluate-matched`, `compare-controls` and `evaluate-cities` score their saved queues. `paired-uncertainty` computes the paired intervals for corrected forecasts. `matched-tables` and `city-tables` render completed aggregate CSVs; they write only to their explicit output directories. No test-period labels enter model or hyperparameter selection.

## Source files

| File | Responsibility |
| --- | --- |
| [`audit_inputs.py`](audit_inputs.py) | Verify acquired sources and construct analysis panels exclusively on AutoDL. |
| [`city_tables.py`](city_tables.py) | Render the completed earlier-period city summaries. |
| [`compare_controls.py`](compare_controls.py) | Paired controls and artifact audit on AutoDL, reusing all original fits. |
| [`evaluate_cities.py`](evaluate_cities.py) | Replay city checkpoints and score the frozen earlier-period control design. |
| [`evaluate_controls.py`](evaluate_controls.py) | Frozen-parameter January validation, with original validation thresholds. |
| [`evaluate_evolvegcn.py`](evaluate_evolvegcn.py) | Frozen corrected EvolveGCN forecasts on the existing January holdout. |
| [`evaluate_matched.py`](evaluate_matched.py) | Evaluate all selected matched controls on AutoDL using the frozen evaluators. |
| [`extract_monthly.py`](extract_monthly.py) | Verify a raw gzip source and package the fixed cohort and two cities. |
| [`matched_tables.py`](matched_tables.py) | Render already computed AutoDL summaries into a compact manuscript table. |
| [`paired_uncertainty.py`](paired_uncertainty.py) | Paired uncertainty for five-seed losses on fixed corrected forecasts. |
| [`prepare_cities.py`](prepare_cities.py) | Construct cohorts from July records, before August validation/September test. |
| [`prepare_january.py`](prepare_january.py) | Extract a later-month holdout without changing the original POI cohort. |

For the full option list, use `python -m exp exp7 <action> --help`. Shared input schemas are documented in [DATA.md](../DATA.md).
