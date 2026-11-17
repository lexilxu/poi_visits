# exp8: Independent parking validation

Compare saved POI forecasts with independent Seattle parking observations. `download` fetches the public source archive. `evaluate` is the payment-based parking analysis; `audit-screening` checks candidate selection. The later availability analysis has separate `inputs.py`, `audit-coverage`, `historical-forecasts`, `evaluate-availability`, `compare-calendar` and `verify` actions.

```bash
python -m exp exp8 download --out data/parking
python -m exp exp8 evaluate --help
python -m exp exp8 evaluate-availability --help
python -m exp exp8 verify --help
python -m exp exp8 tables --source results/extension --out results/exp8/tables
```

Keep the payment and availability protocols distinct. [PARKING_PROTOCOL.md](PARKING_PROTOCOL.md), [PARKING_CANDIDATE_AUDIT.md](PARKING_CANDIDATE_AUDIT.md) and [AVAILABILITY_PROTOCOL.md](AVAILABILITY_PROTOCOL.md) document source fields, coverage, fitting boundaries and candidate screening. These downstream analyses read existing POI predictions; they do not train a new POI forecasting architecture.

## Source files

| File | Responsibility |
| --- | --- |
| [`audit_coverage.py`](audit_coverage.py) | Audit stored parking selections, observation coverage and identification bounds. |
| [`audit_screening.py`](audit_screening.py) | Recompute screening with training-defined candidate sets and audit saved errors. |
| [`compare_calendar.py`](compare_calendar.py) | Compare the prespecified calendar control using preserved forecasts. |
| [`download.py`](download.py) | Acquire official payment-based occupancy records at prespecified clock times. |
| [`evaluate.py`](evaluate.py) | Frozen two-stage paid-parking forecast experiment; execute on AutoDL4. |
| [`evaluate_availability.py`](evaluate_availability.py) | Evaluate dated parking inputs and historical-forecast mapping on AutoDL4. |
| [`historical_forecasts.py`](historical_forecasts.py) | Reuse earlier Seattle temporal models for genuinely held-out mapping inputs. |
| [`inputs.py`](inputs.py) | Construct calendar and explicitly dated lag inputs for the parking audit. |
| [`tables.py`](tables.py) | Render source-backed AutoDL aggregate values without fitting or rescoring. |
| [`verify.py`](verify.py) | Check saved forecast metrics, date boundaries and complete experiment axes. |

For the full option list, use `python -m exp exp8 <action> --help`. Shared input schemas are documented in [DATA.md](../DATA.md).
