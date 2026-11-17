# Input and output contract

Raw SafeGraph inputs, POI-level records, event predictions and fitted weights are excluded from Git. The scripts accept local paths and do not fetch licensed data. Public parking and cartography sources have separate acquisition helpers.

## Canonical panel

The original frozen bundle has SHA-256 `12d6414990eeb5703b1179152b5993518dcfa68361e39e09ae621919950695ec`.

| Bundle member | Contents |
| --- | --- |
| `visits.npy` | float32, shape `(92, 9884)`, date × POI; NaN for missing records |
| `dates.csv` | `date`, October 1–December 31, 2020, sorted |
| `poi_attributes.csv` | Aligned `placekey`, name, coordinates, category, NAICS, brand, parent and opening-hour attributes |
| `dataset_manifest.json` | Date range, cohort policy and input checksums |

There are 905,918 observed POI-days and 3,410 missing records. A recorded zero is an observation; a missing record is not a zero. The original training loader's historical imputation and masks remain separate from descriptive statistics, which exclude missing records from denominators.

Preparation uses these private files:

| Path below the supplied data root | Used by |
| --- | --- |
| `raw/paper_period_audit/Seattle_2020_09_to_12.csv` | `exp0 profile`, `prepare` |
| `raw/Seattle/processed/poi_meta.csv` | Original cohort identification |
| `Seattle_all_poi.csv` | Approved static attribute snapshot |
| `raw/source_manifest.json` and referenced gzip snapshots | `exp0 extract-attributes` |
| `prepared/approved_period_visits_with_missing.npy` | Shared descriptive matrix |
| `prepared/identified_9884_pois.csv`, `approved_dates.csv` | Ordered IDs and dates |
| `prepared/approved_period_source.csv.gz` | Independent monthly reconstruction in `exp0 package` |

`exp0 profile` exports the prepared matrix, IDs, dates and monthly source file. The monthly input is restricted to October–December and the approved IDs, then written as gzip CSV; `package` independently reconstructs and checks all observed values. Exact bundle bytes depend on the original source-file checksums in its manifest. Use the frozen bundle when replaying archived fits.

## Saved runs

Corrected EvolveGCN runs contain `complete.json`, `best_weights.pt`, `predictions.npz` and epoch logs. The prediction archive carries `predicted_visits`, `observed_visits`, `valid`, `forecast_origin`, `dataset_split`, `poi_id` and `horizon_step`. Shared control queues retain `fit.json` per fit and `selected_runs.json` per recipe. Do not rename individual run folders without updating the associated manifests.

Original GraphGRU/static archives use the earlier names such as `gnn-static__identity.pt` and `gnn-static__identity.npy`. The reference-centered attribution action accepts those archives and a matching original FP64 GPU audit. Its `--formal-source` directory also contains the frozen `train_formal.py` used to record source provenance. Classification/export code is now included in `exp5/regimes.py`; an external summary script is no longer needed at runtime.

## Outputs

`data/`, `runs/` and `results/` are ignored. Use explicit `--out` paths when a script requires them. Aggregate tables already under `artifacts/` are historical public results. New per-event outputs stay outside that directory. CSVs intended for release should be reviewed as aggregates before being added explicitly.
