# exp0: Data preparation and temporal sparsity

Prepare the approved panel and reproduce the data-description figures. `plot-sparsity` shows observed zero-to-positive sequences, completed zero-gap prevalence and cross-POI visit density. `describe` computes the daily/category aggregates; `plot` renders the two final separate figures.

```bash
python -m exp exp0 profile --raw-root data/raw --attributes data/Seattle_all_poi.csv --out data/prepared
python -m exp exp0 package --prepared data/prepared --attributes data/Seattle_all_poi.csv --out data/approved_panel.zip
python -m exp exp0 describe --prepared data/prepared --attributes data/Seattle_all_poi.csv --out results/exp0/data
python -m exp exp0 plot --data results/exp0/data --out results/exp0/art
python -m exp exp0 plot-sparsity --prepared data/prepared --out results/exp0/art
```

See [the data contract](../DATA.md) for the monthly reconstruction file needed by `package`. `prepare --mode original-with-mask` retains the original cohort; the other explicit modes are alternative cohort construction tools, not substitutes for the checksum-pinned replay bundle. `extract-attributes` supports those cohort checks. The full-period description and training-defined low-history labels have different scopes.

## Source files

| File | Responsibility |
| --- | --- |
| [`categories.py`](categories.py) | Original nine-category mapping, unchanged from the approved cohort. |
| [`describe.py`](describe.py) | Export the observed-day statistics underlying the descriptive paper figures. |
| [`extract_attributes.py`](extract_attributes.py) | Keep one official 2020 static record per candidate POI with source provenance. |
| [`package.py`](package.py) | Verify and package the approved October--December panel for the formal run. |
| [`plot.py`](plot.py) | Render the final daily and category plots at their original 8.8 by 5.8 inch size. |
| [`plot_sparsity.py`](plot_sparsity.py) | Plot observed zero-to-positive visit sequences and their cohort prevalence. |
| [`prepare.py`](prepare.py) | Build the final experiment inputs using the explicitly selected cohort policy. |
| [`profile.py`](profile.py) | Check the approved experiment period and the identified industry cohort. |

For the full option list, use `python -m exp exp0 <action> --help`. Shared input schemas are documented in [DATA.md](../DATA.md).
