# Original-checkpoint coordinate-origin diagnostic

Fixed before execution. Run only on AutoDL4 with the existing T14/H7 identity
GNN-static checkpoint at `/root/autodl-tmp/poi_formal_2020q4_12d6414990ee`.
No training or checkpoint mutation. Preserve original results. Output to a new
directory; retain summary, POI profiles and two small attribution arrays only.

Use all 22 original December test origins, seven leads and 9,884 POIs. Match the
archived attribution definition: gradient of normalized pre-clipping output,
absolute component scores before summing input lags, then aggregate by feature
group. Check FP32 predictions against the archive with the existing original
attribution tolerances (absolute 0.002, relative 0.001); report actual differences.

Evaluate both coordinate systems in FP64. Subtract the componentwise median of
the stored latitude/longitude values from every location vector and compensate
the input-projection bias by adding the location-weight matrix times that median.
The median uses only the fixed POI metadata. No choice based on prediction error,
attribution changes or desired direction is permitted. Require paired forecasts
to agree with absolute and relative tolerance 1e-9. Check nonlocation absolute
attribution totals with tolerance 1e-8. Report any failed gate.

Report mean location share, location-dominant POI count and standardized regime
agreement in the two coordinate systems. Standardize shares across the common
eligible POIs, with population standard deviation and first-index tie handling.
The test targets origin sensitivity of the original fitted function. It does not
change its forecasting accuracy or provide a feature-removal utility estimate.
This is one original checkpoint; five-seed results already in the appendix use a
different complete-window protocol and remain a separate experiment.
