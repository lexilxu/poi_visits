# Detailed evidence-grid figure

This descriptive display reuses the 20 stored GraphGRU-versus-identity prediction
comparisons for the same 1,935 targets, five seeds, two graphs and two periods.
The existing median-stratum table and all fitted models remain fixed.

For each graph, compute target and non-self-neighbor evidence exactly as in
`neighbor_audit.py`. Interior sextile cutpoints use the eight validation origins
and the matched targets. Collapse duplicate cutpoints; retain every distinct
cutpoint, including zero. Assign values equal to a cutpoint to the lower bin.
The outer intervals include zero and all values above the last cutpoint. Apply
the same validation cutpoints to December and January.

For each cell, report the mean of the five within-seed MAE gains, their sample
standard deviation, scored-event count and represented target count. The figure
uses a shared symmetric zero-centered color scale. Cells with fewer than 30
scored events are marked as limited support; every cell remains in the exported
table. Weighted cell gains must reproduce the corresponding full-cohort gain,
and reconstructed median strata must match the archived assignments exactly.

All statistics run on AutoDL4. Windows performs input packaging, figure
rendering and manuscript compilation. The figure follows the visual style of
the user-supplied `LLM_POI (5).pdf`, Figure 6: separated tiles, signed values,
lettered panels, physical evidence units and a common color scale.
