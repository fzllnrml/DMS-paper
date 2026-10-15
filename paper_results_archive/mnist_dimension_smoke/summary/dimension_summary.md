# Dimension-study numerical summary

Mode candidates; semantic digit classes are external diagnostics.
Primary curve uses seed42 throughout. Independent-model replication is shown separately at fixed anchors.
95% CIs are bootstrap intervals conditional on the fixed representation. No automatic hypothesis conclusion.
RMS minus reference: negative distance difference is favorable. Missing/failed points remain in this table.

| Stratum | Representation | d | Completed / expected | Reconstruction RMS | RMS feature kNN, 95% CI | Flags/status |
|---|---|---:|---:|---:|---|---|
| curve_seed42 | pca | 2 | 1/1 | 0.5092182159423828 | 205.31 [205.31, 205.31] | retained_variance_below_0.90, sanity_outside_fraction_above_0.25, completed |
| curve_seed42 | pca | 4 | 1/1 | 0.49869629740715027 | 288.05 [288.05, 288.05] | retained_variance_below_0.90, sanity_outside_fraction_above_0.25, completed |
| curve_seed42 | pixel | 784 | 1/1 | 0.0 | 29.687 [29.687, 29.687] | training_history_unavailable_in_checkpoint, completed |
| replicated_anchors | pixel | 784 | 1/1 | 0.0 | 29.687 [29.687, 29.687] | training_history_unavailable_in_checkpoint, completed |

Full method/paired metrics, uncertainty, actual steps, compute and failure inventory are in CSV/JSON.
PCA truncation changes retained information. PCA-784 and pixel-CNN-784 are distinct representation/model controls.
The theoretical step ratio is algebraic; it is not itself evidence that support gains are caused by dimension.
