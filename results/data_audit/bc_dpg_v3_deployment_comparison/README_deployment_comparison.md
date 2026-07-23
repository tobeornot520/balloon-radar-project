# BC-DPG-FCN v3 deployment-assumption comparison

This report separates detection performance from deployment assumptions.

## Compared systems

- **raw_dpg**: the frozen DPG detector; one sample can be evaluated independently.
- **sample_independent_bc**: the same calibrator architecture trained with all   12 scan-group features fixed to zero (`no_scan_context`).
- **scan_aware_bc**: the complete v3 calibrator using statistics computed from   the complete scan group.

The third system is currently an offline scan-level calibrator. Its result should not be presented as strictly causal real-time inference because complete-scan statistics can include samples occurring after the sample being classified.

## Six-fold aggregate

| model                 |   false_alarms_sum |   pfa_mean |   pd_mean |   auc_mean |   target_shift_mean | causality                                                  |
|:----------------------|-------------------:|-----------:|----------:|-----------:|--------------------:|:-----------------------------------------------------------|
| raw_dpg               |                186 |  0.206667  |  0.905904 |   0.991155 |          0          | causal/sample-independent                                  |
| sample_independent_bc |                122 |  0.135556  |  0.905904 |   0.991853 |          0.0355492  | causal/sample-independent                                  |
| scan_aware_bc         |                 56 |  0.0622222 |  0.905904 |   0.999309 |          0.00284669 | offline scan-aware; may use future samples within the scan |

## Correct interpretation

The difference between `scan_aware_bc` and `sample_independent_bc` measures the benefit of scan-group context under the current dataset and split. It does not prove that an arbitrary future environment has the same group structure.

For the later self-collected core dataset, the main model should remain sample-independent at test time. A separate causal history module may use only past observations from a continuously operating radar.

## Coverage

Detail rows: 18
Missing summaries: 0
