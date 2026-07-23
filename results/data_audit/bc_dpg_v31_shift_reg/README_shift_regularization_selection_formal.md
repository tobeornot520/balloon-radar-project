# BC-DPG-FCN v3.1 shift regularization validation selection

## Protocol

Each fold selects its shift-regularization weight using validation-side metrics only. The selector never ranks candidates using test Pd, Pfa, AUC, false alarms, or test shift statistics.

Selection order:

1. validation Pd floor and score-never-increased constraints;
2. minimum validation false-alarm count at the frozen DPG threshold;
3. maximum validation Pd;
4. maximum validation AUC;
5. minimum validation target shift;
6. stronger regularization on an exact tie.

Candidate training currently creates test files, but these fields are not read by the selection pass. For a future independent blind dataset, the same selected rule should be frozen before opening the blind test set.

## Selected weight by fold

|   fold |   regularization |   val_false_alarms |   val_pd |   val_auc |   val_target_shift | eligible   |
|-------:|-----------------:|-------------------:|---------:|----------:|-------------------:|:-----------|
|      1 |           0      |                  0 | 0.943396 |  1        |        2.17787e-05 | True       |
|      2 |           0.01   |                  0 | 0.923077 |  1        |        0.000135033 | True       |
|      3 |           0      |                  0 | 0.923077 |  0.999872 |        7.52517e-05 | True       |
|      4 |           0.0025 |                  0 | 0.933333 |  1        |        4.22021e-05 | True       |
|      5 |           0.01   |                  0 | 0.9375   |  0.999638 |        0.02814     | True       |
|      6 |           0      |                  0 | 0.962264 |  0.999497 |        4.94536e-05 | True       |

## Selected candidate test report

|   fold |   selected_regularization |   raw_test_false_alarms |   selected_test_false_alarms |   raw_test_pd |   selected_test_pd |   test_target_shift |
|-------:|--------------------------:|------------------------:|-----------------------------:|--------------:|-------------------:|--------------------:|
|      1 |                    0      |                      86 |                           21 |      1        |           1        |         1.94038e-05 |
|      2 |                    0.01   |                       0 |                            0 |      0.924528 |           0.924528 |         0.000197074 |
|      3 |                    0      |                       0 |                            0 |      0.865385 |           0.865385 |         0.00432038  |
|      4 |                    0.0025 |                     100 |                           14 |      0.903846 |           0.903846 |         5.83884e-05 |
|      5 |                    0.01   |                       0 |                            0 |      0.95     |           0.95     |         0.00102294  |
|      6 |                    0      |                       0 |                            0 |      0.791667 |           0.791667 |         0.0269417   |

Raw false alarms (sum): 186
Selected false alarms (sum): 35
Mean raw Pd: 0.905904
Mean selected Pd: 0.905904
Mean selected target shift: 0.005427

## Candidate coverage

Candidate rows found: 30
Folds selected: 6
Missing candidates: 0

## Interpretation boundary

This remains development-stage internal cross-validation. It optimizes a hyperparameter without using candidate test metrics in the ranking, but it does not replace a frozen-model, new-date, new-environment blind test.
