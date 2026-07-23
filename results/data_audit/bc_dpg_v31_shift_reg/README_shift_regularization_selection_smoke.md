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
|      1 |             0.01 |                  0 | 1        |         1 |         0.00471079 | True       |
|      4 |             0.01 |                  0 | 0.916667 |         1 |         0.0053478  | True       |

## Selected candidate test report

|   fold |   selected_regularization |   raw_test_false_alarms |   selected_test_false_alarms |   raw_test_pd |   selected_test_pd |   test_target_shift |
|-------:|--------------------------:|------------------------:|-----------------------------:|--------------:|-------------------:|--------------------:|
|      1 |                      0.01 |                       7 |                            7 |      1        |           1        |          0.00472023 |
|      4 |                      0.01 |                      11 |                           11 |      0.833333 |           0.833333 |          0.00539122 |

Raw false alarms (sum): 18
Selected false alarms (sum): 18
Mean raw Pd: 0.916667
Mean selected Pd: 0.916667
Mean selected target shift: 0.005056

## Candidate coverage

Candidate rows found: 10
Folds selected: 2
Missing candidates: 0

## Interpretation boundary

This remains development-stage internal cross-validation. It optimizes a hyperparameter without using candidate test metrics in the ranking, but it does not replace a frozen-model, new-date, new-environment blind test.
