# Competition Semantics Audit

## Row-Level Keys

| Split | Rows | Distinct ids | Distinct row keys | Duplicate ids | Duplicate row keys | ID mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 5337414 | 5337414 | 5337414 | 0 | 0 | 0 |
| test | 1447107 | 1447107 | 1447107 | 0 | 0 | 0 |

## Train/Test Overlap

| Level | Columns | Train unique keys | Test unique keys | Overlapping test keys | Unseen test keys | Unseen test rows | Unseen test row rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code | code | 23 | 23 | 23 | 0 | 0 | 0.000000 |
| code_sub_category_horizon | code, sub_category, horizon | 460 | 460 | 460 | 0 | 0 | 0.000000 |
| code_sub_code | code, sub_code | 1856 | 557 | 97 | 460 | 1289694 | 0.891222 |
| triplet | code, sub_code, sub_category | 9270 | 2784 | 485 | 2299 | 1289694 | 0.891222 |
| full_group | code, sub_code, sub_category, horizon | 36923 | 11039 | 1875 | 9164 | 1289694 | 0.891222 |

## Recommendations

- The prediction target is row-level: every train and test row has a unique id and a unique (code, sub_code, sub_category, horizon, ts_index) key.
- Test contains cold-start (code, sub_code, sub_category) groups, so forward holdout alone is not enough; add a cold-start-aware validation slice before trusting feature or group-history gains.
- Do not rely on memorizing full (code, sub_code, sub_category, horizon) histories: many test rows live on groups unseen in train.
- A coarse fallback keyed by (code, sub_category, horizon) has full test coverage and is a defensible leakage-safe baseline level.
