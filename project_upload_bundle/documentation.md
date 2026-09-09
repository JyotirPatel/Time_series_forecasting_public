# Working Log

This file is the shared memory and audit log for the long-horizon run.

## Snapshot

- Prepared on: `2026-04-09 UTC`
- Repo state at latest update: docs, local parquet data, runnable Python scaffold, deterministic forward CV tooling, canonical exact aggregate scoring, extended diagnostics, saved residual-model artifacts on top of the smoothed stable-group baseline, a strict-OOF normalized residual branch with saved OOF artifacts and posthoc sweep outputs, a rebuilt fixed-prefix residual correction path, and reusable warm-start / recent-delta overlay predictors that now run end to end for evaluation and test inference
- Current status: bootstrap scaffold completed, competition semantics audited, cold-start-aware validation implemented, canonical offline scoring fixed, the direct warm-start sanity check still rejects naive fine-group replacement, and the best implemented path is now a stacked train-only overlay on top of `smoothed_stable_horizon_blend` that combines exact warm last-target rules, a recent-vs-full history delta on stable groups for horizons `10` and `25`, and a narrow `K7Y1TTAH / horizon=25` warm overlay
- Current best exact offline score: `0.1450306103` from `smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1`
- Recommended current milestone: `Milestone 5: Failure Analysis And Self-Improvement`
- Long-run autonomy target: keep iterating unattended toward a score path competitive with public target `0.8718`

## Environment

- Default Python environment: repo-local `.venv`
- Dependency source of truth: `requirements.txt`
- Hardware:
  - GPU: `NVIDIA GeForce RTX 4070 Ti` with `12 GB` VRAM
  - RAM: `32 GB`
  - CPU: `Intel Core i5-13600K`
- Bootstrap commands:
  - `python -m venv .venv`
  - `. .venv/bin/activate`
  - `python -m pip install -r requirements.txt`
- Current installed baseline stack in `.venv`:
  - `duckdb==1.5.1`
  - `numpy==2.4.4`
  - `pandas==3.0.2`
  - `pyarrow==23.0.1`
  - `pytest==9.0.3`
  - `scikit-learn==1.8.0`

## Competition Facts Captured At Setup

- Competition: `Hedge fund - Time series forecasting`
- Slug: `ts-forecasting`
- Public description: `Time series forecasting by code, sub-code, sub_category and horizon`
- Metric label: `Skill Score`
- Deadline: `2026-04-07 07:00 UTC`
- Private leaderboard release: `2026-04-14 07:00 UTC`
- Public leaderboard share: `24%`
- Max daily submissions: `1`
- Row id column: `id`
- Total scored rows: `1,447,107`

## Important Assumptions

- The competition deadline appears to have passed before this repo was prepared.
- The next session should verify competition accessibility, rules, and evaluation details before making metric-sensitive decisions.
- The train/test split appears to be a forward time split based on `ts_index` continuity; this is an inference from the local data and should be confirmed in the competition rules.
- User-supplied competition text provides the current working definition of the metric and sequential rules.
- The prediction unit is row-level because both splits have one unique `id` and one unique `(code, sub_code, sub_category, horizon, ts_index)` key per row.

## Local Dataset Inventory

- Files present:
  - `data/train.parquet`
  - `data/test.parquet`
- Shapes:
  - train: `5,337,414 x 94`
  - test: `1,447,107 x 92`
- Shared columns:
  - identifiers/grouping: `id`, `code`, `sub_code`, `sub_category`, `horizon`, `ts_index`
  - features: `86` columns from `feature_a` through `feature_ch`
- Train-only columns:
  - `y_target`
  - `weight`
- Column semantics from the user-provided rules:
  - `id` is the concatenation of `code`, `sub_code`, `sub_category`, `horizon`, and `ts_index` joined by `__`
  - `horizon` is a categorical forecast group and should not be interpreted as the numeric offset from `ts_index`
  - `weight` is used in the loss and must not be used as a feature
- Distinct values:
  - `code`: `23` in train and test
  - `sub_code`: `180` in train, `47` in test
  - `sub_category`: `5` in train and test
  - `(code, sub_code, sub_category)` groups: `9,270` in train, `2,784` in test
  - `horizon`: `[1, 3, 10, 25]`
- Time ranges:
  - train `ts_index`: `1..3601`
  - test `ts_index`: `3602..4376`
- Key integrity and overlap audit:
  - both train and test have zero duplicate `id` values,
  - both train and test have zero duplicate `(code, sub_code, sub_category, horizon, ts_index)` row keys,
  - both train and test have zero `id` reconstruction mismatches against `code__sub_code__sub_category__horizon__ts_index`,
  - test rows on unseen `(code, sub_code)` pairs: `1,289,694 / 1,447,107` (`89.12%`),
  - test rows on unseen `(code, sub_code, sub_category)` triplets: `1,289,694 / 1,447,107` (`89.12%`),
  - test rows on unseen `(code, sub_code, sub_category, horizon)` groups: `1,289,694 / 1,447,107` (`89.12%`),
  - test rows on unseen `(code, sub_category, horizon)` groups: `0 / 1,447,107` (`0%`).
- Row counts by horizon:
  - train:
    - `1`: `1,394,653`
    - `3`: `1,385,816`
    - `10`: `1,337,236`
    - `25`: `1,219,709`
  - test:
    - `1`: `379,617`
    - `3`: `376,558`
    - `10`: `362,057`
    - `25`: `328,875`
- Missingness:
  - several engineered features contain nulls in both splits,
  - notably sparse columns include `feature_at`, `feature_w`, `feature_x`, `feature_y`, `feature_z`, `feature_by`, `feature_cd`, `feature_ce`, and `feature_cf`,
  - null handling must be part of baseline model design.

## Metric And Rule Notes

- Working competition metric:

```python
def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def weighted_rmse_score(y_target, y_pred, w) -> float:
    denom = np.sum(w * y_target ** 2)
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(val))
```

- Public leaderboard uses about `25%` of the test set; private uses the remaining `75%`.
- Canonical offline proxy score in this repo is now the exact metric on the concatenated validation rows across folds, not a row-weighted average of per-fold scores.
- Sequential rule:
  - prediction at time `t` may only use data available up to `t`,
  - preprocessing on test must also be sequential,
  - any use of future test information is a rules violation.
- Host clarification:
  - top public scores may be leakage-driven,
  - final winning code must be reproducible,
  - public ranking is not a reliable proxy for private ranking.
- Target-derived feature clarification:
  - using target and weights on the training set is allowed,
  - but it can inflate CV and public scores,
  - the host said some hidden test points are ignored to reduce this inflation,
  - exact ignored points are undisclosed.
- Leakage hotspot:
  - community discussion specifically flagged `feature_al` as potentially target-reconstructive,
  - any reliance on it should be accompanied by explicit leakage checks.

## Semantics Audit Takeaways

- Submission and scoring are row-level, not an aggregated per-group task.
- The current `id` format is validated directly from local data and can be treated as a strict join key.
- The test split has heavy cold-start behavior below the `(code, sub_category, horizon)` level.
- Any validation scheme that only holds out the last train window will overestimate approaches that depend on `sub_code` or finer group history.
- A defensible early baseline can still use `(code, sub_category, horizon)` because that level has full test coverage without looking ahead.

## Milestone Board

- `Milestone 0`: completed
- `Milestone 1`: completed
- `Milestone 2`: completed
- `Milestone 3`: completed
- `Milestone 4`: completed
- `Milestone 5`: pending
- `Milestone 6`: pending

## Scaffold Added

- Package modules:
  - `src/ts_forecasting/constants.py`
  - `src/ts_forecasting/metrics.py`
  - `src/ts_forecasting/io.py`
  - `src/ts_forecasting/sequential.py`
  - `src/ts_forecasting/baselines.py`
  - `src/ts_forecasting/residuals.py`
  - `src/ts_forecasting/semantics.py`
  - `src/ts_forecasting/validation.py`
- CLI scripts:
  - `scripts/profile_data.py`
  - `scripts/evaluate_baselines.py`
  - `scripts/predict_baseline.py`
  - `scripts/audit_semantics.py`
  - `scripts/analyze_validation_errors.py`
  - `scripts/evaluate_residual_models.py`
- Tests:
  - `tests/test_metrics.py`
  - `tests/test_baselines.py`
  - `tests/test_sequential.py`
  - `tests/test_semantics.py`
  - `tests/test_validation.py`
  - `tests/test_residuals.py`

Current code capabilities:

- exact weighted competition metric helper,
- exact numerator / denominator metric breakdown helper for canonical fold aggregation,
- DuckDB-backed parquet loading helpers,
- sequential inference pipeline that predicts in increasing `ts_index` order,
- frozen train-stat imputer for future feature models,
- leakage-safe weighted mean baselines with fallback hierarchy and hierarchical shrinkage,
- data profiling CLI,
- competition semantics audit CLI,
- deterministic expanding-window fold generation,
- cold-start annotation and coverage summaries aligned to the observed test structure,
- forward-holdout and multi-fold baseline evaluation CLI,
- validation error analysis CLI over saved fold predictions,
- residual-slice helper utilities, prefix-only residual scale helpers, and a residual model evaluation CLI that can emit strict-OOF normalized target artifacts,
- fixed-prefix pseudo-history residual correction predictor for gated weak slices,
- generic fixed-weight linear blending for leakage-safe baseline predictors,
- selectable baseline/model subsets inside the forward-CV evaluator so heavier branches do not slow the default suite,
- baseline submission-generation CLI for grouped, residual-corrected, and blended predictors.
- reusable exact-group warm-start last-target overlays with optional code filters,
- reusable recent-vs-full-history mean-delta corrections on stable groups.

## Decisions

- Use a durable markdown stack modeled on the OpenAI long-horizon Codex workflow.
- Add `AGENTS.md` so the next Codex session is routed into the task automatically.
- Frame the task as offline improvement and reproducibility work by default because the public deadline has already passed.
- Treat the local parquet files as the initial data source of truth.
- Assume a forward time split because `ts_index` is contiguous from train into test, but keep that assumption explicit until validated against the competition rules.
- Standardize on the repo-local `.venv` for future Codex sessions.
- Use `requirements.txt` as the default dependency install source.
- Treat the user-supplied metric and rule summary as the working spec for evaluation and leakage control.
- Ban `weight` from the feature matrix by default.
- Treat public leaderboard chasing as lower priority than robust sequential offline validation.
- Start with a simple, causal scaffold before adding heavier feature-model code.
- Use DuckDB for dataset-scale reads and lightweight aggregation.
- Keep the first baseline family deliberately simple and leakage-resistant, even if the score is weak.
- Treat row-level prediction and cold-start group coverage as resolved competition semantics, not open assumptions.
- Treat `(code, sub_category, horizon)` as the coarsest stable group level currently known to cover the full test split.
- Defer feature-heavy modeling until validation reflects both forward time order and cold-start exposure.
- Record the available hardware so future sessions can consider GPU-capable models without guessing.
- A completed milestone is not, by itself, a stopping condition.
- Future sessions should keep iterating autonomously until the score path is competitive, blocked, or clearly plateaued.
- Use four non-overlapping `360`-step expanding-window folds as the current default offline proxy because they reproduce `75%` to `78%` unseen `(code, sub_code)` validation rows while preserving full `(code, sub_category, horizon)` coverage.
- Treat `(code, sub_category, horizon)` as the safest currently validated model level for target aggregation because it matches full test coverage and avoids relying on unseen `sub_code` histories.
- Prefer all-history stable-group statistics over short recent windows for the current grouped baselines; the recent-window ablation regressed.
- Use smoothed group means rather than raw fine-group means when warm-start performance collapses, because shrinkage improves both warm and cold slices without using future data.
- Use failure analysis on saved fold predictions to guide the next model branch; the current best baseline is weakest on horizons `1` and `3`, and on sub-category slices led by `DPPUO5X2`.
- Treat the exact concatenated-fold metric as the only canonical selector for offline model ranking; fold-average scores are now diagnostic only.
- Build residual models on top of `smoothed_stable_group_mean_pw100`, not as raw feature-only models, because the grouped baseline is the only branch with consistent signal so far.
- Keep `weight` out of the feature matrix and keep `feature_al` excluded from the initial residual branch until explicit leakage checks are added.
- Use small, shrunk residual corrections first; broad unshrunk corrections are not justified while folds `02` and `03` still clip.
- Keep the raw tri-slice residual branch as the best non-baseline residual branch for now; the strict-OOF normalized replacement branch is cleaner but currently tops out at `0.1123268577`, below `0.1128626342`.
- Treat the new research notes as directional guidance, not as evidence; every proposed idea still needs to clear the exact four-fold forward proxy before code is expanded around it.
- Do not replace the stable grouped baseline with a direct smoothed full-group warm-start mean: the naive `(code, sub_code, sub_category, horizon)` translation clipped to `0.0` overall and `0.0` on the warm-start slice under the canonical proxy.
- Treat convex blending of complementary train-only baselines as a first-class modeling path here; `horizon_mean` is weak on its own but materially improves the clipped metric when blended with the smoothed stable-group baseline.
- Promote `smoothed_stable_horizon_blend` to the best implemented offline baseline because it improved the exact four-fold score from `0.1119478178` to `0.1370805421`, raised cold-start score to `0.1562849407`, and lifted warm-start score above zero to `0.0712384530`.
- Keep `smoothed_stable_horizon_blend` as the incumbent even after rebuilding the exploratory raw tri-slice branch, because the clean implementation of `ridge_triweak_h13_tw720_s022_q90 + horizon_mean` scored `0.1370652148`, slightly below the incumbent `0.1370805421`.
- Treat the saved posthoc `raw_best + horizon_mean` edge as unresolved until the discrepancy between the old artifact materialization (`0.1375279103`) and the rebuilt clean path (`0.1370652148`) is explained.
- Deprioritize `feature_al` and other tiny feature-only branches as primary modeling paths; honest four-fold forward checks showed no useful standalone signal and no tiny safe feature set that beat the zero benchmark.
- Treat horizons `10` and `25` as the current highest-leverage drift slices; a train-only recent-vs-full-history delta at `(code, sub_category, horizon)` materially improved both cold and warm aggregate scores on top of the warm last-target overlay.
- Promote `smoothed_stable_horizon_blend_last_target_rules` over the raw residual family because a tightly gated exact-group warm overlay lifted the exact four-fold score to `0.1386781180` without reopening feature-leakage risk.
- Promote `smoothed_stable_horizon_blend_last_target_recent_delta_v1` over the plain warm overlay because it improved the exact four-fold score again to `0.1439924990`, with aggregate cold-start score `0.1601147203` and aggregate warm-start score `0.0937945231`.
- Promote `smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1` to the current best implemented path because a narrow `K7Y1TTAH / horizon=25` warm overlay further improved the exact four-fold score to `0.1450306103` and raised aggregate warm-start score to `0.0991358430`.

## Open Questions

- Is any authenticated Kaggle access available in the next session?
- Does the user still want submission packaging despite the deadline status?
- How should target-derived features, if any, be cross-fitted or constrained so they remain defensible under the host’s leakage warnings?
- Which additional code-specific warm overlays, likely around `OSJL3A7Y` and `X9BZ68VQ` on horizons `25` and `10`, can add to `smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1` without reopening fold `02` damage?
- Can the recent-delta overlay be made code-specific or clipped adaptively so fold `03` warm-start score stops clipping to `0.0` while preserving the fold `02` gain?
- Can a second-stage OOF-normalized correction on top of the current best raw tri-slice branch beat `0.1128626342`, or is a different representation needed?
- Why does the rebuilt clean `ridge_triweak_h13_tw720_s022_q90` path underperform the saved raw artifact materialization by about `0.00078` absolute score, despite matching the same broad config and slice definition?

## Blockers

- No authenticated Kaggle session is available in this setup step.
- The exact subset ignored by the host’s rescoring logic is hidden, so offline validation can only approximate final scoring.
- The best current offline model still clips fold `03` warm-start score to `0.0`, so robustness is not yet adequate.
- A sampled feature-only `HistGradientBoostingRegressor` probe did not beat the zero benchmark, so feature modeling needs better failure analysis before more code is committed.
- The new best implemented blend improves the exact concatenated-fold score materially, but there is still a large gap to the desired score path.
- The strict-OOF normalized replacement branch plateaued at `0.1123268577` after shrink sweeps and narrow-slice probes, so it is not yet a justified replacement for the incumbent raw tri-slice branch.
- The rebuilt fixed-prefix raw tri-slice branch is now reproducible and testable, but its clean implementation did not match the older saved artifact score, so the tiny apparent upside of that stack is not yet trustworthy.

## Suggested First Actions For The Next Session

1. Expand exact warm overlays on top of `smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1`, starting with `OSJL3A7Y` and `X9BZ68VQ` on horizons `25` and `10`, because the narrow `K7Y1TTAH` patch already added a real gain.
2. Test clipped or code-specific recent-delta variants on top of the current incumbent to target the remaining fold `03` warm-start collapse without giving back fold `02`.
3. Refresh the default `outputs/baselines` forward-CV artifact if the canonical one-directory summary needs to reflect the new incumbent, because current best summaries now live under `outputs/advanced_models` and `outputs/stacked_models`.
4. Keep the raw residual reconstruction branch lower priority unless a clear path appears to beat the new warm/delta overlay stack by more than a tiny posthoc margin.
5. Keep `feature_al` off the mainline path unless a future branch shows honest forward-CV value under strict leakage-safe checks.
6. Confirm the competition state and whether authenticated Kaggle access is available.

## Autonomous Stop Rule

The session should not stop merely because a milestone or single experiment completed.

Preferred stopping conditions:

- verified public score is close to or above `0.8718`,
- strongest offline validation has plateaued after several materially different experiment branches,
- a hard external blocker requires user intervention,
- leakage risk cannot be resolved safely.

## Experiment Log

Use this format for every experiment:

| ID | Milestone | Hypothesis | Command / Config | Result | Decision |
| --- | --- | --- | --- | --- | --- |
| E000 | Setup | Created durable markdown task stack for future session handoff | `n/a` | Ready for execution | Start with repo bootstrap |
| E001 | Setup | Local data can replace the initial missing-data assumption | repo-local DuckDB profile over `data/train.parquet` and `data/test.parquet` | Confirmed train/test schema, forward `ts_index` split, train-only `y_target` and `weight` | Update docs and start with local profiling code in the next run |
| E002 | Setup | Standardizing the Python environment now will reduce session drift later | `. .venv/bin/activate && python -m pip install -r requirements.txt` | Installed baseline data and test stack into repo-local venv | Future sessions should activate `.venv` first |
| E003 | Setup | User-provided rules materially narrow the competition ambiguity | Incorporated dataset description, exact metric, sequential rule, and leakage discussion into long-run docs | Reduced ambiguity around scoring and allowed/forbidden processing | Use these notes as the working competition spec |
| E004 | Milestone 0 | A small reusable code scaffold will reduce drift in future sessions | Added `src/`, `scripts/`, and `tests/` with metric, IO, causal inference, and baseline helpers | Runnable project surface now exists | Treat the bootstrap phase as complete |
| E005 | Milestone 0 | The scaffold should validate cleanly before more modeling work | `. .venv/bin/activate && pytest -q -s tests` | `5 passed in 0.29s` | Keep tests green as the codebase grows |
| E006 | Milestone 1 | A dedicated profiling CLI should reproduce the known dataset facts from local parquet files | `. .venv/bin/activate && python scripts/profile_data.py --output-dir outputs/profiles` | Wrote profile artifacts successfully | Use these outputs as local schema references |
| E007 | Milestone 2 | Very simple causal baselines can establish a first honest lower bound | `. .venv/bin/activate && python scripts/evaluate_baselines.py --train-path data/train.parquet --holdout-steps 30 --output-dir outputs/baselines` | `horizon_mean=0.0283453`, `hierarchical_group_mean=0.0` on holdout `ts_index > 3571` | The horizon-only baseline is weak but stable; naive grouped memorization is not robust |
| E008 | Milestone 2 | The baseline submission path should run end to end on the real test file | `. .venv/bin/activate && python scripts/predict_baseline.py --train-path data/train.parquet --test-path data/test.parquet --baseline horizon_mean --output outputs/submissions/horizon_mean_submission.csv` | Wrote submission CSV successfully | Submission generation path works for future experiments |
| E009 | Milestone 1 | A dedicated semantics audit should resolve the prediction unit and quantify train/test overlap before more modeling work | `. .venv/bin/activate && python scripts/audit_semantics.py --train-path data/train.parquet --test-path data/test.parquet --output-dir outputs/semantics` | Confirmed row-level prediction keys; `89.12%` of test rows are on unseen `(code, sub_code)` pairs and also unseen triplet/full-group keys, while `(code, sub_category, horizon)` still covers the entire test set | Mark Milestone 1 complete and prioritize a cold-start-aware validation design |
| E010 | Milestone 1 | The semantics audit and existing scaffold should remain validated after adding the new audit code | `. .venv/bin/activate && pytest -q -s tests/test_semantics.py tests/test_metrics.py tests/test_baselines.py tests/test_sequential.py` | `6 passed in 1.46s` | Keep the semantics audit in the default smoke suite |
| E011 | Setup | Future sessions should know the available hardware up front | Recorded GPU, RAM, and CPU in the durable task files | Reduced ambiguity around when GPU-capable methods are worth trying | Consider GPU use in later model milestones, not by default in every step |
| E012 | Milestone 3 | A deterministic multi-fold validation scheme should match the real test cold-start structure better than a single tail holdout | Added `src/ts_forecasting/validation.py`, extended `scripts/evaluate_baselines.py`, and added `tests/test_validation.py`; `. .venv/bin/activate && pytest -q -s tests` | `10 passed in 1.10s` after the new validation helpers were wired in | Mark Milestone 3 implementation complete and use the new fold loop as the default offline proxy |
| E013 | Milestone 3 | Four non-overlapping `360`-step expanding folds can approximate the test cold-start rate while preserving stable-group coverage | `. .venv/bin/activate && python scripts/evaluate_baselines.py --train-path data/train.parquet --test-path data/test.parquet --holdout-steps 360 --n-folds 4 --step-size 360 --min-train-steps 2000 --cold-start-level code_sub_code --output-dir outputs/baselines` | Validation folds showed `75.04%` to `77.98%` unseen `(code, sub_code)` rows vs real-test `89.12%`, while `(code, sub_category, horizon)` stayed fully covered in every fold | Treat this fold scheme as the current best offline proxy and evaluate models against it |
| E014 | Milestone 4 | A sampled feature-only tree model might beat the weak grouped baselines on the newest long fold | Ad hoc `.venv` `HistGradientBoostingRegressor` probe on fold `3241 -> 3601` using `200k` sampled recent rows, `(code, sub_category, horizon)`, and a 26-feature subset | Scored `0.0` with ratio `1.0014`, so it failed to beat the zero benchmark | Do not formalize raw feature-only modeling until failure analysis explains why it is underperforming |
| E015 | Milestone 4 | A baseline restricted to the stable coverage level `(code, sub_category, horizon)` should outperform horizon-only means under cold-start-aware CV | Re-ran `scripts/evaluate_baselines.py` with a new `stable_group_mean` config on the four-fold scheme | `stable_group_mean` reached row-weighted score `0.07760` vs `0.00569` for `horizon_mean`; the full hierarchical baseline stayed at `0.0` overall | Promote stable-group aggregation as the new causal reference model and demote full-group memorization |
| E016 | Milestone 4 | Recent-window grouped averages might handle drift better than all-history grouped averages | Ad hoc window ablation over stable-group means with windows `720`, `1440`, and full history across the four long folds | Mean scores were `0.02391` for `720`, `0.06590` for `1440`, and `0.07780` for full history | Keep all-history grouped statistics for now; shorter windows are not the fix |
| E017 | Milestone 4 | Shrinking stable-group means toward coarser priors should reduce overfitting on noisy groups | Added `SmoothedWeightedGroupMeanRegressor` plus tests, then re-ran `. .venv/bin/activate && python scripts/evaluate_baselines.py --train-path data/train.parquet --test-path data/test.parquet --holdout-steps 360 --n-folds 4 --step-size 360 --min-train-steps 2000 --cold-start-level code_sub_code --output-dir outputs/baselines` | `smoothed_stable_group_mean_pw100` became best known with row-weighted score `0.09817`, mean cold-start score `0.12998`, and mean warm-start score `0.06733` | Mark Milestone 4 complete at the grouped-model level and shift the next loop to failure analysis plus residual/hybrid improvements |
| E018 | Milestone 6 | The current best-known grouped baseline should be runnable end to end on the real test file | `. .venv/bin/activate && python scripts/predict_baseline.py --train-path data/train.parquet --test-path data/test.parquet --baseline smoothed_stable_group_mean_pw100 --output outputs/submissions/smoothed_stable_group_mean_pw100_submission.csv` | Wrote the full submission-format CSV successfully | Use this artifact as the current reproducible inference path while better models are developed |
| E019 | Milestone 5 | Saved fold predictions should reveal which slices still cause the strongest baseline to fail | `. .venv/bin/activate && python scripts/analyze_validation_errors.py --summary-path outputs/baselines/forward_cv_h360_f4_s360_code_sub_code/summary.json --train-path data/train.parquet --baseline smoothed_stable_group_mean_pw100 --output-dir outputs/analysis --top-k 3` | Aggregate score on concatenated fold rows was `0.11195`; horizons `1` and `3` were the weakest, and sub-category `DPPUO5X2` was the worst persistent slice, with folds `02` and `03` also collapsing on `PZ9S1Z4V` and `PHHHVYZI` across multiple horizons | Use the next model loop to attack these slices specifically instead of broad untargeted feature experiments |
| E020 | Milestone 5 | The offline loop must rank models by the same exact aggregate metric used in downstream diagnostics | Updated `src/ts_forecasting/metrics.py`, `scripts/evaluate_baselines.py`, and tests, then re-ran `. .venv/bin/activate && python scripts/evaluate_baselines.py --train-path data/train.parquet --test-path data/test.parquet --holdout-steps 360 --n-folds 4 --step-size 360 --min-train-steps 2000 --cold-start-level code_sub_code --output-dir outputs/baselines` | `summary.json` now uses canonical `aggregate_overall_score`, and the best baseline is `smoothed_stable_group_mean_pw100` at exact score `0.1119478178` rather than the stale row-weighted `0.09817` selector | Treat the exact concatenated-fold score as the only canonical offline objective from here forward |
| E021 | Milestone 5 | Stronger diagnostics should explain whether zero-scored folds are true failures or denominator pathologies | Extended `scripts/analyze_validation_errors.py` and re-ran `. .venv/bin/activate && python scripts/analyze_validation_errors.py --summary-path outputs/baselines/forward_cv_h360_f4_s360_code_sub_code/summary.json --train-path data/train.parquet --baseline smoothed_stable_group_mean_pw100 --output-dir outputs/analysis --top-k 10` | Added `code / sub_category / horizon` slices, fold `02` vs `03` comparisons, warm vs cold splits inside each slice, and per-slice energy denominators; aggregate weak slices remain concentrated in short horizons and in `DPPUO5X2`, `PZ9S1Z4V`, and `PHHHVYZI` | Use these tables to target the next residual experiments and to distinguish real numerator failures from small-denominator noise |
| E022 | Milestone 5 | A residual model on top of the smoothed stable-group baseline should beat the base model before any leakage-risky feature expansion | Added `src/ts_forecasting/residuals.py`, `tests/test_residuals.py`, and `scripts/evaluate_residual_models.py`; ran `. .venv/bin/activate && python scripts/evaluate_residual_models.py --base-summary-path outputs/baselines/forward_cv_h360_f4_s360_code_sub_code/summary.json --train-path data/train.parquet --base-baseline smoothed_stable_group_mean_pw100 --output-dir outputs/residual_models` | DPP-only residuals gave small gains: best exact score was `0.1122213767` from `ridge_dppuo5x2_h13_tw720_s020_q90`; `Huber` improved only slightly and shallow `HistGradientBoostingRegressor` regressed | Keep the residual-on-baseline direction, but broaden the weak slice set before considering heavier models |
| E023 | Milestone 5 | Adding `PZ9S1Z4V` and `PHHHVYZI` to the short-horizon residual slice should outperform the DPP-only residual branch | Expanded `scripts/evaluate_residual_models.py` with tri-slice Ridge/Huber configs over horizons `1` and `3`, excluding `feature_al` and shrinking corrections toward zero | `ridge_triweak_h13_tw720_s010_q90` improved the exact score to `0.1125969381`, with numerator reductions in folds `01`, `02`, `03`, and `04` even though folds `02` and `03` still clipped to `0.0` | Promote the tri-slice Ridge residual family as the new best branch and tune shrinkage before widening feature usage |
| E024 | Milestone 5 | The best tri-slice residual family may need a larger global shrinkage multiplier than the first conservative setting | Posthoc exact rescaling of saved `ridge_triweak_h13_tw720_s010_q90` corrections against the base predictions, then materialized `ridge_triweak_h13_tw720_s022_q90` predictions and summary rows under `outputs/residual_models` | Best exact offline score improved again to `0.1128626342`, a gain of `+0.0009148164` over the canonical smoothed baseline, with the strongest numerator reduction in fold `02` and smaller positive gains in folds `01` and `04` | Current best branch is now the shrunk tri-slice Ridge residual; next work should focus on better residual targets and leakage-safe feature expansion rather than more blind scale sweeps |
| E025 | Milestone 5 | A strict-OOF residual target with prefix-only group normalization may be cleaner and stronger than the current fixed-prefix pseudo-history | Extended `src/ts_forecasting/residuals.py`, rewrote `scripts/evaluate_residual_models.py`, and added `tests/test_residuals.py` coverage for scale lookup and normalized correction handling; validated with `. .venv/bin/activate && PYTHONPATH=src pytest -q -s tests/test_residuals.py tests/test_metrics.py tests/test_validation.py` and `. .venv/bin/activate && PYTHONPATH=src python -m py_compile scripts/evaluate_residual_models.py` | The helper layer passed cleanly and the residual CLI now writes `oof_base_predictions.parquet` and `oof_residual_pool.parquet` under `outputs/residual_models/forward_cv_h360_f4_s360_code_sub_code_oof_norm` | Use the new OOF artifacts as the clean target-construction path for all follow-up probes on this branch |
| E026 | Milestone 5 | Horizon-specific OOF-normalized tri-slice residual models may improve folds `02` and `03` more safely than the incumbent raw branch | `. .venv/bin/activate && PYTHONPATH=src python scripts/evaluate_residual_models.py --base-summary-path outputs/baselines/forward_cv_h360_f4_s360_code_sub_code/summary.json --train-path data/train.parquet --base-baseline smoothed_stable_group_mean_pw100 --output-dir outputs/residual_models` | Untuned strict-OOF normalized configs regressed sharply; the best direct run was only `0.1101621570` from `ridge_oofnorm_triweak_h1_tw720_s015_q85`, while folds `02` and `03` stayed clipped and fold `04` degraded materially | Do not trust unshrunk OOF-normalized corrections; inspect shrink sensitivity before promoting or discarding the branch |
| E027 | Milestone 5 | The OOF-normalized branch may still recover if its corrections are shrunk far more aggressively than the first direct run used | One-off posthoc exact rescale sweep over the saved `outputs/residual_models/forward_cv_h360_f4_s360_code_sub_code_oof_norm` fold predictions, then materialized `ridge_oofnorm_triweak_h13_tw720_s003_q85_posthoc` plus `posthoc_rescale_sweep.json` under the same directory | The best tuned OOF-normalized score was `0.1123268577` from `ridge_oofnorm_triweak_h13_tw720_s015_q85` rescaled to `0.03`; fold `02` error sum fell to `47,497,660.60` and fold `03` to `48,600,062.22`, but the branch still trailed the incumbent `0.1128626342` | Mark the strict-OOF normalized replacement path as cleaner but currently weaker than the incumbent raw tri-slice branch |
| E028 | Milestone 5 | If OOF normalization is sound but the broad tri-slice fit is too diffuse, scale-aware weighting or a narrower `DPPUO5X2` slice might recover the gap | Ad hoc probes on the saved OOF parquet artifacts: one used training weights proportional to `weight * residual_scale^2`, and another trained DPP-only short-horizon Ridge variants with posthoc shrink sweeps | Neither rescue path beat the tuned OOF-normalized tri-slice result; the scale-aware weighting probe peaked at `0.1120444512`, and the best DPP-only probe stayed at `0.1120033400` | Treat this replacement branch as plateaued for now and move the next effort to a stack or materially different representation instead of more micro-tuning |
| E029 | Milestone 5 | The newly added research may justify a simple warm-start baseline branch if a direct smoothed fine-group translation survives the canonical forward proxy | Ad hoc `.venv` probe using `SmoothedWeightedGroupMeanRegressor` with fallback levels `[(code, sub_code, sub_category, horizon), (code, sub_category, horizon), (sub_category, horizon), (horizon)]`, `prior_weight=100.0`, and the exact four-fold forward proxy with `code_sub_code` cold-start annotation | The direct warm-start translation failed completely: every fold scored `0.0`, aggregate overall score was `0.0`, aggregate cold-start score stayed `0.1451415055`, and aggregate warm-start score was `0.0` | Reject naive fine-group replacement means; only pursue research-inspired warm-start ideas as additive, heavily gated overlays or as stacked experts on top of the stable parent baseline |
| E030 | Milestone 5 | The saved fold artifacts may already contain a stronger leakage-safe stack than the incumbent raw tri-slice residual branch | Exact quadratic posthoc search over saved OOF prediction artifacts from `horizon_mean`, `smoothed_stable_group_mean_pw100`, `ridge_triweak_h13_tw720_s022_q90`, and the best OOF-normalized branch; plus forward-calibrated scalar sanity checks on older folds only | The strongest implemented-safe candidate was the baseline-only `smoothed_stable_group_mean_pw100 + horizon_mean` blend at exact score `0.1370805421`; an exploratory `raw_best + horizon_mean` pair reached `0.1375279103` posthoc and `0.1209444254` under forward-calibrated scalar updates | Formalize the baseline-only blend first because it has a clean test-time inference path, then revisit the residual-plus-horizon blend as a follow-up |
| E031 | Milestone 5 | A fixed linear blend of `smoothed_stable_group_mean_pw100` and `horizon_mean` can turn the weak horizon baseline into a useful risk-reducing correction without adding leakage-sensitive features | Added `LinearBlendRegressor`, exposed `smoothed_stable_horizon_blend` in the baseline CLIs with weights `[0.6355710510, 0.3644289490]`, then ran `. .venv/bin/activate && PYTHONPATH=src pytest -q -s tests`, `. .venv/bin/activate && PYTHONPATH=src python scripts/evaluate_baselines.py --train-path data/train.parquet --test-path data/test.parquet --holdout-steps 360 --n-folds 4 --step-size 360 --min-train-steps 2000 --cold-start-level code_sub_code --output-dir outputs/baselines`, and `. .venv/bin/activate && PYTHONPATH=src python scripts/predict_baseline.py --train-path data/train.parquet --test-path data/test.parquet --baseline smoothed_stable_horizon_blend --output outputs/submissions/smoothed_stable_horizon_blend_submission.csv` | The new blend became the best implemented offline path with exact score `0.1370805421`, cold-start score `0.1562849407`, warm-start score `0.0712384530`, and a full submission-format test prediction file | Promote `smoothed_stable_horizon_blend` to the current best-known implemented pipeline and use the new submission artifact as the default reproducible inference path |
| E032 | Milestone 5 | Rebuilding the raw tri-slice residual branch as a reusable fixed-prefix predictor might reproduce the saved artifact edge and make the `raw + horizon` stack promotion-worthy | Added `FixedPrefixResidualCorrectionRegressor`, exposed `ridge_triweak_h13_tw720_s022_q90` plus `ridge_triweak_h13_tw720_s022_q90_horizon_blend` in the baseline CLIs, validated with `. .venv/bin/activate && PYTHONPATH=src pytest -q -s tests`, ran `. .venv/bin/activate && PYTHONPATH=src python scripts/evaluate_baselines.py --train-path data/train.parquet --test-path data/test.parquet --holdout-steps 360 --n-folds 4 --step-size 360 --min-train-steps 2000 --cold-start-level code_sub_code --baselines smoothed_stable_horizon_blend ridge_triweak_h13_tw720_s022_q90 ridge_triweak_h13_tw720_s022_q90_horizon_blend --output-dir outputs/advanced_models`, and ran `. .venv/bin/activate && PYTHONPATH=src python scripts/predict_baseline.py --train-path data/train.parquet --test-path data/test.parquet --baseline ridge_triweak_h13_tw720_s022_q90_horizon_blend --output outputs/submissions/ridge_triweak_h13_tw720_s022_q90_horizon_blend_submission.csv` | The clean raw branch scored `0.1120846714`, and its formalized horizon blend scored `0.1370652148`, which is reproducible and end-to-end valid but still `0.0000153273` below the incumbent `smoothed_stable_horizon_blend`; the new test prediction CSV was written successfully and the smoke suite finished at `20 passed in 1.86s` | Keep the rebuilt branch available for follow-up analysis, but do not promote it over the incumbent until the artifact discrepancy is explained |
| E033 | Milestone 5 | A tightly gated exact-group warm overlay may recover warm-start drift without destabilizing cold rows | Added `WarmStartLastTargetBlendRegressor`, exposed `smoothed_stable_horizon_blend_last_target_rules`, validated with `. .venv/bin/activate && PYTHONPATH=src pytest -q -s tests`, ran `. .venv/bin/activate && PYTHONPATH=src python scripts/evaluate_baselines.py --train-path data/train.parquet --test-path data/test.parquet --holdout-steps 360 --n-folds 4 --step-size 360 --min-train-steps 2000 --cold-start-level code_sub_code --baselines smoothed_stable_horizon_blend smoothed_stable_horizon_blend_last_target_rules --output-dir outputs/advanced_models`, and ran `. .venv/bin/activate && PYTHONPATH=src python scripts/predict_baseline.py --train-path data/train.parquet --test-path data/test.parquet --baseline smoothed_stable_horizon_blend_last_target_rules --output outputs/submissions/smoothed_stable_horizon_blend_last_target_rules_submission.csv` | The exact warm overlay improved the canonical four-fold score from `0.1370805421` to `0.1386781180`, mainly by lifting warm rows while leaving cold-start behavior unchanged; the full submission-format test prediction file was written successfully | Promote the warm last-target overlay over the raw residual family and keep using tightly gated additive warm-start ideas rather than direct fine-group replacements |
| E034 | Milestone 5 | A stable-group recent-vs-full-history delta on horizons `10` and `25` may correct the remaining drift on top of the warm last-target overlay | Added `RecentMeanDeltaRegressor`, exposed `smoothed_stable_horizon_blend_last_target_recent_delta_v1`, validated with `. .venv/bin/activate && PYTHONPATH=src pytest -q -s tests/test_baselines.py tests/test_residuals.py`, ran `. .venv/bin/activate && PYTHONPATH=src python scripts/evaluate_baselines.py --train-path data/train.parquet --test-path data/test.parquet --holdout-steps 360 --n-folds 4 --step-size 360 --min-train-steps 2000 --cold-start-level code_sub_code --baselines smoothed_stable_horizon_blend_last_target_rules smoothed_stable_horizon_blend_last_target_recent_delta_v1 --output-dir outputs/advanced_models`, and ran `. .venv/bin/activate && PYTHONPATH=src python scripts/predict_baseline.py --train-path data/train.parquet --test-path data/test.parquet --baseline smoothed_stable_horizon_blend_last_target_recent_delta_v1 --output outputs/submissions/smoothed_stable_horizon_blend_last_target_recent_delta_v1_submission.csv` | The recent-delta overlay improved the canonical four-fold score again to `0.1439924990`, with aggregate cold-start score `0.1601147203` and aggregate warm-start score `0.0937945231`; the full submission-format test prediction file was written successfully and the full suite reached `23 passed in 2.07s` before the next edit round | Promote the recent-delta branch as the new mainline incumbent and focus follow-up work on narrow code-specific overlays or clipped delta refinements |
| E035 | Milestone 5 | A narrow code-specific warm overlay may still add on top of the recent-delta incumbent if it only touches the strongest recurring miss slice | Generalized `WarmStartLastTargetBlendRegressor` to support optional code filters, exposed `smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1`, validated with `. .venv/bin/activate && PYTHONPATH=src pytest -q -s tests`, ran `. .venv/bin/activate && PYTHONPATH=src python scripts/evaluate_baselines.py --train-path data/train.parquet --test-path data/test.parquet --holdout-steps 360 --n-folds 4 --step-size 360 --min-train-steps 2000 --cold-start-level code_sub_code --baselines smoothed_stable_horizon_blend_last_target_recent_delta_v1 smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1 --output-dir outputs/stacked_models`, and ran `. .venv/bin/activate && PYTHONPATH=src python scripts/predict_baseline.py --train-path data/train.parquet --test-path data/test.parquet --baseline smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1 --output outputs/submissions/smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1_submission.csv` | The `K7Y1TTAH / horizon=25` warm overlay improved the canonical four-fold score further to `0.1450306103`, raising aggregate warm-start score to `0.0991358430`; the full suite finished at `24 passed in 2.34s` and the full submission-format test prediction file was written successfully | Promote `smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1` to the current best implemented path and extend the same narrow-overlay pattern to other recurring miss codes only if they clear the exact proxy |

## Commands That Matter

- Read order at session start:
  - `prompt.md`
  - `plans.md`
  - `documentation.md`
  - `implement.md`
- Default Python environment:
  - `. .venv/bin/activate`
- Default dependency install:
  - `python -m pip install -r requirements.txt`
- Local inspection reference used during setup:
  - `. .venv/bin/activate && python -c "import duckdb; ..."`
- Run tests:
  - `. .venv/bin/activate && pytest -q -s tests`
- Regenerate profiles:
  - `. .venv/bin/activate && python scripts/profile_data.py --output-dir outputs/profiles`
- Evaluate current baselines:
  - `. .venv/bin/activate && python scripts/evaluate_baselines.py --train-path data/train.parquet --holdout-steps 30 --output-dir outputs/baselines`
- Evaluate the current best offline proxy:
  - `. .venv/bin/activate && python scripts/evaluate_baselines.py --train-path data/train.parquet --test-path data/test.parquet --holdout-steps 360 --n-folds 4 --step-size 360 --min-train-steps 2000 --cold-start-level code_sub_code --output-dir outputs/baselines`
- Evaluate the rebuilt raw residual branch and its clean horizon blend without disturbing the default baseline summary:
  - `. .venv/bin/activate && PYTHONPATH=src python scripts/evaluate_baselines.py --train-path data/train.parquet --test-path data/test.parquet --holdout-steps 360 --n-folds 4 --step-size 360 --min-train-steps 2000 --cold-start-level code_sub_code --baselines smoothed_stable_horizon_blend ridge_triweak_h13_tw720_s022_q90 ridge_triweak_h13_tw720_s022_q90_horizon_blend --output-dir outputs/advanced_models`
- Evaluate the warm last-target overlay against the old baseline incumbent:
  - `. .venv/bin/activate && PYTHONPATH=src python scripts/evaluate_baselines.py --train-path data/train.parquet --test-path data/test.parquet --holdout-steps 360 --n-folds 4 --step-size 360 --min-train-steps 2000 --cold-start-level code_sub_code --baselines smoothed_stable_horizon_blend smoothed_stable_horizon_blend_last_target_rules --output-dir outputs/advanced_models`
- Evaluate the recent-delta overlay against the warm last-target incumbent:
  - `. .venv/bin/activate && PYTHONPATH=src python scripts/evaluate_baselines.py --train-path data/train.parquet --test-path data/test.parquet --holdout-steps 360 --n-folds 4 --step-size 360 --min-train-steps 2000 --cold-start-level code_sub_code --baselines smoothed_stable_horizon_blend_last_target_rules smoothed_stable_horizon_blend_last_target_recent_delta_v1 --output-dir outputs/advanced_models`
- Evaluate the current best stacked incumbent against its recent-delta parent:
  - `. .venv/bin/activate && PYTHONPATH=src python scripts/evaluate_baselines.py --train-path data/train.parquet --test-path data/test.parquet --holdout-steps 360 --n-folds 4 --step-size 360 --min-train-steps 2000 --cold-start-level code_sub_code --baselines smoothed_stable_horizon_blend_last_target_recent_delta_v1 smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1 --output-dir outputs/stacked_models`
- Generate the current best-known submission-format predictions:
  - `. .venv/bin/activate && python scripts/predict_baseline.py --train-path data/train.parquet --test-path data/test.parquet --baseline smoothed_stable_horizon_blend --output outputs/submissions/smoothed_stable_horizon_blend_submission.csv`
- Generate the warm last-target overlay submission-format predictions:
  - `. .venv/bin/activate && PYTHONPATH=src python scripts/predict_baseline.py --train-path data/train.parquet --test-path data/test.parquet --baseline smoothed_stable_horizon_blend_last_target_rules --output outputs/submissions/smoothed_stable_horizon_blend_last_target_rules_submission.csv`
- Generate the recent-delta incumbent submission-format predictions:
  - `. .venv/bin/activate && PYTHONPATH=src python scripts/predict_baseline.py --train-path data/train.parquet --test-path data/test.parquet --baseline smoothed_stable_horizon_blend_last_target_recent_delta_v1 --output outputs/submissions/smoothed_stable_horizon_blend_last_target_recent_delta_v1_submission.csv`
- Generate the current best stacked submission-format predictions:
  - `. .venv/bin/activate && PYTHONPATH=src python scripts/predict_baseline.py --train-path data/train.parquet --test-path data/test.parquet --baseline smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1 --output outputs/submissions/smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1_submission.csv`
- Generate the rebuilt residual-plus-horizon blend predictions:
  - `. .venv/bin/activate && PYTHONPATH=src python scripts/predict_baseline.py --train-path data/train.parquet --test-path data/test.parquet --baseline ridge_triweak_h13_tw720_s022_q90_horizon_blend --output outputs/submissions/ridge_triweak_h13_tw720_s022_q90_horizon_blend_submission.csv`
- Analyze the saved validation predictions:
  - `. .venv/bin/activate && python scripts/analyze_validation_errors.py --summary-path outputs/baselines/forward_cv_h360_f4_s360_code_sub_code/summary.json --train-path data/train.parquet --baseline smoothed_stable_group_mean_pw100 --output-dir outputs/analysis --top-k 3`
- Analyze the canonical weak slices in more detail:
  - `. .venv/bin/activate && python scripts/analyze_validation_errors.py --summary-path outputs/baselines/forward_cv_h360_f4_s360_code_sub_code/summary.json --train-path data/train.parquet --baseline smoothed_stable_group_mean_pw100 --output-dir outputs/analysis --top-k 10`
- Evaluate residual models on top of the best grouped baseline:
  - `. .venv/bin/activate && python scripts/evaluate_residual_models.py --base-summary-path outputs/baselines/forward_cv_h360_f4_s360_code_sub_code/summary.json --train-path data/train.parquet --base-baseline smoothed_stable_group_mean_pw100 --output-dir outputs/residual_models`
- Inspect the strict-OOF normalized residual artifacts and best shrink sweep:
  - `outputs/residual_models/forward_cv_h360_f4_s360_code_sub_code_oof_norm/summary.json`
  - `outputs/residual_models/forward_cv_h360_f4_s360_code_sub_code_oof_norm/posthoc_rescale_sweep.json`
  - `outputs/residual_models/forward_cv_h360_f4_s360_code_sub_code_oof_norm/ridge_oofnorm_triweak_h13_tw720_s003_q85_posthoc/summary.json`
- Generate the previous best grouped-only submission-format predictions:
  - `. .venv/bin/activate && python scripts/predict_baseline.py --train-path data/train.parquet --test-path data/test.parquet --baseline smoothed_stable_group_mean_pw100 --output outputs/submissions/smoothed_stable_group_mean_pw100_submission.csv`
- Audit competition semantics:
  - `. .venv/bin/activate && python scripts/audit_semantics.py --train-path data/train.parquet --test-path data/test.parquet --output-dir outputs/semantics`
- Run the current smoke test suite:
  - `. .venv/bin/activate && pytest -q -s tests`

## Notes For Future Updates

- Keep this file append-only where practical.
- Record failures, not just successes.
- When a milestone completes, update:
  - milestone board,
  - decisions,
  - experiment log,
  - commands that matter,
  - next actions.
