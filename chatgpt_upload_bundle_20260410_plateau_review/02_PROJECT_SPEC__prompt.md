# Long-Horizon Codex Task: `ts-forecasting`

## Objective

Build a reproducible, evidence-driven forecasting project for Kaggle competition `ts-forecasting`, using a long-horizon Codex workflow that can improve itself over time through logged experiments, milestone-based execution, and continuous validation.

Because the public Kaggle competition metadata shows the submission deadline already passed on `2026-04-07 07:00 UTC`, the default goal for this run is:

- reproduce the competition locally,
- build the strongest offline validation setup possible,
- iterate on models and ensembles with measured evidence,
- leave behind a clean project that can be resumed, audited, and extended.

If the user later confirms that submissions are still possible by some other path, adapt the workflow without discarding the offline-first discipline.

## Long-Run Success Target

The intended behavior is autonomous iteration, not a short assisted burst.

- Keep working through milestones and experiments without waiting for manual monitoring.
- External target score to chase when submissions are possible: current best public score approximately `0.8718`.
- Treat `0.8718` as a public-leaderboard reference, not as a credible offline planning target under the current exact proxy. Until the metric or validation proxy changes with evidence, optimize for relative offline gains, colder-slice robustness, and leakage-safe improvement rather than trying to match public-LB magnitude.
- If direct submission feedback is unavailable, use increasingly realistic offline validation as the optimization target and keep improving until progress clearly plateaus.
- Do not stop just because one milestone or one analysis task finished.
- Stop only for a real blocker, severe leakage uncertainty, or convincingly exhausted progress.

## Competition Snapshot

- Competition: `Hedge fund - Time series forecasting`
- Slug: `ts-forecasting`
- Public description: `Time series forecasting by code, sub-code, sub_category and horizon`
- Metric label: `Skill Score`
- Metric direction: maximize
- Row id column: `id`
- Public leaderboard share: `24%`
- Max daily submissions: `1`
- Total scored rows: `1,447,107`
- Total teams at capture time: `1,443`
- Deadline: `2026-04-07 07:00 UTC`
- Private leaderboard release: `2026-04-14 07:00 UTC`
- Workspace prep date: `2026-04-09 UTC`

Known gap:

- The exact metric formula and detailed evaluation rules were not available from the public unauthenticated endpoints used during setup. The next run should pull them from the Kaggle Evaluation tab or authenticated competition metadata before making any metric-specific assumptions.

Competition details supplied by the user after setup materially reduce that gap and should be treated as authoritative working guidance unless contradicted by the official competition page.

## Local Data Snapshot

Competition data is now present locally:

- `data/train.parquet`
- `data/test.parquet`

Observed structure:

- Train shape: `5,337,414` rows, `94` columns
- Test shape: `1,447,107` rows, `92` columns
- Shared feature columns: `86` columns named `feature_a` through `feature_ch`
- Identifier and grouping columns: `id`, `code`, `sub_code`, `sub_category`, `horizon`, `ts_index`
- Train-only columns: `y_target`, `weight`
- Horizon values present in both splits: `[1, 3, 10, 25]`
- `ts_index` spans:
  - train: `1` to `3601`
  - test: `3602` to `4376`
- Distinct group counts:
  - train: `23` codes, `180` sub-codes, `5` sub-categories, `9,270` `(code, sub_code, sub_category)` groups
  - test: `23` codes, `47` sub-codes, `5` sub-categories, `2,784` `(code, sub_code, sub_category)` groups

Useful implications:

- This appears to be a forward time split, so validation should also be forward-chaining and time-aware.
- `weight` is part of evaluation and should be incorporated into every offline metric prototype, but must not be used as a feature.
- Many feature columns contain missing values, so null handling is part of the baseline, not an optional refinement.
- Horizon values are categorical forecast groups, not numeric time differences from `ts_index`.

## Available Hardware

- GPU: `NVIDIA GeForce RTX 4070 Ti` with `12 GB` VRAM
- RAM: `32 GB`
- CPU: `Intel Core i5-13600K`

Implications:

- GPU acceleration is available for later model iterations if it offers real value.
- Early profiling, metric work, leakage checks, and simple baselines do not require GPU use.
- Before adding GPU-specific model stacks or dependencies, verify that the runtime can actually access the GPU and record the decision in `documentation.md`.

## Evaluation Metric

Use this metric for offline evaluation unless the official competition material later contradicts it:

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

Implications:

- Higher is better.
- The score depends on `weight` and `y_target`.
- If the weighted squared error exceeds the weighted target-energy denominator, the clipped score collapses toward `0`.
- A standard RMSE objective may be directionally useful, but the actual offline validation function should match the competition metric exactly.

## Sequential Prediction Constraint

This is a hard rule:

- For prediction at time `t`, the model may depend only on information available up to `t`.
- On the test period, all inference must be processed strictly sequentially.
- Test-time preprocessing must also be sequential.
- Any normalization, scaling, encoding, feature engineering, or post-processing that uses future test rows is invalid.

In practice:

- no global standardization over the full test set,
- no fitting imputers or encoders on the full test set,
- no using future test rows to create lagged, rolling, grouped, or target-proxy features,
- no validation design that silently relies on future information.

## Leakage And Discussion Notes

Important user-supplied competition discussion context:

- The host explicitly warned that unusually high public scores may be driven by forward-looking information and may be disqualified.
- The host stated that public leaderboard rank is a poor proxy for private leaderboard rank.
- The host clarified that target-derived features on train are allowed, but they can inflate CV and public scores.
- To counter this, the host updated scoring logic to ignore some hidden test points; the exact ignored set is undisclosed.
- The host stated that train/test targets overlap by construction across adjacent timestamps and long horizons, even when input features do not overlap.
- Community discussion flagged `feature_al` as a possible target-proxy or leakage vector. Treat any unusually predictive proxy feature with extreme caution and document exactly how leakage risk is ruled out before relying on it.

## Primary Goals

1. Create a real project structure for data loading, feature engineering, training, validation, inference, and experiment tracking.
2. Reconstruct the competition problem as precisely as possible from the available data and rules.
3. Establish a trustworthy offline validation scheme before spending significant time on model complexity.
4. Implement simple but strong baselines first, then iterate using measured gains only.
5. Maintain durable project memory so the next Codex session can continue without drift.
6. Produce a final reproducible best-known pipeline, not a pile of disconnected notebooks.

## Non-Goals

- Do not chase leaderboard-style improvements without offline evidence.
- Do not rely on hidden or leaked information.
- Do not create large exploratory notebooks as the main artifact if scripts/configs will do.
- Do not broaden scope into unrelated modeling research unless it directly supports this competition.
- Do not rewrite working code purely for aesthetics during an active milestone.

## Hard Constraints

- Every claimed improvement must be backed by a reproducible command and logged metric result.
- Validation comes before ambition. If validation is weak, improve validation before model sophistication.
- Prefer small, reviewable steps over large speculative rewrites.
- Keep all important decisions in `documentation.md`.
- If a validation command fails, repair it before moving on.
- If Kaggle access is blocked, continue with code scaffolding, synthetic smoke tests, and exact unblock instructions.
- Use the repo-local `.venv` for Python work unless there is a documented reason not to.
- Use the local parquet files as the initial source of truth for schema, split boundaries, and target availability.
- Do not use `weight` as a feature.
- Evaluate with the exact weighted metric, not a rough proxy, whenever feasible.
- Any preprocessing applied to test data must be causally valid at each `ts_index`.
- Treat suspicious target-proxy features and target-derived encodings as leakage-sensitive and require explicit justification.
- Save repeatable experiment outputs under stable paths such as `outputs/`, `artifacts/`, or equivalent directories created during the run.

## Deliverables

By the end of the long-horizon run, the repo should contain:

1. A runnable project layout with scripts or entrypoints for setup, training, evaluation, and prediction.
2. A documented understanding of the data schema, forecasting target, and evaluation assumptions.
3. A baseline model suite with recorded offline scores.
4. At least one stronger iteration beyond baseline, with ablation notes explaining why it helped or failed.
5. A best-known reproducible pipeline and config.
6. A continuously updated `documentation.md` containing decisions, scores, commands, blockers, and next actions.
7. If still useful and permitted, a submission-generation path. If not useful, document why it was skipped.
8. A documented causal inference path for test-time preprocessing and sequential prediction.

## Done When

The run is successful when all of the following are true:

- The project can be set up and executed locally from documented commands.
- There is a trustworthy offline evaluation loop.
- Baseline and improved models have been run and compared.
- The best-known approach is reproducible from repo state alone.
- `documentation.md` tells a future session what happened, what worked, what failed, and what to do next.

And for unattended long-run behavior:

- the session should keep iterating automatically until it either reaches a leaderboard-competitive solution, hits a real blocker, or exhausts several strong experiment branches without meaningful gains.

## Recommended Project Shape

The next session may create or adapt a structure like this if the repo is still empty:

```text
configs/
data/
notebooks/
outputs/
src/
tests/
```

Inside `src/`, prefer a layout close to:

```text
src/
  data/
  features/
  metrics/
  models/
  training/
  inference/
  utils/
```

## Quality Bar

- Baselines first.
- One major hypothesis per experiment cycle.
- Fast smoke checks for every new script.
- Milestone-level validation before moving forward.
- Clear logging of both wins and failures.

## References

- OpenAI reference for the durable markdown stack: [Run long horizon tasks with Codex](https://developers.openai.com/blog/run-long-horizon-tasks-with-codex)
- Kaggle competition overview: [ts-forecasting overview](https://www.kaggle.com/competitions/ts-forecasting/overview)
- Public competition metadata endpoint used during setup: [GetCompetition](https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition?competitionName=ts-forecasting)
