# Long-Horizon Runbook

This file defines how Codex should operate during the run.

## Read Order At Session Start

1. Read `prompt.md`.
2. Read `plans.md`.
3. Read `documentation.md`.
4. Use this file as the operating policy.

## Source Of Truth

- `prompt.md` defines the target and constraints.
- `plans.md` defines milestone order and completion rules.
- `documentation.md` is the live memory of what already happened.
- `.venv` plus `requirements.txt` define the default Python execution environment.

If these files disagree:

1. obey `prompt.md` for scope,
2. obey `plans.md` for milestone sequencing,
3. update `documentation.md` with the conflict and the chosen resolution.

## Default Operating Loop

For each work cycle:

1. Ensure `.venv` is activated. If missing, create it and install from `requirements.txt`.
2. Identify the highest-priority incomplete milestone.
3. Read the relevant code and notes before editing.
4. Make the smallest coherent change set that advances that milestone.
5. Run the fastest relevant validation first.
6. If that passes, run the milestone-level validation.
7. If validation fails, repair before moving on.
8. Update `documentation.md` with:
   - what changed,
   - commands run,
   - result metrics,
   - decisions made,
   - blockers,
   - next step.

Do not skip step 8.

After step 8:

9. Continue automatically into the next most valuable experiment or milestone unless a hard stop condition has been reached.

Do not interpret a completed milestone as a cue to stop the session.

## Self-Improvement Policy

- Treat every experiment as a learning loop, not just a score hunt.
- For every experiment, log:
  - hypothesis,
  - exact command or config,
  - observed result,
  - interpretation,
  - next action.
- Prefer one major idea per experiment cycle so results stay interpretable.
- Do not repeat a failed idea unless new evidence suggests the previous setup was flawed.
- If the score worsens, log the failure clearly and pivot intentionally.
- Treat unusually strong gains as suspicious until leakage has been actively ruled out.
- After each experiment, decide the next experiment immediately and keep going unless blocked.

## Scope Control

- Stay inside the active milestone unless there is a blocking dependency.
- Keep diffs scoped. Avoid large opportunistic refactors.
- Do not replace a simple working path with a more complex one unless the gain is measured or the old path is clearly broken.
- Once a milestone is complete, advance to the next one without waiting for user approval.

## Competition-Specific Rules

- Public metadata indicates the Kaggle deadline passed on `2026-04-07 07:00 UTC`.
- Default assumption: this run is for offline improvement, reproducibility, and postmortem-quality solution building.
- Local parquet data is already present under `data/`, so do not waste time trying to rediscover the basic schema from scratch.
- Use `.venv` as the default Python environment for all commands in this repo.
- Hardware available for this repo:
  - `NVIDIA GeForce RTX 4070 Ti` (`12 GB` VRAM)
  - `32 GB` RAM
  - `Intel Core i5-13600K`
- Do not spend major effort on submission automation unless:
  - the user asks for it, or
  - the next session verifies a still-active submission path.
- Use the exact weighted competition metric in local evaluation.
- If authenticated Kaggle access is available, verify the user-provided rule summary against the official page; otherwise proceed using the supplied summary.
- Treat `weight` as part of the scoring function and never as an input feature.
- Assume the train/test boundary is time-based because `ts_index` is contiguous across splits; validate that assumption before locking in cross-validation.
- Implement inference so that test rows are processed causally in `ts_index` order.
- Any preprocessing on test data must also be causal:
  - no full-test scaling,
  - no full-test imputation statistics,
  - no encoders or transforms fit on future test rows.
- Be cautious with target-derived features:
  - train-time target-derived features may be allowed by the host,
  - but they can inflate validation and public scores,
  - they require explicit leakage analysis and documentation before being trusted.
- Treat `feature_al` and any similarly suspicious target-proxy signal as a leakage review hotspot.
- Public leaderboard results should be treated as weak evidence; prioritize robust offline validation over leaderboard mimicry.
- Treat `0.8718` as a public-score reference only. Under the current exact offline proxy, do not plan around matching that magnitude unless the metric or validation proxy is materially revised with evidence.
- Prefer GPU-capable approaches only when they are justified by the current milestone and fit comfortably within `12 GB` VRAM.

## Validation Discipline

At minimum, keep these categories covered once the project grows enough to support them:

- unit or smoke tests,
- metric self-tests,
- train command,
- evaluation command,
- prediction or submission-format command,
- reproducibility rerun for the best-known config.
- leakage checks for preprocessing and feature generation.

Validation should escalate in this order:

1. fast smoke check,
2. milestone validation,
3. best-known pipeline rerun when a milestone completes.

Selector policy for the current repo state:

- keep `aggregate_overall_score` as the canonical offline selector,
- once the stress report exists, treat `aggregate_cold_start_score`, combined `fold_02_03_overall_score`, and the long-horizon watchlist slices as veto diagnostics,
- do not promote a candidate on a tiny aggregate gain if it materially regresses the colder stress view.

## Near-Term Priority Order

Until the current safe branch is exhausted, prefer this sequence:

1. extend evaluation summaries with stress diagnostics and fold/watchlist subset aggregation,
2. test clipped or gated `RecentMeanDeltaRegressor` variants on horizons `10` and `25`,
3. expand exact warm overlays for `OSJL3A7Y` and `X9BZ68VQ` in narrow single-bundle sweeps,
4. only then revisit short-horizon drift overlays or incumbent-based residuals,
5. keep generic boosting, deep sequence models, and broad feature sweeps lower priority.

## Session Stop Conditions

Keep the session running unless one of these is true:

1. A direct public score is close to or above the current target of `0.8718`.
2. Several materially different experiment branches have failed to produce meaningful gains under the strongest available offline validation.
3. A hard blocker requires user action, credentials, or non-recoverable clarification.
4. Continuing would require violating the leakage or sequential-processing rules.

If none of those conditions holds, continue working.

## Documentation Discipline

`documentation.md` must stay current enough that someone can leave for hours and still understand:

- what is done,
- what is next,
- what worked,
- what failed,
- what commands matter,
- what external blockers remain.

## When Blocked

If blocked by missing data, credentials, or hidden competition details:

1. make all progress that does not require the missing dependency,
2. create scaffolding, interfaces, tests, and placeholders that will be needed anyway,
3. log the blocker in `documentation.md`,
4. write the exact command, credential, or file needed to unblock.

Only stop when there is no meaningful local progress left.

## Submission Format Reminder

If a submission path is built, it must output:

- `id`
- `prediction`

and preserve the exact row identities from `test.parquet`.
