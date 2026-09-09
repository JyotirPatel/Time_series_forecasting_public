# Long-Horizon Plan

`prompt.md` is the source of truth for scope. This file turns that scope into milestones with validation rules.

## Stop-And-Fix Rule

- If a milestone validation fails, stop and repair the failure before starting the next milestone.
- If a result is inconclusive, log it as inconclusive instead of treating it as progress.
- If a new idea requires changing the plan, update this file and explain why in `documentation.md`.
- Completing a milestone is not a reason to end the session; it is a reason to move to the next highest-leverage milestone.

## Intended Architecture

Target a repo shape that supports:

- reproducible config-driven runs,
- reusable data loading and feature code,
- explicit validation and metric code,
- multiple model families,
- logged artifacts and experiment summaries.

## Milestone 0: Bootstrap The Repo

Purpose:

- turn the blank repo into a runnable forecasting project shell.

Acceptance criteria:

- choose and document the Python environment strategy,
- standardize on the repo-local `.venv`,
- create `requirements.txt` or equivalent dependency source of truth,
- create the initial directory layout,
- add a minimal command surface for setup and smoke checks,
- create at least one passing local smoke test,
- document the bootstrap commands in `documentation.md`.

Validation:

- `.venv` activation works,
- dependency install from `requirements.txt` completes,
- `pytest -q` passes for the initial smoke test set,
- at least one project entrypoint or script prints help or runs a no-op smoke path.

## Milestone 1: Recover Competition Semantics

Purpose:

- understand exactly what must be predicted and how it will be evaluated.

Acceptance criteria:

- local data ingestion path exists or the blocker is precisely documented,
- dataset inventory is written down: file names, row counts, key columns, target columns, horizon structure, and missingness patterns,
- the official weighted metric is implemented,
- the no-leakage sequential constraint is translated into concrete engineering rules,
- a candidate offline validation strategy is proposed and justified.

Validation:

- data inspection command runs end to end,
- metric self-test runs on a toy example,
- schema summary is logged in `documentation.md`.

Specific direction for this repo:

- start from the known local split:
  - train `ts_index`: `1..3601`
  - test `ts_index`: `3602..4376`
- treat `weight` as evaluation-only and not as a feature,
- explicitly test whether the correct unit of prediction is per-row or per-group-horizon aggregation,
- document which transformations are legal at train time but illegal at test time due to sequential constraints.

## Milestone 2: Establish Strong Baselines

Purpose:

- create simple reference points before higher-complexity work.

Acceptance criteria:

- at least two baseline approaches exist,
- offline validation can score them reproducibly,
- predictions can be generated in the competition format or a faithful local proxy,
- the best baseline and its weaknesses are documented.

Suggested baseline families:

- naive or repeated-last-value,
- grouped historical average or seasonal average,
- simple weighted regression or tree baseline if appropriate.

Baseline restrictions:

- avoid any baseline that uses future test rows, even implicitly,
- avoid target encoding unless its leakage implications are clearly controlled and documented,
- log whether suspiciously strong features such as `feature_al` are used, excluded, or transformed.

Validation:

- baseline training command runs,
- baseline evaluation command runs,
- baseline inference or submission-format command runs on a sample or full local dataset.

## Milestone 3: Build A Better Validation Loop

Purpose:

- make the score trustworthy enough to guide self-improvement.

Acceptance criteria:

- time-aware folds or competition-faithful splits are implemented,
- leakage checks are documented,
- score variance across folds is visible,
- the team can explain why the chosen validation scheme is the best current proxy.

Likely starting point:

- use anchored or rolling forward splits near the end of the train range,
- preserve the observed horizon structure `[1, 3, 10, 25]`,
- keep group identifiers available so leakage across `(code, sub_code, sub_category)` is testable,
- verify that every preprocessing step used in CV can also be executed sequentially at test time.

Validation:

- validation split generation is deterministic,
- fold-level evaluation runs,
- leakage smoke checks pass.

## Milestone 4: Improve The Model Stack

Purpose:

- move beyond baseline with focused, testable ideas.

Acceptance criteria:

- at least two materially different improvement paths are tested,
- each path has a short hypothesis, result, and next action,
- the best model clearly outperforms the baseline on the chosen offline metric,
- model configs and artifacts are saved reproducibly.

Candidate directions:

- grouped feature engineering,
- horizon-aware models,
- gradient boosting families,
- global sequence models only if justified by the data shape and runtime budget,
- hybrid residual modeling,
- simple but justified ensembling.

Validation:

- improved model training completes,
- offline comparison table is updated,
- best run can be reproduced from config and logged command.

## Milestone 5: Failure Analysis And Self-Improvement

Purpose:

- use evidence from bad predictions to decide what to try next instead of brute-force searching.

Acceptance criteria:

- error analysis exists by code, sub-code, sub-category, and horizon if those fields exist in the data,
- recurring failure modes are identified,
- at least one follow-up change is directly motivated by analysis,
- dead ends are explicitly recorded to avoid repeating them.

Specific analysis topics:

- public-score-chasing risk vs. genuinely robust validation gains,
- dependence on leakage-sensitive proxy features,
- stability across recent vs. older train windows,
- behavior on horizons `1`, `3`, `10`, and `25` separately.

Validation:

- analysis script or notebook runs on saved predictions,
- output tables or plots are generated,
- follow-up action is logged in `documentation.md`.

## Milestone 6: Finalize The Best-Known Solution

Purpose:

- leave the repo in a state that another session can run without rediscovering everything.

Acceptance criteria:

- best-known pipeline is reproducible from repo state,
- final commands for train, evaluate, and predict are documented,
- known limitations and likely next wins are listed,
- `documentation.md` contains a concise project summary and the next recommended move.

Validation:

- clean rerun of the best-known pipeline succeeds,
- final smoke test set passes,
- reproducibility notes are complete enough for another Codex session to continue.

## Persistent Run Policy

After Milestone 6 or whenever a local best-known solution exists:

- continue with targeted improvement loops instead of stopping immediately,
- prefer the next branch that has the best ratio of expected gain to leakage risk,
- keep iterating until one of these conditions holds:
  - a direct public score is close to or above `0.8718`,
  - offline validation has plateaued across several materially different experiment branches,
  - a hard blocker requires user intervention.
