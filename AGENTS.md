# Long-Horizon Session Bootstrap

This repository is configured for a long-running Codex task.

Environment bootstrap:

- Use the repo-local virtual environment at `.venv` by default.
- If `.venv` exists, activate it before running Python commands:
  - `. .venv/bin/activate`
- If `.venv` does not exist, create it and install dependencies:
  - `python -m venv .venv`
  - `. .venv/bin/activate`
  - `python -m pip install -r requirements.txt`
- Prefer `python -m pip install -r requirements.txt` over ad hoc installs unless the active milestone requires a new dependency.

Hardware:

- GPU: `NVIDIA RTX A5000` with `24564 MiB` VRAM
- RAM: `251 GiB` visible to the OS
- CPU: `AMD EPYC 7713P` with `128` logical CPUs
- Storage: `WD_BLACK SN850X`
- Prefer GPU acceleration when it is materially helpful for the active milestone, but verify runtime availability before adding GPU-specific dependencies or training code paths.
- For CPU-bound LightGBM work, expect better multicore throughput than the previous workstation, but respect the repo runtime defaults that cap auto `num_threads` at `64` and enable `force_col_wise` on many-core hosts.
- Do not introduce heavyweight GPU libraries unless the expected gain justifies the added complexity and the choice is logged in `documentation.md`.

Before making changes, read these files in order:

1. `prompt.md`
2. `plans.md`
3. `documentation.md`
4. `implement.md`

Operating rules:

- Treat `prompt.md` as the project spec and source of truth for goals, constraints, and deliverables.
- Treat `plans.md` as the milestone list and acceptance criteria.
- Treat `implement.md` as the runbook for how to operate.
- Treat `documentation.md` as the live memory, audit log, and current status.
- Treat competition leakage rules as hard constraints, not suggestions.
- Default autonomy mode: keep working without waiting for user confirmation between milestones or experiments.
- Complete work one milestone at a time.
- Run validation after each milestone and repair failures before moving on.
- Keep diffs scoped to the current milestone.
- Update `documentation.md` after every meaningful change, experiment, or decision.
- If Kaggle credentials, data access, or submission access are missing, do not stall. Build the local scaffold, validate what can be validated, and log the exact blocker and unblocking step.
- The public competition metadata shows the deadline was `2026-04-07 07:00 UTC`, while this repo was prepared on `2026-04-09 UTC`. Assume the task is offline improvement and postmortem work unless the user explicitly confirms submissions are still possible.
- Do not use `weight` as a model feature.
- Do not use any preprocessing on test data that depends on future test rows.
- Be extremely skeptical of any method that reconstructs target information from suspicious proxy features or target overlap.

Autonomous stopping policy:

- Target outcome: continue iterating until the approach is plausibly competitive with the current best public score target of `0.8718`, or until hard blockers or diminishing returns make further unattended work unjustified.
- Do not stop after a single milestone just because one audit, baseline, or scaffold task completed.
- After each milestone, immediately continue to the next highest-leverage task unless blocked.
- Only stop and return control early if one of these is true:
  - a verified public score is close to or above `0.8718`,
  - offline validation has stalled after several materially different attempts with no meaningful gain,
  - a hard blocker requires user input, credentials, or policy clarification,
  - continuing would require a leakage-risky shortcut that cannot be justified.
