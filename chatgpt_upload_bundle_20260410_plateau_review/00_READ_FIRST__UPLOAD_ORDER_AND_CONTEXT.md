## ChatGPT Upload Bundle

This folder is prepared for a new `GPT-5.4 Pro` project prompt after the latest offline-improvement loop.

### Current winner

- Baseline: `smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_h3_beta_up_v1`
- Exact offline score: `0.1451839276`
- Exact winner summary: `14_FINAL_WINNER_EXACT_FORWARD_CV_SUMMARY__h13_pentaslice_h3_followup_exact.json`
- Triplet confirmation summary: `15_FINAL_WINNER_TRIPLET_STRESS_SUMMARY__h13_pentaslice_h3_followup_triplet.json`

### Why this bundle exists

- The short-horizon drift family still improved once more with the final `h=3` beta ratchet.
- The long-horizon warm-overlay family now looks plateaued.
- This is the right point to ask for only materially different ideas, not more micro-tuning in the same family.

### Recommended upload order

1. `00_READ_FIRST__UPLOAD_ORDER_AND_CONTEXT.md`
2. `01_PROJECT_INSTRUCTIONS__AGENTS.md`
3. `02_PROJECT_SPEC__prompt.md`
4. `03_MILESTONE_PLAN__plans.md`
5. `04_CURRENT_STATUS_AND_WORKING_LOG__documentation.md`
6. `05_RUNBOOK_AND_OPERATING_RULES__implement.md`
7. `06_BASELINE_REGISTRY_AND_WINNER_CONFIGS__baseline_registry.py`
8. `07_BASELINE_MODEL_IMPLEMENTATIONS__baselines.py`
9. `08_VALIDATION_AND_FORWARD_CV_UTILS__validation.py`
10. `11_BASELINE_EVALUATION_DRIVER__evaluate_baselines.py`
11. `12_PREDICTION_DRIVER__predict_baseline.py`
12. `14_FINAL_WINNER_EXACT_FORWARD_CV_SUMMARY__h13_pentaslice_h3_followup_exact.json`
13. `15_FINAL_WINNER_TRIPLET_STRESS_SUMMARY__h13_pentaslice_h3_followup_triplet.json`
14. `16_NON_PROMOTED_X9_H25_CORE_PROBE_SUMMARY__x9_h25_core_exact.json`
15. `17_NON_PROMOTED_OSJL_H25_CORE_PROBE_SUMMARY__osjl_h25_core_exact.json`
16. `18_SHORT_HORIZON_ERROR_ANALYSIS_CONTEXT__h13_beta_hi_summary.md`

### What each file is for

- `01` to `05`: repo rules, task spec, current milestone plan, live memory, and operating runbook.
- `06` to `13`: implementation and evaluation code for the current modeling stack.
- `14` and `15`: exact evidence for the current promoted winner.
- `16` and `17`: evidence that the tested long-horizon warm-overlay family improved local stress slices but still failed canonical promotion.
- `18`: saved error-analysis context that explains why the short-horizon family mattered.

### Recommended prompt focus

Ask for at most `3` materially different leakage-safe next branches.

Do not ask for more alpha/beta micro-tuning of the current warm-overlay family unless the proposal is clearly different from:

- `NQ58FVQM / horizon=25` retries
- `X9BZ68VQ / horizon=25` core warm overlays
- `OSJL3A7Y / horizon=25` long-core warm overlays
- simple local beta sweeps around the pentaslice `h=1/3` family

If no materially different branch is compelling, the model should say the repo is at a real unattended plateau.
