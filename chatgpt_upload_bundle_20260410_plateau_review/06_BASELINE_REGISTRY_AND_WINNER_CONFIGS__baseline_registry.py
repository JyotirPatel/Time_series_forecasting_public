from __future__ import annotations

from ts_forecasting.baselines import (
    LinearBlendRegressor,
    RecentMeanDeltaRegressor,
    SmoothedWeightedGroupMeanRegressor,
    WarmStartLastTargetBlendRegressor,
    WeightedGroupMeanRegressor,
)
from ts_forecasting.constants import GROUP_COLUMNS
from ts_forecasting.residuals import FixedPrefixResidualCorrectionRegressor


SMOOTHED_STABLE_GROUP_CONFIG: dict[str, object] = {
    "regressor": "smoothed_group_mean",
    "fallback_levels": [
        ["code", "sub_category", "horizon"],
        ["sub_category", "horizon"],
        ["horizon"],
    ],
    "prior_weight": 100.0,
}
HORIZON_MEAN_CONFIG: dict[str, object] = {
    "regressor": "group_mean",
    "fallback_levels": [["horizon"]],
}
SMOOTHED_STABLE_HORIZON_BLEND_CONFIG: dict[str, object] = {
    "regressor": "linear_blend",
    "weights": [0.6355710510414874, 0.3644289489585126],
    "components": [
        SMOOTHED_STABLE_GROUP_CONFIG,
        HORIZON_MEAN_CONFIG,
    ],
}
WARM_LAST_TARGET_BLEND_RULES: list[dict[str, object]] = [
    {"horizon": 3, "sub_category": "NQ58FVQM", "alpha": 0.05},
    {"horizon": 10, "sub_category": "NQ58FVQM", "alpha": 0.05},
    {"horizon": 10, "sub_category": "V8BKY1IV", "alpha": 0.04},
    {"horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.12},
    {"horizon": 25, "sub_category": "PHHHVYZI", "alpha": 0.01},
    {"horizon": 25, "sub_category": "PZ9S1Z4V", "alpha": 0.04},
    {"horizon": 25, "sub_category": "V8BKY1IV", "alpha": 0.075},
]
K7_H25_WARM_RULES: list[dict[str, object]] = [
    {"code": "K7Y1TTAH", "horizon": 25, "sub_category": "PHHHVYZI", "alpha": 0.15},
    {"code": "K7Y1TTAH", "horizon": 25, "sub_category": "PZ9S1Z4V", "alpha": 0.20},
    {"code": "K7Y1TTAH", "horizon": 25, "sub_category": "DPPUO5X2", "alpha": 0.03},
]
OSJL_DPPU_H10_WARM_PROBE_RULES: list[dict[str, object]] = [
    {"code": "OSJL3A7Y", "horizon": 10, "sub_category": "DPPUO5X2", "alpha": 0.03},
]
OSJL_PH_H10_WARM_PROBE_RULES: list[dict[str, object]] = [
    {"code": "OSJL3A7Y", "horizon": 10, "sub_category": "PHHHVYZI", "alpha": 0.03},
]
OSJL_PZ_H10_WARM_PROBE_RULES: list[dict[str, object]] = [
    {"code": "OSJL3A7Y", "horizon": 10, "sub_category": "PZ9S1Z4V", "alpha": 0.03},
]
X9_PH_H10_WARM_PROBE_RULES: list[dict[str, object]] = [
    {"code": "X9BZ68VQ", "horizon": 10, "sub_category": "PHHHVYZI", "alpha": 0.03},
]
OSJL_H3_CORE_WARM_PROBE_A003_RULES: list[dict[str, object]] = [
    {"code": "OSJL3A7Y", "horizon": 3, "sub_category": "DPPUO5X2", "alpha": 0.03},
    {"code": "OSJL3A7Y", "horizon": 3, "sub_category": "PHHHVYZI", "alpha": 0.03},
    {"code": "OSJL3A7Y", "horizon": 3, "sub_category": "PZ9S1Z4V", "alpha": 0.03},
]
OSJL_H3_CORE_WARM_PROBE_A005_RULES: list[dict[str, object]] = [
    {"code": "OSJL3A7Y", "horizon": 3, "sub_category": "DPPUO5X2", "alpha": 0.05},
    {"code": "OSJL3A7Y", "horizon": 3, "sub_category": "PHHHVYZI", "alpha": 0.05},
    {"code": "OSJL3A7Y", "horizon": 3, "sub_category": "PZ9S1Z4V", "alpha": 0.05},
]
X9_H3_CORE_WARM_PROBE_A003_RULES: list[dict[str, object]] = [
    {"code": "X9BZ68VQ", "horizon": 3, "sub_category": "DPPUO5X2", "alpha": 0.03},
    {"code": "X9BZ68VQ", "horizon": 3, "sub_category": "PHHHVYZI", "alpha": 0.03},
    {"code": "X9BZ68VQ", "horizon": 3, "sub_category": "PZ9S1Z4V", "alpha": 0.03},
]
X9_H3_CORE_WARM_PROBE_A005_RULES: list[dict[str, object]] = [
    {"code": "X9BZ68VQ", "horizon": 3, "sub_category": "DPPUO5X2", "alpha": 0.05},
    {"code": "X9BZ68VQ", "horizon": 3, "sub_category": "PHHHVYZI", "alpha": 0.05},
    {"code": "X9BZ68VQ", "horizon": 3, "sub_category": "PZ9S1Z4V", "alpha": 0.05},
]
OSJL_H25_NQ_WARM_PROBE_A003_RULES: list[dict[str, object]] = [
    {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.03},
]
OSJL_H25_NQ_WARM_PROBE_A005_RULES: list[dict[str, object]] = [
    {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.05},
]
X9_H25_NQ_WARM_PROBE_A003_RULES: list[dict[str, object]] = [
    {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.03},
]
X9_H25_NQ_WARM_PROBE_A005_RULES: list[dict[str, object]] = [
    {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.05},
]
X9_H25_CORE_WARM_PROBE_A003_RULES: list[dict[str, object]] = [
    {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "DPPUO5X2", "alpha": 0.03},
    {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "PHHHVYZI", "alpha": 0.03},
    {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "PZ9S1Z4V", "alpha": 0.03},
    {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.03},
]
X9_H25_CORE_WARM_PROBE_A005_RULES: list[dict[str, object]] = [
    {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "DPPUO5X2", "alpha": 0.05},
    {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "PHHHVYZI", "alpha": 0.05},
    {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "PZ9S1Z4V", "alpha": 0.05},
    {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.05},
]
OSJL_H25_CORE_WARM_PROBE_A003_RULES: list[dict[str, object]] = [
    {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "DPPUO5X2", "alpha": 0.03},
    {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "PHHHVYZI", "alpha": 0.03},
    {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "PZ9S1Z4V", "alpha": 0.03},
]
OSJL_H25_CORE_WARM_PROBE_A005_RULES: list[dict[str, object]] = [
    {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "DPPUO5X2", "alpha": 0.05},
    {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "PHHHVYZI", "alpha": 0.05},
    {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "PZ9S1Z4V", "alpha": 0.05},
]
H13_DPPU_RULES: list[dict[str, object]] = [
    {"sub_category": "DPPUO5X2", "horizon": 1},
    {"sub_category": "DPPUO5X2", "horizon": 3},
]
H13_TRISLICE_RULES: list[dict[str, object]] = [
    *H13_DPPU_RULES,
    {"sub_category": "PZ9S1Z4V", "horizon": 1},
    {"sub_category": "PZ9S1Z4V", "horizon": 3},
    {"sub_category": "PHHHVYZI", "horizon": 1},
    {"sub_category": "PHHHVYZI", "horizon": 3},
]
H13_PENTASLICE_RULES: list[dict[str, object]] = [
    *H13_TRISLICE_RULES,
    {"sub_category": "NQ58FVQM", "horizon": 1},
    {"sub_category": "NQ58FVQM", "horizon": 3},
    {"sub_category": "V8BKY1IV", "horizon": 1},
    {"sub_category": "V8BKY1IV", "horizon": 3},
]
RECENT_DELTA_V1_CONFIG: dict[str, object] = {
    "regressor": "recent_mean_delta",
    "base_predictor": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": SMOOTHED_STABLE_HORIZON_BLEND_CONFIG,
        "blend_rules": WARM_LAST_TARGET_BLEND_RULES,
    },
    "recent_window": 180,
    "min_recent_rows": 50,
    "beta": 0.15,
    "horizons": [10, 25],
}
RECENT_DELTA_CLIP_V1_CONFIG: dict[str, object] = {
    **RECENT_DELTA_V1_CONFIG,
    "clip_quantile_by_horizon": {10: 0.8, 25: 0.9},
    "min_recent_weight_sum": 25.0,
}
RECENT_DELTA_CLIP_HBETA_V1_CONFIG: dict[str, object] = {
    **RECENT_DELTA_CLIP_V1_CONFIG,
    "beta_by_horizon": {10: 0.08, 25: 0.15},
}
RECENT_DELTA_CLIP_OSJLX9_V1_CONFIG: dict[str, object] = {
    **RECENT_DELTA_CLIP_V1_CONFIG,
    "apply_rules": [
        {"code": "OSJL3A7Y", "horizon": 10},
        {"code": "OSJL3A7Y", "horizon": 25},
        {"code": "X9BZ68VQ", "horizon": 10},
        {"code": "X9BZ68VQ", "horizon": 25},
    ],
}
RECENT_DELTA_K7_V1_CONFIG: dict[str, object] = {
    "regressor": "warm_start_last_target_blend",
    "base_predictor": RECENT_DELTA_V1_CONFIG,
    "blend_rules": K7_H25_WARM_RULES,
}
RECENT_DELTA_K7_H13_DPPU_V1_CONFIG: dict[str, object] = {
    "regressor": "recent_mean_delta",
    "base_predictor": RECENT_DELTA_K7_V1_CONFIG,
    "recent_window": 180,
    "min_recent_rows": 100,
    "min_recent_weight_sum": 50.0,
    "beta": 0.0,
    "horizons": [1, 3],
    "beta_by_horizon": {1: 0.04, 3: 0.06},
    "clip_quantile_by_horizon": {1: 0.7, 3: 0.8},
    "apply_rules": H13_DPPU_RULES,
}
RECENT_DELTA_K7_H13_TRISLICE_V1_CONFIG: dict[str, object] = {
    **RECENT_DELTA_K7_H13_DPPU_V1_CONFIG,
    "apply_rules": H13_TRISLICE_RULES,
}
RECENT_DELTA_K7_H13_PENTASLICE_V1_CONFIG: dict[str, object] = {
    **RECENT_DELTA_K7_H13_DPPU_V1_CONFIG,
    "apply_rules": H13_PENTASLICE_RULES,
}
RECENT_DELTA_K7_H13_TRISLICE_BETA_LO_V1_CONFIG: dict[str, object] = {
    **RECENT_DELTA_K7_H13_TRISLICE_V1_CONFIG,
    "beta_by_horizon": {1: 0.03, 3: 0.05},
}
RECENT_DELTA_K7_H13_TRISLICE_BETA_HI_V1_CONFIG: dict[str, object] = {
    **RECENT_DELTA_K7_H13_TRISLICE_V1_CONFIG,
    "beta_by_horizon": {1: 0.05, 3: 0.08},
}
RECENT_DELTA_K7_H13_TRISLICE_RW360_V1_CONFIG: dict[str, object] = {
    **RECENT_DELTA_K7_H13_TRISLICE_V1_CONFIG,
    "recent_window": 360,
}
RECENT_DELTA_K7_H13_PENTASLICE_BETA_HI_V1_CONFIG: dict[str, object] = {
    **RECENT_DELTA_K7_H13_PENTASLICE_V1_CONFIG,
    "beta_by_horizon": {1: 0.05, 3: 0.08},
}
RECENT_DELTA_K7_H13_PENTASLICE_H3_BETA_UP_V1_CONFIG: dict[str, object] = {
    **RECENT_DELTA_K7_H13_PENTASLICE_V1_CONFIG,
    "beta_by_horizon": {1: 0.05, 3: 0.10},
}
RIDGE_TRIWEAK_RAW_CONFIG: dict[str, object] = {
    "regressor": "fixed_prefix_residual",
    "base_predictor": SMOOTHED_STABLE_GROUP_CONFIG,
    "model_type": "ridge",
    "horizons": [1, 3],
    "sub_categories": ["DPPUO5X2", "PZ9S1Z4V", "PHHHVYZI"],
    "residual_train_steps": 720,
    "correction_scale": 0.22,
    "correction_clip_quantile": 0.9,
    "max_train_rows": 180_000,
    "alpha": 10.0,
    "banned_feature_columns": ["feature_al"],
}


BASELINE_CONFIGS: dict[str, dict[str, object]] = {
    "horizon_mean": HORIZON_MEAN_CONFIG,
    "stable_group_mean": {
        "regressor": "group_mean",
        "fallback_levels": [
            ["code", "sub_category", "horizon"],
            ["sub_category", "horizon"],
            ["horizon"],
        ],
    },
    "smoothed_stable_group_mean_pw100": SMOOTHED_STABLE_GROUP_CONFIG,
    "smoothed_stable_horizon_blend": SMOOTHED_STABLE_HORIZON_BLEND_CONFIG,
    "smoothed_stable_horizon_blend_last_target_rules": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": SMOOTHED_STABLE_HORIZON_BLEND_CONFIG,
        "blend_rules": WARM_LAST_TARGET_BLEND_RULES,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_v1": RECENT_DELTA_V1_CONFIG,
    "smoothed_stable_horizon_blend_last_target_recent_delta_clip_v1": {
        **RECENT_DELTA_CLIP_V1_CONFIG,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_clip_hbeta_v1": {
        **RECENT_DELTA_CLIP_HBETA_V1_CONFIG,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_clip_osjlx9_v1": {
        **RECENT_DELTA_CLIP_OSJLX9_V1_CONFIG,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1": RECENT_DELTA_K7_V1_CONFIG,
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_osjl_dppu10_a003_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_V1_CONFIG,
        "blend_rules": OSJL_DPPU_H10_WARM_PROBE_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_osjl_ph10_a003_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_V1_CONFIG,
        "blend_rules": OSJL_PH_H10_WARM_PROBE_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_osjl_pz10_a003_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_V1_CONFIG,
        "blend_rules": OSJL_PZ_H10_WARM_PROBE_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_x9_ph10_a003_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_V1_CONFIG,
        "blend_rules": X9_PH_H10_WARM_PROBE_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_dppu_v1": {
        **RECENT_DELTA_K7_H13_DPPU_V1_CONFIG,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_v1": {
        **RECENT_DELTA_K7_H13_TRISLICE_V1_CONFIG,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_v1": {
        **RECENT_DELTA_K7_H13_PENTASLICE_V1_CONFIG,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_beta_hi_v1": {
        **RECENT_DELTA_K7_H13_PENTASLICE_BETA_HI_V1_CONFIG,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_h3_beta_up_v1": {
        **RECENT_DELTA_K7_H13_PENTASLICE_H3_BETA_UP_V1_CONFIG,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_lo_v1": {
        **RECENT_DELTA_K7_H13_TRISLICE_BETA_LO_V1_CONFIG,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_v1": {
        **RECENT_DELTA_K7_H13_TRISLICE_BETA_HI_V1_CONFIG,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_rw360_v1": {
        **RECENT_DELTA_K7_H13_TRISLICE_RW360_V1_CONFIG,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_osjl_h3_core_a003_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_H13_TRISLICE_BETA_HI_V1_CONFIG,
        "blend_rules": OSJL_H3_CORE_WARM_PROBE_A003_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_osjl_h3_core_a005_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_H13_TRISLICE_BETA_HI_V1_CONFIG,
        "blend_rules": OSJL_H3_CORE_WARM_PROBE_A005_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_x9_h3_core_a003_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_H13_TRISLICE_BETA_HI_V1_CONFIG,
        "blend_rules": X9_H3_CORE_WARM_PROBE_A003_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_x9_h3_core_a005_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_H13_TRISLICE_BETA_HI_V1_CONFIG,
        "blend_rules": X9_H3_CORE_WARM_PROBE_A005_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_osjl_h25_nq_a003_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_H13_TRISLICE_BETA_HI_V1_CONFIG,
        "blend_rules": OSJL_H25_NQ_WARM_PROBE_A003_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_osjl_h25_nq_a005_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_H13_TRISLICE_BETA_HI_V1_CONFIG,
        "blend_rules": OSJL_H25_NQ_WARM_PROBE_A005_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_x9_h25_nq_a003_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_H13_TRISLICE_BETA_HI_V1_CONFIG,
        "blend_rules": X9_H25_NQ_WARM_PROBE_A003_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_x9_h25_nq_a005_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_H13_TRISLICE_BETA_HI_V1_CONFIG,
        "blend_rules": X9_H25_NQ_WARM_PROBE_A005_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_beta_hi_x9_h25_core_a003_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_H13_PENTASLICE_BETA_HI_V1_CONFIG,
        "blend_rules": X9_H25_CORE_WARM_PROBE_A003_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_beta_hi_x9_h25_core_a005_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_H13_PENTASLICE_BETA_HI_V1_CONFIG,
        "blend_rules": X9_H25_CORE_WARM_PROBE_A005_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_beta_hi_osjl_h25_core_a003_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_H13_PENTASLICE_BETA_HI_V1_CONFIG,
        "blend_rules": OSJL_H25_CORE_WARM_PROBE_A003_RULES,
        "run_by_default": False,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_beta_hi_osjl_h25_core_a005_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_K7_H13_PENTASLICE_BETA_HI_V1_CONFIG,
        "blend_rules": OSJL_H25_CORE_WARM_PROBE_A005_RULES,
        "run_by_default": False,
    },
    "ridge_triweak_h13_tw720_s022_q90": {
        **RIDGE_TRIWEAK_RAW_CONFIG,
        "run_by_default": False,
    },
    "ridge_triweak_h13_tw720_s022_q90_horizon_blend": {
        "regressor": "linear_blend",
        "weights": [0.63786, 0.36214],
        "components": [
            RIDGE_TRIWEAK_RAW_CONFIG,
            HORIZON_MEAN_CONFIG,
        ],
        "run_by_default": False,
    },
    "hierarchical_group_mean": {
        "regressor": "group_mean",
        "fallback_levels": [
            [*GROUP_COLUMNS],
            ["code", "sub_category", "horizon"],
            ["sub_category", "horizon"],
            ["horizon"],
        ],
    },
}


def build_predictor(config: dict[str, object]):
    regressor_type = str(config["regressor"])
    if regressor_type == "group_mean":
        return WeightedGroupMeanRegressor(
            fallback_levels=[list(level) for level in config["fallback_levels"]],
        )
    if regressor_type == "smoothed_group_mean":
        return SmoothedWeightedGroupMeanRegressor(
            fallback_levels=[list(level) for level in config["fallback_levels"]],
            prior_weight=float(config["prior_weight"]),
        )
    if regressor_type == "linear_blend":
        return LinearBlendRegressor(
            predictors=[
                build_predictor(component_config)
                for component_config in config["components"]
            ],
            weights=[float(weight) for weight in config["weights"]],
        )
    if regressor_type == "warm_start_last_target_blend":
        return WarmStartLastTargetBlendRegressor(
            base_predictor=build_predictor(dict(config["base_predictor"])),
            blend_rules=[dict(rule) for rule in config["blend_rules"]],
        )
    if regressor_type == "recent_mean_delta":
        return RecentMeanDeltaRegressor(
            base_predictor=build_predictor(dict(config["base_predictor"])),
            recent_window=int(config["recent_window"]),
            min_recent_rows=int(config["min_recent_rows"]),
            beta=float(config["beta"]),
            level_columns=list(config.get("level_columns", ["code", "sub_category", "horizon"])),
            horizons=(
                [int(horizon) for horizon in config["horizons"]]
                if config.get("horizons") is not None
                else None
            ),
            beta_by_horizon=(
                {
                    int(horizon): float(beta)
                    for horizon, beta in dict(config["beta_by_horizon"]).items()
                }
                if config.get("beta_by_horizon") is not None
                else None
            ),
            clip_quantile_by_horizon=(
                {
                    int(horizon): float(quantile)
                    for horizon, quantile in dict(config["clip_quantile_by_horizon"]).items()
                }
                if config.get("clip_quantile_by_horizon") is not None
                else None
            ),
            apply_rules=(
                [dict(rule) for rule in config["apply_rules"]]
                if config.get("apply_rules") is not None
                else None
            ),
            min_recent_weight_sum=float(config.get("min_recent_weight_sum", 0.0)),
        )
    if regressor_type == "fixed_prefix_residual":
        return FixedPrefixResidualCorrectionRegressor(
            base_predictor=build_predictor(dict(config["base_predictor"])),
            model_type=str(config["model_type"]),
            horizons=config.get("horizons"),
            sub_categories=config.get("sub_categories"),
            codes=config.get("codes"),
            residual_train_steps=int(config["residual_train_steps"]),
            correction_scale=float(config["correction_scale"]),
            correction_clip_quantile=float(config["correction_clip_quantile"]),
            max_train_rows=int(config["max_train_rows"]),
            alpha=float(config.get("alpha", 10.0)),
            epsilon=float(config.get("epsilon", 1.35)),
            max_iter=int(config.get("max_iter", 200)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            max_depth=int(config.get("max_depth", 3)),
            min_samples_leaf=int(config.get("min_samples_leaf", 200)),
            banned_feature_columns=set(config.get("banned_feature_columns", [])),
        )
    raise ValueError(f"unknown regressor type: {regressor_type}")
