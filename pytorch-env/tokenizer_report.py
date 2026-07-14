"""
tokenizer_report.py — Báo cáo đánh giá quantile / phân phối dữ liệu.
─────────────────────────────────────────────────────────────────────────
Tách khỏi tabular_tokenizer.py: module này CHỈ lo xuất báo cáo (JSON),
không tham gia vào logic tokenize. Hai báo cáo:

  1. feature_groups_report.json  — profile quantile + group + strategy mỗi feature
                                    (kết quả Phase 1 + Phase 2).
  2. post_scaling_report.json    — thống kê phân phối token SAU khi scale
                                    (kết quả Phase 3).
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Thứ tự cột cho báo cáo feature-groups (profile quantile per-feature).
# [REM-3] Đã bỏ các trường clip (clipped/clip_lo/clip_hi) — cơ chế clip không còn dùng.
# Thêm marker piecewise + giá trị đuôi sau biến đổi (pw_split, pw_tail_after).
_FEATURE_GROUPS_COL_ORDER = [
    "feature", "group", "strategy", "heavy_tail", "piecewise",
    "pw_split", "pw_tail_after", "scaler", "reason",
    "p0", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p100",
    "iqr", "tail_ratio", "tail_75_ratio", "tail_90_ratio",
    "left_ratio", "spread_ratio", "zero_ratio", "n_unique",
    "mean", "std", "skew", "kurtosis", "cv",
]

# Nhóm liên tục (token = giá trị đã scale [0, scale_factor]) — để chuẩn hoá khi report
_DEFAULT_CONTINUOUS_GROUPS = {"normal", "skewed"}


def export_feature_groups_report(rows: list, report_path: str) -> None:
    """
    Xuất báo cáo profile quantile + grouping + strategy mỗi feature ra JSON.
    `rows`: list dict do phase2_group dựng (mỗi feature 1 dict).
    """
    (pd.DataFrame(rows)[_FEATURE_GROUPS_COL_ORDER]
       .sort_values(["group", "feature"])
       .to_json(report_path, orient="records", indent=4, force_ascii=False))


def export_post_scaling_report(X_tokens, features, groups, scale_factor,
                               continuous_groups=None,
                               report_path="post_scaling_report.json",
                               strategies=None) -> None:
    """
    Xuất báo cáo thống kê phân phối token SAU khi scale (min/max/percentile,
    zero-ratio…) cho từng feature. `strategies` (tùy chọn): map feature→chiến lược
    để đánh dấu cột nào dùng piecewise — các scaled_* chính là GIÁ TRỊ SAU biến đổi.
    """
    cont = continuous_groups if continuous_groups is not None else _DEFAULT_CONTINUOUS_GROUPS
    feat_group = {c: g for g, cols in groups.items() for c in cols}
    strategies = strategies or {}

    rows = []
    for i, col in enumerate(features):
        vals   = X_tokens[:, i].astype(float)
        grp    = feat_group.get(col, "unknown")
        normed = vals / max(scale_factor, 1) if grp in cont else vals

        rows.append({
            "feature":           col,
            "group":             grp,
            "strategy":          strategies.get(col, grp),
            "piecewise":         strategies.get(col) == "piecewise_sqrt",
            "token_min":         int(vals.min()),
            "token_max":         int(vals.max()),
            "token_mean":        round(float(vals.mean()), 2),
            "scaled_mean":       round(float(normed.mean()), 6),
            "scaled_std":        round(float(normed.std()),  6),
            "scaled_p0":         round(float(np.percentile(normed,   0)), 6),
            "scaled_p1":         round(float(np.percentile(normed,   1)), 6),
            "scaled_p5":         round(float(np.percentile(normed,   5)), 6),
            "scaled_p25":        round(float(np.percentile(normed,  25)), 6),
            "scaled_p50":        round(float(np.percentile(normed,  50)), 6),
            "scaled_p75":        round(float(np.percentile(normed,  75)), 6),
            "scaled_p95":        round(float(np.percentile(normed,  95)), 6),
            "scaled_p99":        round(float(np.percentile(normed,  99)), 6),
            "scaled_p100":       round(float(np.percentile(normed, 100)), 6),
            "scaled_zero_ratio": round(float((normed == 0).mean()), 4),
        })

    (pd.DataFrame(rows)
       .sort_values(["group", "feature"])
       .to_json(report_path, orient="records", indent=4, force_ascii=False))
    print(f"\n  Post-scaling report → {Path(report_path).resolve()}")
