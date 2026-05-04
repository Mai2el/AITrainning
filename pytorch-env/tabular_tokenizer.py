"""
tabular_tokenizer.py  —  3-Phase Tabular Processing Pipeline
─────────────────────────────────────────────────────────────
PHASE 1 — QUANTILE PROFILING  (fit on train split only)
  Percentiles: p0 p1 p5 p10 p25 p50 p75 p90 p95 p99 p100
  Derived indicators:
    iqr          = p75 - p25
    tail_ratio   = (p99 - p50) / (p50 - p1 + ε)   right tail vs left body
    spread_ratio = iqr / (p99 - p1 + ε)            how spread the IQR is
    zero_ratio   = fraction of zeros
    n_unique     = distinct value count

PHASE 2 — QUANTILE-BASED GROUPING + REPORT EXPORT
  Decision (in priority order):
    port       → col name contains 'port'
    binary     → unique ⊆ {0, 1}
    low_card   → n_unique < 20
    zero_heavy → p50 == 0   (≥50 % zeros)
    skewed     → tail_ratio > 5  AND  p99 > 100
    sparse     → spread_ratio < 0.05
    smooth     → spread_ratio > 0.30  AND  tail_ratio < 3
    moderate   → everything else
  Export: feature_groups_report.json  (all quantile stats + group + reason)

PHASE 3 — SCALE + TOKENISE
  QuantileTransformer (train) → [0,1] → × scale_factor → round → int token
  Example:  duration=694250.49 → 0.8234 → ×30000 → 24702.7 → token 24703
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────
_VAL_TEST_FRACTION         = 0.30
_TEST_WITHIN_TEMP_FRACTION = 0.50
_SPLIT_RANDOM_STATE        = 42
DEFAULT_SCALE_FACTOR       = 30_000
_PORT_VOCAB                = 7
_EPS                       = 1e-9
_PERCENTILES               = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

# ── Group labels ──────────────────────────────────────────────
GROUP_BINARY     = "binary"
GROUP_PORT       = "port"
GROUP_LOWCARD    = "low_card"
GROUP_ZERO_HEAVY = "zero_heavy"
GROUP_SKEWED     = "skewed"
GROUP_ANOMALY    = "anomaly"
GROUP_NORMAL     = "normal"
_ALL_GROUPS = [
    GROUP_BINARY, GROUP_PORT, GROUP_LOWCARD,
    GROUP_ZERO_HEAVY, GROUP_SKEWED, GROUP_ANOMALY, GROUP_NORMAL
]


# ═══════════════════════════════════════════════════════════════
# PHASE 1 — QUANTILE PROFILING
# ═══════════════════════════════════════════════════════════════
def _profile_one(col_name: str, series: pd.Series) -> dict:
    """
    Compute quantile profile for one feature (call on train split only).

    Percentile columns : p0, p1, p5, p10, p25, p50, p75, p90, p95, p99, p100
    Derived indicators :
      iqr          = p75 - p25
      tail_ratio   = (p99-p50) / (p50-p1+ε)   → right-tail heaviness
      left_ratio   = (p50-p1)  / (p99-p1+ε)   → body left fraction
      spread_ratio = iqr       / (p99-p1+ε)   → IQR relative to body
      zero_ratio   = fraction of zeros
      n_unique     = distinct values count
    """
    v = series.dropna().values.astype(float)
    n = len(v)

    pct = {f"p{p}": (float(np.percentile(v, p)) if n > 0 else 0.0)
           for p in _PERCENTILES}

    p1, p25, p50, p75, p90, p99 = pct["p1"], pct["p25"], pct["p50"], pct["p75"], pct["p90"], pct["p99"]
    iqr          = p75 - p25
    body         = p99 - p1
    tail_ratio   = (p99 - p50) / (p50 - p1  + _EPS)
    tail_75_ratio= (p99 - p75) / (iqr       + _EPS)
    tail_90_ratio= (p99 - p90) / (p90 - p50 + _EPS)
    left_ratio   = (p50 - p1)  / (body       + _EPS)
    spread_ratio = iqr          / (body       + _EPS)
    zero_ratio   = float((v == 0).mean()) if n > 0 else 0.0
    n_unique     = int(np.unique(v).size)

    return {
        "feature":      col_name,
        **{k: round(val, 6) for k, val in pct.items()},
        "iqr":          round(iqr,          6),
        "tail_ratio":   round(tail_ratio,   4),
        "tail_75_ratio":round(tail_75_ratio, 4),
        "tail_90_ratio":round(tail_90_ratio, 4),
        "left_ratio":   round(left_ratio,   4),
        "spread_ratio": round(spread_ratio, 4),
        "zero_ratio":   round(zero_ratio,   4),
        "n_unique":     n_unique,
        "mean":         round(float(np.mean(v)),  6) if n > 0 else 0.0,
        "std":          round(float(np.std(v)),   6) if n > 0 else 0.0,
        "skew":         round(float(pd.Series(v).skew()),     4) if n > 2 else 0.0,
        "kurtosis":     round(float(pd.Series(v).kurtosis()), 4) if n > 3 else 0.0,
        "cv":           round(float(np.std(v) / (np.mean(v) + _EPS)), 4) if n > 0 else 0.0,
        "is_binary":    set(np.unique(v)).issubset({0.0, 1.0}),
        "is_low_card":  n_unique < 20,
        "is_port":      "port" in col_name.lower(),
    }


def phase1_profile(df: pd.DataFrame,
                   features: list,
                   train_idx: np.ndarray) -> dict:
    """PHASE 1: Quantile-profile all features using training rows only."""
    print(f"\n[Phase 1] Quantile-profiling {len(features)} features "
          f"({len(train_idx):,} train rows) …")
    profiles = {col: _profile_one(col, df.loc[train_idx, col]) for col in features}
    print(f"          Percentiles: {_PERCENTILES}")
    return profiles


# ═══════════════════════════════════════════════════════════════
# PHASE 2 — QUANTILE-BASED GROUPING + REPORT EXPORT
# ═══════════════════════════════════════════════════════════════
_GROUP_CRITERIA = {
    GROUP_BINARY:     "unique ⊆ {0,1}",
    GROUP_PORT:       "col name contains 'port'",
    GROUP_LOWCARD:    "n_unique < 20  (AND zero_ratio < 0.5)",
    GROUP_ZERO_HEAVY: "zero_ratio ≥ 0.50",
    GROUP_SKEWED:     "tail_ratio > 5 OR spread_ratio < 0.05",
    GROUP_ANOMALY:    "p99 > 1e8 AND p10 < 100 (mixed timestamps)",
    GROUP_NORMAL:     "everything else (well-behaved distribution)",
}


def _assign_group(p: dict) -> tuple:
    """
    Return (group, reason) purely from quantile indicators.

    Priority order (critical fixes):
      1. port       — name-based, always first
      2. binary     — unique values ⊆ {0, 1}
      3. zero_heavy — p50 == 0 (≥50% zeros) checked BEFORE low_card
                      so high-zero discrete features (e.g. fin_flag_counts
                      with n_unique=12 but zero_ratio=0.89) are handled
                      by the two-phase zero/non-zero scaler, not factorize.
      4. low_card   — n_unique < 20, only when NOT zero_heavy
      5. skewed     — heavy right tail (tail_ratio>5 and p99>100)
      6. sparse     — IQR very narrow (spread_ratio<0.05), zero_ratio<0.5
                      If zero_ratio>=0.5 and p50==0, prefer zero_heavy.
      7. smooth     — well-spread, low tail
      8. moderate   — everything else
    """
    if p["is_port"]:
        return GROUP_PORT, "col name contains 'port' → hand-crafted bucket (0-6)"

    if p["is_binary"]:
        return GROUP_BINARY, f"unique⊆{{0,1}}, n_unique={p['n_unique']}"

    tr, sr = p["tail_ratio"], p["spread_ratio"]
    p50, p99 = p["p50"], p["p99"]
    zero_r = p["zero_ratio"]

    # ── Zero-heavy ────────────────────────────────────────────────────────
    # ✅ FIX: Hạ ngưỡng 0.50 → 0.45 để bắt các feature như `duration`
    # (zero_ratio=0.4996 sát ngưỡng; sau log-tail scale vẫn tạo 67% zeros)
    if zero_r >= 0.45:
        return (GROUP_ZERO_HEAVY,
                f"zero_ratio={zero_r:.3f}≥0.45 → dominated by zeros; "
                f"two-phase piecewise scaling preserves sparse nature")

    # ── Low cardinality (only when NOT zero-dominated) ────────────────────
    if p["is_low_card"]:
        return GROUP_LOWCARD, f"n_unique={p['n_unique']} < 20 (zero_ratio={zero_r:.3f}<0.5) → label-encode"

    # ── Timestamp Anomaly ─────────────────────────────────────────────────
    if p["p99"] > 1e8 and p["p10"] < 100:
        return (GROUP_ANOMALY, 
                f"Mixed timestamps (p99={p99:.0f}, p10={p['p10']:.2f}) → cap tail and scale body linearly")

    # ── Skewed or Sparse ──────────────────────────────────────────────────
    if tr > 5.0 or sr < 0.05 or p["left_ratio"] > 0.8 or p["left_ratio"] < 0.2:
        return (GROUP_SKEWED,
                f"tail_ratio={tr:.2f} or spread_ratio={sr:.4f} → skewed/sparse distribution; "
                f"piecewise scaling handles body linearly and compresses tail")

    # ── Normal / Well-behaved ─────────────────────────────────────────────
    return (GROUP_NORMAL,
            f"well-behaved distribution (tail_ratio={tr:.2f}, spread_ratio={sr:.3f}) → robust linear")


def phase2_group(profiles: dict,
                 report_path: str = "feature_groups_report.json") -> dict:
    """
    PHASE 2: Group features by quantile criteria; export JSON report.

    Report columns:
      feature | group | reason | p0…p100 | iqr | tail_ratio |
      left_ratio | spread_ratio | zero_ratio | n_unique
    """
    print(f"\n[Phase 2] Grouping by quantile analysis → '{report_path}' …")

    groups = {g: [] for g in _ALL_GROUPS}
    rows   = []

    for col, p in profiles.items():
        grp, reason = _assign_group(p)
        groups[grp].append(col)
        rows.append({
            "feature": col, "group": grp, "reason": reason,
            "scaler": _SCALER_LABELS.get(grp, "unknown"),
            **{k: p[k] for k in
               ["p0","p1","p5","p10","p25","p50","p75","p90","p95","p99","p100",
                "iqr","tail_ratio","tail_75_ratio","tail_90_ratio","left_ratio","spread_ratio","zero_ratio","n_unique",
                "mean","std","skew","kurtosis","cv"]},
        })

    # Export JSON
    col_order = ["feature","group","scaler","reason",
                 "p0","p1","p5","p10","p25","p50","p75","p90","p95","p99","p100",
                 "iqr","tail_ratio","tail_75_ratio","tail_90_ratio","left_ratio","spread_ratio","zero_ratio","n_unique",
                 "mean","std","skew","kurtosis","cv"]
    (pd.DataFrame(rows)[col_order]
       .sort_values(["group","feature"])
       .to_json(report_path, orient="records", indent=4, force_ascii=False))

    # Print summary
    W = 68
    print(f"\n  {'─'*W}")
    print(f"  {'Feature Group Summary  (Quantile-Based)':^{W}}")
    print(f"  {'─'*W}")
    print(f"  {'Group':<12}  {'N':>4}  {'Criteria':<35}  Example features")
    print(f"  {'─'*W}")
    for grp in _ALL_GROUPS:
        cols = groups[grp]
        if not cols:
            continue
        ex = ", ".join(cols[:2]) + ("…" if len(cols) > 2 else "")
        print(f"  {grp:<12}  {len(cols):>4}  {_GROUP_CRITERIA[grp]:<35}  {ex}")
    print(f"  {'─'*W}")
    print(f"\n  Report → {Path(report_path).resolve()}")
    return groups


# ═══════════════════════════════════════════════════════════════
# PORT BUCKET MAPPING
# ═══════════════════════════════════════════════════════════════
_PORT_MAP = {0: 0, 22: 1, 53: 2, 80: 3, 443: 4}

def _port_to_token(port) -> int:
    try:
        port = int(port)
    except (TypeError, ValueError):
        return 0
    if port in _PORT_MAP:
        return _PORT_MAP[port]
    return 5 if port < 1024 else 6


# ═══════════════════════════════════════════════════════════════
# PHASE 3 — SCALE + TOKENISE  (data-driven per-group strategy)
# ═══════════════════════════════════════════════════════════════
# Scaling Strategy by Group:
#   smooth     → Robust Linear (p1/p99 clip → linear [0,1])
#                Preserves linear relationships for well-spread data
#   moderate   → QuantileTransformer (uniform output)
#                Adapts to mixed distributions
#   skewed     → QuantileTransformer (uniform output)
#                Compresses heavy right tails
#   sparse     → Rank-based (rankdata → linear [0,1])
#                Handles ties & low resolution / clumped values
#   zero_heavy → Two-phase: zero→0, non-zero→QuantileTransformer→[1, SF]
#                Preserves the zero/non-zero boundary
#   binary     → Keep as-is (0/1)
#   port       → Hand-crafted bucket mapping (0-6)
#   low_card   → Label-encode (factorize, ordinal)
# ─────────────────────────────────────────────────────────────


def _fit_qt(X_train: np.ndarray) -> QuantileTransformer:
    """Fit QuantileTransformer (uniform output) on training data."""
    n_q = min(1000, max(100, X_train.shape[0] // 50))
    qt  = QuantileTransformer(
        n_quantiles=n_q, output_distribution="uniform",
        subsample=min(300_000, X_train.shape[0]), random_state=42,
    )
    qt.fit(X_train)
    return qt


def _qt_scale(X_train, X_all, scale_factor) -> np.ndarray:
    """QuantileTransform → [0,1] → ×scale → round → int token."""
    scaled = _fit_qt(X_train).transform(X_all)
    return np.clip(np.round(scaled * scale_factor).astype(np.int64),
                   0, scale_factor)


def _robust_linear_scale(X_train, X_all, scale_factor,
                          lo_pct=1, hi_pct=99) -> np.ndarray:
    """
    Robust linear scaler: clip to [p_lo, p_hi] of training data,
    then linearly map to [0, scale_factor].
    Preserves linear relationships — ideal for smooth / well-spread features.
    Relative ordering is preserved within the clip bounds.
    """
    lo = np.percentile(X_train, lo_pct, axis=0)    # (n_cols,)
    hi = np.percentile(X_train, hi_pct, axis=0)    # (n_cols,)
    denom = hi - lo
    denom[denom < _EPS] = _EPS          # avoid division by zero
    clipped = np.clip(X_all, lo, hi)
    normed  = (clipped - lo) / denom    # → [0, 1]
    return np.clip(np.round(normed * scale_factor).astype(np.int64),
                   0, scale_factor)


def _piecewise_linear_log_scale(X_train, X_all, scale_factor, split_pcts=None, body_ratios=None) -> np.ndarray:
    """
    Dynamic Log-Tail Scaler: 
    - Body (min to split_pct): Linear (no transformation).
    - Tail (split_pct to max): Log transformation: x' = s + log1p(x - s).
    - Finally, Min-Max scale the entire transformed feature to [0, scale_factor].
    (body_ratios is kept in signature for compatibility but is intentionally ignored)
    """
    n_cols = X_train.shape[1]
    if split_pcts is None:
        split_pcts = [99] * n_cols

    result = np.zeros_like(X_all, dtype=np.float64)
    
    for j in range(n_cols):
        col_tr = X_train[:, j]
        col_all = X_all[:, j]
        s_pct = split_pcts[j]
        
        s = np.percentile(col_tr, s_pct)
        l = np.percentile(col_tr, 1)  # robust minimum
        
        if s - l < _EPS:
            s = l + _EPS
            
        # 1. Transform training data to find global maximum
        tr_trans = col_tr.copy().astype(np.float64)
        tr_tail_mask = tr_trans > s
        if np.any(tr_tail_mask):
            tr_trans[tr_tail_mask] = s + np.log1p(tr_trans[tr_tail_mask] - s)
            
        m_trans = np.max(tr_trans)
        
        # 2. Transform all data
        all_trans = col_all.copy().astype(np.float64)
        all_tail_mask = all_trans > s
        if np.any(all_tail_mask):
            all_trans[all_tail_mask] = s + np.log1p(all_trans[all_tail_mask] - s)
            
        # 3. Global Min-Max Scale
        denom = m_trans - l
        if denom < _EPS:
            denom = _EPS
            
        normed = (all_trans - l) / denom
        result[:, j] = np.clip(normed * scale_factor, 0.0, scale_factor)

    return np.clip(np.round(result).astype(np.int64), 0, scale_factor)


def _rank_scale(X_train, X_all, scale_factor) -> np.ndarray:
    """
    Rank-based scaler: fit rank mapping on training data,
    apply to all data via percentile interpolation.
    Handles ties gracefully (average rank).
    Ideal for sparse / clumped features with many ties.
    Preserves relative ordering.
    """
    n_cols = X_train.shape[1]
    result = np.zeros_like(X_all, dtype=np.int64)

    for j in range(n_cols):
        train_col = X_train[:, j]
        all_col   = X_all[:, j]

        # Build rank mapping from training data
        sorted_unique = np.unique(train_col)
        n_uniq = len(sorted_unique)

        if n_uniq <= 1:
            # Constant column → all tokens = 0
            result[:, j] = 0
            continue

        # Map each unique value to a [0,1] position via its rank
        rank_positions = np.linspace(0.0, 1.0, n_uniq)
        # Use searchsorted to find where each value falls among train uniques
        indices = np.searchsorted(sorted_unique, all_col, side='right') - 1
        indices = np.clip(indices, 0, n_uniq - 1)
        normed  = rank_positions[indices]

        result[:, j] = np.clip(
            np.round(normed * scale_factor).astype(np.int64),
            0, scale_factor)

    return result


_SCALER_LABELS = {
    GROUP_BINARY:     "keep {0,1}",
    GROUP_PORT:       "bucket map (0-6)",
    GROUP_LOWCARD:    "factorize (ordinal)",
    GROUP_ZERO_HEAVY: "two_phase (0→0, nz→log_tail_minmax)",
    GROUP_SKEWED:     "log_tail_minmax (dynamic token ratio)",
    GROUP_ANOMALY:    "clip_to_1000 → robust_linear",
    GROUP_NORMAL:     "robust_linear (p1/p99 → [0,1])",
}


def phase3_scale_and_tokenize(
    df, features, groups, idx_train, profiles,
    scale_factor: int = DEFAULT_SCALE_FACTOR,
) -> np.ndarray:
    """
    PHASE 3: Per-group scaling → integer token matrix (N, F).

    Each group uses a scaling strategy matched to its statistical profile:
      normal       → Robust linear (p1/p99 clip, linear map)
      skewed       → Log-Tail MinMax (log tail then global minmax)
      zero_heavy   → Two-phase (zero→0, non-zero→log_tail_minmax)
      binary       → Keep as-is
      port         → Bucket map
      low_card     → Factorize
    """
    print(f"\n[Phase 3] Scaling & tokenising  (scale_factor={scale_factor:,}) …")
    print(f"  ┌{'─'*60}┐")
    print(f"  │ {'Group':<12} │ {'Strategy':<38} │ {'N':>3} │")
    print(f"  ├{'─'*60}┤")
    for grp in _ALL_GROUPS:
        if groups.get(grp):
            print(f"  │ {grp:<12} │ {_SCALER_LABELS[grp]:<38} │ {len(groups[grp]):>3} │")
    print(f"  └{'─'*60}┘")

    n, F      = len(df), len(features)
    feat_idx  = {col: i for i, col in enumerate(features)}
    X_tokens  = np.zeros((n, F), dtype=np.int64)

    for grp, cols in groups.items():
        if not cols:
            continue
        ci = [feat_idx[c] for c in cols]
        print(f"\n  [{grp:<12}] {len(cols):>3} features  "
              f"({_SCALER_LABELS.get(grp, '?')})", end="")

        # ── Binary: keep {0, 1} ────────────────────────────────
        if grp == GROUP_PORT:
            for col, i in zip(cols, ci):
                X_tokens[:, i] = df[col].apply(_port_to_token).values
            print(f"  →  token ∈ {{0..{_PORT_VOCAB-1}}}")
            continue

        if grp == GROUP_BINARY:
            for col, i in zip(cols, ci):
                X_tokens[:, i] = df[col].astype(np.int64).values.clip(0, 1)
            print("  →  token ∈ {0, 1}")
            continue

        # ── Low cardinality: stable sort-based ordinal encoding ──
        # ✅ FIX: pd.factorize() phụ thuộc thứ tự hàng → encoding thay đổi nếu data bị shuffle
        # Thay bằng mapping dựa trên sorted unique values từ training set → stable & reproducible
        if grp == GROUP_LOWCARD:
            for col, i in zip(cols, ci):
                train_vals = sorted(df.loc[idx_train, col].dropna().unique())
                val_map = {v: k for k, v in enumerate(train_vals)}
                X_tokens[:, i] = (
                    df[col].map(val_map).fillna(0).astype(np.int64).values.clip(0, 19)
                )
            print("  →  token ∈ [0, n_unique-1]  (train-fitted, sort-stable)")
            continue

        # ── Continuous groups: extract train/all matrices ──────
        X_tr  = df.loc[idx_train, cols].values.astype(float)
        X_all = df[cols].values.astype(float)

        # ── Zero-heavy: two-phase (piecewise linear+log) ──────────────
        if grp == GROUP_ZERO_HEAVY:
            tokens = np.zeros_like(X_all, dtype=np.int64)
            for k in range(X_all.shape[1]):
                col_name = cols[k]
                col_all  = X_all[:, k]
                col_tr   = X_tr[:, k]
                
                nz_mask_tr = (col_tr != 0)
                nz_tr      = col_tr[nz_mask_tr].reshape(-1, 1)
                
                if len(nz_tr) > 0:
                    col_all_2d = col_all.reshape(-1, 1)
                    
                    # Adaptive Piecewise Linear-Log
                    p = profiles[col_name]
                    s_pct, b_rat = 99, 0.90
                    if p.get("tail_75_ratio", 0) > 5.0:
                        s_pct, b_rat = 75, 0.75
                    elif p.get("tail_90_ratio", 0) > 5.0:
                        s_pct, b_rat = 90, 0.90
                        
                    tk = _piecewise_linear_log_scale(nz_tr, col_all_2d, scale_factor - 1, split_pcts=[s_pct], body_ratios=[b_rat]) + 1
                        
                    tokens[:, k] = tk.flatten()
                    
                tokens[col_all == 0, k] = 0

        # ── Skewed: Adaptive Piecewise linear+log ──────────────────────────────
        elif grp == GROUP_SKEWED:
            s_pcts, b_rats = [], []
            for col_name in cols:
                p = profiles[col_name]
                if p.get("tail_75_ratio", 0) > 5.0:
                    s_pcts.append(75)
                    b_rats.append(0.75)
                elif p.get("tail_90_ratio", 0) > 5.0:
                    s_pcts.append(90)
                    b_rats.append(0.90)
                else:
                    s_pcts.append(99)
                    b_rats.append(0.90)
            tokens = _piecewise_linear_log_scale(X_tr, X_all, scale_factor, split_pcts=s_pcts, body_ratios=b_rats)

        # ── Anomaly: Cap Timestamps and Linear Scale Body ──────────────────
        elif grp == GROUP_ANOMALY:
            tokens = np.zeros_like(X_all, dtype=np.int64)
            for k in range(X_all.shape[1]):
                col_tr  = X_tr[:, k]
                col_all = X_all[:, k]
                
                # Find valid max (highest percentile < 1e6, fallback to 1000)
                valid_max = 1000.0
                for pct in [99, 95, 90, 75, 50, 25, 10]:
                    val = np.percentile(col_tr, pct)
                    if val < 1e6:
                        valid_max = max(val * 1.5, 1000.0)
                        break
                        
                clipped_tr  = np.clip(col_tr, 0, valid_max)
                clipped_all = np.clip(col_all, 0, valid_max)
                
                tokens[:, k] = _robust_linear_scale(clipped_tr.reshape(-1, 1), 
                                                    clipped_all.reshape(-1, 1), 
                                                    scale_factor, lo_pct=0, hi_pct=100).flatten()

        # ── Normal: robust linear scaler ──────────────────────────────
        elif grp == GROUP_NORMAL:
            tokens = _robust_linear_scale(X_tr, X_all, scale_factor,
                                           lo_pct=1, hi_pct=99)

        X_tokens[:, ci] = tokens
        print(f"  →  token ∈ [{int(tokens.min())}, {int(tokens.max())}]")

    vocab = int(X_tokens.max()) + 1
    print(f"\n  {'─'*50}")
    print(f"  Vocab size = {vocab:,}  → use as NUM_BINS in start.py")
    return X_tokens


# ═══════════════════════════════════════════════════════════════
# POST-SCALING REPORT  (continuous values before discretisation)
# ═══════════════════════════════════════════════════════════════
def _export_post_scaling_report(X_tokens, features, groups, scale_factor,
                                report_path="post_scaling_report.json"):
    """
    Export per-feature statistics of the scaled (continuous) data.
    Continuous groups: tokens / scale_factor → approximate [0,1].
    Discrete groups (binary, port, low_card): raw token stats.
    """
    _CONTINUOUS = {GROUP_NORMAL, GROUP_SKEWED, GROUP_ZERO_HEAVY, GROUP_ANOMALY}
    feat_group = {}
    for g, cols in groups.items():
        for c in cols:
            feat_group[c] = g

    rows = []
    for i, col in enumerate(features):
        vals = X_tokens[:, i].astype(float)
        grp  = feat_group.get(col, "unknown")
        normed = vals / max(scale_factor, 1) if grp in _CONTINUOUS else vals

        rows.append({
            "feature":      col,
            "group":        grp,
            "token_min":    int(vals.min()),
            "token_max":    int(vals.max()),
            "token_mean":   round(float(vals.mean()), 2),
            "scaled_mean":  round(float(normed.mean()), 6),
            "scaled_std":   round(float(normed.std()),  6),
            "scaled_min":   round(float(normed.min()),  6),
            "scaled_p0":    round(float(np.percentile(normed, 0)), 6),
            "scaled_p1":    round(float(np.percentile(normed, 1)), 6),
            "scaled_p5":    round(float(np.percentile(normed, 5)), 6),
            "scaled_p10":   round(float(np.percentile(normed, 10)), 6),
            "scaled_p25":   round(float(np.percentile(normed, 25)), 6),
            "scaled_p50":   round(float(np.percentile(normed, 50)), 6),
            "scaled_p75":   round(float(np.percentile(normed, 75)), 6),
            "scaled_p90":   round(float(np.percentile(normed, 90)), 6),
            "scaled_p95":   round(float(np.percentile(normed, 95)), 6),
            "scaled_p99":   round(float(np.percentile(normed, 99)), 6),
            "scaled_p100":  round(float(np.percentile(normed, 100)), 6),
            "scaled_max":   round(float(normed.max()),  6),
            "scaled_zero_ratio": round(float((normed == 0).mean()), 4),
        })

    (pd.DataFrame(rows)
       .sort_values(["group", "feature"])
       .to_json(report_path, orient="records", indent=4, force_ascii=False))
    print(f"\n  Post-scaling report → {Path(report_path).resolve()}")


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def smart_process_and_tokenize(
    input_csv:    str,
    output_txt:   str,
    label_col:    str   = "activity",
    scale_factor: int   = DEFAULT_SCALE_FACTOR,
    report_path:  str   = "feature_groups_report.json",
    post_scaling_report_path: str = "post_scaling_report.json",
    val_test_fraction:         float = _VAL_TEST_FRACTION,
    test_within_temp_fraction: float = _TEST_WITHIN_TEMP_FRACTION,
    random_state:              int   = _SPLIT_RANDOM_STATE,
) -> int:
    """Run the full 3-phase pipeline. Returns vocab_size (use as NUM_BINS)."""
    t0 = time.time()

    print(f"\n{'='*60}\n  Tabular Tokenizer  |  scale_factor={scale_factor:,}\n{'='*60}")

    # Load & clean
    df = pd.read_csv(input_csv, low_memory=False)
    df.columns = df.columns.str.strip()
    df.drop(columns=[c for c in ["label","flow_id","timestamp","src_ip","dst_ip","src_port"]
                     if c in df.columns], inplace=True)
    for col in df.columns:
        if col != label_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    print(f"[Load]  shape={df.shape}  label='{label_col}'")

    # Labels + split indices
    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].values)
    print(f"        Labels: {list(le.classes_)}")

    features = [c for c in df.columns if c != label_col]
    n = len(df)
    idx_train, idx_temp = train_test_split(
        np.arange(n), test_size=val_test_fraction, stratify=y, random_state=random_state)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=test_within_temp_fraction,
        stratify=y[idx_temp], random_state=random_state)
    print(f"        train={len(idx_train):,}  val={len(idx_val):,}  test={len(idx_test):,}")

    # Phase 1 — Quantile Profiling
    profiles  = phase1_profile(df, features, idx_train)
    # Phase 2 — Quantile-Based Grouping
    groups    = phase2_group(profiles, report_path=report_path)
    # Phase 3 — Scale + Tokenise
    X_tokens  = phase3_scale_and_tokenize(df, features, groups, idx_train, profiles, scale_factor)

    # Post-scaling report (continuous stats before discretisation)
    _export_post_scaling_report(X_tokens, features, groups, scale_factor,
                                report_path=post_scaling_report_path)

    # Save
    df_out = pd.DataFrame(X_tokens, columns=features)
    df_out[label_col] = df[label_col].values
    df_out.to_csv(output_txt, sep="\t", index=False)

    vocab_size = int(X_tokens.max()) + 1
    print(f"\n{'='*60}")
    print(f"  ✅  Done in {time.time()-t0:.1f}s")
    print(f"  Rows={n:,}  Features={len(features)}  Vocab={vocab_size:,}")
    print(f"  Output : {Path(output_txt).resolve()}")
    print(f"  Report : {Path(report_path).resolve()}")
    print(f"{'='*60}\n")
    return vocab_size


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    vocab_size = smart_process_and_tokenize(
        input_csv    = "dataset_ready.csv",
        output_txt   = "dataset_tokenized.txt",
        label_col    = "activity",
        scale_factor = DEFAULT_SCALE_FACTOR,
        report_path  = "feature_groups_report.json",
    )
    print(f"→  NUM_BINS = {vocab_size}")
