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
GROUP_SPARSE     = "sparse"
GROUP_SMOOTH     = "smooth"
GROUP_MODERATE   = "moderate"
_ALL_GROUPS = [
    GROUP_BINARY, GROUP_PORT, GROUP_LOWCARD,
    GROUP_ZERO_HEAVY, GROUP_SKEWED, GROUP_SPARSE,
    GROUP_SMOOTH, GROUP_MODERATE,
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

    p1, p50, p99 = pct["p1"], pct["p50"], pct["p99"]
    iqr        = pct["p75"] - pct["p25"]
    body       = p99 - p1
    tail_ratio   = (p99 - p50) / (p50 - p1  + _EPS)
    left_ratio   = (p50 - p1)  / (body       + _EPS)
    spread_ratio = iqr          / (body       + _EPS)
    zero_ratio   = float((v == 0).mean()) if n > 0 else 0.0
    n_unique     = int(np.unique(v).size)

    return {
        "feature":      col_name,
        **{k: round(val, 6) for k, val in pct.items()},
        "iqr":          round(iqr,          6),
        "tail_ratio":   round(tail_ratio,   4),
        "left_ratio":   round(left_ratio,   4),
        "spread_ratio": round(spread_ratio, 4),
        "zero_ratio":   round(zero_ratio,   4),
        "n_unique":     n_unique,
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
    GROUP_LOWCARD:    "n_unique < 20",
    GROUP_ZERO_HEAVY: "p50 = 0  (≥50% zeros)",
    GROUP_SKEWED:     "tail_ratio > 5  AND  p99 > 100",
    GROUP_SPARSE:     "spread_ratio < 0.05  (IQR narrow)",
    GROUP_SMOOTH:     "spread_ratio > 0.30  AND  tail_ratio < 3",
    GROUP_MODERATE:   "everything else",
}


def _assign_group(p: dict) -> tuple:
    """Return (group, reason) purely from quantile indicators."""
    if p["is_port"]:
        return GROUP_PORT, "col name contains 'port' → hand-crafted bucket (0-6)"
    if p["is_binary"]:
        return GROUP_BINARY, f"unique⊆{{0,1}}, n_unique={p['n_unique']}"
    if p["is_low_card"]:
        return GROUP_LOWCARD, f"n_unique={p['n_unique']} < 20 → label-encode"

    tr, sr = p["tail_ratio"], p["spread_ratio"]
    p50, p99 = p["p50"], p["p99"]

    if p50 == 0:
        return (GROUP_ZERO_HEAVY,
                f"p50=0 → ≥50% zeros; zero_ratio={p['zero_ratio']:.3f}")
    if tr > 5.0 and p99 > 100:
        return (GROUP_SKEWED,
                f"tail_ratio={tr:.2f}>5 and p99={p99:.1f}>100 → heavy right tail; "
                f"p25={p['p25']:.1f} p50={p50:.1f} p99={p99:.1f}")
    if sr < 0.05:
        return (GROUP_SPARSE,
                f"spread_ratio={sr:.4f}<0.05 → IQR={p['iqr']:.2f} narrow; "
                f"feature is clumped/discrete-like")
    if sr > 0.30 and tr < 3.0:
        return (GROUP_SMOOTH,
                f"spread_ratio={sr:.3f}>0.30 and tail_ratio={tr:.2f}<3 → "
                f"well-spread; p25={p['p25']:.1f} p75={p['p75']:.1f}")
    return (GROUP_MODERATE,
            f"tail_ratio={tr:.2f}, spread_ratio={sr:.3f}; "
            f"p25={p['p25']:.1f} p50={p50:.1f} p75={p['p75']:.1f}")


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
            **{k: p[k] for k in
               ["p0","p1","p5","p10","p25","p50","p75","p90","p95","p99","p100",
                "iqr","tail_ratio","left_ratio","spread_ratio","zero_ratio","n_unique"]},
        })

    # Export JSON
    col_order = ["feature","group","reason",
                 "p0","p1","p5","p10","p25","p50","p75","p90","p95","p99","p100",
                 "iqr","tail_ratio","left_ratio","spread_ratio","zero_ratio","n_unique"]
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
# PHASE 3 — SCALE + TOKENISE
# ═══════════════════════════════════════════════════════════════
def _fit_qt(X_train: np.ndarray) -> QuantileTransformer:
    n_q = min(1000, max(100, X_train.shape[0] // 50))
    qt  = QuantileTransformer(
        n_quantiles=n_q, output_distribution="uniform",
        subsample=min(300_000, X_train.shape[0]), random_state=42,
    )
    qt.fit(X_train)
    return qt


def _qt_discretise(X_train, X_all, scale_factor) -> np.ndarray:
    """QuantileTransform → ×scale → round → int token."""
    scaled = _fit_qt(X_train).transform(X_all)
    return np.clip(np.round(scaled * scale_factor).astype(np.int64), 0, scale_factor)


def phase3_scale_and_tokenize(
    df, features, groups, idx_train,
    scale_factor: int = DEFAULT_SCALE_FACTOR,
) -> np.ndarray:
    """PHASE 3: Per-group scaling → integer token matrix (N, F)."""
    print(f"\n[Phase 3] Scaling & tokenising  (scale_factor={scale_factor:,}) …")

    n, F      = len(df), len(features)
    feat_idx  = {col: i for i, col in enumerate(features)}
    X_tokens  = np.zeros((n, F), dtype=np.int64)

    for grp, cols in groups.items():
        if not cols:
            continue
        ci = [feat_idx[c] for c in cols]
        print(f"  [{grp:<12}] {len(cols):>3} features", end="  →  ")

        if grp == GROUP_PORT:
            for col, i in zip(cols, ci):
                X_tokens[:, i] = df[col].apply(_port_to_token).values
            print(f"token ∈ {{0..{_PORT_VOCAB-1}}}")
            continue

        if grp == GROUP_BINARY:
            for col, i in zip(cols, ci):
                X_tokens[:, i] = df[col].astype(np.int64).values.clip(0, 1)
            print("token ∈ {0, 1}")
            continue

        if grp == GROUP_LOWCARD:
            for col, i in zip(cols, ci):
                enc, _ = pd.factorize(df[col])
                X_tokens[:, i] = enc.clip(0, 19)
            print("token ∈ [0, n_unique-1]")
            continue

        X_tr  = df.loc[idx_train, cols].values.astype(float)
        X_all = df[cols].values.astype(float)

        if grp == GROUP_ZERO_HEAVY:
            # Zero → token 0; non-zero → [1, scale_factor]
            zmask     = (X_all == 0)
            nz_tr     = X_tr.copy()
            col_means = np.where(np.nanmean(np.where(nz_tr == 0, np.nan, nz_tr), axis=0) > 0,
                                 np.nanmean(np.where(nz_tr == 0, np.nan, nz_tr), axis=0), 1)
            for k in range(nz_tr.shape[1]):
                nz_tr[nz_tr[:, k] == 0, k] = col_means[k]
            tokens = _qt_discretise(nz_tr, X_all, scale_factor - 1) + 1
            tokens[zmask] = 0
        else:
            tokens = _qt_discretise(X_tr, X_all, scale_factor)

        X_tokens[:, ci] = tokens
        print(f"token ∈ [{int(tokens.min())}, {int(tokens.max())}]")

    print(f"\n  Vocab size = {int(X_tokens.max()) + 1:,}  → use as NUM_BINS in start.py")
    return X_tokens


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def smart_process_and_tokenize(
    input_csv:    str,
    output_txt:   str,
    label_col:    str   = "activity",
    scale_factor: int   = DEFAULT_SCALE_FACTOR,
    report_path:  str   = "feature_groups_report.json",
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
    X_tokens  = phase3_scale_and_tokenize(df, features, groups, idx_train, scale_factor)

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
        input_csv    = "dataset_ready_for_model.csv",
        output_txt   = "dataset_tokenized.txt",
        label_col    = "activity",
        scale_factor = DEFAULT_SCALE_FACTOR,
        report_path  = "feature_groups_report.json",
    )
    print(f"→  NUM_BINS = {vocab_size}")
