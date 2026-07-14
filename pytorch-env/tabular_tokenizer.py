"""
tabular_tokenizer.py  —  3-Phase Tabular Processing Pipeline
─────────────────────────────────────────────────────────────────────────────────────────
PHASE 1 — QUANTILE PROFILING  (fit on train split only)
  Percentiles: p0 p1 p5 p10 p25 p50 p75 p90 p95 p99 p100
  Derived indicators: iqr, tail_ratio, spread_ratio, zero_ratio, n_unique …

PHASE 2 — QUANTILE-BASED GROUPING
  Decision (in priority order):
    port     → col name contains 'port'
    binary   → unique ⊆ {0, 1}
    low_card → n_unique < 20
    skewed   → tail_ratio > 5  OR  spread_ratio < 0.05
               OR  left_ratio > 0.8  OR  left_ratio < 0.2
    normal   → everything else

PHASE 3 — SCALE → TOKENISE  (KHÔNG CLIP)
  Đã BỎ cơ chế clip. Cột liên tục VƯỢT QUÁ độ phân giải token (range>scale_factor
  hoặc n_unique>scale_factor) được nén đuôi ĐƠN ĐIỆU bằng piecewise sqrt (thân tuyến
  tính, chỉ nén phần > split) THAY cho clip; cbrt/quantile làm fallback cho đuôi rất
  nặng / lưỡng đỉnh. Cột nằm gọn trong ngân sách token → minmax tuyến tính không mất
  mát. Xem _fits_token_budget.

BÁO CÁO ĐÁNH GIÁ QUANTILE  → đã tách sang module `tokenizer_report.py`
  (export_feature_groups_report, export_post_scaling_report). File này chỉ lo tokenize.

  CHANGELOG:
    [REM-1] Bỏ GROUP_ZERO_HEAVY  → chuyển về SKEWED hoặc NORMAL theo tail_ratio
    [REM-2] Bỏ GROUP_ANOMALY     → chuyển về SKEWED (timestamps cực lớn bị clip trước)
    [NEW-1] Universal clip p0.1/p99.9 (train-fitted) áp dụng cho nhóm liên tục
            trước bất kỳ bước scale nào
    [REV-1] Clip + nén giờ CÓ ĐIỀU KIỆN qua _fits_token_budget: chỉ clip/nén cột
            vượt độ phân giải scale_factor (range>SF hoặc n_unique>SF). Cột nhỏ
            (vd bwd_ece_flag_counts: 9 giá trị nguyên ∈[0,10]) → minmax, không clip,
            không cbrt → giữ nguyên giá trị biên
    [FIX-1] GROUP_NORMAL dùng lo_pct=1, hi_pct=99 → tránh p0/p100 nhạy outlier
    [FIX-2] Zero check dùng np.abs < _EPS thay vì == 0
    [FIX-3] GROUP_LOWCARD OOV → token = len(train_vals) thay vì 0
    [REM-3] BỎ cơ chế CLIP (universal p0.1/p99.9 + clip nội bộ p1/p99 của linear).
            Cột vượt độ phân giải → PIECEWISE sqrt (thân linear, nén đuôi phải đơn
            điệu) thay clip; giữ cbrt/quantile làm fallback. `_apply_universal_clip`
            còn lại CHỈ phục vụ _eval_scaling.py, không nằm trong đường sản xuất.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split

from tokenizer_report import export_feature_groups_report, export_post_scaling_report

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────
_VAL_TEST_FRACTION         = 0.30
_TEST_WITHIN_TEMP_FRACTION = 0.50
_SPLIT_RANDOM_STATE        = 42
DEFAULT_SCALE_FACTOR       = 30_000   
_PORT_VOCAB                = 7
_EPS                       = 1e-9
_PERCENTILES               = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

# Ngưỡng clip universal (train-fitted)
_CLIP_LO_PCT = 0.1
_CLIP_HI_PCT = 99.9

# ── Group labels ───────────────────────────────────────────────
GROUP_BINARY  = "binary"
GROUP_PORT    = "port"
GROUP_LOWCARD = "low_card"
GROUP_SKEWED  = "skewed"
GROUP_NORMAL  = "normal"

_ALL_GROUPS = [GROUP_BINARY, GROUP_PORT, GROUP_LOWCARD, GROUP_SKEWED, GROUP_NORMAL]

# ── Per-feature continuous scaling strategies ──────────────────
# Mỗi feature liên tục được gán MỘT chiến lược scale phù hợp với hình dạng
# phân phối của nó. KHÔNG dùng log — thay bằng signed power-root (căn bậc 2 /
# căn bậc 3) để nén đuôi/outlier nhẹ nhàng hơn và GIỮ ĐƯỢC dấu (âm/dương).
#   signed power:  sign(x) * |x|^p   với  p = 1/2 (sqrt)  hoặc  1/3 (cbrt)
#   - cbrt (p=1/3): nén mạnh hơn sqrt → đuôi rất nặng / lưỡng cực biên độ lớn.
#   - sqrt (p=1/2): nén nhẹ          → đuôi vừa.
STRAT_LINEAR    = "robust_linear"    # phân phối tương đối đều, đuôi nhẹ
STRAT_QUANTILE  = "quantile_rank"    # rank-CDF → uniform [0,1] (robust nhất)
STRAT_SQRT      = "signed_sqrt"      # đuôi vừa → căn bậc 2 (có dấu)
STRAT_CBRT      = "signed_cbrt"      # đuôi rất nặng / lưỡng cực → căn bậc 3 (có dấu)
STRAT_PIECEWISE = "piecewise_sqrt"   # thân TUYẾN TÍNH, CHỈ nén đuôi phải (thay cho clip)

# Override CỤC BỘ: các cột IAT bị LƯỠNG ĐỈNH (nhiễm timestamp ~1.7e9 trộn với IAT
# nhỏ). minmax sẽ nén chúng về 2 token, cbrt nén mode thấp → cả hai đều hỏng.
# quantile trải đều theo rank, tách sạch 2 mode → đây là cách xử lý hợp lý nhất.
FORCE_QUANTILE_FEATURES = {
    "packets_IAT_mode",
    "packets_IAT_median",
    "fwd_packets_IAT_median",
    "fwd_packets_IAT_mean",
    "packet_IAT_total",
    "packet_IAT_max",
}


# ═══════════════════════════════════════════════════════════════
# PHASE 1 — QUANTILE PROFILING
# ═══════════════════════════════════════════════════════════════
def _profile_one(col_name: str, series: pd.Series) -> dict:
    v = series.dropna().values.astype(float)
    n = len(v)

    pct = {f"p{p}": (float(np.percentile(v, p)) if n > 0 else 0.0)
           for p in _PERCENTILES}

    p1, p25, p50, p75, p90, p99 = (
        pct["p1"], pct["p25"], pct["p50"],
        pct["p75"], pct["p90"], pct["p99"]
    )
    iqr           = p75 - p25
    body          = p99 - p1
    tail_ratio    = (p99 - p50) / (p50 - p1  + _EPS)
    tail_75_ratio = (p99 - p75) / (iqr        + _EPS)
    tail_90_ratio = (p99 - p90) / (p90 - p50  + _EPS)
    left_ratio    = (p50 - p1)  / (body        + _EPS)
    spread_ratio  = iqr          / (body        + _EPS)
    zero_ratio    = float((v == 0).mean()) if n > 0 else 0.0
    n_unique      = int(np.unique(v).size)

    # Clip bounds (dùng cho Phase 3)
    clip_lo = float(np.percentile(v, _CLIP_LO_PCT)) if n > 0 else 0.0
    clip_hi = float(np.percentile(v, _CLIP_HI_PCT)) if n > 0 else 0.0

    return {
        "feature":       col_name,
        "clip_lo":       round(clip_lo, 6),   # [NEW-1]
        "clip_hi":       round(clip_hi, 6),   # [NEW-1]
        **{k: round(val, 6) for k, val in pct.items()},
        "iqr":           round(iqr,           6),
        "tail_ratio":    round(tail_ratio,    4),
        "tail_75_ratio": round(tail_75_ratio, 4),
        "tail_90_ratio": round(tail_90_ratio, 4),
        "left_ratio":    round(left_ratio,    4),
        "spread_ratio":  round(spread_ratio,  4),
        "zero_ratio":    round(zero_ratio,    4),
        "n_unique":      n_unique,
        "mean":          round(float(np.mean(v)),  6) if n > 0 else 0.0,
        "std":           round(float(np.std(v)),   6) if n > 0 else 0.0,
        "skew":          round(float(pd.Series(v).skew()),     4) if n > 2 else 0.0,
        "kurtosis":      round(float(pd.Series(v).kurtosis()), 4) if n > 3 else 0.0,
        "cv":            round(float(np.std(v) / (np.mean(v) + _EPS)), 4) if n > 0 else 0.0,
        "is_binary":     set(np.unique(v)).issubset({0.0, 1.0}),
        "is_low_card":   n_unique < 20,
        "is_port":       "port" in col_name.lower(),
    }


def phase1_profile(df: pd.DataFrame,
                   features: list,
                   train_idx: np.ndarray) -> dict:
    print(f"\n[Phase 1] Quantile-profiling {len(features)} features "
          f"({len(train_idx):,} train rows) …")
    profiles = {col: _profile_one(col, df.loc[train_idx, col]) for col in features}
    print(f"          Percentiles : {_PERCENTILES}")
    print(f"          Clip bounds : p{_CLIP_LO_PCT} – p{_CLIP_HI_PCT}  (train-fitted)")
    return profiles


# ═══════════════════════════════════════════════════════════════
# PHASE 2 — GROUPING
# ═══════════════════════════════════════════════════════════════
_GROUP_CRITERIA = {
    GROUP_BINARY:  "unique ⊆ {0,1}",
    GROUP_PORT:    "col name contains 'port'",
    GROUP_LOWCARD: "n_unique < 20",
    GROUP_SKEWED:  "tail_ratio > 5  OR  spread_ratio < 0.05  OR  left_ratio extreme",
    GROUP_NORMAL:  "everything else (well-behaved distribution)",
}


def _assign_group(p: dict) -> tuple:
    if p["is_port"]:
        return GROUP_PORT, "col name contains 'port' → hand-crafted bucket (0-6)"

    if p["is_binary"]:
        return GROUP_BINARY, f"unique⊆{{0,1}}, n_unique={p['n_unique']}"

    if p["is_low_card"]:
        return GROUP_LOWCARD, f"n_unique={p['n_unique']} < 20 → label-encode"

    tr, sr = p["tail_ratio"], p["spread_ratio"]
    lr     = p["left_ratio"]

    if tr > 5.0 or sr < 0.05 or lr > 0.8 or lr < 0.2:
        return (GROUP_SKEWED,
                f"tail_ratio={tr:.2f}, spread_ratio={sr:.4f}, left_ratio={lr:.3f} "
                f"→ piecewise sqrt / symmetric log")

    return (GROUP_NORMAL,
            f"well-behaved (tail_ratio={tr:.2f}, spread_ratio={sr:.3f}) → robust minmax")


# Ngưỡng phân giải tối thiểu: thân (giữa 80% data) phải trải >= bao nhiêu token
# dưới minmax thì coi là "đủ phân giải, không cần nén". Phụ thuộc scale_factor.
_MIN_BULK_RES = 800


def _fits_token_budget(p: dict, scale_factor: int = DEFAULT_SCALE_FACTOR) -> bool:
    """
    Feature có "nằm gọn" trong ngân sách token (độ phân giải = scale_factor) không?

    Một feature VƯỢT QUÁ độ phân giải scale_factor khi:
      • dải giá trị (p100 − p0) > scale_factor   → minmax buộc nhiều giá trị nguyên
        liền kề rơi vào CÙNG token (va chạm), HOẶC
      • n_unique > scale_factor                  → nhiều giá trị rời rạc hơn số token
        có sẵn (pigeonhole ⇒ chắc chắn va chạm).

    Khi CẢ HAI điều kiện đều ≤ scale_factor → minmax tuyến tính tách được MỌI giá
    trị mà không va chạm → KHÔNG cần clip, KHÔNG cần nén (cbrt/sqrt chỉ bóp méo
    khoảng cách độ lớn vô ích). Đây là predicate dùng chung cho cả quyết định clip
    (Phase 3) lẫn quyết định strategy.
    """
    full_range = p["p100"] - p["p0"]
    n_unique   = p.get("n_unique", 0)
    return full_range <= scale_factor and n_unique <= scale_factor


# Ngưỡng "đuôi phải THỰC SỰ nặng": độ lớn của top 1% (p99→p100) phải vượt
# _HEAVY_TAIL_RATIO lần độ rộng thân (p1→p99). Dưới ngưỡng = không có đuôi độ lớn
# (vd cột tỉ lệ ∈[0,1]: "đuôi" chỉ là giá trị rời rạc tới 1.0) → KHÔNG nén piecewise.
_HEAVY_TAIL_RATIO = 2.0


def _has_heavy_tail(p: dict) -> bool:
    """
    Cột có đuôi PHẢI nặng thật không? Chỉ khi đúng vậy mới đáng nén bằng piecewise/cbrt.
    Cột bị gán nhóm 'skewed' do tail_ratio = (p99-p50)/(p50-p1) bùng nổ khi median≈p1≈0
    (vd flag-percentage 81% zero) NHƯNG range chỉ [0,1] → không có đuôi độ lớn → False,
    để rơi về min-max tuyến tính thay vì bị co nối tiếp vô ích.
    """
    tail_span = p["p100"] - p["p99"]
    body_span = p["p99"]  - p["p1"]
    if body_span <= _EPS:
        return tail_span > _EPS            # thân dẹt nhưng vẫn có đuôi độ lớn
    return (tail_span / body_span) >= _HEAVY_TAIL_RATIO


def _body_resolution(tokens: np.ndarray) -> int:
    """
    Độ phân giải THÂN thực tế: số token PHÂN BIỆT mà khối giữa (p5–p95 theo hàng)
    chiếm dụng sau khi scale. Đây là thước đo trực tiếp — đếm trên token đã sinh,
    KHÔNG suy ra từ tỉ lệ phân vị (vì phân vị bị đánh lừa bởi dữ liệu lưỡng đỉnh /
    nhiễm timestamp: p25..p75 có thể trải rất rộng nhưng chỉ chứa 2 cụm điểm).
    """
    t = np.asarray(tokens).flatten()
    if t.size == 0:
        return 0
    lo, hi = np.percentile(t, 5), np.percentile(t, 95)
    mid = t[(t >= lo) & (t <= hi)]
    return int(np.unique(mid).size) if mid.size else 0


def _select_strategy_percentile(p: dict, scale_factor: int) -> str:
    """Fallback KHI KHÔNG có dữ liệu cột (chỉ có profile): dùng bulk_res phân vị.
    Kém chính xác với lưỡng đỉnh — chỉ dùng cho báo cáo nhanh không kèm dữ liệu."""
    p0, p99 = p["p0"], p["p99"]
    skew    = abs(p.get("skew", 0.0))
    tail    = p.get("tail_ratio", 0.0)
    bipolar = p0 < 0 and abs(p0) > 0.05 * (abs(p99) + _EPS)
    bulk_res = (p["p90"] - p["p10"]) / (p99 - p["p1"] + _EPS) * scale_factor
    if bulk_res >= _MIN_BULK_RES:
        return STRAT_LINEAR
    if bipolar or skew > 4.0 or tail > 6.0:
        return STRAT_CBRT
    return STRAT_PIECEWISE if _has_heavy_tail(p) else STRAT_LINEAR


def _select_continuous_strategy(p: dict, scale_factor: int = DEFAULT_SCALE_FACTOR,
                                col_train: np.ndarray = None) -> str:
    """
    Chọn chiến lược scale. Thứ tự quyết định:
      1. FORCE_QUANTILE_FEATURES → quantile (ép tay).
      2. _fits_token_budget → robust_linear: cả dải lẫn n_unique đều nằm trong ngân
         sách token nên minmax tách sạch mọi giá trị, KHÔNG mất mát, KHÔNG clip.
      3. Còn lại (vượt độ phân giải) → ĐO THỰC NGHIỆM trên train (nếu có col_train),
         KHÔNG clip; áp lần lượt linear → piecewise → cbrt → quantile, đo
         _body_resolution, chọn cái ĐẦU TIÊN đạt >= _MIN_BULK_RES theo thứ tự ưu tiên
         GIỮ ĐỘ LỚN:
           • robust_linear  — giữ đúng tỉ lệ độ lớn (tốt nhất cho bộ mã hoá đơn điệu)
           • piecewise_sqrt — thân tuyến tính, CHỈ nén đuôi phải (thay clip; giữ thân)
           • signed_cbrt    — đuôi rất nặng đơn-mode: nén cả thân, vẫn giữ thứ tự
           • quantile_rank  — lưỡng đỉnh/nhiễm timestamp: chỉ rank mới cứu được thân
         Không có col_train → fallback bulk_res phân vị (_select_strategy_percentile).
    """
    if p.get("feature") in FORCE_QUANTILE_FEATURES:
        return STRAT_QUANTILE

    # [RANGE-GUARD] Nằm gọn trong ngân sách token → minmax đủ, không cần nén
    if _fits_token_budget(p, scale_factor):
        return STRAT_LINEAR

    if col_train is None:
        return _select_strategy_percentile(p, scale_factor)

    # ── Đo thực nghiệm trên train (KHÔNG clip — đuôi do piecewise xử lý) ─────
    ctr = np.asarray(col_train, dtype=float).reshape(-1, 1)

    # Ưu tiên GIỮ ĐỘ LỚN: linear > piecewise > cbrt > quantile. Dùng cái ĐẦU TIÊN
    # đạt đủ phân giải thân (khỏi thử bước nén mạnh hơn → giữ magnitude tối đa).
    lin_res = _body_resolution(
        _robust_linear_scale(ctr, ctr, scale_factor, lo_pct=0, hi_pct=100))
    if lin_res >= _MIN_BULK_RES:
        return STRAT_LINEAR

    # PIECEWISE chỉ đủ điều kiện khi đuôi phải THỰC SỰ nặng (top 1% trải độ lớn lớn).
    # Cột range hẹp/tỉ lệ (vd flag-percentage [0,1]) → không nén nối tiếp, dùng minmax.
    heavy = _has_heavy_tail(p)
    pw_res = 0
    if heavy:
        pw_res = _body_resolution(_piecewise_sqrt_scale(ctr, ctr, scale_factor))
        if pw_res >= _MIN_BULK_RES:
            return STRAT_PIECEWISE
    cb_res = _body_resolution(
        _signed_power_scale(ctr, ctr, scale_factor, 1.0 / 3.0))
    if cb_res >= _MIN_BULK_RES:
        return STRAT_CBRT

    # Tất cả đều yếu thân → so thêm quantile, CHỌN ĐỘ PHÂN GIẢI CAO NHẤT
    # (không rơi mù quáng vào quantile nếu nó còn tệ hơn — xem bwd_init_win_bytes).
    qt_res = _body_resolution(_quantile_scale(ctr, ctr, scale_factor))
    cands  = [(STRAT_LINEAR, lin_res), (STRAT_CBRT, cb_res), (STRAT_QUANTILE, qt_res)]
    if heavy:                                   # chỉ đưa piecewise vào đua khi đuôi nặng
        cands.append((STRAT_PIECEWISE, pw_res))
    best   = max(r for _, r in cands)
    pref   = {STRAT_LINEAR: 0, STRAT_PIECEWISE: 1, STRAT_CBRT: 2, STRAT_QUANTILE: 3}
    # Trong số ứng viên đạt >= 95% best, chọn cái GIỮ ĐỘ LỚN nhất
    ok = [s for s, r in cands if r >= 0.95 * best]
    return min(ok, key=lambda s: pref[s])


def phase2_group(profiles: dict,
                 fixed_normal_features: list = None,
                 report_path: str = "feature_groups_report.json",
                 scale_factor: int = DEFAULT_SCALE_FACTOR,
                 df: pd.DataFrame = None,
                 idx_train: np.ndarray = None) -> tuple:
    """
    Trả về (groups, strategies). Nếu truyền df + idx_train, chiến lược cho nhóm
    liên tục được CHỌN THỰC NGHIỆM (đo _body_resolution trên train); nếu không,
    fallback bulk_res phân vị. `strategies` dùng lại y nguyên ở Phase 3 để báo cáo
    và tokenize KHỚP nhau.
    """
    print(f"\n[Phase 2] Grouping → '{report_path}' …")

    fixed_normal_set = set(fixed_normal_features) if fixed_normal_features else set()
    if fixed_normal_set:
        print(f"      ℹ️  {len(fixed_normal_set)} features forced → NORMAL")
    empirical = df is not None and idx_train is not None
    print(f"      Strategy: {'THỰC NGHIỆM (đo body-resolution trên train)' if empirical else 'phân vị (fallback)'}")

    groups     = {g: [] for g in _ALL_GROUPS}
    strategies = {}
    rows       = []

    for col, p in profiles.items():
        if col in fixed_normal_set:
            grp, reason = GROUP_NORMAL, "Manually forced to normal group"
        else:
            grp, reason = _assign_group(p)

        # Chiến lược scale per-feature (chỉ áp cho group liên tục).
        if grp in (GROUP_NORMAL, GROUP_SKEWED):
            col_train = (df.loc[idx_train, col].values.astype(float)
                         if empirical else None)
            strategy = _select_continuous_strategy(p, scale_factor, col_train)
        else:
            strategy = grp
        strategies[col] = strategy

        # Đánh dấu piecewise + giá trị đuôi SAU biến đổi (s + sqrt(p100 − s)):
        #   pw_split      = điểm split (p99) — nơi thân tuyến tính kết thúc
        #   pw_tail_after = đỉnh đuôi sau khi nén sqrt (so với p100 gốc để thấy mức nén)
        heavy = _has_heavy_tail(p)
        is_pw = (strategy == STRAT_PIECEWISE)
        if is_pw:
            s             = float(p["p99"])
            pw_split      = round(s, 6)
            pw_tail_after = round(float(s + np.sqrt(max(p["p100"] - s, 0.0))), 6)
        else:
            pw_split = pw_tail_after = None

        groups[grp].append(col)
        rows.append({
            "feature": col, "group": grp, "strategy": strategy,
            "heavy_tail": heavy, "piecewise": is_pw,
            "pw_split": pw_split, "pw_tail_after": pw_tail_after,
            "reason": reason,
            "scaler": _SCALER_LABELS.get(grp, "unknown"),
            **{k: p[k] for k in [
                "p0","p1","p5","p10","p25","p50","p75","p90","p95","p99","p100",
                "iqr","tail_ratio","tail_75_ratio","tail_90_ratio",
                "left_ratio","spread_ratio","zero_ratio","n_unique",
                "mean","std","skew","kurtosis","cv",
            ]},
        })

    # Báo cáo profile quantile + grouping (tách sang tokenizer_report.py)
    export_feature_groups_report(rows, report_path)

    W = 68
    print(f"\n  {'─'*W}")
    print(f"  {'Feature Group Summary':^{W}}")
    print(f"  {'─'*W}")
    print(f"  {'Group':<12}  {'N':>4}  {'Criteria':<38}  Example")
    print(f"  {'─'*W}")
    for grp in _ALL_GROUPS:
        cols = groups[grp]
        if not cols:
            continue
        ex = ", ".join(cols[:2]) + ("…" if len(cols) > 2 else "")
        print(f"  {grp:<12}  {len(cols):>4}  {_GROUP_CRITERIA[grp]:<38}  {ex}")
    print(f"  {'─'*W}")
    print(f"\n  Report → {Path(report_path).resolve()}")
    return groups, strategies


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
# PHASE 3 — HELPER SCALERS
# ═══════════════════════════════════════════════════════════════

def _apply_universal_clip(X_train: np.ndarray,
                          X_all: np.ndarray,
                          profiles_list: list,
                          scale_factor: int = DEFAULT_SCALE_FACTOR) -> tuple:
    """
    [NEW-1][REV] Clip CÓ ĐIỀU KIỆN: chỉ clip cột VƯỢT QUÁ độ phân giải token
    (scale_factor) về [clip_lo, clip_hi] (p0.1/p99.9 fit từ train). Cột đã nằm gọn
    trong ngân sách token (_fits_token_budget) thì GIỮ NGUYÊN min/max — không clip,
    để không phá huỷ giá trị biên hợp lệ (vd count=10 trước đây bị cắt còn 2).
    Trả về (X_train_clipped, X_all_clipped).
    """
    lo = np.array([p["clip_lo"] for p in profiles_list], dtype=float)
    hi = np.array([p["clip_hi"] for p in profiles_list], dtype=float)

    # Cột nằm gọn trong ngân sách token → bỏ clip (đặt biên = min/max thật)
    for j, p in enumerate(profiles_list):
        if _fits_token_budget(p, scale_factor):
            lo[j] = p["p0"]
            hi[j] = p["p100"]

    # Nếu lo == hi (cột hằng số), mở rộng nhỏ để tránh chia 0
    degenerate = (hi - lo) < _EPS
    hi[degenerate] = lo[degenerate] + _EPS

    X_tr_c  = np.clip(X_train, lo, hi)
    X_all_c = np.clip(X_all,   lo, hi)
    return X_tr_c, X_all_c


def _robust_linear_scale(X_train, X_all, scale_factor,
                         lo_pct=1, hi_pct=99) -> np.ndarray:
    """[FIX-1] lo_pct=1, hi_pct=99 để tránh p0/p100 nhạy outlier"""
    lo    = np.percentile(X_train, lo_pct, axis=0)
    hi    = np.percentile(X_train, hi_pct, axis=0)
    denom = hi - lo
    denom[denom < _EPS] = _EPS
    normed = (np.clip(X_all, lo, hi) - lo) / denom
    return np.clip(np.round(normed * scale_factor).astype(np.int64),
                   0, scale_factor)


def _piecewise_sqrt_scale(X_train, X_all, scale_factor,
                          split_pcts=None) -> np.ndarray:
    """
    Shifted Square Root (THAY cho clip — nén đuôi phải ĐƠN ĐIỆU, giữ nguyên thân):
      x ≤ s  →  tuyến tính (thân giữ đúng tỉ lệ độ lớn)
      x > s  →  s + sqrt(x - s)   (đuôi phải bị nén nhưng VẪN tăng đơn điệu, không cắt
                                   phẳng như clip → outlier vẫn phân biệt được)
    Neo đáy l = min(train) (KHÔNG floor p1) để không clip phần dưới. split mặc định p99.
    """
    n_cols = X_train.shape[1]
    if split_pcts is None:
        split_pcts = [99] * n_cols

    result = np.zeros_like(X_all, dtype=np.float64)

    for j in range(n_cols):
        col_tr  = X_train[:, j].astype(np.float64)
        col_all = X_all[:,   j].astype(np.float64)

        s = float(np.percentile(col_tr, split_pcts[j]))
        l = float(np.min(col_tr))
        if s - l < _EPS:
            s = l + _EPS

        def _transform(c):
            t = c.copy()
            mask = t > s
            t[mask] = s + np.sqrt(t[mask] - s)
            return t

        tr_t  = _transform(col_tr)
        all_t = _transform(col_all)

        m     = float(np.max(tr_t))
        denom = m - l
        if denom < _EPS:
            denom = _EPS

        normed = (all_t - l) / denom
        result[:, j] = np.clip(normed * scale_factor, 0.0, scale_factor)

    return np.clip(np.round(result).astype(np.int64), 0, scale_factor)


def _signed_power_scale(X_train, X_all, scale_factor, power: float) -> np.ndarray:
    """
    Signed power-root transform:  t = sign(x) * |x|^power   (KHÔNG dùng log).
      - power = 1/2 → căn bậc 2 (nén nhẹ đuôi).
      - power = 1/3 → căn bậc 3 (nén mạnh hơn; xử lý tốt cả giá trị âm lớn).
    Giữ dấu nên áp được cho dữ liệu lưỡng cực. Fit min/max trên TRAIN rồi
    min-max về [0, scale_factor].
    """
    n_cols = X_train.shape[1]
    result = np.zeros_like(X_all, dtype=np.float64)

    for j in range(n_cols):
        col_tr  = X_train[:, j].astype(np.float64)
        col_all = X_all[:,   j].astype(np.float64)

        tr_t  = np.sign(col_tr)  * np.abs(col_tr)  ** power
        all_t = np.sign(col_all) * np.abs(col_all) ** power

        l, m  = float(tr_t.min()), float(tr_t.max())
        denom = m - l if (m - l) > _EPS else _EPS

        normed = (all_t - l) / denom
        result[:, j] = np.clip(normed * scale_factor, 0.0, scale_factor)

    return np.clip(np.round(result).astype(np.int64), 0, scale_factor)


def _quantile_scale(X_train, X_all, scale_factor) -> np.ndarray:
    """
    Rank / CDF transform (fit trên train) → uniform [0,1].
    Đơn điệu, miễn nhiễm outlier, trải token đều nhất → tốt nhất cho feature
    lành tính. Dùng QuantileTransformer của sklearn theo từng cột.
    """
    n_cols = X_train.shape[1]
    result = np.zeros_like(X_all, dtype=np.float64)

    for j in range(n_cols):
        col_tr  = X_train[:, j].astype(np.float64).reshape(-1, 1)
        col_all = X_all[:,   j].astype(np.float64).reshape(-1, 1)

        n_q = int(min(1000, max(10, np.unique(col_tr).size)))
        qt  = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution="uniform",
            subsample=1_000_000_000,
            random_state=_SPLIT_RANDOM_STATE,
        )
        qt.fit(col_tr)
        normed = qt.transform(col_all).flatten()          # ∈ [0, 1]
        result[:, j] = np.clip(normed * scale_factor, 0.0, scale_factor)

    return np.clip(np.round(result).astype(np.int64), 0, scale_factor)


_SCALER_LABELS = {
    GROUP_BINARY:  "keep {0,1}",
    GROUP_PORT:    "bucket map (0-6)",
    GROUP_LOWCARD: "factorize (ordinal)",
    GROUP_SKEWED:  "no clip → per-feat (piecewise/cbrt/quantile/minmax)",
    GROUP_NORMAL:  "no clip → per-feat (minmax/piecewise/cbrt)",
}


# ═══════════════════════════════════════════════════════════════
# PHASE 3 — MAIN
# ═══════════════════════════════════════════════════════════════
def phase3_scale_and_tokenize(
    df, features, groups, idx_train, profiles,
    scale_factor: int = DEFAULT_SCALE_FACTOR,
    strategies: dict = None,
) -> np.ndarray:
    print(f"\n[Phase 3] Scaling & tokenising  (scale_factor={scale_factor:,}) …")
    print(f"  Clip: ĐÃ BỎ — đuôi nén ĐƠN ĐIỆU bằng piecewise sqrt cho cột vượt độ phân giải {scale_factor:,} (range>SF hoặc n_unique>SF)")
    print(f"  ┌{'─'*62}┐")
    print(f"  │ {'Group':<12} │ {'Strategy':<42} │ {'N':>3} │")
    print(f"  ├{'─'*62}┤")
    for grp in _ALL_GROUPS:
        if groups.get(grp):
            print(f"  │ {grp:<12} │ {_SCALER_LABELS[grp]:<42} │ {len(groups[grp]):>3} │")
    print(f"  └{'─'*62}┘")

    n, F     = len(df), len(features)
    feat_idx = {col: i for i, col in enumerate(features)}
    X_tokens = np.zeros((n, F), dtype=np.int64)

    for grp, cols in groups.items():
        if not cols:
            continue
        ci = [feat_idx[c] for c in cols]
        print(f"\n  [{grp:<12}] {len(cols):>3} cols  ({_SCALER_LABELS.get(grp,'?')})", end="")

        # ── Categorical / discrete groups ──────────────────────
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

        if grp == GROUP_LOWCARD:
            for col, i in zip(cols, ci):
                train_vals = sorted(df.loc[idx_train, col].dropna().unique())
                val_map    = {v: k for k, v in enumerate(train_vals)}
                # [FIX-3] OOV → len(train_vals), không phải 0
                oov_token  = len(train_vals)
                X_tokens[:, i] = (
                    df[col].map(val_map).fillna(oov_token)
                    .astype(np.int64).values
                )
            print("  →  token ∈ [0, n_unique]  (OOV = n_unique)")
            continue

        # ── Continuous groups ─────────────────────────────────
        X_tr  = df.loc[idx_train, cols].values.astype(float)
        X_all = df[cols].values.astype(float)
        profs = [profiles[c] for c in cols]

        # KHÔNG clip nữa: đuôi được nén đơn điệu bằng piecewise sqrt (per-feature).
        X_tr_c, X_all_c = X_tr, X_all

        # ── Chọn chiến lược scale theo TỪNG feature (per-feature dispatch) ──
        tokens      = np.zeros((n, len(cols)), dtype=np.int64)
        strat_count = {}
        for k, (col, p) in enumerate(zip(cols, profs)):
            # Dùng strategy ĐÃ CHỌN ở Phase 2 (khớp báo cáo); nếu thiếu thì chọn
            # thực nghiệm tại chỗ trên train (không clip).
            if strategies is not None and col in strategies:
                strat = strategies[col]
            else:
                strat = _select_continuous_strategy(
                    p, scale_factor, X_tr_c[:, k])
            strat_count[strat] = strat_count.get(strat, 0) + 1

            col_tr_2d  = X_tr_c[:,  k].reshape(-1, 1)
            col_all_2d = X_all_c[:, k].reshape(-1, 1)

            if strat == STRAT_CBRT:
                tk = _signed_power_scale(col_tr_2d, col_all_2d, scale_factor, 1.0 / 3.0)
            elif strat == STRAT_SQRT:
                tk = _signed_power_scale(col_tr_2d, col_all_2d, scale_factor, 1.0 / 2.0)
            elif strat == STRAT_PIECEWISE:
                tk = _piecewise_sqrt_scale(col_tr_2d, col_all_2d, scale_factor)
            elif strat == STRAT_QUANTILE:
                tk = _quantile_scale(col_tr_2d, col_all_2d, scale_factor)
            else:  # STRAT_LINEAR — min-max thuần (không clip; full min/max 0–100)
                tk = _robust_linear_scale(col_tr_2d, col_all_2d, scale_factor,
                                          lo_pct=0, hi_pct=100)
            tokens[:, k] = tk.flatten()

        X_tokens[:, ci] = tokens
        strat_str = ", ".join(f"{s}×{c}" for s, c in sorted(strat_count.items()))
        print(f"  →  token ∈ [{int(tokens.min())}, {int(tokens.max())}]  "
              f"({strat_str})")

    vocab = int(X_tokens.max()) + 1
    print(f"\n  {'─'*52}")
    print(f"  Vocab size = {vocab:,}  → dùng làm NUM_BINS trong start.py")
    return X_tokens


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def smart_process_and_tokenize(
    input_csv:    str,
    output_txt:   str,
    label_col:    str   = "activity",
    scale_factor: int   = DEFAULT_SCALE_FACTOR,
    fixed_normal_features: list = None,
    report_path:  str   = "feature_groups_report.json",
    post_scaling_report_path: str = "post_scaling_report.json",
    val_test_fraction:         float = _VAL_TEST_FRACTION,
    test_within_temp_fraction: float = _TEST_WITHIN_TEMP_FRACTION,
    random_state:              int   = _SPLIT_RANDOM_STATE,
) -> int:
    t0 = time.time()
    print(f"\n{'='*62}\n  Tabular Tokenizer  |  scale_factor={scale_factor:,}\n{'='*62}")

    df = pd.read_csv(input_csv, low_memory=False)
    df.columns = df.columns.str.strip()
    df.drop(columns=[c for c in [
        "label","flow_id","timestamp","src_ip","dst_ip","src_port",
        "active_std","active_variance",
    ] if c in df.columns], inplace=True)

    for col in df.columns:
        if col != label_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    print(f"[Load]  shape={df.shape}  label='{label_col}'")

    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].values)
    print(f"        Labels: {list(le.classes_)}")

    features = [c for c in df.columns if c != label_col]
    n        = len(df)

    idx_train, idx_temp = train_test_split(
        np.arange(n), test_size=val_test_fraction,
        stratify=y, random_state=random_state)
    idx_val, idx_test   = train_test_split(
        idx_temp, test_size=test_within_temp_fraction,
        stratify=y[idx_temp], random_state=random_state)
    print(f"        train={len(idx_train):,}  val={len(idx_val):,}  test={len(idx_test):,}")

    profiles = phase1_profile(df, features, idx_train)
    groups, strategies = phase2_group(profiles,
                            fixed_normal_features=fixed_normal_features,
                            report_path=report_path,
                            scale_factor=scale_factor,
                            df=df, idx_train=idx_train)
    X_tokens = phase3_scale_and_tokenize(df, features, groups,
                                         idx_train, profiles, scale_factor,
                                         strategies=strategies)

    export_post_scaling_report(X_tokens, features, groups, scale_factor,
                               continuous_groups={GROUP_NORMAL, GROUP_SKEWED},
                               report_path=post_scaling_report_path,
                               strategies=strategies)

    df_out           = pd.DataFrame(X_tokens, columns=features)
    df_out[label_col] = df[label_col].values
    df_out.to_csv(output_txt, sep="\t", index=False)

    vocab_size = int(X_tokens.max()) + 1
    print(f"\n{'='*62}")
    print(f"  ✅  Done in {time.time()-t0:.1f}s")
    print(f"  Rows={n:,}  Features={len(features)}  Vocab={vocab_size:,}")
    print(f"  Output : {Path(output_txt).resolve()}")
    print(f"  Report : {Path(report_path).resolve()}")
    print(f"{'='*62}\n")
    return vocab_size


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    my_fixed_features = [
        "std_fwd_header_bytes_delta_len", "std_bwd_packets_delta_time",
        "fwd_max_header_bytes", "fwd_median_header_bytes",
        "fwd_min_header_bytes", "fwd_mode_header_bytes",
        "max_header_bytes", "median_header_bytes",
        "min_header_bytes", "mode_header_bytes",
        "delta_start", "syn_flag_percentage_in_total",
        "mean_header_bytes", "fwd_init_win_bytes",
        "bwd_max_header_bytes", "bwd_ece_flag_percentage_in_total",
        "bwd_ece_flag_counts", "bwd_packets_IAT_median",
        "bwd_payload_bytes_max", "bwd_psh_flag_percentage_in_bwd_packets",
        "ece_flag_percentage_in_total", "fwd_cov_header_bytes",
        "fwd_ece_flag_counts", "fwd_psh_flag_percentage_in_fwd_packets",
    ]

    vocab_size = smart_process_and_tokenize(
        input_csv              = "dataset_bin60_noIAT.csv",
        output_txt             = "dataset_tokenized.txt",
        label_col              = "activity",
        scale_factor           = DEFAULT_SCALE_FACTOR,
        fixed_normal_features  = my_fixed_features,
        report_path            = "feature_groups_report.json",
    )
    print(f"→  NUM_BINS = {vocab_size}")