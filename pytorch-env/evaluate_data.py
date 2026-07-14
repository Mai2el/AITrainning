"""
evaluate_data.py — Đánh giá dữ liệu thô (độc lập, KHÔNG tokenize/train).
─────────────────────────────────────────────────────────────────────────
Chỉ phụ thuộc numpy + pandas. Dùng để soi nhanh một dataset CSV:

  1. Tổng quan        : shape, số feature, số lớp, phân phối nhãn.
  2. Chất lượng       : NaN/Inf, dòng trùng lặp, cột hằng/zero-heavy.
  3. Profile quantile : percentile, skew, kurtosis, tail_ratio mỗi feature.
  4. Phân bố độ lệch  : tóm tắt feature đối xứng / lệch vừa / nặng / cực.

Cách chạy:
    python evaluate_data.py <duong_dan.csv> [ten_cot_nhan] [--save]

Ví dụ:
    python evaluate_data.py dataset_bin60.csv activity --save
"""
import sys
import numpy as np
import pandas as pd

_EPS = 1e-9
_PERCENTILES = [0, 1, 5, 25, 50, 75, 95, 99, 100]


def _line(title=""):
    print("=" * 64)
    if title:
        print(title)
        print("=" * 64)


def evaluate(csv_path: str, label_col: str = "activity", save: bool = False):
    print(f"\nĐang tải: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    if label_col not in df.columns:
        print(f"⚠️  Không thấy cột nhãn '{label_col}'. Các cột: {list(df.columns)[:8]}…")
        label_col = None

    feats = [c for c in df.columns if c != label_col]

    # ── 1. TỔNG QUAN ───────────────────────────────────────────
    _line("1. TỔNG QUAN")
    print(f"  Số dòng    : {len(df):,}")
    print(f"  Số feature : {len(feats)}")
    if label_col:
        vc = df[label_col].astype(str).str.strip().value_counts()
        print(f"  Số lớp     : {len(vc)}")
        print(f"\n  Phân phối nhãn:")
        for cls, cnt in vc.items():
            print(f"    {str(cls):28s} {cnt:>10,}  ({cnt/len(df)*100:5.2f}%)")
        imbalance = vc.max() / max(vc.min(), 1)
        print(f"  Tỷ lệ mất cân bằng (max/min): {imbalance:.1f}×")

    # Ép numeric để phân tích
    X = df[feats].apply(pd.to_numeric, errors="coerce")

    # ── 2. CHẤT LƯỢNG ──────────────────────────────────────────
    _line("2. CHẤT LƯỢNG DỮ LIỆU")
    n_nan = int(X.isna().sum().sum())
    n_inf = int(np.isinf(X.replace([np.inf, -np.inf], np.nan).fillna(0)).sum().sum())
    raw_inf = int(np.isinf(df[feats].apply(pd.to_numeric, errors="coerce")).sum().sum())
    print(f"  Ô NaN (sau ép numeric)      : {n_nan:,}")
    print(f"  Ô Inf (±vô cực)             : {raw_inf:,}")

    dup_full = int(df.duplicated().sum())
    dup_feat = int(df[feats].duplicated().sum())
    conflict = int(df.duplicated(subset=feats, keep=False).sum()
                   - df.duplicated(keep=False).sum())
    print(f"  Dòng trùng HOÀN TOÀN         : {dup_full:,} ({dup_full/len(df)*100:.2f}%)")
    print(f"  Dòng trùng FEATURE (bỏ nhãn) : {dup_feat:,}")
    print(f"  Feature giống, KHÁC nhãn     : {conflict:,}  (nhiễu nhãn)")

    Xf = X.fillna(0)
    const, near_const, zero_heavy = [], [], []
    for c in feats:
        v = Xf[c].values
        nun = np.unique(v).size
        if nun <= 1:
            const.append(c)
        elif pd.Series(v).value_counts(normalize=True).iloc[0] > 0.99:
            near_const.append(c)
        if (v == 0).mean() > 0.95:
            zero_heavy.append(c)
    print(f"  Cột hằng số (vô dụng)        : {len(const):2d}  {const[:5]}")
    print(f"  Cột gần-hằng (>99% 1 giá trị): {len(near_const):2d}  {near_const[:5]}")
    print(f"  Cột >95% giá trị 0           : {len(zero_heavy):2d}  {zero_heavy[:5]}")

    # ── 3. PROFILE QUANTILE per-feature ────────────────────────
    _line("3. PROFILE QUANTILE / ĐỘ LỆCH per-feature")
    rows = []
    for c in feats:
        v = Xf[c].values.astype(float)
        pc = np.percentile(v, _PERCENTILES)
        p = dict(zip([f"p{q}" for q in _PERCENTILES], pc))
        tail = (p["p99"] - p["p50"]) / (p["p50"] - p["p1"] + _EPS)
        s = pd.Series(v)
        rows.append({
            "feature": c,
            "min": v.min(), "p50": p["p50"], "max": v.max(),
            "mean": float(v.mean()), "std": float(v.std()),
            "skew": float(s.skew()), "kurtosis": float(s.kurtosis()),
            "tail_ratio": float(tail),
            "zero_%": float((v == 0).mean() * 100),
            "n_unique": int(np.unique(v).size),
            "has_neg": bool((v < 0).any()),
        })
    rep = pd.DataFrame(rows)
    rep["abs_skew"] = rep["skew"].abs()

    print(f"  {'feature':30s} {'skew':>8} {'kurt':>10} {'tail':>8} {'zero%':>6}")
    for _, r in rep.sort_values("abs_skew", ascending=False).head(12).iterrows():
        print(f"  {r['feature'][:30]:30s} {r['skew']:8.1f} {r['kurtosis']:10.0f} "
              f"{r['tail_ratio']:8.1f} {r['zero_%']:6.0f}")

    # ── 4. PHÂN BỐ ĐỘ LỆCH ─────────────────────────────────────
    _line("4. PHÂN BỐ ĐỘ LỆCH (|skew|)")
    a = rep["abs_skew"]
    print(f"  Đối xứng  |skew|<1   : {(a < 1).sum():3d} feature")
    print(f"  Lệch vừa  1–3        : {((a >= 1) & (a < 3)).sum():3d} feature")
    print(f"  Lệch nặng 3–10       : {((a >= 3) & (a < 10)).sum():3d} feature")
    print(f"  Lệch CỰC  |skew|>=10 : {(a >= 10).sum():3d} feature")
    print(f"  Kurtosis>50          : {(rep['kurtosis'] > 50).sum():3d} feature")
    print(f"  Có giá trị âm        : {rep['has_neg'].sum():3d} feature")

    if save:
        out = csv_path.rsplit(".", 1)[0] + "_eval_report.csv"
        rep.drop(columns=["abs_skew"]).to_csv(out, index=False)
        print(f"\n  → Đã lưu báo cáo per-feature: {out}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Dùng: python evaluate_data.py <duong_dan.csv> [ten_cot_nhan] [--save]")
        sys.exit(0)
    path  = sys.argv[1]
    label = "activity"
    save  = "--save" in sys.argv
    for a in sys.argv[2:]:
        if a != "--save":
            label = a
    evaluate(path, label, save)
