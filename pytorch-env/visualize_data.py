"""
visualize_data.py — Trực quan hóa báo cáo đánh giá dữ liệu (độc lập).
─────────────────────────────────────────────────────────────────────────
Vẽ dashboard PNG 6 panel từ một dataset CSV:
  1. Phân phối nhãn (bar, log-scale)
  2. Phân bố độ lệch |skew| (bar buckets)
  3. Top-15 feature lệch nặng nhất (|skew|, horizontal bar)
  4. Phân bố tỷ lệ zero per-feature (histogram)
  5. Phân bố kurtosis (histogram)
  6. Tóm tắt chất lượng (NaN/Inf/trùng/mâu thuẫn — text panel)

Cách chạy:
    python visualize_data.py <duong_dan.csv> [ten_cot_nhan]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_EPS = 1e-9


def build_stats(csv_path, label_col):
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    if label_col not in df.columns:
        label_col = None
    feats = [c for c in df.columns if c != label_col]
    X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # quality
    n_nan = int(X.isna().sum().sum())
    n_inf = int(np.isinf(df[feats].apply(pd.to_numeric, errors="coerce")).sum().sum())
    dup_full = int(df.duplicated().sum())
    conflict = int(df.duplicated(subset=feats, keep=False).sum()
                   - df.duplicated(keep=False).sum())
    Xf = X.fillna(0)

    rows = []
    for c in feats:
        v = Xf[c].values.astype(float)
        s = pd.Series(v)
        rows.append({"feature": c, "skew": float(s.skew()),
                     "kurtosis": float(s.kurtosis()),
                     "zero_pct": float((v == 0).mean() * 100)})
    rep = pd.DataFrame(rows)
    rep["abs_skew"] = rep["skew"].abs()

    vc = (df[label_col].astype(str).str.strip().value_counts()
          if label_col else pd.Series(dtype=int))
    quality = {"rows": len(df), "feats": len(feats), "classes": len(vc),
               "nan": n_nan, "inf": n_inf, "dup_full": dup_full,
               "conflict": conflict}
    return rep, vc, quality, csv_path


def plot(rep, vc, quality, csv_path):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Đánh giá dữ liệu — {csv_path}", fontsize=15, fontweight="bold")

    # 1. Phân phối nhãn (log)
    if len(vc):
        top = vc.head(20)
        ax[0, 0].barh(range(len(top)), top.values, color="#4C72B0")
        ax[0, 0].set_yticks(range(len(top)))
        ax[0, 0].set_yticklabels([str(x)[:22] for x in top.index], fontsize=8)
        ax[0, 0].invert_yaxis(); ax[0, 0].set_xscale("log")
        ax[0, 0].set_title(f"Phân phối nhãn ({quality['classes']} lớp, log)")
        ax[0, 0].set_xlabel("Số mẫu")
    else:
        ax[0, 0].axis("off")

    # 2. Phân bố |skew| buckets
    a = rep["abs_skew"]
    buckets = [(a < 1).sum(), ((a >= 1) & (a < 3)).sum(),
               ((a >= 3) & (a < 10)).sum(), (a >= 10).sum()]
    labels = ["<1\nđối xứng", "1–3\nvừa", "3–10\nnặng", "≥10\ncực"]
    colors = ["#55A868", "#DDCC77", "#E1812C", "#C44E52"]
    ax[0, 1].bar(labels, buckets, color=colors)
    for i, b in enumerate(buckets):
        ax[0, 1].text(i, b, str(int(b)), ha="center", va="bottom", fontweight="bold")
    ax[0, 1].set_title("Phân bố độ lệch |skew|")
    ax[0, 1].set_ylabel("Số feature")

    # 3. Top-15 feature lệch nặng
    t = rep.sort_values("abs_skew", ascending=False).head(15)
    ax[0, 2].barh(range(len(t)), t["abs_skew"].values, color="#C44E52")
    ax[0, 2].set_yticks(range(len(t)))
    ax[0, 2].set_yticklabels([f[:24] for f in t["feature"]], fontsize=8)
    ax[0, 2].invert_yaxis()
    ax[0, 2].set_title("Top-15 feature lệch nặng nhất (|skew|)")
    ax[0, 2].set_xlabel("|skew|")

    # 4. Phân bố zero%
    ax[1, 0].hist(rep["zero_pct"], bins=20, color="#8172B3", edgecolor="white")
    ax[1, 0].set_title("Phân bố tỷ lệ giá trị 0 per-feature")
    ax[1, 0].set_xlabel("% giá trị 0"); ax[1, 0].set_ylabel("Số feature")

    # 5. Phân bố kurtosis (clip để dễ nhìn)
    kc = rep["kurtosis"].clip(upper=rep["kurtosis"].quantile(0.95))
    ax[1, 1].hist(kc, bins=20, color="#937860", edgecolor="white")
    ax[1, 1].set_title("Phân bố kurtosis (clip p95)")
    ax[1, 1].set_xlabel("kurtosis"); ax[1, 1].set_ylabel("Số feature")

    # 6. Panel tóm tắt chất lượng (dùng font mặc định để render đủ tiếng Việt)
    ax[1, 2].axis("off")
    q = quality
    lines = [
        ("Số dòng", f"{q['rows']:,}"),
        ("Feature", f"{q['feats']}"),
        ("Số lớp", f"{q['classes']}"),
        ("NaN", f"{q['nan']:,}"),
        ("Inf", f"{q['inf']:,}"),
        ("Trùng hoàn toàn", f"{q['dup_full']:,} ({q['dup_full']/q['rows']*100:.1f}%)"),
        ("Mâu thuẫn nhãn", f"{q['conflict']:,}"),
    ]
    y = 0.92
    ax[1, 2].text(0.04, y, "TÓM TẮT CHẤT LƯỢNG", va="top", ha="left",
                  fontsize=13, fontweight="bold", transform=ax[1, 2].transAxes)
    y -= 0.13
    for k, v in lines:
        ax[1, 2].text(0.06, y, k, va="top", ha="left", fontsize=12,
                      transform=ax[1, 2].transAxes)
        ax[1, 2].text(0.62, y, v, va="top", ha="left", fontsize=12,
                      fontweight="bold", transform=ax[1, 2].transAxes)
        y -= 0.11
    ax[1, 2].add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False,
                       edgecolor="#999", lw=1, transform=ax[1, 2].transAxes))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = csv_path.rsplit(".", 1)[0] + "_dashboard.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"→ Đã lưu dashboard: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Dùng: python visualize_data.py <duong_dan.csv> [ten_cot_nhan]")
        sys.exit(0)
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "activity"
    rep, vc, quality, _ = build_stats(path, label)
    plot(rep, vc, quality, path)
