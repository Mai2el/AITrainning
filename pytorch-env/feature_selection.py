"""
feature_selection.py  (DDoS-aware version v2)
----------------------------------------------
Cải tiến so với v1:
  1. [Bước 0] engineer_ddos_features(): thêm 5 features tổ hợp quan trọng
     (syn_per_packet, iat_x_rate, bwd_fwd_ratio, syn_no_fin, bytes_per_packet)
  2. [Bước 2] Correlation filter: ưu tiên giữ feature có importance CAO HƠN
     thay vì chỉ giữ feature đến trước.
  3. [Bước 3] Tree importance dùng CV 5-fold → importance ổn định hơn,
     không bị bias theo 1 lần split.
  4. [Bước 3] Hỗ trợ chọn top_k theo cumulative importance thay vì số cứng.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
import time


# ── Features phải giữ lại bất kể correlation hay importance ──────────────
DDOS_CRITICAL_FEATURES = {
    # Tốc độ — DDoS flood đẩy rate lên bất thường
    "packets_rate", "fwd_packets_rate", "bwd_packets_rate",
    "bytes_rate", "fwd_bytes_rate", "bwd_bytes_rate", "down_up_rate",
    # Inter-Arrival Time — flood có IAT rất nhỏ, gần như 0
    "packets_IAT_mean", "packet_IAT_std", "packet_IAT_min", "packet_IAT_max",
    "fwd_packets_IAT_mean", "bwd_packets_IAT_mean",
    # TCP flags — SYN flood, ACK flood, RST flood
    "syn_flag_counts", "ack_flag_counts", "rst_flag_counts", "fin_flag_counts",
    "fwd_syn_flag_counts", "bwd_syn_flag_counts",
    "syn_flag_percentage_in_total", "ack_flag_percentage_in_total",
    # Flow cơ bản
    "duration", "packets_count", "fwd_packets_count", "bwd_packets_count",
    # Handshake — SYN flood không hoàn thành handshake
    "handshake_duration", "handshake_state",
    #MỚI: Features tổ hợp từ engineer_ddos_features()
    "syn_per_packet", "iat_x_rate",
    "bwd_fwd_ratio", "syn_no_fin", "bytes_per_packet",
}


# ── Bước 0: Feature Engineering ──────────────────────────────────────────
def engineer_ddos_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm 5 features tổ hợp đặc trưng cho DDoS.
    Gọi TRƯỚC khi chạy FeatureSelector.fit_transform().

    Parameters
    ----------
    df : DataFrame đã có các cột gốc (packets_count, syn_flag_counts, ...)

    Returns
    -------
    df : DataFrame với 5 cột mới được thêm vào
    """
    eps = 1e-9  # tránh chia 0

    # Tỉ lệ SYN/packet — SYN flood ≈ 1.0, traffic bình thường << 1
    if "syn_flag_counts" in df.columns and "packets_count" in df.columns:
        df["syn_per_packet"] = df["syn_flag_counts"] / (df["packets_count"] + eps)

    # IAT nhỏ + rate cao = dấu hiệu flood rõ nhất
    if "packets_IAT_mean" in df.columns and "packets_rate" in df.columns:
        df["iat_x_rate"] = df["packets_IAT_mean"] * df["packets_rate"]

    # Mất cân bằng chiều — DDoS thường một chiều (nhiều fwd, ít bwd)
    if "bwd_packets_count" in df.columns and "fwd_packets_count" in df.columns:
        df["bwd_fwd_ratio"] = (
            (df["bwd_packets_count"] + eps) / (df["fwd_packets_count"] + eps)
        )

    # SYN không có FIN → handshake không hoàn chỉnh (SYN flood)
    if "syn_flag_counts" in df.columns and "fin_flag_counts" in df.columns:
        df["syn_no_fin"] = (df["syn_flag_counts"] - df["fin_flag_counts"]).clip(lower=0)

    # Bytes trên mỗi packet — DDoS packet thường rất nhỏ (chỉ header)
    if "bytes_rate" in df.columns and "packets_rate" in df.columns:
        df["bytes_per_packet"] = (df["bytes_rate"] + eps) / (df["packets_rate"] + eps)

    new_cols = ["syn_per_packet", "iat_x_rate", "bwd_fwd_ratio",
                "syn_no_fin", "bytes_per_packet"]
    added = [c for c in new_cols if c in df.columns]
    print(f"[engineer_ddos_features] Đã thêm: {added}")
    return df


# ── Selector chính ────────────────────────────────────────────────────────
class FeatureSelector:
    def __init__(self,
                 variance_threshold: float = 0.01,
                 corr_threshold: float = 0.95,
                 top_k: int = 80,
                 cumulative_importance: float = None,
                 cv_folds: int = 5,
                 ddos_whitelist=None):
        """
        Parameters
        ----------
        variance_threshold   : ngưỡng loại feature gần hằng số
        corr_threshold       : ngưỡng correlation Pearson
        top_k                : số feature giữ tối đa ở bước tree importance
                               (bị bỏ qua nếu cumulative_importance được đặt)
        cumulative_importance: nếu đặt (vd 0.995), giữ số feature TỐI THIỂU
                               cover đủ % importance thay vì top_k cứng
        cv_folds             : số fold để tính importance ổn định (mặc định 5)
        ddos_whitelist       : set tên feature luôn được bảo vệ
        """
        self.variance_threshold    = variance_threshold
        self.corr_threshold        = corr_threshold
        self.top_k                 = top_k
        self.cumulative_importance = cumulative_importance
        self.cv_folds              = cv_folds
        self.ddos_whitelist        = (ddos_whitelist if ddos_whitelist is not None
                                      else DDOS_CRITICAL_FEATURES)
        self.selected_cols_        = None
        self._y_cache              = None   # dùng nội bộ cho correlation step

    # ------------------------------------------------------------------
    def fit_transform(self, X, y, feature_names):
        """Chỉ gọi trên tập TRAIN."""
        print(f"\n{'='*64}")
        print(f"  FEATURE SELECTION — DDoS-aware v2  ({len(feature_names)} features)")
        print(f"{'='*64}")

        present_wl = [c for c in self.ddos_whitelist if c in feature_names]
        print(f"\n⚑  Whitelist: {len(present_wl)}/{len(self.ddos_whitelist)} "
              f"DDoS-critical features có trong dataset → luôn bảo vệ.")

        self._y_cache = y
        cols, X_cur   = list(feature_names), X.copy()

        X_cur, cols = self._step_variance(X_cur, cols)
        X_cur, cols = self._step_correlation(X_cur, cols)
        X_cur, cols = self._step_tree_importance(X_cur, y, cols)

        self.selected_cols_ = cols
        print(f"\n✓ Kết quả: {len(self.selected_cols_)} features được giữ lại\n")
        return X_cur, self.selected_cols_

    def transform(self, X, original_feature_names):
        if self.selected_cols_ is None:
            raise RuntimeError("Gọi fit_transform() trước.")
        idx = [original_feature_names.index(c) for c in self.selected_cols_]
        return X[:, idx]

    # ------------------------------------------------------------------
    # Bước 1: Variance threshold
    # ------------------------------------------------------------------
    def _step_variance(self, X, cols):
        t0 = time.time()
        wl_mask  = np.array([c in self.ddos_whitelist for c in cols])
        non_wl_X = X[:, ~wl_mask]
        non_wl_c = [c for c, m in zip(cols, wl_mask) if not m]

        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(non_wl_X)
        keep_non_wl = selector.get_support()

        kept_wl      = [c for c, m in zip(cols, wl_mask) if m]
        kept_non_wl  = [c for c, m in zip(non_wl_c, keep_non_wl) if m]
        kept_set     = set(kept_wl + kept_non_wl)
        final_idx    = [i for i, c in enumerate(cols) if c in kept_set]
        kept_ordered = [cols[i] for i in final_idx]

        print(f"\nBước 1 — Variance threshold (< {self.variance_threshold})")
        print(f"  Loại: {len(cols)-len(kept_ordered)}"
              f"  →  còn {len(kept_ordered)}  [{time.time()-t0:.1f}s]")
        return X[:, final_idx], kept_ordered

    # ------------------------------------------------------------------
    # Bước 2: Correlation filter —GIỮ feature có importance CAO HƠN
    # ------------------------------------------------------------------
    def _step_correlation(self, X, cols):
        t0 = time.time()

        # MỚI: Tính importance sơ bộ (nhanh) để quyết định DROP cái nào
        print(f"\nBước 2 — Correlation filter (corr > {self.corr_threshold})")
        print(f"  [2a] Tính importance sơ bộ để chọn feature giữ lại...")
        quick_clf = ExtraTreesClassifier(
            n_estimators=50, max_depth=8,
            n_jobs=-1, random_state=42,
            class_weight="balanced",
        )
        quick_clf.fit(X, self._y_cache)
        importance = dict(zip(cols, quick_clf.feature_importances_))

        df    = pd.DataFrame(X, columns=cols)
        corr  = df.corr(method='pearson').abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop   = set()
        protected = 0

        for col in upper.columns:
            if col in to_drop:
                continue
            if col in self.ddos_whitelist:
                continue   # ← Không bao giờ drop whitelist

            correlated_cols = upper[col][upper[col] > self.corr_threshold]
            if correlated_cols.empty:
                continue

            for partner in correlated_cols.index:
                if partner in to_drop or partner in self.ddos_whitelist:
                    continue
                # MỚI: Drop feature có importance THẤP HƠN trong cặp
                if importance.get(col, 0) < importance.get(partner, 0):
                    to_drop.add(col)
                    break
                else:
                    to_drop.add(partner)

        # Kiểm tra whitelist bị bảo vệ
        for c in cols:
            if c in self.ddos_whitelist:
                s = upper[c][upper[c] > self.corr_threshold]
                if not s.empty:
                    protected += 1

        kept = [c for c in cols if c not in to_drop]
        idx  = [cols.index(c) for c in kept]

        print(f"  Loại: {len(to_drop)}  →  còn {len(kept)}  [{time.time()-t0:.1f}s]")
        if protected:
            print(f"  ⚑  Bảo vệ {protected} DDoS-critical features dù có corr cao")

        # In mẫu các cặp bị loại
        sample = []
        for col in list(to_drop)[:4]:
            if col in upper.columns:
                s = upper[col][upper[col] > self.corr_threshold]
                if not s.empty:
                    best = s.idxmax()
                    sample.append((col, best, s[best]))
        for a, b, r in sample:
            print(f"    DROP {a[:30]:<30} ↔ KEEP {b[:30]:<30}  r={r:.3f}")
        if len(to_drop) > 4:
            print(f"    ... và {len(to_drop)-4} cặp khác")

        return X[:, idx], kept

    # ------------------------------------------------------------------
    # Bước 3: Tree importance —  CV 5-fold + cumulative option
    # ------------------------------------------------------------------
    def _step_tree_importance(self, X, y, cols):
        t0 = time.time()

        # MỚI: Tính importance trung bình qua CV → ổn định hơn
        print(f"\nBước 3 — Tree importance "
              f"(CV {self.cv_folds}-fold, class_weight='balanced')")

        skf         = StratifiedKFold(n_splits=self.cv_folds,
                                      shuffle=True, random_state=42)
        importances = np.zeros(len(cols))

        for fold, (tr_idx, _) in enumerate(skf.split(X, y), 1):
            clf = ExtraTreesClassifier(
                n_estimators=150, max_depth=15,
                n_jobs=-1, random_state=fold,
                class_weight="balanced",
            )
            clf.fit(X[tr_idx], y[tr_idx])
            importances += clf.feature_importances_
            print(f"  Fold {fold}/{self.cv_folds} done", end="\r")

        importances /= self.cv_folds
        ranked_idx   = np.argsort(importances)[::-1]
        wl_idx       = [i for i, c in enumerate(cols) if c in self.ddos_whitelist]

        # MỚI: Chọn top_k theo cumulative importance nếu được đặt
        if self.cumulative_importance is not None:
            k = self._select_by_cumulative(importances, ranked_idx,
                                           self.cumulative_importance)
        else:
            k = min(self.top_k, len(cols))

        print(f"\n  Giữ top {k} features (CV avg importance)  [{time.time()-t0:.1f}s]")

        top_idx_set = set(ranked_idx[:k].tolist()) | set(wl_idx)

        # Nếu whitelist làm vượt k: cắt bớt non-whitelist
        if len(top_idx_set) > k:
            wl_set        = set(wl_idx)
            non_wl_ranked = [i for i in ranked_idx if i not in wl_set]
            top_idx_set   = wl_set | set(non_wl_ranked[:k - len(wl_set)])

        top_idx = sorted(top_idx_set)
        kept    = [cols[i] for i in top_idx]

        print(f"  Loại: {len(cols)-len(kept)}  →  còn {len(kept)}")
        print("\n  Top-10 features (CV importance trung bình):")
        for name, imp in sorted(zip(cols, importances), key=lambda x: -x[1])[:10]:
            mark = " ⚑" if name in self.ddos_whitelist else ""
            print(f"    {name[:40]:<40}{mark}  {imp:.5f}  {'█'*int(imp*300)}")

        low_wl = [(c, importances[cols.index(c)]) for c in self.ddos_whitelist
                  if c in cols and importances[cols.index(c)] < 0.001]
        if low_wl:
            print(f"\n  ⚠  {len(low_wl)} whitelist features importance < 0.001 "
                  f"(vẫn giữ, nhưng nên kiểm tra dữ liệu):")
            for name, imp in sorted(low_wl, key=lambda x: x[1])[:5]:
                print(f"     {name}: {imp:.6f}")

        return X[:, top_idx], kept

    # ------------------------------------------------------------------
    def _select_by_cumulative(self, importances, ranked_idx, threshold):
        """Trả về số feature tối thiểu cover đủ `threshold` importance."""
        cumsum = np.cumsum(importances[ranked_idx])
        cutoff = int(np.searchsorted(cumsum, threshold * cumsum[-1])) + 1
        cutoff = max(cutoff, len([i for i, c in enumerate(
            [None]*len(importances)) if i in
            [j for j, col in enumerate(self.ddos_whitelist)
             if col in self.ddos_whitelist]]))
        print(f"  → cumulative {threshold*100:.1f}%: cần {cutoff} features")
        return min(cutoff, len(importances))


# ------------------------------------------------------------------
# STANDALONE DEMO
# ------------------------------------------------------------------
if __name__ == "__main__":
    CSV_PATH  = "output.csv"
    LABEL_COL = "label"
    PORT_COLS = ["src_port", "dst_port"]
    DROP_COLS = ["activity"]
    TOP_K     = 80

    print("Đang load dữ liệu...")
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[df[LABEL_COL] != "Suspicious"].reset_index(drop=True)

    drop = [c for c in DROP_COLS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    # ✅ MỚI: Thêm features tổ hợp trước khi select
    df = engineer_ddos_features(df)

    labels_raw = df[LABEL_COL].values
    if isinstance(labels_raw[0], str):
        enc = {l: i for i, l in enumerate(np.unique(labels_raw))}
        y   = np.array([enc[l] for l in labels_raw])
        print("Labels:", enc)
    else:
        y = labels_raw.astype(int)

    print("\nPhân phối class:")
    for cls, cnt in zip(*np.unique(y, return_counts=True)):
        print(f"  Class {cls}: {cnt:,} ({cnt/len(y)*100:.1f}%)")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in PORT_COLS + [LABEL_COL]]

    # FIX: fillna thông minh thay vì dùng mean cho tất cả
    #
    # - ZERO_FILL: các cột mà giá trị null có nghĩa là "không tồn tại"
    #   (vd: flow UDP/ICMP không có handshake → handshake_duration = 0)
    #
    # - MEDIAN_FILL: các cột dạng rate/IAT dùng median thay vì mean
    #   vì DDoS tạo outlier cực lớn → mean bị kéo lệch, median ổn hơn
    #
    # - Các cột còn lại: vẫn dùng mean như cũ
    ZERO_FILL_COLS = [
        "handshake_duration",   # UDP/ICMP không handshake → 0
        "handshake_state",      # không có handshake → trạng thái 0
        "syn_no_fin",           # feature tổ hợp, null = 0
    ]
    MEDIAN_FILL_COLS = [
        "bytes_rate", "fwd_bytes_rate", "bwd_bytes_rate",
        "packets_rate", "fwd_packets_rate", "bwd_packets_rate",
        "packet_IAT_std", "packets_IAT_mean",
        "fwd_packets_IAT_mean", "bwd_packets_IAT_mean",
        "packet_IAT_min", "packet_IAT_max",
    ]

    df_num = df[num_cols].copy()

    # Bước 1: fill 0 cho các cột "không tồn tại"
    for col in ZERO_FILL_COLS:
        if col in df_num.columns:
            null_count = df_num[col].isna().sum()
            if null_count > 0:
                print(f"  fillna(0)      {col}: {null_count:,} nulls")
            df_num[col] = df_num[col].fillna(0)

    # Bước 2: fill median cho các cột rate/IAT (tránh bị lệch bởi outlier)
    for col in MEDIAN_FILL_COLS:
        if col in df_num.columns:
            null_count = df_num[col].isna().sum()
            if null_count > 0:
                print(f"  fillna(median) {col}: {null_count:,} nulls")
            df_num[col] = df_num[col].fillna(df_num[col].median())

    # Bước 3: các cột còn lại dùng mean như cũ
    remaining_nulls = df_num.columns[df_num.isna().any()].tolist()
    if remaining_nulls:
        print(f"  fillna(mean)   {len(remaining_nulls)} cột còn lại")
        df_num[remaining_nulls] = df_num[remaining_nulls].fillna(
            df_num[remaining_nulls].mean()
        )

    X = df_num.values

    # MỚI: Thêm cv_folds=5, có thể dùng cumulative_importance=0.995
    #         thay cho top_k nếu muốn tự động chọn số feature
    fs = FeatureSelector(
        variance_threshold=0.01,
        corr_threshold=0.95,
        top_k=TOP_K,
        # cumulative_importance=0.995,  # bỏ comment để dùng thay top_k
        cv_folds=5,
    )
    X_reduced, selected = fs.fit_transform(X, y, num_cols)

    with open("selected_features.txt", "w") as f:
        f.write("\n".join(selected))
    print(f"\n✓ Đã lưu {len(selected)} features → selected_features.txt")
    print("\nThêm vào train_optimized.py sau preprocess_features():")
    print("  df = engineer_ddos_features(df)  # ← thêm dòng này")
    print("  processor.select_features(corr_threshold=0.95, top_k=80)")