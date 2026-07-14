"""
feature_selection_v2.py
----------------------------------------------
Multi-class (~10 labels) optimized version

Pipeline:
  - Variance filter
  - Ensemble global (ANOVA + MI + ExtraTrees CV)
  - Per-class importance (với CV để tránh leakage)
  - Mandatory + Soft whitelist
  - Merge + top_k
  - [MỚI] Tự động kéo theo flag columns (_is_posinf, _is_neginf, _is_missing)

CHANGELOG:
  [FIX-1] Tự động include flag columns của feature được chọn
  [FIX-2] Per-class importance dùng CV (tránh leakage)
  [FIX-3] Giới hạn per-class đóng góp tối đa = top_k // 2
          để global ranking luôn có chỗ
  [FIX-4] Dùng dict lookup thay vì list.index() — O(1)
  [FIX-5] In bảng rank score để debug
"""

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

import warnings
import gc

warnings.filterwarnings("ignore")


# =========================================================
# MANDATORY FEATURES
# =========================================================
DDOS_MANDATORY_FEATURES = {
    "active_variance",
    "active_std",
    "bwd_ack_flag_counts",
    "fwd_bulk_state_count",
}

# =========================================================
# SOFT WHITELIST
# =========================================================
DDOS_CRITICAL_FEATURES = {
    "dst_port",
    "packets_rate",
    "bytes_rate",
    "syn_flag_counts",
    "ack_flag_counts",
    "duration",
    "packets_count",
}

# =========================================================
# FLAG SUFFIXES (sinh ra bởi dataset_preprocessor.py)
# =========================================================
_FLAG_SUFFIXES = ["_is_posinf", "_is_neginf", "_is_missing"]


def _collect_flag_cols(selected: list, all_cols: set) -> list:
    """
    Với mỗi feature được chọn, kiểm tra xem có cột flag tương ứng
    trong all_cols không. Nếu có → thêm vào danh sách.
    """
    flags = []
    for col in selected:
        for suffix in _FLAG_SUFFIXES:
            flag_col = f"{col}{suffix}"
            if flag_col in all_cols and flag_col not in flags:
                flags.append(flag_col)
    return flags


# =========================================================
# FEATURE SELECTOR
# =========================================================
class FeatureSelector:

    def __init__(self, top_k: int = 60, cv_folds: int = 5,
                 per_class_top: int = 5, per_class_budget_ratio: float = 0.4):
        """
        Parameters
        ----------
        top_k               : Số feature tối đa (không tính flag columns)
        cv_folds            : Số fold cho ExtraTrees CV
        per_class_top       : Top-N feature mỗi class (per-class step)
        per_class_budget_ratio : Tối đa bao nhiêu % top_k dành cho per-class
                              (còn lại dành cho global ranking)
                              Ví dụ: 0.4 → per-class tối đa 24/60
        """
        self.top_k                  = top_k
        self.cv_folds               = cv_folds
        self.per_class_top          = per_class_top
        self.per_class_budget       = int(top_k * per_class_budget_ratio)
        self.selected_cols_         = None
        self.rank_df_               = None   # debug

    # ----------------------------------------------------------
    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                      feature_names: list):
        cols = list(feature_names)
        # [FIX-4] dict lookup O(1)
        col2idx = {c: i for i, c in enumerate(cols)}

        print("\n" + "=" * 70)
        print(" FEATURE SELECTION V2 (MULTI-CLASS)")
        print("=" * 70)

        # =====================================================
        # 0. Variance Filter
        # =====================================================
        non_mand_mask = np.array([c not in DDOS_MANDATORY_FEATURES for c in cols])
        mand_mask     = ~non_mand_mask

        var = VarianceThreshold(0.0)
        var.fit(X[:, non_mand_mask])
        support = var.get_support()

        keep = mand_mask.copy()
        keep[non_mand_mask] = support

        X_cur = X[:, keep]
        cols  = [c for c, k in zip(cols, keep) if k]
        print(f"[Variance] Remaining: {len(cols)} features")

        # Khởi tạo bảng rank (thấp = tốt hơn)
        ranks = {c: 0 for c in cols}

        # =====================================================
        # 1. ANOVA F-score
        # =====================================================
        print("[ANOVA]")
        f_vals, _ = f_classif(X_cur, y)
        for r, i in enumerate(np.argsort(np.nan_to_num(f_vals))[::-1]):
            ranks[cols[i]] += r

        # =====================================================
        # 2. Mutual Information
        # =====================================================
        print("[Mutual Info]")
        sample_size = min(50_000, len(X_cur))
        idx_sample  = np.random.choice(len(X_cur), sample_size, replace=False)
        mi = mutual_info_classif(X_cur[idx_sample], y[idx_sample], random_state=42)
        for r, i in enumerate(np.argsort(np.nan_to_num(mi))[::-1]):
            ranks[cols[i]] += r

        # =====================================================
        # 3. ExtraTrees Global (CV — không leakage)
        # =====================================================
        print("[ExtraTrees - Global CV]")
        skf    = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        et_imp = np.zeros(len(cols))
        for fold, (tr, _) in enumerate(skf.split(X_cur, y), 1):
            clf = ExtraTreesClassifier(
                n_estimators=150, max_depth=20,
                min_samples_leaf=5, n_jobs=-1, random_state=fold
            )
            clf.fit(X_cur[tr], y[tr])
            et_imp += clf.feature_importances_
        et_imp /= self.cv_folds
        for r, i in enumerate(np.argsort(et_imp)[::-1]):
            ranks[cols[i]] += r

        # =====================================================
        # 4. Per-class Importance  [FIX-2: dùng CV]
        #                          [FIX-3: giới hạn budget]
        # =====================================================
        print("[Per-class Importance  (CV, no leakage)]")
        per_class_features = []   # dùng list để giữ thứ tự ưu tiên
        per_class_seen     = set()

        unique_classes = np.unique(y)
        skf_pc = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

        for c in unique_classes:
            print(f"   → Class {c}")
            y_bin   = (y == c).astype(int)
            imp_acc = np.zeros(len(cols))

            for tr, _ in skf_pc.split(X_cur, y_bin):
                clf = ExtraTreesClassifier(
                    n_estimators=100, max_depth=15,
                    min_samples_leaf=5, n_jobs=-1, random_state=42
                )
                clf.fit(X_cur[tr], y_bin[tr])
                imp_acc += clf.feature_importances_

            top_idx = np.argsort(imp_acc)[::-1][:self.per_class_top]
            for i in top_idx:
                feat = cols[i]
                if feat not in per_class_seen:
                    per_class_seen.add(feat)
                    per_class_features.append(feat)

        # [FIX-3] Cắt per-class theo budget
        per_class_features = per_class_features[:self.per_class_budget]
        print(f"   Collected {len(per_class_seen)} unique → kept {len(per_class_features)} (budget={self.per_class_budget})")

        # =====================================================
        # 5. Merge (thứ tự ưu tiên rõ ràng)
        # =====================================================
        global_sorted = sorted(cols, key=lambda c: ranks[c])  # thấp = tốt
        final = []
        added = set()

        def _add(c):
            if c in cols and c not in added:
                final.append(c)
                added.add(c)

        # (a) Mandatory luôn vào trước
        for c in DDOS_MANDATORY_FEATURES:
            _add(c)

        # (b) Per-class (đã giới hạn budget)
        for c in per_class_features:
            _add(c)

        # (c) Soft whitelist
        for c in DDOS_CRITICAL_FEATURES:
            _add(c)

        # (d) Fill còn lại bằng global ranking
        for c in global_sorted:
            if len(final) >= self.top_k:
                break
            _add(c)

        self.selected_cols_ = final[:self.top_k]

        # =====================================================
        # 6. [FIX-5] Debug rank table
        # =====================================================
        self.rank_df_ = (
            pd.DataFrame({
                "feature":  cols,
                "rank_sum": [ranks[c] for c in cols],
                "et_imp":   et_imp,
            })
            .sort_values("rank_sum")
            .reset_index(drop=True)
        )
        self.rank_df_["selected"] = self.rank_df_["feature"].isin(set(self.selected_cols_))

        print("\n" + "=" * 70)
        print(f" FINAL SELECTED: {len(self.selected_cols_)} features")
        print("=" * 70)
        print(self.rank_df_.head(20).to_string(index=False))

        # [FIX-4] O(1) index lookup
        out_idx = [col2idx[c] for c in self.selected_cols_ if c in col2idx]
        return X[:, out_idx], self.selected_cols_


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    CSV_PATH  = "/kaggle/working/dataset_optimized_multi1.csv"
    LABEL_COL = "activity"
    DROP_COLS = ["label", "flow_id", "timestamp", "src_ip", "dst_ip", "src_port"]
    TOP_K     = 60

    # =====================================================
    # Load
    # =====================================================
    print("1. Loading data...")
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    df = df[df[LABEL_COL] != "Suspicious"]
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Tập hợp toàn bộ tên cột (dùng để tìm flag columns)
    all_col_set = set(df.columns)

    # =====================================================
    # Prepare target
    # =====================================================
    print("2. Preparing matrix...")
    y_raw = df[LABEL_COL].values
    enc   = {l: i for i, l in enumerate(np.unique(y_raw))}
    y     = np.array([enc[l] for l in y_raw], dtype=np.int32)

    # =====================================================
    # Feature columns (CHỈ feature gốc, KHÔNG gồm flag)
    # =====================================================
    feature_cols = [
        c for c in df.columns
        if c != LABEL_COL and not any(c.endswith(s) for s in _FLAG_SUFFIXES)
    ]

    print(f"   -> Feature gốc: {len(feature_cols)}")
    print(f"   -> Flag cols  : {len([c for c in df.columns if any(c.endswith(s) for s in _FLAG_SUFFIXES)])}")

    # =====================================================
    # Build numeric matrix (chỉ feature gốc)
    # =====================================================
    X = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        X[:, i] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .astype(np.float32)
            .values
        )

    # =====================================================
    # Feature Selection
    # =====================================================
    print("3. Running Feature Selection...")
    fs = FeatureSelector(
        top_k=TOP_K,
        cv_folds=5,
        per_class_top=5,           # top 5 feature mỗi class
        per_class_budget_ratio=0.4 # tối đa 40% = 24 slot cho per-class
    )
    _, selected = fs.fit_transform(X, y, feature_cols)

    del X
    gc.collect()

    # =====================================================
    # [FIX-1] Tự động kéo theo flag columns
    # =====================================================
    flag_cols = _collect_flag_cols(selected, all_col_set)
    print(f"\n   -> Tự động thêm {len(flag_cols)} flag columns:")
    for fc in flag_cols:
        print(f"      + {fc}")

    # =====================================================
    # Save feature list (chỉ feature gốc)
    # =====================================================
    with open("selected_features.txt", "w") as f:
        f.write("\n".join(selected))

    # =====================================================
    # Export dataset: feature gốc + flag + label
    # =====================================================
    print("\n4. Exporting dataset...")
    final_cols = selected + flag_cols + [LABEL_COL]
    df_final   = df[final_cols]

    df_final.to_csv("dataset_ready_multilabel1.csv", index=False)

    print(f"\n✅ DONE")
    print(f"   Features gốc  : {len(selected)}")
    print(f"   Flag columns  : {len(flag_cols)}")
    print(f"   Tổng columns  : {len(final_cols) - 1}  (không tính label)")
    print(f"   Rows          : {len(df_final):,}")
    print(f"   Output        : dataset_ready_multilabel1.csv")