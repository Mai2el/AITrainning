"""
feature_selection_v2.py
----------------------------------------------
Multi-class (~10 labels) optimized version:

Pipeline:
  - Variance filter (bỏ constant, giữ mandatory)
  - Ensemble global (ANOVA + MI + ExtraTrees)
  - Per-class importance (One-vs-Rest ExtraTrees)
  - Mandatory + Soft whitelist
  - Merge + chọn top_k

Giữ nguyên:
  - Raw data export (không fillna)
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, f_classif, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
import gc

warnings.filterwarnings("ignore")

# =========================================================
# 🔒 MANDATORY (KHÔNG BAO GIỜ BỎ)
# =========================================================
DDOS_MANDATORY_FEATURES = {
    "active_variance",
    "active_std",
    "bwd_ack_flag_counts",
    "fwd_bulk_state_count",
}

# =========================================================
# ⚖️ SOFT WHITELIST (NHẸ, KHÔNG BIAS)
# =========================================================
DDOS_CRITICAL_FEATURES = {
    "dst_port",
    "packets_rate", "bytes_rate",
    "syn_flag_counts", "ack_flag_counts",
    "duration", "packets_count",
    "bwd_fwd_ratio", "bytes_per_packet",
}

# =========================================================
# Feature Engineering
# =========================================================
def engineer_ddos_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9

    def get_num(col):
        return pd.to_numeric(df[col], errors='coerce')

    if "syn_flag_counts" in df.columns and "packets_count" in df.columns:
        df["syn_per_packet"] = get_num("syn_flag_counts") / (get_num("packets_count") + eps)

    if "bwd_packets_count" in df.columns and "fwd_packets_count" in df.columns:
        df["bwd_fwd_ratio"] = (
            get_num("bwd_packets_count") / (get_num("fwd_packets_count") + 1.0)
        ).clip(0, 100)

    if "bytes_rate" in df.columns and "packets_rate" in df.columns:
        df["bytes_per_packet"] = (get_num("bytes_rate") + eps) / (get_num("packets_rate") + eps)

    print("[engineer] Added features: syn_per_packet, bwd_fwd_ratio, bytes_per_packet")
    return df


# =========================================================
# 🔥 FEATURE SELECTOR V2
# =========================================================
class FeatureSelector:
    def __init__(self, top_k=60, cv_folds=5):
        self.top_k = top_k
        self.cv_folds = cv_folds
        self.selected_cols_ = None

    def fit_transform(self, X, y, feature_names):
        cols = list(feature_names)

        print("\n" + "="*70)
        print(" FEATURE SELECTION V2 (MULTI-CLASS)")
        print("="*70)

        # -------------------------------------------------
        # 0. Variance Filter (giữ mandatory)
        # -------------------------------------------------
        non_mand_idx = [i for i, c in enumerate(cols) if c not in DDOS_MANDATORY_FEATURES]
        mand_idx = [i for i, c in enumerate(cols) if c in DDOS_MANDATORY_FEATURES]

        var = VarianceThreshold(0.0)
        var.fit(X[:, non_mand_idx])
        keep_non_mand = var.get_support()

        keep = np.zeros(len(cols), dtype=bool)

        for k, i in enumerate(non_mand_idx):
            keep[i] = keep_non_mand[k]
        for i in mand_idx:
            keep[i] = True

        X_cur = X[:, keep]
        cols = [cols[i] for i, k in enumerate(keep) if k]

        print(f"[Variance] Remaining: {len(cols)} features")

        ranks = {c: 0 for c in cols}

        # -------------------------------------------------
        # 1. ANOVA
        # -------------------------------------------------
        print("[ANOVA]")
        f_vals, _ = f_classif(X_cur, y)
        order = np.argsort(np.nan_to_num(f_vals))[::-1]
        for r, i in enumerate(order):
            ranks[cols[i]] += r

        # -------------------------------------------------
        # 2. Mutual Information
        # -------------------------------------------------
        print("[Mutual Info]")
        sample_size = min(50000, len(X_cur))
        idx = np.random.choice(len(X_cur), sample_size, replace=False)

        mi = mutual_info_classif(X_cur[idx], y[idx], random_state=42)
        order = np.argsort(np.nan_to_num(mi))[::-1]
        for r, i in enumerate(order):
            ranks[cols[i]] += r

        # -------------------------------------------------
        # 3. Extra Trees (global)
        # -------------------------------------------------
        print("[ExtraTrees - Global]")
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        et_imp = np.zeros(len(cols))

        for fold, (tr, _) in enumerate(skf.split(X_cur, y), 1):
            clf = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=15,
                n_jobs=-1,
                random_state=fold
            )
            clf.fit(X_cur[tr], y[tr])
            et_imp += clf.feature_importances_

        order = np.argsort(et_imp)[::-1]
        for r, i in enumerate(order):
            ranks[cols[i]] += r

        # -------------------------------------------------
        # 🔥 4. Per-class Importance (QUAN TRỌNG NHẤT)
        # -------------------------------------------------
        print("[Per-class Importance]")

        per_class_features = set()
        unique_classes = np.unique(y)

        for c in unique_classes:
            print(f"   → Class {c}")

            y_bin = (y == c).astype(int)

            clf = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=15,
                n_jobs=-1,
                random_state=42
            )
            clf.fit(X_cur, y_bin)

            imp = clf.feature_importances_
            top_idx = np.argsort(imp)[::-1][:10]   # top 10 mỗi class

            for i in top_idx:
                per_class_features.add(cols[i])

        print(f"   Collected {len(per_class_features)} per-class features")

        # -------------------------------------------------
        # 5. Merge tất cả
        # -------------------------------------------------
        sorted_global = sorted(cols, key=lambda c: ranks[c])

        final = []

        # (1) Mandatory
        for c in DDOS_MANDATORY_FEATURES:
            if c in cols:
                final.append(c)

        # (2) Per-class
        for c in per_class_features:
            if c not in final:
                final.append(c)

        # (3) Soft whitelist
        for c in DDOS_CRITICAL_FEATURES:
            if c in cols and c not in final:
                final.append(c)

        # (4) Fill bằng global ranking
        for c in sorted_global:
            if len(final) >= self.top_k:
                break
            if c not in final:
                final.append(c)

        self.selected_cols_ = final[:self.top_k]

        print("\n" + "="*70)
        print(f" FINAL SELECTED: {len(self.selected_cols_)} features")
        print("="*70)

        idx = [feature_names.index(c) for c in self.selected_cols_]
        return X[:, idx], self.selected_cols_


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    CSV_PATH = "dataset_optimized.csv"
    LABEL_COL = "activity"
    DROP_COLS = ["label", "flow_id", "timestamp", "src_ip", "dst_ip", "src_port"]
    TOP_K = 60

    print("1. Loading data...")
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    df = df[df[LABEL_COL] != "Suspicious"]

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    df = engineer_ddos_features(df)

    print("2. Preparing matrix...")

    y_raw = df[LABEL_COL].values
    enc = {l: i for i, l in enumerate(np.unique(y_raw))}
    y = np.array([enc[l] for l in y_raw], dtype=np.int32)

    feature_cols = [c for c in df.columns if c != LABEL_COL]

    X = np.zeros((len(df), len(feature_cols)), dtype=np.float32)

    for i, col in enumerate(feature_cols):
        col_data = pd.to_numeric(df[col], errors='coerce')
        col_data = col_data.replace([np.inf, -np.inf], np.nan).fillna(0)
        X[:, i] = col_data.astype(np.float32)

    print("3. Running Feature Selection...")

    fs = FeatureSelector(top_k=TOP_K)
    _, selected = fs.fit_transform(X, y, feature_cols)

    # cleanup
    del X
    gc.collect()

    # save feature list
    with open("selected_features.txt", "w") as f:
        f.write("\n".join(selected))

    # export RAW data
    print("4. Exporting dataset...")

    df_final = df[selected + [LABEL_COL]]
    df_final.to_csv("dataset_ready.csv", index=False)

    print("✓ DONE")