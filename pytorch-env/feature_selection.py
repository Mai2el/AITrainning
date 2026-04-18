"""
feature_selection.py  (DDoS-aware v8 - FULL SMART PREPROCESS)
------------------------------------------------------------
✓ Drop ID columns
✓ Fix handshake_duration
✓ Feature engineering
✓ Smart INF handling (clamp + NaN)
✓ Smart NaN fill (by feature type)
✓ CV 5-fold feature selection
✓ Keep label
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
import time


# ── CONFIG ─────────────────────────────────────────────
CSV_PATH = "dataset_grouped.csv"
LABEL_COL = "activity"

ID_COLS = [
    "flow_id",
    "src_ip",
    "dst_ip",
    "timestamp",
    "label",
    "src_port",

]


# ── FEATURE ENGINEERING ────────────────────────────────
def engineer_ddos_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9

    if "syn_flag_counts" in df.columns and "packets_count" in df.columns:
        df["syn_per_packet"] = df["syn_flag_counts"] / (df["packets_count"] + eps)

    if "packets_IAT_mean" in df.columns and "packets_rate" in df.columns:
        df["iat_x_rate"] = df["packets_IAT_mean"] * df["packets_rate"]

    if "bwd_packets_count" in df.columns and "fwd_packets_count" in df.columns:
        df["bwd_fwd_ratio"] = (
            (df["bwd_packets_count"] + eps) / (df["fwd_packets_count"] + eps)
        )

    if "syn_flag_counts" in df.columns and "fin_flag_counts" in df.columns:
        df["syn_no_fin"] = (df["syn_flag_counts"] - df["fin_flag_counts"]).clip(lower=0)

    if "bytes_rate" in df.columns and "packets_rate" in df.columns:
        df["bytes_per_packet"] = (df["bytes_rate"] + eps) / (df["packets_rate"] + eps)

    return df


# ── FEATURE SELECTOR ───────────────────────────────────
class FeatureSelector:
    def __init__(self, variance_threshold=0.01, corr_threshold=0.95,
                 top_k=80, cv_folds=5):
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.top_k = top_k
        self.cv_folds = cv_folds
        self.selected_cols_ = None

    # 🔥 PREPROCESS (QUAN TRỌNG NHẤT)
    def _preprocess(self, X, cols):
        print("[Preprocess] Smart handling INF & NaN...")

        df = pd.DataFrame(X, columns=cols)

        ZERO_FILL_KEYWORDS = ["rate", "count", "duration", "bytes", "packets"]
        STAT_FILL_KEYWORDS = ["skew", "cov", "std"]

        for col in df.columns:
            col_lower = col.lower()

            # ── STEP 1: xử lý INF ───────────────────
            if "ratio" in col_lower or "per" in col_lower:
                # clamp INF → percentile 99
                val_99 = df[col].replace(np.inf, np.nan).quantile(0.99)
                df[col] = df[col].replace(np.inf, val_99)
            else:
                df[col] = df[col].replace(np.inf, np.nan)

            # ── STEP 2: fill NaN ───────────────────
            if df[col].isna().sum() == 0:
                continue

            if any(k in col_lower for k in ZERO_FILL_KEYWORDS):
                df[col] = df[col].fillna(0)

            elif any(k in col_lower for k in STAT_FILL_KEYWORDS):
                df[col] = df[col].fillna(df[col].median())

            elif "iat" in col_lower:
                df[col] = df[col].fillna(df[col].median())

            else:
                df[col] = df[col].fillna(df[col].mean())

        return df.values

    # ─────────────────────────────
    def fit_transform(self, X, y, cols):
        print("\n=== FEATURE SELECTION (CV 5-FOLD) ===")

        X = self._preprocess(X, cols)

        X, cols = self._variance(X, cols)
        X, cols = self._correlation(X, cols)
        X, cols = self._importance_cv(X, y, cols)

        self.selected_cols_ = cols
        print(f"\n✓ Final: {len(cols)} features")
        return X, cols

    # ─────────────────────────────
    def _variance(self, X, cols):
        t0 = time.time()

        selector = VarianceThreshold(self.variance_threshold)
        X_new = selector.fit_transform(X)

        cols_new = [c for c, m in zip(cols, selector.get_support()) if m]

        print(f"[Variance] {len(cols)} → {len(cols_new)} ({time.time()-t0:.1f}s)")
        return X_new, cols_new

    # ─────────────────────────────
    def _correlation(self, X, cols):
        t0 = time.time()

        df = pd.DataFrame(X, columns=cols)
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        drop = set()

        for col in upper.columns:
            if col in drop:
                continue

            high_corr = upper[col][upper[col] > self.corr_threshold]

            for partner in high_corr.index:
                if partner not in drop:
                    drop.add(partner)

        cols_new = [c for c in cols if c not in drop]
        idx = [cols.index(c) for c in cols_new]

        print(f"[Correlation] drop {len(drop)} → {len(cols_new)} ({time.time()-t0:.1f}s)")
        return X[:, idx], cols_new

    # ─────────────────────────────
    def _importance_cv(self, X, y, cols):
        t0 = time.time()

        print(f"\n[Importance] CV {self.cv_folds}-fold...")

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        importances = np.zeros(len(cols))

        for fold, (tr_idx, _) in enumerate(skf.split(X, y), 1):
            clf = ExtraTreesClassifier(
                n_estimators=150,
                max_depth=15,
                n_jobs=-1,
                random_state=fold,
                class_weight="balanced"
            )

            clf.fit(X[tr_idx], y[tr_idx])
            importances += clf.feature_importances_

            print(f"  Fold {fold}/{self.cv_folds} done", end="\r")

        importances /= self.cv_folds

        ranked_idx = np.argsort(importances)[::-1]
        top_idx = ranked_idx[:self.top_k]

        print("\nTop 10 features:")
        for i in ranked_idx[:10]:
            print(f"  {cols[i]:30} {importances[i]:.5f}")

        cols_new = [cols[i] for i in top_idx]

        print(f"[Importance] keep {len(cols_new)} ({time.time()-t0:.1f}s)")
        return X[:, top_idx], cols_new


# ── MAIN ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading dataset...")

    df = pd.read_csv(CSV_PATH, low_memory=False)

    # ── DROP ID ───────────────────────────────────────
    drop_cols = [c for c in ID_COLS if c in df.columns]
    if drop_cols:
        print(f"Drop ID columns: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)

    # ── FIX HANDSHAKE ───────────────────────────────
    if "handshake_duration" in df.columns:
        print("Fix handshake_duration...")

        df["is_handshake_complete"] = (
            df["handshake_duration"] != "not a complete handshake"
        ).astype(int)

        df["handshake_duration_clean"] = pd.to_numeric(
            df["handshake_duration"], errors="coerce"
        )

        df.drop(columns=["handshake_duration"], inplace=True)

    # ── LABEL ───────────────────────────────────────
    labels_raw = df[LABEL_COL].values
    y = pd.Series(labels_raw).astype("category").cat.codes.values

    # ── FEATURE ENGINEERING ─────────────────────────
    df = engineer_ddos_features(df)

    # ── TÁCH FEATURE / LABEL ────────────────────────
    df_features = df.drop(columns=[LABEL_COL])

    # ── NUMERIC ONLY ───────────────────────────────
    df_num = df_features.select_dtypes(include=[np.number]).copy()

    X = df_num.values
    cols = df_num.columns.tolist()

    # ── FEATURE SELECTION ───────────────────────────
    fs = FeatureSelector()
    X_new, selected = fs.fit_transform(X, y, cols)

    print(f"\nSelected {len(selected)} features")

    # ── SAVE ───────────────────────────────────────
    df_out = pd.DataFrame(X_new, columns=selected)
    df_out[LABEL_COL] = labels_raw

    df_out.to_csv("dataset_selected_processed.csv", index=False)

    print("Done.")