"""
tabular_tokenizer.py
--------------------
Script tiền xử lý và nhúng dữ liệu Tabular thành Token ID.
Đã cập nhật: Tách riêng nhóm Đặc trưng phân loại (Categorical/Flags) để tránh mất mát dữ liệu.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import train_test_split
import time
import warnings

warnings.filterwarnings("ignore")

_VAL_TEST_FRACTION = 0.30
_TEST_WITHIN_TEMP_FRACTION = 0.50
_SPLIT_RANDOM_STATE = 42

PORT_TO_TOKEN_MAP = {
    0: 0,
    22: 1,
    53: 2,
    80: 3,
    443: 4,
    1024: 5,   
    65535: 6   
}

def direct_port_tokenize(port):
    """Phân nhóm và nhúng trực tiếp port thành Token ID"""
    try:
        port = int(port)
    except:
        return PORT_TO_TOKEN_MAP[0]
        
    if port in [80, 443, 53, 22]:
        return PORT_TO_TOKEN_MAP[port]
    elif port < 1024:
        return PORT_TO_TOKEN_MAP[1024]
    else:
        return PORT_TO_TOKEN_MAP[65535]


def _assign_feature_groups(df: pd.DataFrame, features: list, train_idx: np.ndarray):
    sparse_cols, heavy_cols, dense_cols, zero_skew_cols, embed_cols, cat_cols = [], [], [], [], [], []
    sparse_idx, heavy_idx, dense_idx, zero_skew_idx, embed_idx, cat_idx = [], [], [], [], [], []

    for idx, col in enumerate(features):
        if "port" in col.lower():
            embed_cols.append(col)
            embed_idx.append(idx)
            continue

        s = df.loc[train_idx, col]
        n_unique = s.nunique()
        zero_ratio = (s == 0).mean()
        max_val = s.max()

        # [SỬA LỖI TẠI ĐÂY] Tách riêng các cờ (Flags) có ít hơn 15 giá trị duy nhất
        if n_unique < 15:
            cat_cols.append(col)
            cat_idx.append(idx)
        elif zero_ratio > 0.10 and max_val > 100 and n_unique > 20:
            zero_skew_cols.append(col)
            zero_skew_idx.append(idx)
        elif n_unique <= 256 or zero_ratio > 0.85:
            sparse_cols.append(col)
            sparse_idx.append(idx)
        elif max_val > 10000:
            heavy_cols.append(col)
            heavy_idx.append(idx)
        else:
            dense_cols.append(col)
            dense_idx.append(idx)

    return (
        sparse_cols, heavy_cols, dense_cols, zero_skew_cols, embed_cols, cat_cols,
        sparse_idx, heavy_idx, dense_idx, zero_skew_idx, embed_idx, cat_idx
    )


def _safe_discretizer_fit_transform(
    discretizer: KBinsDiscretizer, X_train: np.ndarray, X_full: np.ndarray
) -> np.ndarray:
    nb = discretizer.n_bins
    if not np.isscalar(nb):
        nb = int(nb[0])
    try:
        discretizer.fit(X_train)
        out = discretizer.transform(X_full)
        cap = int(discretizer.n_bins) if np.isscalar(discretizer.n_bins) else int(discretizer.n_bins[0])
    except ValueError:
        cap = min(nb, max(2, X_train.shape[0] // 2))
        fallback = KBinsDiscretizer(
            n_bins=cap, encode="ordinal", strategy="uniform",
            subsample=min(200000, X_train.shape[0]),
        )
        fallback.fit(X_train)
        out = fallback.transform(X_full)
        cap = int(fallback.n_bins) if np.isscalar(fallback.n_bins) else int(fallback.n_bins[0])
    out = np.asarray(out, dtype=np.float64)
    return np.clip(np.rint(out), 0, cap - 1).astype(int)


def _zero_isolated_log_quantile_transform(
    X_train: np.ndarray, X_full: np.ndarray, n_bins: int
) -> np.ndarray:
    out_full = np.zeros_like(X_full, dtype=int)

    for col_idx in range(X_train.shape[1]):
        col_train = X_train[:, col_idx]
        col_full = X_full[:, col_idx]

        non_zero_mask_train = col_train > 0
        train_non_zero = col_train[non_zero_mask_train]

        if len(train_non_zero) < 10:
            continue

        train_nz_log = np.log1p(train_non_zero).reshape(-1, 1)

        disc = KBinsDiscretizer(
            n_bins=n_bins - 1, encode="ordinal", strategy="quantile",
            subsample=min(200000, len(train_nz_log))
        )

        try:
            disc.fit(train_nz_log)
        except ValueError:
            disc = KBinsDiscretizer(
                n_bins=n_bins - 1, encode="ordinal", strategy="uniform",
            )
            disc.fit(train_nz_log)

        non_zero_mask_full = col_full > 0
        if np.any(non_zero_mask_full):
            full_nz_log = np.log1p(col_full[non_zero_mask_full]).reshape(-1, 1)
            binned_nz = disc.transform(full_nz_log).flatten() + 1
            out_full[non_zero_mask_full, col_idx] = binned_nz

    return out_full


def smart_process_and_tokenize(
    input_csv, output_txt, n_bins=128, label_col="activity",
    val_test_fraction=_VAL_TEST_FRACTION,
    test_within_temp_fraction=_TEST_WITHIN_TEMP_FRACTION,
    random_state=_SPLIT_RANDOM_STATE,
):
    t0 = time.time()
    print(f"1. Đang đọc dữ liệu từ: '{input_csv}'...")

    df = pd.read_csv(input_csv, low_memory=False)
    df.columns = df.columns.str.strip()

    drop_cols = ["label", "flow_id", "timestamp", "src_ip", "dst_ip", "src_port"]
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 🔥 Map port trực tiếp sang Token ID ngay từ đầu
    for col in df.columns:
        if "port" in col.lower():
            df[col] = df[col].apply(direct_port_tokenize)

    for col in df.columns:
        if col != label_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)

    print("\n2. Encode Label...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].values)

    features = [c for c in df.columns if c != label_col]
    n = len(df)
    indices = np.arange(n)

    idx_train, idx_temp = train_test_split(
        indices, test_size=val_test_fraction, stratify=y, random_state=random_state,
    )
    y_temp = y[idx_temp]
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=test_within_temp_fraction, stratify=y_temp, random_state=random_state,
    )

    X_tokenized = np.zeros((n, len(features)), dtype=int)

    # Nhận thêm cat_cols và cat_idx
    (
        sparse_cols, heavy_cols, dense_cols, zero_skew_cols, embed_cols, cat_cols,
        sparse_idx, heavy_idx, dense_idx, zero_skew_idx, embed_idx, cat_idx
    ) = _assign_feature_groups(df, features, idx_train)

    print("\n3. Feature groups:")
    print(f"   Cat (Flags): {len(cat_cols)}")
    print(f"   Sparse: {len(sparse_cols)}")
    print(f"   Heavy : {len(heavy_cols)}")
    print(f"   Dense : {len(dense_cols)}")
    print(f"   ZeroSkew: {len(zero_skew_cols)}")
    print(f"   Embed(port): {len(embed_cols)}")

    print("\n4. Transform...")

    if cat_cols:
        for col, idx in zip(cat_cols, cat_idx):
            encoded_vals, _ = pd.factorize(df[col])
            X_tokenized[:, idx] = encoded_vals

    if sparse_cols:
        X_tr = df.loc[idx_train, sparse_cols].values
        X_all = df[sparse_cols].values
        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
        X_tokenized[:, sparse_idx] = _safe_discretizer_fit_transform(disc, X_tr, X_all)

    if heavy_cols:
        X_tr = df.loc[idx_train, heavy_cols].values
        X_all = df[heavy_cols].values
        scaler = StandardScaler()
        scaler.fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_all = scaler.transform(X_all)
        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
        X_tokenized[:, heavy_idx] = _safe_discretizer_fit_transform(disc, X_tr, X_all)

    if dense_cols:
        X_tr = df.loc[idx_train, dense_cols].values
        X_all = df[dense_cols].values
        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        X_tokenized[:, dense_idx] = _safe_discretizer_fit_transform(disc, X_tr, X_all)

    if zero_skew_cols:
        X_tr = df.loc[idx_train, zero_skew_cols].values
        X_all = df[zero_skew_cols].values
        X_tokenized[:, zero_skew_idx] = _zero_isolated_log_quantile_transform(X_tr, X_all, n_bins)

    if embed_cols:
        for col, idx in zip(embed_cols, embed_idx):
            X_tokenized[:, idx] = df[col].astype(int).values

    print(f"\n5. Save: {output_txt}")
    df_tokenized = pd.DataFrame(X_tokenized, columns=features)
    df_tokenized[label_col] = df[label_col].values
    df_tokenized.to_csv(output_txt, sep="\t", index=False)

    print(f"✅ Done! Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    smart_process_and_tokenize(
        "dataset_ready_for_model.csv",
        "dataset_tokenized.txt",
        n_bins=256,
        label_col="activity"
    )