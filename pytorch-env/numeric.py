import pandas as pd
import numpy as np
import time
import os

def process_chunk(
    chunk: pd.DataFrame,
    selected_features: list,
    label_col: str,
    is_first_chunk: bool
) -> pd.DataFrame:
    """
    Xử lý từng chunk dữ liệu để tránh OOM RAM.
    """

    # ==========================================================
    # 1. Chỉ giữ feature + label thật sự tồn tại
    # ==========================================================
    cols_to_keep = [
        c for c in selected_features + [label_col]
        if c in chunk.columns
    ]

    chunk_clean = chunk[cols_to_keep].copy()

    # ==========================================================
    # 2. Xử lý đặc biệt handshake_duration
    # ==========================================================
    if 'handshake_duration' in chunk_clean.columns:
        chunk_clean['handshake_duration'] = (
            chunk_clean['handshake_duration']
            .replace('not a complete handshake', 0)
        )

        chunk_clean['handshake_duration'] = pd.to_numeric(
            chunk_clean['handshake_duration'],
            errors='coerce'
        )

    # ==========================================================
    # 3. Chỉ lấy các feature còn tồn tại
    # ==========================================================
    valid_features = [
        c for c in selected_features
        if c in chunk_clean.columns
    ]

    # ==========================================================
    # 4. Ép kiểu numeric + factorize object/category
    # ==========================================================
    for col in valid_features:

        if (
            chunk_clean[col].dtype == 'object'
            or chunk_clean[col].dtype.name == 'category'
        ):

            chunk_clean[col] = (
                chunk_clean[col]
                .astype(str)
                .str.strip()
            )

            # Encode nhanh
            chunk_clean[col] = pd.factorize(
                chunk_clean[col]
            )[0]

        chunk_clean[col] = pd.to_numeric(
            chunk_clean[col],
            errors='coerce'
        )

    # ==========================================================
    # 5. Replace INF/-INF
    # ==========================================================
    for col in valid_features:

        finite_series = (
            chunk_clean[col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        has_inf = np.isinf(chunk_clean[col]).any()

        if has_inf:

            if not finite_series.empty:

                max_finite = float(finite_series.max())
                min_finite = float(finite_series.min())

                inf_replacement = max(max_finite, 1000.0)

                neg_inf_replacement = min(min_finite, -1000.0)

            else:

                inf_replacement = 99999.0
                neg_inf_replacement = -99999.0

            chunk_clean[col] = chunk_clean[col].replace(
                np.inf,
                inf_replacement
            )

            chunk_clean[col] = chunk_clean[col].replace(
                -np.inf,
                neg_inf_replacement
            )

    # ==========================================================
    # 6. Fill NaN
    # ==========================================================
    chunk_clean[valid_features] = (
        chunk_clean[valid_features]
        .fillna(0.0)
    )

    # ==========================================================
    # 7. Làm sạch timestamp lỗi trong IAT
    # ==========================================================
    _TIMESTAMP_THRESHOLD = 1e8

    iat_cols = [
        c for c in valid_features
        if "iat" in c.lower()
    ]

    for col in iat_cols:

        mask_ts = chunk_clean[col] > _TIMESTAMP_THRESHOLD

        if mask_ts.any():
            chunk_clean.loc[mask_ts, col] = 0.0

    return chunk_clean


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":

    t0 = time.time()

    INPUT_CSV = "multilabels.csv"

    OUTPUT_CSV = "dataset_optimized_multi1.csv"

    LABEL_COL = "activity"

    CHUNK_SIZE = 100000

    # ==========================================================
    # Xóa output cũ
    # ==========================================================
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    print(
        f"1. Đang khởi tạo bộ đọc Chunking cho "
        f"'{INPUT_CSV}' "
        f"(Size: {CHUNK_SIZE:,} dòng/chunk)..."
    )

    try:

        # ======================================================
        # Đọc preview
        # ======================================================
        preview_df = pd.read_csv(
            INPUT_CSV,
            nrows=2
        )

        preview_df.columns = (
            preview_df.columns.str.strip()
        )

        # ======================================================
        # Kiểm tra label
        # ======================================================
        if LABEL_COL not in preview_df.columns:

            print(
                f"❌ LỖI: Không tìm thấy cột nhãn "
                f"'{LABEL_COL}'"
            )

            print("\nCác cột hiện có:\n")

            print(list(preview_df.columns))

        else:

            # ==================================================
            # Feature list
            # Loại bỏ:
            # - activity (label thật)
            # - label (cột dư)
            # ==================================================
            selected_features = [

                col for col in preview_df.columns

                if col not in [
                    LABEL_COL,
                    'label'
                ]
            ]

            print(
                f"   -> Tổng số features xác định được: "
                f"{len(selected_features)}"
            )

            # ==================================================
            # Chunk iterator
            # ==================================================
            chunk_iterator = pd.read_csv(
                INPUT_CSV,
                chunksize=CHUNK_SIZE,
                low_memory=False
            )

            # ==================================================
            # Loop chunk
            # ==================================================
            for i, chunk in enumerate(chunk_iterator):

                chunk.columns = (
                    chunk.columns.str.strip()
                )

                # ==============================================
                # Xóa cột label dư
                # ==============================================
                if 'label' in chunk.columns:

                    chunk.drop(
                        columns=['label'],
                        inplace=True
                    )

                is_first = (i == 0)

                # ==============================================
                # Process chunk
                # ==============================================
                chunk_ready = process_chunk(
                    chunk,
                    selected_features,
                    LABEL_COL,
                    is_first
                )

                # ==============================================
                # Ghi append
                # ==============================================
                chunk_ready.to_csv(
                    OUTPUT_CSV,
                    mode='a',
                    index=False,
                    header=is_first
                )

                processed_rows = (
                    (i + 1) * CHUNK_SIZE
                )

                print(
                    f"   -> Đã xử lý Chunk {i+1} "
                    f"(≈ {processed_rows:,} dòng)"
                )

            print(
                f"\n✅ XONG!"
                f"\nTổng thời gian: "
                f"{time.time() - t0:.2f} giây"
            )

            print(
                f"📁 Output: {OUTPUT_CSV}"
            )

    except FileNotFoundError:

        print(
            f"❌ Không tìm thấy file "
            f"'{INPUT_CSV}'"
        )