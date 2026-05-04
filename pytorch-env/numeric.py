import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time

def prepare_final_data(df: pd.DataFrame, selected_features: list, label_col: str) -> pd.DataFrame:
    """
    Hàm chuẩn hóa: Xử lý ngoại lệ, Label Encode cột chữ, và ép toàn bộ về số.
    """
    print("\n[Data Preparation] Đang chuẩn hóa tập dữ liệu cuối cùng...")
    
    # 1. Lọc DataFrame chỉ lấy các feature đã chọn và cột nhãn
    cols_to_keep = selected_features + [label_col]
    df_clean = df[cols_to_keep].copy()

    # =====================================================================
    # 🌟 BƯỚC XỬ LÝ NGOẠI LỆ (TRƯỚC KHI LABEL ENCODE)
    # =====================================================================
    # Xử lý cột handshake_duration bị trộn lẫn chữ và số thời gian (Mixed types)
    if 'handshake_duration' in df_clean.columns:
        # Thay thế chuỗi bằng số 0
        df_clean['handshake_duration'] = df_clean['handshake_duration'].replace('not a complete handshake', 0)
        # Ép ngay về kiểu float để vòng lặp LabelEncode bên dưới bỏ qua nó
        df_clean['handshake_duration'] = pd.to_numeric(df_clean['handshake_duration'], errors='coerce')
        print("   -> Đã xử lý ngoại lệ cột 'handshake_duration': Đổi chữ thành số 0 và ép về float.")
    # =====================================================================

    le = LabelEncoder()
    encoded_cols = []

    # 2. Tự động tìm và Label Encode các cột feature (chỉ những cột thực sự là chữ)
    for col in selected_features:
        if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category':
            # Ép về string trước khi encode để tránh lỗi mixed types
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            encoded_cols.append(col)
            
    if encoded_cols:
        print(f"   -> Đã tự động Label Encode các cột categorical: {encoded_cols}")

    # 3. Ép TẤT CẢ các feature về kiểu số thực (float) an toàn
    for col in selected_features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 4. Xử lý các giá trị vô cực (Infinity) và khuyết thiếu (NaN)
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    missing_count = df_clean[selected_features].isna().sum().sum()
    if missing_count > 0:
        # ⚠️  QUAN TRỌNG: Dùng 0 cho TẤT CẢ (kể cả IAT/duration).
        # Lý do: IAT features chứa mixed data (duration thực ~µs lẫn Unix timestamp ~1.7e9).
        # Nếu fill median sẽ inject giá trị timestamp vào các hàng NaN → nhiễm loạn group anomaly.
        # Tokenizer sẽ xử lý đúng: 0 → giữ nguyên trong zero_heavy / anomaly group.
        print(f"   -> Đang lấp đầy {missing_count:,} giá trị NaN/Inf bằng 0 (bao gồm IAT/duration)...")
        for col in selected_features:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(0.0)

    # =====================================================================
    # 🌟 BƯỚC LÀM SẠCH TIMESTAMPS TRONG IAT FEATURES
    # =====================================================================
    # Các features IAT đôi khi chứa Unix timestamp (~1.7e9 giây)
    # thay vì duration thực (~µs đến giây) khi flow chỉ có 1 packet.
    # Phải reset về 0 NGAY TẠI ĐÂY – trước khi select_dta.py tính
    # engineered features (iat_x_rate, bwd_fwd_ratio) sử dụng chúng.
    _TIMESTAMP_THRESHOLD = 1e8  # IAT hợp lệ không bao giờ vượt quá 1e8 giây (~3.17 năm)
    _iat_cols = [c for c in selected_features if "iat" in c.lower()]
    _cleaned_ts = []
    for col in _iat_cols:
        if col in df_clean.columns:
            mask_ts = df_clean[col] > _TIMESTAMP_THRESHOLD
            if mask_ts.any():
                df_clean.loc[mask_ts, col] = 0.0
                _cleaned_ts.append(f"{col}({mask_ts.sum():,})")
    if _cleaned_ts:
        print(f"   -> [Timestamp Clean] Reset timestamp → 0 trong: {_cleaned_ts}")
    # =====================================================================

    if df_clean[label_col].dtype == 'object':
        print(f"   -> Giữ nguyên cột nhãn gốc '{label_col}' để tokenizer và start.py dùng.")

    print("[Data Preparation] Hoàn tất! Dữ liệu đã sạch và sẵn sàng 100% dạng số.")
    return df_clean

# ------------------------------------------------------------------
# STANDALONE DEMO (MAIN)
# ------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    
    INPUT_CSV = "multilabels.csv"
    OUTPUT_CSV = "dataset_optimized.csv"
    LABEL_COL = "activity" # Đổi tên nếu cột nhãn của bạn khác

    print(f"1. Đang đọc dữ liệu từ '{INPUT_CSV}'...")
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False)
        
        # Xóa khoảng trắng thừa ở tên cột
        df.columns = df.columns.str.strip()

        if LABEL_COL not in df.columns:
            print(f"❌ LỖI: Không tìm thấy cột nhãn '{LABEL_COL}' trong file CSV.")
            print(f"Các cột hiện có: {list(df.columns)}")
        else:
            # Lấy danh sách tất cả các feature (loại trừ cột nhãn)
            selected_features = [col for col in df.columns if col != LABEL_COL]
            print(f"   -> Tổng số features: {len(selected_features)}")
            print(f"   -> Tổng số dòng dữ liệu hợp lệ: {len(df):,}")
            
            # Thực thi hàm chuẩn hóa
            df_ready = prepare_final_data(df, selected_features, LABEL_COL)
            
            # Lưu ra file mới
            print(f"\n2. Đang lưu dữ liệu đã chuẩn hóa ra file '{OUTPUT_CSV}'...")
            df_ready.to_csv(OUTPUT_CSV, index=False)
            
            print(f"\n✅ XONG! Tổng thời gian chạy: {time.time() - t0:.2f} giây.")
            print("File '.csv' đã hoàn hảo để đưa vào train Deep Learning/Machine Learning!")
            
    except FileNotFoundError:
         print(f"❌ LỖI: Không tìm thấy file '{INPUT_CSV}'. Hãy đảm bảo file đang nằm cùng thư mục với script này.")