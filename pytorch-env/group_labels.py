import pandas as pd

def process_and_export_data(input_file, output_file, label_col='activity'):
    print(f"Đang đọc dữ liệu từ: {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{input_file}'. Vui lòng kiểm tra lại đường dẫn.")
        return

    # Xóa khoảng trắng thừa ở tên cột
    df.columns = df.columns.str.strip()

    if label_col not in df.columns:
        print(f"Lỗi: Không tìm thấy cột '{label_col}' trong file CSV.")
        return

    print(f"Số lượng mẫu ban đầu: {len(df)}")

    # Chuẩn hóa label (strip khoảng trắng)
    df[label_col] = df[label_col].astype(str).str.strip()

    # 1. Loại bỏ 'Suspicious'
    df = df[df[label_col] != 'Suspicious'].copy()
    print(f"Số lượng mẫu sau khi xóa 'Suspicious': {len(df)}")

    # 2. Gộp tất cả Benign → 'Benign'
    df[label_col] = df[label_col].apply(
        lambda x: 'Benign' if x.startswith('Benign') else x
    )

    # 3. (Optional) Cảnh báo nếu có label lạ (NaN hoặc rỗng)
    invalid = df[df[label_col].isna() | (df[label_col] == "")]
    if not invalid.empty:
        print("\nCảnh báo: Có các dòng có label không hợp lệ:")
        print(invalid.head())
        df = df.dropna(subset=[label_col])

    # 4. Thống kê cuối
    print("\n✓ Thống kê phân phối nhãn sau khi xử lý:")
    print(df[label_col].value_counts())

    print(f"\nTổng số mẫu cuối cùng: {len(df)}")

    # 5. Xuất file
    print(f"\nĐang lưu dữ liệu ra file: {output_file}...")
    df.to_csv(output_file, index=False)
    print("Hoàn tất!")


# ==========================================
# CẤU HÌNH VÀ CHẠY
# ==========================================
if __name__ == "__main__":
    INPUT_CSV = "merged_CSVs.csv"
    OUTPUT_CSV = "multilabels.csv"

    process_and_export_data(INPUT_CSV, OUTPUT_CSV, label_col='activity')