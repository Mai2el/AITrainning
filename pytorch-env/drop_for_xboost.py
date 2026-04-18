import pandas as pd
import os

def drop_overfitted_features(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"❌ Lỗi: Không tìm thấy file {input_path}")
        return

    print(f"📥 1. Đang tải dữ liệu từ: {input_path}...")
    df = pd.read_csv(input_path)
    
    initial_cols = len(df.columns)
    print(f"   -> Tổng số cột ban đầu: {initial_cols}")
    
    # Từ khóa nhận diện các cột "học vẹt" cần xóa
    keywords_to_drop = ["header_bytes"]
    
    # Tìm kiếm các cột khớp với từ khóa
    cols_to_drop = []
    for col in df.columns:
        for keyword in keywords_to_drop:
            if keyword.lower() in col.lower():
                cols_to_drop.append(col)
                break 
                
    if not cols_to_drop:
        print("⚠️ Không tìm thấy cột nào chứa từ khóa cần xóa.")
    else:
        print(f"\n🗑️ 2. Đã tìm thấy {len(cols_to_drop)} tính năng gây nhiễu/học vẹt:")
        for c in cols_to_drop:
            print(f"   - {c}")
            
        # Thực hiện DROP (xóa) cột
        df = df.drop(columns=cols_to_drop)
        print(f"\n✅ Đã xóa thành công! Số lượng cột còn lại: {len(df.columns)}")

    # Lưu file mới
    print(f"\n💾 3. Đang xuất ra file CSV mới: {output_path}...")
    df.to_csv(output_path, index=False)
    print("🎉 Hoàn tất! Hãy dùng file mới này để huấn luyện lại mô hình.")

if __name__ == "__main__":
    # Tên file hiện tại bạn đang dùng cho XGBoost
    INPUT_CSV = "dataset_ready_for_model.csv" 
    
    # Tên file xuất ra sau khi đã cắt bỏ features
    OUTPUT_CSV = "dataset_no_header_bytes.csv"
    
    drop_overfitted_features(INPUT_CSV, OUTPUT_CSV)