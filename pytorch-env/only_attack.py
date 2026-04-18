import pandas as pd
import os

def apply_ultimate_grouping(input_path, output_path, label_col='activity'):
    if not os.path.exists(input_path):
        print(f"❌ Lỗi: Không tìm thấy file {input_path}")
        return

    print(f"📥 1. Đang tải dữ liệu từ: {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    
    # Đảm bảo nhãn là chuỗi và không có khoảng trắng thừa
    df[label_col] = df[label_col].astype(str).str.strip()

    print("🔄 2. Đang thực hiện gom nhóm (Ultimate Taxonomy)...")
    
    # Từ điển gom nhóm (Bao trùm cả nhãn gốc 17 lớp và nhãn đã gom 6 lớp)
    ultimate_mapping = {
        # --- GOM TẤT CẢ CÁC LUỒNG TCP FLOOD VÀO 1 NHÓM ---
        # 1. Nếu data đang ở dạng 6 nhóm
        'Attack-SYN-Group': 'Attack-TCP-Flood',
        'Attack-ACK-RST-Group': 'Attack-TCP-Flood',
        'Attack-Volumetric-Group': 'Attack-TCP-Flood',
        'Attack-Mixed-Flags': 'Attack-TCP-Flood',
        
        # 2. Nếu data đang ở dạng mười mấy nhãn chi tiết gốc
        'Attack-TCP-Flag-SYN': 'Attack-TCP-Flood',
        'Attack-TCP-SYN': 'Attack-TCP-Flood',
        'Attack-TCP-Valid-SYN': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-SYN-ACK': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-SYN-TFO': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-SYN-TIME': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-OSYN': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-OSYNP': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-ACK': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-ACK-PSH': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-RST-ACK': 'Attack-TCP-Flood',
        'Attack-Killall-v2': 'Attack-TCP-Flood',
        'Attack-Killer-TCP': 'Attack-TCP-Flood',
        'Attack-TCP-Control': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-MIX': 'Attack-TCP-Flood',

        # --- CÁC LỚP GIỮ NGUYÊN HOẶC CHUẨN HOÁ TÊN ---
        'Attack-TCP-IGMP': 'Attack-IGMP',
        'Attack-IGMP': 'Attack-IGMP',
        'Attack-TCP-BYPass-V1': 'Attack-TCP-BYPass-V1'
    }

    # Áp dụng thay thế nhãn
    df[label_col] = df[label_col].map(ultimate_mapping).fillna(df[label_col])
    
    # In ra báo cáo các nhóm sau khi gom
    print("\n✅ Thống kê số lượng mẫu sau khi gom nhóm:")
    print(df[label_col].value_counts())

    print(f"\n💾 3. Đang xuất ra file CSV mới: {output_path}...")
    df.to_csv(output_path, index=False)
    print("🎉 Hoàn tất! Hãy dùng file mới này để huấn luyện mô hình XGBoost.")

if __name__ == "__main__":
    # Tên file đầu vào (Dùng file đã cắt bỏ header_bytes của bạn)
    INPUT_CSV = "dataset_no_header_bytes.csv" 
    
    # Tên file đầu ra
    OUTPUT_CSV = "dataset_ultimate_3classes.csv"
    
    apply_ultimate_grouping(INPUT_CSV, OUTPUT_CSV)