import pandas as pd

def process_and_export_data(input_file, output_file, label_col='activity'):
    print(f"Đang đọc dữ liệu từ: {input_file}...")
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{input_file}'. Vui lòng kiểm tra lại đường dẫn.")
        return

    # Xóa khoảng trắng thừa ở tên cột (đề phòng lỗi do file CSV)
    df.columns = df.columns.str.strip()

    if label_col not in df.columns:
        print(f"❌ Lỗi: Không tìm thấy cột '{label_col}' trong file CSV.")
        return

    print(f"Số lượng mẫu ban đầu: {len(df):,}")

    # 1. Loại bỏ các dòng có nhãn 'Suspicious'
    df = df[df[label_col].astype(str).str.strip() != 'Suspicious'].copy()
    print(f"Số lượng mẫu sau khi xóa 'Suspicious': {len(df):,}")

    mapping_dict = {
        'Benign': 'Benign',
        'Benign-Web_Browsing_HTTP-S': 'Benign',
        'Benign-Systemic': 'Benign',
        'Benign-Telnet': 'Benign',
        'Benign-SSH': 'Benign',
        'Benign-Email-Send': 'Benign',
        'Benign-Email-Receive': 'Benign',
        'Benign-FTP': 'Benign',
        
        'Attack-TCP-BYPass-V1': 'Attack-TCP-BYPass-V1',
        
        'Attack-TCP-IGMP': 'Attack-IGMP',
        
        'Attack-TCP-Flag-ACK': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-SYN-ACK': 'Attack-TCP-Flood',
        'Attack-TCP-Valid-SYN': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-SYN': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-ACK-PSH': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-MIX': 'Attack-TCP-Flood',
        'Attack-TCP-SYN': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-SYN-TFO': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-RST-ACK': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-SYN-TIME': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-OSYNP': 'Attack-TCP-Flood',
        'Attack-TCP-Flag-OSYN': 'Attack-TCP-Flood',
        'Attack-Killer-TCP': 'Attack-TCP-Flood',
        'Attack-Killall-v2': 'Attack-TCP-Flood',
        'Attack-TCP-Control': 'Attack-TCP-Flood'
    }

    df[label_col] = df[label_col].astype(str).str.strip()

    # 3. Áp dụng mapping
    df['new_activity'] = df[label_col].map(mapping_dict)

    # 4. Kiểm tra xem có nhãn nào bị lọt lưới (chưa được map) không
    unmapped = df[df['new_activity'].isna()]
    if not unmapped.empty:
        print("\n⚠️ Cảnh báo: Có các nhãn sau chưa được định nghĩa trong từ điển mapping và sẽ bị xóa:")
        print(unmapped[label_col].value_counts())
        
    # Loại bỏ các dòng không map được và ghi đè lại cột nhãn
    df = df.dropna(subset=['new_activity'])
    df[label_col] = df['new_activity']
    df = df.drop(columns=['new_activity'])

    # 5. In thống kê cuối cùng
    print("\n✓ Thống kê phân phối nhãn mới (Ultimate Taxonomy):")
    print(df[label_col].value_counts())
    print(f"\nTổng số mẫu cuối cùng: {len(df):,}")

    # 6. Xuất ra file CSV mới
    print(f"\n💾 Đang lưu dữ liệu ra file: {output_file}...")
    df.to_csv(output_file, index=False)
    print("🎉 Hoàn tất!")

# ==========================================
# CẤU HÌNH VÀ CHẠY
# ==========================================
if __name__ == "__main__":
    # Nguồn dữ liệu (File dataset gốc của bạn)
    INPUT_CSV = "merged_CSVs.csv"          
    
    # Đích đến (File đã được gom nhóm)
    OUTPUT_CSV = "dataset_grouped.csv"
    
    process_and_export_data(INPUT_CSV, OUTPUT_CSV, label_col='activity')