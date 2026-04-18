import pandas as pd

def process_and_export_data(input_file, output_file, label_col='activity'):
    print(f"Đang đọc dữ liệu từ: {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{input_file}'. Vui lòng kiểm tra lại đường dẫn.")
        return

    # Xóa khoảng trắng thừa ở tên cột (đề phòng lỗi do file CSV)
    df.columns = df.columns.str.strip()

    if label_col not in df.columns:
        print(f"Lỗi: Không tìm thấy cột '{label_col}' trong file CSV.")
        return

    print(f"Số lượng mẫu ban đầu: {len(df)}")

    # 1. Loại bỏ các dòng có nhãn 'Suspicious'
    df = df[df[label_col].str.strip() != 'Suspicious'].copy()
    print(f"Số lượng mẫu sau khi xóa 'Suspicious': {len(df)}")

    mapping_dict = {
        'Benign': 'Benign',
        'Benign-Web_Browsing_HTTP-S': 'Benign',
        'Benign-Systemic': 'Benign',
        'Benign-Telnet': 'Benign',
        'Benign-SSH': 'Benign',
        'Benign-Email-Send': 'Benign',
        'Benign-Email-Receive': 'Benign',
        'Benign-FTP': 'Benign',
        
        'Attack-TCP-BYPass-V1': 'Attack-Bypass',
        
        'Attack-TCP-Flag-ACK': 'Attack-TCP-Anomaly',
        'Attack-TCP-Flag-SYN-ACK': 'Attack-TCP-Anomaly',
        'Attack-TCP-Valid-SYN': 'Attack-TCP-Anomaly',
        'Attack-TCP-Flag-SYN': 'Attack-TCP-Anomaly',
        'Attack-TCP-Flag-ACK-PSH': 'Attack-TCP-Anomaly',
        'Attack-TCP-Flag-MIX': 'Attack-TCP-Anomaly',
        'Attack-TCP-SYN': 'Attack-TCP-Anomaly',
        'Attack-TCP-Flag-SYN-TFO': 'Attack-TCP-Anomaly',
        'Attack-TCP-Flag-RST-ACK': 'Attack-TCP-Anomaly',
        'Attack-TCP-Flag-SYN-TIME': 'Attack-TCP-Anomaly',
        'Attack-TCP-Flag-OSYNP': 'Attack-TCP-Anomaly',
        'Attack-TCP-Flag-OSYN': 'Attack-TCP-Anomaly',
        
        'Attack-TCP-IGMP': 'Attack-TCP-Anomaly',
        'Attack-Killer-TCP': 'Attack-TCP-Anomaly',
        'Attack-Killall-v2': 'Attack-TCP-Anomaly',
        'Attack-TCP-Control': 'Attack-TCP-Anomaly'
    }

    df[label_col] = df[label_col].str.strip()

    # 3. Áp dụng mapping
    df['new_activity'] = df[label_col].map(mapping_dict)

    unmapped = df[df['new_activity'].isna()]
    if not unmapped.empty:
        print("\nCảnh báo: Có các nhãn sau chưa được định nghĩa trong từ điển mapping và sẽ bị xóa:")
        print(unmapped[label_col].value_counts())
        
    df = df.dropna(subset=['new_activity'])
    df[label_col] = df['new_activity']
    df = df.drop(columns=['new_activity'])

    # 5. In thống kê cuối cùng
    print("\n✓ Thống kê phân phối nhãn mới:")
    print(df[label_col].value_counts())
    print(f"\nTổng số mẫu cuối cùng: {len(df)}")

    # 6. Xuất ra file CSV mới
    print(f"\nĐang lưu dữ liệu ra file: {output_file}...")
    df.to_csv(output_file, index=False)
    print("Hoàn tất!")

# ==========================================
# CẤU HÌNH VÀ CHẠY
# ==========================================
if __name__ == "__main__":
    INPUT_CSV = "merged_CSVs.csv"          
    OUTPUT_CSV = "dataset_grouped.csv"
    
    process_and_export_data(INPUT_CSV, OUTPUT_CSV, label_col='activity')