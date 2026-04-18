import pandas as pd
import glob
import os

def analyze_misclassifications(filepath=None):
    # Nếu không truyền tên file cụ thể, tự động tìm file txt mới nhất
    if filepath is None:
        files = glob.glob('misclassified_test_*.txt')
        if not files:
            print("❌ LỖI: Không tìm thấy file 'misclassified_test_*.txt' nào trong thư mục.")
            print("Hãy chắc chắn rằng bạn đã chạy mô hình và file txt đã được tạo ra.")
            return
        # Lấy file được tạo gần đây nhất
        filepath = max(files, key=os.path.getctime)

    print(f"\n📊 ĐANG PHÂN TÍCH TỆP: {filepath}")
    print("="*70)

    # Đọc dữ liệu, bỏ qua cột Features để tiết kiệm RAM nếu file quá lớn
    df = pd.read_csv(filepath, sep='\t', usecols=['True_Label', 'Predicted_Label'])
    total_errors = len(df)
    
    if total_errors == 0:
        print("Tuyệt vời! Không có mẫu nào bị dự đoán sai trong file này.")
        return
        
    print(f"Tổng số mẫu dự đoán sai: {total_errors}\n")

    # ---------------------------------------------------------
    # 1. Các cặp dự đoán sai phổ biến nhất
    # ---------------------------------------------------------
    print("1️⃣ TOP 10 CẶP NHẦM LẪN NGHIÊM TRỌNG NHẤT (Thực tế -> Dự đoán):")
    pair_stats = df.groupby(['True_Label', 'Predicted_Label']).size().reset_index(name='Số lượng')
    pair_stats['Tỷ lệ (%)'] = (pair_stats['Số lượng'] / total_errors * 100).round(2)
    pair_stats = pair_stats.sort_values('Số lượng', ascending=False).head(10)
    print(pair_stats.to_string(index=False))
    print("-" * 70)

    # ---------------------------------------------------------
    # 2. Nhãn thực tế bị nhận diện sai nhiều nhất
    # ---------------------------------------------------------
    print("2️⃣ NHÃN THỰC TẾ BỊ NHẬN DIỆN MẤT NHIỀU NHẤT (Recall yếu):")
    true_stats = df['True_Label'].value_counts().reset_index()
    true_stats.columns = ['Nhãn Thực Tế', 'Số lần bị đoán sai']
    true_stats['Tỷ lệ trong tổng lỗi (%)'] = (true_stats['Số lần bị đoán sai'] / total_errors * 100).round(2)
    print(true_stats.to_string(index=False))
    print("-" * 70)

    # ---------------------------------------------------------
    # 3. Lớp bị mô hình "đổ oan" nhiều nhất
    # ---------------------------------------------------------
    print("3️⃣ NHÃN DỰ ĐOÁN SAI NHIỀU NHẤT (Precision yếu / Hay bị đoán nhầm vào):")
    pred_stats = df['Predicted_Label'].value_counts().reset_index()
    pred_stats.columns = ['Nhãn Dự Đoán', 'Số lần đoán oan']
    pred_stats['Tỷ lệ trong tổng lỗi (%)'] = (pred_stats['Số lần đoán oan'] / total_errors * 100).round(2)
    print(pred_stats.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    # Bạn có thể truyền thẳng tên file vào đây nếu muốn, ví dụ:
    analyze_misclassifications("misclassified_test_20260411_1632.txt")
