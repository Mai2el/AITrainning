import os
import pandas as pd

# 1. Định nghĩa đường dẫn file dữ liệu
# (Bạn hãy thay 'dataset.csv' bằng tên hoặc đường dẫn file thực tế của bạn)
path_to_dataset = "filtered.csv" 
label_column_name = "label"  # Đổi thành 'Label' hoặc 'Class' nếu dữ liệu viết hoa

# Kiểm tra xem file có tồn tại hay không trước khi xử lý
if not os.path.exists(path_to_dataset):
    print(f"❌ Lỗi: Không tìm thấy file tại '{path_to_dataset}'. Vui lòng kiểm tra lại đường dẫn!")
else:
    # 2. Đọc tập dữ liệu
    print("⏳ Bước 1: Đang tải tập dữ liệu...")
    df = pd.read_csv(path_to_dataset)
    print(f"✅ Tải thành công! Kích thước dữ liệu gốc: {df.shape[0]} dòng, {df.shape[1]} cột.\n")
    print("-" * 50)

    # 3. Liệt kê toàn bộ đặc trưng (features) ban đầu
    print("📋 Bước 2: Danh sách toàn bộ thuộc tính ban đầu:")
    initial_features = df.columns.tolist()
    for index, feature in enumerate(initial_features, 1):
        print(f"  {index}. {feature}")
    print(f"👉 Tổng số lượng cột ban đầu: {len(initial_features)}\n")
    print("-" * 50)

    # 4. Kiểm tra và thực hiện drop cột nhãn để chia tập X (Đặc trưng) và y (Nhãn)
    if label_column_name in df.columns:
        print(f"✂️ Bước 3: Đang tiến hành xóa/tách cột nhãn '{label_column_name}'...")
        
        # Tập X: Chứa toàn bộ các feature dùng để học (đã drop cột label)
        X = df.drop(columns=[label_column_name])
        
        # Tập y: Chỉ chứa riêng cột nhãn để làm mục tiêu dự báo
        y = df[label_column_name]
        
        print("✅ Đã tách dữ liệu thành công!\n")
        print("-" * 50)

        # 5. Kiểm tra và liệt kê lại danh sách đặc trưng sau khi đã drop cột label
        print("📊 Bước 4: Kiểm tra lại danh sách đặc trưng trong tập X (Dùng để train):")
        final_features = X.columns.tolist()
        for index, feature in enumerate(final_features, 1):
            print(f"  {index}. {feature}")
            
        print(f"\n📈 KẾT QUẢ CUỐI CÙNG:")
        print(f"  - Số lượng đặc trưng còn lại trong tập X: {len(final_features)} cột.")
        print(f"  - Kích thước Ma trận đặc trưng X: {X.shape}")
        print(f"  - Kích thước Vector nhãn y: {y.shape}")
        
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy cột nào có tên chính xác là '{label_column_name}' trong dataset!")
        print("Vui lòng kiểm tra xem tên cột nhãn trong file của bạn có viết hoa hay thay đổi gì không (Ví dụ: 'Label', 'class', 'target'...).")