import pandas as pd

# 1. Đọc dữ liệu từ file CSV ban đầu (thay đổi tên file cho phù hợp với bạn)
input_file = 'dataset_ready_for_model.csv'
output_file = 'filtered_dataset.csv'

print(f"Đang đọc dữ liệu từ {input_file}...")
df = pd.read_csv(input_file, low_memory=False)

# 2. Xóa khoảng trắng thừa ở tên cột (để đảm bảo không bị lỗi khi gọi cột 'activity')
df.columns = df.columns.str.strip()

# 3. Lọc các hàng có activity là 'Attack-TCP-Flags' hoặc 'Attack-Tools-Control'
# Sử dụng phương thức .isin() để lọc nhiều giá trị cùng lúc
target_activities = ['Attack-TCP-Flags', 'Attack-Tools-Control']
df_filtered = df[df['activity'].isin(target_activities)]

# 4. Lưu dữ liệu đã lọc ra file mới
print(f"Đã tìm thấy {len(df_filtered)} hàng phù hợp.")
print(f"Đang lưu dữ liệu mới ra {output_file}...")
df_filtered.to_csv(output_file, index=False)

print("Hoàn tất!")