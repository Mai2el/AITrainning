import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# =============================================================================
# 1. THUẬT TOÁN TRÍCH CHỌN ĐẶC TRƯNG LAI (HYBRID FEATURE SELECTION)
# =============================================================================
def hybrid_feature_selection(X, y, k_anova=100, k_ig=100, k_et=100, final_k=40):
    """
    Quy trình lai kết hợp ANOVA, Information Gain và Extra Trees.
    Sử dụng Điểm số Lai tổng hợp chuẩn hóa nhằm tối ưu hóa độ chính xác.
    
    Parameters:
    - X: DataFrame dữ liệu đã scale tạm thời
    - y: Mảng/Series nhãn mục tiêu (dạng số)
    - final_k: Số lượng đặc trưng cốt lõi cuối cùng (Bài báo dùng 40)
    """
    feature_names = np.array(X.columns)
    num_initial_features = len(feature_names)
    print(f"🚀 Tổng số đặc trưng số đầu vào thuật toán chọn: {num_initial_features}")
    
    # Đảm bảo các cấu hình k không vượt quá tổng số đặc trưng sẵn có
    k_anova = min(k_anova, num_initial_features)
    k_ig = min(k_ig, num_initial_features)
    k_et = min(k_et, num_initial_features)
    final_k = min(final_k, num_initial_features)

    # --- VÒNG 1: ANOVA F-value ---
    print("\n[Vòng 1] Đang tính toán ANOVA F-value...")
    anova_scores, _ = f_classif(X, y)
    anova_scores = np.nan_to_num(anova_scores, nan=0.0) # Xử lý an toàn nếu chia cho 0
    anova_top_indices = np.argsort(anova_scores)[::-1][:k_anova]
    anova_features = set(feature_names[anova_top_indices])
    print(f"-> ANOVA đã chọn lọc {len(anova_features)} đặc trưng tiềm năng.")

    # --- VÒNG 2: Information Gain (Mutual Information) ---
    print("\n[Vòng 2] Đang tính toán Information Gain (Mutual Information)...")
    ig_scores = mutual_info_classif(X, y, random_state=42)
    ig_top_indices = np.argsort(ig_scores)[::-1][:k_ig]
    ig_features = set(feature_names[ig_top_indices])
    print(f"-> Information Gain đã chọn lọc {len(ig_features)} đặc trưng tiềm năng.")

    # --- VÒNG 3: Extra Trees Feature Importance ---
    print("\n[Vòng 3] Đang huấn luyện mô hình Extra Trees Classifier...")
    et_model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    et_model.fit(X, y)
    et_scores = et_model.feature_importances_
    et_top_indices = np.argsort(et_scores)[::-1][:k_et]
    et_features = set(feature_names[et_top_indices])
    print(f"-> Extra Trees đã chọn lọc {len(et_features)} đặc trưng tiềm năng.")

    # --- KẾT HỢP VÀ TÍNH ĐIỂM LAI CHUẨN HÓA AMALGAMATED ---
    print("\n[Kết hợp] Đang tìm phần giao đồng thuận của cả 3 phương pháp...")
    intersection_features = anova_features.intersection(ig_features).intersection(et_features)
    print(f"-> Số đặc trưng ĐỒNG THUẬN bởi cả 3 phương pháp: {len(intersection_features)}")

    # Hàm chuẩn hóa đưa các thang đo (F-score, Entropy, Gini) về cùng dải [0, 1]
    def min_max_scale(array):
        arr_min, arr_max = array.min(), array.max()
        if arr_max == arr_min:
            return np.zeros_like(array)
        return (array - arr_min) / (arr_max - arr_min)

    anova_norm = min_max_scale(anova_scores)
    ig_norm = min_max_scale(ig_scores)
    et_norm = min_max_scale(et_scores)
    
    # Tính toán Điểm số Lai (Hybrid Score)
    hybrid_scores = (anova_norm + ig_norm + et_norm) / 3.0
    
    # Tối ưu hóa O(1) bằng Dictionary Mapping để tăng tốc độ tìm kiếm
    feature_to_hybrid_score = dict(zip(feature_names, hybrid_scores))
    
    # Khởi tạo danh sách ứng viên từ tập giao nhau
    candidate_list = list(intersection_features)
    
    # KỊCH BẢN 1: Nếu tập giao nhau ít hơn final_k -> Tự động bổ sung dựa trên Điểm số Lai cao nhất
    if len(candidate_list) < final_k:
        print(f"⚠️ Số đặc trưng giao nhau ít hơn {final_k}, tự động bổ sung dựa trên Hybrid Score...")
        all_features_sorted = feature_names[np.argsort(hybrid_scores)[::-1]]
        for feat in all_features_sorted:
            if feat not in intersection_features:
                candidate_list.append(feat)
            if len(candidate_list) == final_k:
                break
                
    # KỊCH BẢN 2 & BƯỚC CHỐT: Sắp xếp lại danh sách ứng viên và cắt lấy chính xác đúng final_k đặc trưng tốt nhất
    final_features = sorted(candidate_list, key=lambda f: feature_to_hybrid_score[f], reverse=True)[:final_k]
    
    print(f"\n🏆 Hoàn thành! Đã chọn được {len(final_features)} đặc trưng tối ưu nhất theo bài báo.")
    return final_features


# =============================================================================
# 2. LUỒNG XỬ LÝ CHÍNH ĐỐI VỚI FILE CSV THỰC TẾ
# =============================================================================
if __name__ == "__main__":
    
    # --- CẤU HÌNH THÔNG TIN FILE CỦA BẠN TẠI ĐÂY ---
    csv_path = "/kaggle/working/multilabels.csv"  # Đường dẫn file của bạn trên Kaggle
    target_column = "activity"                       # <-- Hãy chắc chắn cột nhãn của bạn tên là "Label"
    final_num_features = 40                       # Số lượng đặc trưng cuối cùng muốn lấy (40 đặc trưng)

    if not os.path.exists(csv_path):
        print(f"❌ KHÔNG TÌM THẤY FILE '{csv_path}'. Vui lòng kiểm tra lại đường dẫn!")
    else:
        # --- BƯỚC A: ĐỌC VÀ LÀM SẠCH DỮ LIỆU THÔ ---
        print(f"--- Bước A: Tải và làm sạch dữ liệu từ file: {csv_path} ---")
        df = pd.read_csv(csv_path)
        print(f"📊 Kích thước file gốc: {df.shape[0]} dòng, {df.shape[1]} cột.")

        # Xóa khoảng trắng ẩn ở tên cột nếu có
        df.columns = df.columns.str.strip()

        # Thay thế các giá trị vô hạn (inf) thành NaN, sau đó lấp đầy bằng 0 thay vì xóa dòng
        # Điều này giúp giữ nguyên >640k dòng dữ liệu ban đầu
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        print(f"✅ Kích thước sau khi xử lý giá trị lỗi NaN/Inf (Lấp đầy bằng 0): {df.shape[0]} dòng.")

        # Tách Nhãn (y) và Đặc trưng (X)
        y_raw = df[target_column]
        X_raw = df.drop(columns=[target_column])

        # Loại bỏ các cột định danh dạng chữ (IP, Timestamp...) - Chỉ giữ cột số hợp lệ
        X_raw = X_raw.select_dtypes(include=[np.number])

        # --- LỌC BỎ CỘT HẰNG SỐ (CONSTANT FEATURES) TRƯỚC KHI CHỌN ---
        # Loại bỏ các cột hoàn toàn không biến đổi để tránh gây nhiễu cho ANOVA (lỗi chia cho 0)
        constant_cols = [col for col in X_raw.columns if X_raw[col].nunique() <= 1]
        if len(constant_cols) > 0:
            print(f"❌ Phát hiện và loại bỏ {len(constant_cols)} cột hằng số (không có giá trị phân loại).")
            X_raw = X_raw.drop(columns=constant_cols)

        # Mã hóa nhãn chữ thành nhãn số (Ví dụ: Benign -> 0, DDoS -> 1)
        if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
            print("🔤 Phát hiện nhãn dạng chữ. Đang tiến hành mã hóa tự động...")
            le = LabelEncoder()
            y_raw = le.fit_transform(y_raw)
            for idx, class_name in enumerate(le.classes_):
                print(f"   🔹 Lớp {idx} <==> {class_name}")

        # --- BƯỚC B: TẠO BẢN SCALE TẠM THỜI VỚI FACTOR 30.000 ĐỂ CHỌN ĐẶC TRƯNG ---
        print(f"\n--- Bước B: Khởi tạo bản Scale tạm thời với dải dữ liệu (0, 30000) ---")
        temp_scaler = MinMaxScaler(feature_range=(0, 30000))
        X_temp_scaled = pd.DataFrame(temp_scaler.fit_transform(X_raw), columns=X_raw.columns)

        # --- BƯỚC C: THỰC THI TRÍCH CHỌN ĐẶC TRƯNG LAI TRÊN BẢN SCALE TẠM ---
        selected_features = hybrid_feature_selection(
            X_temp_scaled, y_raw, 
            k_anova=80, k_ig=80, k_et=80, 
            final_k=final_num_features
        )

        # --- BƯỚC D: KHÔI PHỤC DỮ LIỆU CỦA 40 ĐẶC TRƯNG TỪ TẬP THÔ GỐC BAN ĐẦU ---
        print("\n--- Bước D: Khôi phục dải giá trị gốc hoàn toàn của các đặc trưng được chọn ---")
        # Lọc danh sách đặc trưng trực tiếp trên X_raw ban đầu (chưa dính scale tạm)
        X_final_unscaled = X_raw[selected_features]
        
        print(f"✅ Đã khôi phục thành công dữ liệu thô gốc của {X_final_unscaled.shape[1]} đặc trưng tối ưu.")

        # --- BƯỚC E: ĐÓNG GÓI VÀ XUẤT FILE KẾT QUẢ ---
        output_file = "BCCC_selected_features_raw.csv"
        df_final = X_final_unscaled.copy()
        df_final['activity'] = y_raw
        
        df_final.to_csv(output_file, index=False)
        print(f"\n💾 ĐÃ LƯU THÀNH CÔNG file: '{output_file}'")
        print("👉 Bây giờ bạn có thể mang file này đưa vào bước SCALE CHUYÊN NGHIỆP phía sau của bạn!")