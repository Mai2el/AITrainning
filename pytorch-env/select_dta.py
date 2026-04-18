"""
feature_selection.py  (DDoS-aware version - ENSEMBLE + WHITELIST + RAW DATA PRESERVATION)
----------------------------------------------
Cải tiến:
  - Chỉ tính toán Feature Selection trên một bản sao tạm thời.
  - Sử dụng Ensemble Feature Selection: ANOVA, Information Gain, Extra Trees.
  - Tích hợp Whitelist bảo vệ các tính năng quan trọng và lấy đủ 80 features.
  - Dữ liệu xuất ra file CSV cuối cùng giữ nguyên 100% trạng thái gốc 
    (không fillna, giữ nguyên định dạng ban đầu).
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, f_classif, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
import time
import warnings
import gc

warnings.filterwarnings("ignore")

# ── Features phải giữ lại bất kể correlation hay importance ──────────────
DDOS_CRITICAL_FEATURES = { 
    "dst_port", "protocol", # Thêm protocol để hỗ trợ phân biệt IGMP/TCP
    "packets_rate", "fwd_packets_rate", "bwd_packets_rate",
    "bytes_rate", "fwd_bytes_rate", "bwd_bytes_rate", "down_up_rate",
    "packets_IAT_mean", "packet_IAT_std", "packet_IAT_min", "packet_IAT_max",
    "fwd_packets_IAT_mean", "bwd_packets_IAT_mean",
    "syn_flag_counts", "ack_flag_counts", "rst_flag_counts", "fin_flag_counts",
    "fwd_syn_flag_counts", "bwd_syn_flag_counts",
    "syn_flag_percentage_in_total", "ack_flag_percentage_in_total",
    "duration", "packets_count", "fwd_packets_count", "bwd_packets_count",
    "handshake_duration", "handshake_state",
    "syn_per_packet", "iat_x_rate",
    "bwd_fwd_ratio", "syn_no_fin", "bytes_per_packet",
}

# ── Bước 0: Feature Engineering ──────────────────────────────────────────
def engineer_ddos_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9  
    
    def get_num(col_name):
        return pd.to_numeric(df[col_name], errors='coerce')

    if "syn_flag_counts" in df.columns and "packets_count" in df.columns:
        df["syn_per_packet"] = get_num("syn_flag_counts") / (get_num("packets_count") + eps)
        
    if "packets_IAT_mean" in df.columns and "packets_rate" in df.columns:
        df["iat_x_rate"] = get_num("packets_IAT_mean") * get_num("packets_rate")
        
    if "bwd_packets_count" in df.columns and "fwd_packets_count" in df.columns:
        df["bwd_fwd_ratio"] = (get_num("bwd_packets_count") + eps) / (get_num("fwd_packets_count") + eps)
        
    if "syn_flag_counts" in df.columns and "fin_flag_counts" in df.columns:
        df["syn_no_fin"] = (get_num("syn_flag_counts") - get_num("fin_flag_counts")).clip(lower=0)
        
    if "bytes_rate" in df.columns and "packets_rate" in df.columns:
        df["bytes_per_packet"] = (get_num("bytes_rate") + eps) / (get_num("packets_rate") + eps)

    new_cols = ["syn_per_packet", "iat_x_rate", "bwd_fwd_ratio", "syn_no_fin", "bytes_per_packet"]
    added = [c for c in new_cols if c in df.columns]
    print(f"[engineer_ddos_features] Đã thêm các features tổ hợp: {added}")
    return df

# ── Selector chính (Ensemble: ANOVA + InfoGain + ExtraTrees) ──────────────
class FeatureSelector:
    def __init__(self, top_k=80, cv_folds=5, ddos_whitelist=None):
        self.top_k = top_k
        self.cv_folds = cv_folds
        self.ddos_whitelist = ddos_whitelist if ddos_whitelist is not None else DDOS_CRITICAL_FEATURES
        self.selected_cols_ = None

    def fit_transform(self, X, y, feature_names):
        print(f"\n{'='*64}")
        print(f" ENSEMBLE FEATURE SELECTION + WHITELIST (Target: {self.top_k} features)")
        print(f"{'='*64}")
        cols = list(feature_names)
        
        # 0. Variance Filter (Bỏ các cột có giá trị hằng số)
        print("\n[0] Lọc các features hằng số (Variance = 0)...")
        var_filter = VarianceThreshold(threshold=0.0)
        var_filter.fit(X)
        keep_var = var_filter.get_support()
        X_cur = X[:, keep_var]
        cols = [cols[i] for i, k in enumerate(keep_var) if k]
        ranks = {col: 0 for col in cols}

        # 1. ANOVA
        print("\n[1] Đang tính toán ANOVA (F-Value)...")
        f_vals, _ = f_classif(X_cur, y)
        anova_ranks = np.argsort(np.nan_to_num(f_vals))[::-1]
        for rank, idx in enumerate(anova_ranks):
            ranks[cols[idx]] += rank

        # 2. Information Gain
        print("[2] Đang tính toán Information Gain (Mutual Information)...")
        sample_size = min(X_cur.shape[0], 50000) 
        sample_idx = np.random.choice(X_cur.shape[0], sample_size, replace=False)
        mi_scores = mutual_info_classif(X_cur[sample_idx], y[sample_idx], random_state=42)
        ig_ranks = np.argsort(np.nan_to_num(mi_scores))[::-1]
        for rank, idx in enumerate(ig_ranks):
            ranks[cols[idx]] += rank

        # 3. Extra Trees
        print(f"[3] Đang tính toán Extra Trees Importance (CV {self.cv_folds}-Fold)...")
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        et_importances = np.zeros(len(cols))
        for fold, (tr_idx, _) in enumerate(skf.split(X_cur, y), 1):
            clf = ExtraTreesClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=fold)
            clf.fit(X_cur[tr_idx], y[tr_idx])
            et_importances += clf.feature_importances_
        et_ranks = np.argsort(et_importances / self.cv_folds)[::-1]
        for rank, idx in enumerate(et_ranks):
            ranks[cols[idx]] += rank

        # 4. Tích hợp Whitelist & Ensemble Voting
        print("\n[4] Tiến hành kết hợp Whitelist và kết quả Ensemble...")
        sorted_by_ensemble = sorted(cols, key=lambda c: ranks[c])
        
        whitelist_present = [c for c in self.ddos_whitelist if c in cols]
        print(f"    ⚑ Đã nhặt {len(whitelist_present)}/{len(self.ddos_whitelist)} features từ Whitelist đưa vào danh sách bảo vệ.")

        final_list = list(whitelist_present)
        
        for col in sorted_by_ensemble:
            if len(final_list) >= self.top_k:
                break
            if col not in final_list:
                final_list.append(col)
                
        self.selected_cols_ = final_list
        
        print(f"\n{'='*64}")
        print(f" ✓ KẾT QUẢ CUỐI CÙNG: Top {len(self.selected_cols_)} features xuất sắc nhất")
        print(f"{'='*64}\n")

        final_idx = [list(feature_names).index(c) for c in self.selected_cols_]
        return X[:, final_idx], self.selected_cols_

# ------------------------------------------------------------------
# STANDALONE DEMO
# ------------------------------------------------------------------
if __name__ == "__main__":
    import gc # Thư viện dọn dẹp RAM

    CSV_PATH  = "dataset_optimized.csv"
    LABEL_COL = "activity"
    # LƯU Ý: Không loại bỏ "dst_port" và "protocol"
    DROP_COLS = ["label", "flow_id", "timestamp", "src_ip", "dst_ip", "src_port"]  
    TOP_K     = 80

    print("1. Đang load dữ liệu gốc (Đọc theo từng chunk để tránh Out of Memory)...")
    
    chunk_list = []
    initial_len = 0
    
    try:
        for chunk in pd.read_csv(CSV_PATH, chunksize=100000):
            chunk.columns = chunk.columns.str.strip()
            initial_len += len(chunk)
            
            if LABEL_COL in chunk.columns:
                chunk = chunk[chunk[LABEL_COL] != "Suspicious"]
                
            chunk_list.append(chunk)
            
        df_raw = pd.concat(chunk_list, ignore_index=True)
        # Giải phóng list tạm ngay lập tức
        del chunk_list 
        gc.collect()
        
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file {CSV_PATH}")
        exit(1)

    if LABEL_COL in df_raw.columns:
        print(f"   Đã xóa {initial_len - len(df_raw):,} dòng 'Suspicious'. Còn lại {len(df_raw):,} dòng.")

    drop = [c for c in DROP_COLS if c in df_raw.columns]
    if drop:
        df_raw = df_raw.drop(columns=drop)

    df_raw = engineer_ddos_features(df_raw)

    print("\n2. Khởi tạo ma trận dữ liệu (Ép kiểu Float32 để giảm 50% RAM)...")

    labels_raw = df_raw[LABEL_COL].values
    if isinstance(labels_raw[0], str):
        enc = {l: i for i, l in enumerate(np.unique(labels_raw))}
        y_work = np.array([enc[l] for l in labels_raw], dtype=np.int32)
    else:
        y_work = labels_raw.astype(np.int32)

    feature_cols = [c for c in df_raw.columns if c != LABEL_COL]

    # CẤP PHÁT MA TRẬN NUMPY TRỰC TIẾP (Bỏ qua việc copy DataFrame)
    X_work = np.zeros((len(df_raw), len(feature_cols)), dtype=np.float32)

    print("   -> Đang xử lý từng cột dữ liệu...")
    for i, col in enumerate(feature_cols):
        # Chuyển đổi an toàn, điền NaN/Inf bằng 0
        col_data = pd.to_numeric(df_raw[col], errors='coerce').astype(np.float32)
        col_data = col_data.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_work[:, i] = col_data.values

    # Dọn dẹp biến tạm
    del col_data
    gc.collect()

    # 3. Chạy Feature Selection
    print("\n3. Đang chạy thuật toán học máy Feature Selection...")
    fs = FeatureSelector(top_k=TOP_K, cv_folds=5)
    _, selected_features = fs.fit_transform(X_work, y_work, feature_cols)

    # Giải phóng ma trận numpy khổng lồ sau khi hoàn tất
    del X_work
    gc.collect()

    with open("selected_features.txt", "w") as f:
        f.write("\n".join(selected_features))
    
    # ------------------------------------------------------------------
    # 4. XUẤT CSV TỪ DỮ LIỆU GỐC
    # ------------------------------------------------------------------
    print("\n4. Đang tạo file Dataset mới từ dữ liệu GỐC...")
    final_cols = selected_features + [LABEL_COL]
    
    # Lọc lại từ df_raw (giữ nguyên gốc)
    df_final = df_raw[final_cols]
    
    OUTPUT_CSV = "dataset_ready_for_model.csv"
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print(f"✓ Hoàn tất! Đã trích xuất đúng {len(selected_features)} cột.")
    print(f"✓ Lưu tại: {OUTPUT_CSV} (Tổng số dòng: {len(df_final):,})")