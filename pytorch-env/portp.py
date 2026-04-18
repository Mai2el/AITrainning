import pandas as pd
import numpy as np

# ==========================================
# CONFIG
# ==========================================
INPUT_PATH = "dataset_ready_for_model.csv"
PORT_COL = "dst_port"
LABEL_COL = "activity"

# ==========================================
# LOAD DATA
# ==========================================
print("Đang load dữ liệu...")
df = pd.read_csv(INPUT_PATH, low_memory=False)

if PORT_COL not in df.columns:
    raise ValueError(f"❌ Không tìm thấy cột '{PORT_COL}'")

print(f"✔ Tổng số dòng: {len(df):,}")

# ==========================================
# CLEAN
# ==========================================
df[PORT_COL] = pd.to_numeric(df[PORT_COL], errors="coerce")
df = df.dropna(subset=[PORT_COL])

# ép int cho dễ nhìn
df[PORT_COL] = df[PORT_COL].astype(int)

# ==========================================
# 1. THỐNG KÊ CƠ BẢN
# ==========================================
print("\n===== BASIC STATS =====")
print("Unique ports:", df[PORT_COL].nunique())
print("Min port:", df[PORT_COL].min())
print("Max port:", df[PORT_COL].max())

# ==========================================
# 2. ĐẾM TỪNG PORT
# ==========================================
print("\n===== COUNT TỪNG PORT =====")
port_counts = df[PORT_COL].value_counts().sort_index()

# in 50 port đầu tiên
print(port_counts.head(50))

# ==========================================
# 3. TOP PORT PHỔ BIẾN
# ==========================================
print("\n===== TOP 30 PORT PHỔ BIẾN =====")
top_ports = df[PORT_COL].value_counts().head(30)
print(top_ports)

# ==========================================
# 4. TỶ LỆ (%)
# ==========================================
print("\n===== TOP PORT (%) =====")
port_ratio = df[PORT_COL].value_counts(normalize=True) * 100
print(port_ratio.head(30))

# ==========================================
# 5. PHÂN LOẠI PORT RANGE
# ==========================================
print("\n===== PORT RANGE =====")

well_known = ((df[PORT_COL] >= 0) & (df[PORT_COL] <= 1023)).mean()
registered = ((df[PORT_COL] >= 1024) & (df[PORT_COL] <= 49151)).mean()
dynamic = ((df[PORT_COL] >= 49152) & (df[PORT_COL] <= 65535)).mean()

print(f"Well-known (0-1023): {well_known:.3f}")
print(f"Registered (1024-49151): {registered:.3f}")
print(f"Dynamic (49152-65535): {dynamic:.3f}")

# ==========================================
# 6. HISTOGRAM THÔ
# ==========================================
print("\n===== HISTOGRAM (10 BINS) =====")
hist, bins = np.histogram(df[PORT_COL], bins=10)

for i in range(len(hist)):
    print(f"{int(bins[i])} - {int(bins[i+1])}: {hist[i]}")

# ==========================================
# 7. THEO CLASS (RẤT QUAN TRỌNG)
# ==========================================
if LABEL_COL in df.columns:
    print("\n===== TOP PORT THEO CLASS =====")
    for cls in df[LABEL_COL].unique():
        print(f"\n--- CLASS: {cls} ---")
        subset = df[df[LABEL_COL] == cls]
        print(subset[PORT_COL].value_counts().head(10))

# ==========================================
# 8. SAVE FILE
# ==========================================
print("\nĐang lưu file kết quả...")

port_counts.to_csv("dst_port_counts.csv")

port_ratio.to_csv("dst_port_ratio.csv")

if LABEL_COL in df.columns:
    all_class_stats = []
    for cls in df[LABEL_COL].unique():
        subset = df[df[LABEL_COL] == cls]
        vc = subset[PORT_COL].value_counts().head(20)
        tmp = vc.reset_index()
        tmp.columns = ["port", "count"]
        tmp["class"] = cls
        all_class_stats.append(tmp)

    pd.concat(all_class_stats).to_csv("dst_port_by_class.csv", index=False)

print("✅ DONE!")
print("File xuất:")
print(" - dst_port_counts.csv")
print(" - dst_port_ratio.csv")
print(" - dst_port_by_class.csv")