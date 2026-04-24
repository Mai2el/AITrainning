import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import time

# 1. Load data
input_file = "dataset_ultimate_3classes.csv"
print(f"Đang tải dữ liệu từ {input_file}...")
df = pd.read_csv(input_file, low_memory=False)

# Drop cột không cần thiết
drop_cols = ["label", "flow_id", "timestamp", "src_ip", "dst_ip"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

LABEL_COL = 'activity'

# Convert numeric
for col in df.columns:
    if col != LABEL_COL:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# 2. Split X, y
X = df.drop(columns=[LABEL_COL]).values
y_raw = df[LABEL_COL].values

le = LabelEncoder()
y = le.fit_transform(y_raw)
target_names = le.classes_

print(f"Label mapping: {dict(zip(target_names, range(len(target_names))))}")

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")


# LƯU Ý: ĐÃ TẮT BƯỚC TÍNH TOÁN SAMPLE_WEIGHT ĐỂ CHỐNG NHIỄU (OVERFITTING LỚP NHỎ)
# 4. Model (Multi-class)
model = xgb.XGBClassifier(
    objective='multi:softprob',     
    num_class=len(target_names),    
    eval_metric='auc',              
    n_estimators=1000,              
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.4,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50        
)

# 5. Train MÔ HÌNH THUẦN TÚY (Không ép trọng số)
t0 = time.time()

print("Bắt đầu huấn luyện mô hình (Chế độ tự nhiên)...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50                      
)

print(f"Training time: {time.time() - t0:.2f}s")

# 6. Predict
# Lấy xác suất của tất cả các class
y_proba = model.predict_proba(X_test) 

# Lấy nhãn dự đoán
y_pred = model.predict(X_test)

# 7. Evaluation
print("\n" + "="*50)
print("KẾT QUẢ TEST")
print("="*50)

print(f"AUC: {roc_auc_score(y_test, y_proba, multi_class='ovr'):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Feature importance
print("\n" + "="*50)
print("TOP 10 FEATURES")
print("="*50)

feature_names = df.drop(columns=[LABEL_COL]).columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(min(10, len(feature_names))):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")