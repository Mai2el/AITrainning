"""
eval_per_layer.py
-----------------
So sánh 3 cách chấm điểm hệ thống Hybrid (XGBoost gatekeeper + MHA attack expert):

  (1) END-TO-END  : đúng như pipeline thật. Lỗi định tuyến của XGBoost (FN: attack bị
                    cho là benign) BỊ cộng dồn vào kết quả cuối. -> số trung thực nhất.

  (2) PER-LAYER (oracle routing) : giống cách bài báo BCCC chấm. Mỗi mẫu attack đi THẲNG
                    vào MHA theo NHÃN THẬT (định tuyến hoàn hảo, không qua cổng XGBoost),
                    mẫu benign vẫn do XGBoost gác cổng. -> bỏ lỗi FN của Layer 1.

  (3) LAYER ĐỘC LẬP : báo cáo riêng từng tầng (Layer 1 nhị phân trên toàn test;
                    Layer 2 MHA chỉ trên attack thật) = số "Task 1" và "Task 2" của bài báo.

Dùng đúng model đã train: xgb_gatekeeper.json + best_hybrid_mha_expert.pt
"""
import os
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Lấy lại kiến trúc + CONFIG từ script gốc (import an toàn nhờ __main__ guard)
from hybrid_xgb_mha import Deep_MHA_Tabular, CONFIG

DATA = "dataset_tokenized.txt"
LABEL_COL = "activity"

print("1. Nạp dữ liệu + tái tạo split giống lúc train ...")
df = pd.read_csv(DATA, sep="\t", low_memory=False)
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
df = df[~df[LABEL_COL].str.lower().eq("suspicious")].copy()
df[LABEL_COL] = df[LABEL_COL].apply(lambda x: "Benign" if x.lower().startswith("benign") else x)

features = [c for c in df.columns if c != LABEL_COL]
X = df[features].values
y_raw = df[LABEL_COL].values

le = LabelEncoder()
y = le.fit_transform(y_raw)
target_names = [str(c) for c in le.classes_]
# LƯU Ý: nhãn trong file là SỐ ('17' = Benign). LabelEncoder mã hoá theo thứ tự chuỗi
# nên '17' -> encoded 9, KHÔNG phải 17. Script gốc dùng BENIGN_INDEX=17 là SAI (bug).
BENIGN_INDEX = target_names.index("17") if "17" in target_names else (
    target_names.index("Benign") if "Benign" in target_names else 17)
NUM_CLASSES = len(target_names)
NUM_BINS = int(df[features].max().max()) + 1
y_binary = (y != BENIGN_INDEX).astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=CONFIG["test_size_total"], stratify=y, random_state=CONFIG["seed"])
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=CONFIG["val_test_split"], stratify=y_temp, random_state=CONFIG["seed"])
print(f"   -> Test = {len(X_test)} mẫu | {NUM_CLASSES} lớp | Benign idx = {BENIGN_INDEX}")

print("2. Nạp model đã train ...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_gatekeeper.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mha = Deep_MHA_Tabular(
    num_numeric_features=len(features), num_bins=NUM_BINS,
    value_dim=CONFIG["mha_value_dim"], feat_dim=CONFIG["mha_feat_dim"],
    n_segments=CONFIG["mha_n_segments"], num_layers=CONFIG["mha_num_layers"],
    num_branches=CONFIG["mha_num_branches"], num_heads_per_branch=CONFIG["mha_num_heads_per_branch"],
    num_classes=NUM_CLASSES, mlp_dropout=CONFIG["mha_mlp_dropout"], value_embed=CONFIG["mha_value_embed"],
).to(device)
mha.load_state_dict(torch.load("best_hybrid_mha_expert.pt", map_location=device))
mha.eval()


def mha_predict(X_subset):
    """Trả về nhãn attack MHA dự đoán (đã chặn benign = -inf)."""
    preds = []
    bs = CONFIG["inference_batch_size"]
    with torch.no_grad():
        for i in range(0, len(X_subset), bs):
            bx = torch.tensor(X_subset[i:i+bs], dtype=torch.long).to(device)
            logits = mha(bx)
            logits[:, BENIGN_INDEX] = -float("inf")
            preds.extend(logits.argmax(1).cpu().numpy())
    return np.array(preds)


def summary(tag, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n{'='*70}\n{tag}\n{'='*70}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1 weighted   : {f1w:.4f}   <-- cột bài báo báo cáo (Table 5)")
    print(f"  F1 macro      : {f1m:.4f}")
    return acc, f1w, f1m


# ---------- (1) END-TO-END (pipeline thật) ----------
test_bin = xgb_model.predict(X_test)
e2e = np.full(len(X_test), BENIGN_INDEX)
atk_idx = np.where(test_bin == 1)[0]
if len(atk_idx):
    e2e[atk_idx] = mha_predict(X_test[atk_idx])
summary("(1) END-TO-END  — lỗi định tuyến XGBoost ĐƯỢC cộng dồn (số của bạn)", y_test, e2e)

# ---------- (2) PER-LAYER oracle routing (cách bài báo) ----------
# attack thật -> thẳng vào MHA (định tuyến hoàn hảo); benign thật -> XGBoost gác cổng
oracle = np.empty(len(X_test), dtype=int)
true_atk = np.where(y_test != BENIGN_INDEX)[0]
true_ben = np.where(y_test == BENIGN_INDEX)[0]
oracle[true_atk] = mha_predict(X_test[true_atk])                       # Layer 2, perfect routing
oracle[true_ben] = np.where(xgb_model.predict(X_test[true_ben]) == 1,  # Layer 1 trên benign
                            -1, BENIGN_INDEX)                          # FP benign -> sai (-1)
summary("(2) PER-LAYER oracle routing — giống Task 6 bài báo (định tuyến hoàn hảo)", y_test, oracle)

# ---------- (3) Từng layer độc lập ----------
print(f"\n{'='*70}\n(3) TỪNG LAYER ĐỘC LẬP (= Task 1 & Task 2 bài báo)\n{'='*70}")
y_test_bin = (y_test != BENIGN_INDEX).astype(int)
b_acc = accuracy_score(y_test_bin, test_bin)
b_f1 = f1_score(y_test_bin, test_bin, zero_division=0)
print(f"  LAYER 1 (XGBoost nhị phân, toàn test)  : acc={b_acc:.4f}  f1={b_f1:.4f}   (~Task 1)")
atk_only_pred = mha_predict(X_test[true_atk])
a_acc = accuracy_score(y_test[true_atk], atk_only_pred)
a_f1w = f1_score(y_test[true_atk], atk_only_pred, average="weighted", zero_division=0)
a_f1m = f1_score(y_test[true_atk], atk_only_pred, average="macro", zero_division=0)
print(f"  LAYER 2 (MHA, chỉ trên attack thật)    : acc={a_acc:.4f}  f1_w={a_f1w:.4f}  f1_m={a_f1m:.4f}   (~Task 2)")

print("\nXong. So sánh (1) vs (2) để thấy phần 'lạm phát' do bỏ lỗi định tuyến.")
