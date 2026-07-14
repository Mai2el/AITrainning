"""TẦNG 1 — XGBoost phân loại nhị phân (Benign vs Attack), full feature, nhắm ~99%."""
import time, numpy as np, pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_auc_score)

LABEL_COL = "activity"
print("Đang tải multilabels.csv (full feature)...")
df = pd.read_csv("multilabels.csv", low_memory=False)
df.columns = df.columns.str.strip()
df.drop(columns=[c for c in ["flow_id","timestamp","src_ip","dst_ip","label"]
                 if c in df.columns], inplace=True)

# Nhãn nhị phân
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
df = df[~df[LABEL_COL].str.lower().eq("suspicious")].copy()
y_bin = (~df[LABEL_COL].str.lower().str.startswith("benign")).astype(int)  # 1=Attack,0=Benign

for c in df.columns:
    if c != LABEL_COL:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df.replace([np.inf,-np.inf], np.nan, inplace=True); df.fillna(0, inplace=True)
feat_df = df.drop(columns=[LABEL_COL])
nunq = feat_df.nunique(); feat_df = feat_df.drop(columns=nunq[nunq<=1].index.tolist())

X = feat_df.values; y = y_bin.values
print(f"shape={X.shape}  Benign={int((y==0).sum()):,}  Attack={int((y==1).sum()):,}")

# Split 70/15/15 (khớp pipeline deep)
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=42)
print(f"train={len(Xtr):,}  val={len(Xval):,}  test={len(Xte):,}")

spw = (y==0).sum() / (y==1).sum()   # cân bằng lớp cho XGBoost
clf = xgb.XGBClassifier(
    objective="binary:logistic", eval_metric="logloss",
    n_estimators=1200, learning_rate=0.05, max_depth=12,
    min_child_weight=2, gamma=0.0,
    subsample=0.85, colsample_bytree=0.6,
    reg_lambda=1.5, reg_alpha=0.0,
    scale_pos_weight=spw, tree_method="hist", n_jobs=10,
    random_state=42, early_stopping_rounds=50,
)
t0 = time.time()
clf.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=100)
print(f"train time {time.time()-t0:.0f}s  best_iter={clf.best_iteration}")

# ── Đánh giá trên TEST ──
proba = clf.predict_proba(Xte)[:, 1]
pred  = (proba >= 0.5).astype(int)
acc = accuracy_score(yte, pred)
f1m = f1_score(yte, pred, average="macro")
auc = roc_auc_score(yte, proba)
print(f"\n==== TẦNG 1 — XGBoost BINARY (TEST) ====")
print(f"Accuracy={acc:.4f}  F1-macro={f1m:.4f}  ROC-AUC={auc:.5f}")
print(classification_report(yte, pred, target_names=["Benign","Attack"], digits=4))
print("Confusion matrix [rows=true(Benign,Attack), cols=pred]:")
print(confusion_matrix(yte, pred))

clf.save_model("stage1_xgb_binary.json")
print("\n→ Đã lưu model: stage1_xgb_binary.json")
