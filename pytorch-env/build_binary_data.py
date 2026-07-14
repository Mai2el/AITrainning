"""Bước 1: chọn top-60 feature TỐT NHẤT cho binary (honest) + verify trần XGBoost."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

ID_LEAK = {'flow_id','timestamp','src_ip','dst_ip','source ip','destination ip',
           'src_port','dst_port','source port','destination port','label','flow id'}

# Đọc pool ứng viên (top-150 theo ranking cũ) qua usecols → tránh OOM
rank = pd.read_csv('xgb_feature_ranking.txt', header=None)[0].str.strip().tolist()
pool = rank[:150] + ['activity']
df = pd.read_csv('multilabels.csv', usecols=lambda c: c.strip() in pool, low_memory=False)
df.columns = df.columns.str.strip()
# lọc định danh (honest)
leak = [c for c in df.columns if c.lower() in ID_LEAK]
df = df.drop(columns=leak, errors='ignore')
df['activity'] = df['activity'].astype(str).str.strip()
df = df[~df['activity'].str.lower().eq('suspicious')]
y = (~df['activity'].str.lower().str.startswith('benign')).astype(int).values
feats = [c for c in df.columns if c != 'activity']
for c in feats: df[c] = pd.to_numeric(df[c], errors='coerce')
df[feats] = df[feats].replace([np.inf,-np.inf], np.nan).fillna(0)
print(f'Pool {len(feats)} feature (đã bỏ định danh: {leak})')

# Rank lại bằng XGBoost BINARY
X = df[feats].values
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
clf = xgb.XGBClassifier(objective='binary:logistic',n_estimators=400,learning_rate=0.1,max_depth=10,
    subsample=0.85,colsample_bytree=0.6,tree_method='hist',n_jobs=10,random_state=42,
    eval_metric='logloss',early_stopping_rounds=25)
clf.fit(Xtr,ytr,eval_set=[(Xte,yte)],verbose=False)
imp = clf.feature_importances_
order = np.argsort(imp)[::-1]
top60 = [feats[i] for i in order[:60]]
pd.Series(top60).to_csv('binary_top60.txt', index=False, header=False)
print('Đã lưu binary_top60.txt')

# Verify trần XGBoost trên top-60
Xt = df[top60].values
Xtr,Xte,ytr,yte = train_test_split(Xt,y,test_size=0.3,stratify=y,random_state=42)
clf2 = xgb.XGBClassifier(objective='binary:logistic',n_estimators=500,learning_rate=0.1,max_depth=10,
    subsample=0.85,colsample_bytree=0.6,tree_method='hist',n_jobs=10,random_state=42,
    eval_metric='logloss',early_stopping_rounds=25)
clf2.fit(Xtr,ytr,eval_set=[(Xte,yte)],verbose=False)
yp = clf2.predict(Xte)
_acc=accuracy_score(yte,yp); _f1=f1_score(yte,yp,average='macro')
print(f'TRAN XGBoost binary (top-60 honest): Acc={_acc:.4f} F1m={_f1:.4f}')

# Lưu CSV top-60 + activity để tokenize
out = df[top60 + ['activity']].copy()
out.to_csv('dataset_bin60.csv', index=False)
print(f'→ dataset_bin60.csv shape={out.shape}')
