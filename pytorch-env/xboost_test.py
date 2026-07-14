import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)
import time
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD DATA
# =========================================================

input_file = "dataset_cic_sampled.csv"

print(f"Đang tải dữ liệu từ {input_file}...")

df = pd.read_csv(input_file, low_memory=False)

# Drop cột không cần thiết
drop_cols = ["label", "flow_id", "timestamp", "src_ip", "dst_ip"]

df = df.drop(
    columns=[c for c in drop_cols if c in df.columns],
    errors="ignore"
)

LABEL_COL = 'activity'

# =========================================================
# 2. PREPROCESSING
# =========================================================

for col in df.columns:
    if col != LABEL_COL:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# =========================================================
# 3. SPLIT X / y
# =========================================================

X = df.drop(columns=[LABEL_COL]).values
y_raw = df[LABEL_COL].values

le = LabelEncoder()
y = le.fit_transform(y_raw)

target_names = le.classes_

print(f"Label mapping: {dict(zip(target_names, range(len(target_names))))}")

# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# =========================================================
# 5. DETECT TASK TYPE
# =========================================================

num_classes = len(target_names)

if num_classes == 2:
    print("\n[INFO] Binary Classification Mode")

    objective = 'binary:logistic'
    eval_metric = ['logloss', 'auc']

else:
    print("\n[INFO] Multi-class Classification Mode")

    objective = 'multi:softprob'
    eval_metric = ['mlogloss', 'auc']

# =========================================================
# 6. MODEL
# =========================================================

model = xgb.XGBClassifier(
    objective=objective,

    # Chỉ dùng cho multi-class
    num_class=num_classes if num_classes > 2 else None,

    eval_metric=eval_metric,

    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,

    subsample=0.8,
    colsample_bytree=0.4,

    random_state=42,
    n_jobs=-1,

    early_stopping_rounds=50
)

# =========================================================
# 7. TRAIN
# =========================================================

t0 = time.time()

print("\nBắt đầu huấn luyện mô hình...")

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_train, y_train),
        (X_test, y_test)
    ],

    verbose=50
)

print(f"\nTraining time: {time.time() - t0:.2f}s")

# =========================================================
# 8. PLOT LOSS CURVES
# =========================================================

try:
    evals_result = model.evals_result()

    if num_classes == 2:
        train_loss = evals_result.get('validation_0', {}).get('logloss')
        val_loss = evals_result.get('validation_1', {}).get('logloss')
        ylabel = 'Binary Log Loss'
    else:
        train_loss = evals_result.get('validation_0', {}).get('mlogloss')
        val_loss = evals_result.get('validation_1', {}).get('mlogloss')
        ylabel = 'Multi-class Log Loss'

    if train_loss and val_loss:

        epochs = list(range(1, len(train_loss) + 1))

        plt.figure(figsize=(8, 5))

        plt.plot(epochs, train_loss, label='train_loss')
        plt.plot(epochs, val_loss, label='val_loss')

        plt.xlabel('Epoch')
        plt.ylabel(ylabel)

        plt.title('Training vs Validation Loss')

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        out_png = 'loss_curves.png'

        plt.savefig(out_png)

        try:
            plt.show()
        except Exception:
            pass

        print(f"Saved loss curves to '{out_png}'")

        # =====================================================
        # DIAGNOSTICS
        # =====================================================

        t0_loss = train_loss[0]
        t_end = train_loss[-1]

        v0_loss = val_loss[0]
        v_end = val_loss[-1]

        train_impr = (
            (t0_loss - t_end) / t0_loss
            if t0_loss != 0 else 0.0
        )

        val_impr = (
            (v0_loss - v_end) / v0_loss
            if v0_loss != 0 else 0.0
        )

        if train_impr > 0.01 and val_impr < 0:

            print("\n[DIAGNOSTIC] Overfitting detected")
            print(" - Training loss giảm nhưng validation loss tăng")
            print(" - Gợi ý:")
            print("   + giảm max_depth")
            print("   + tăng regularization")
            print("   + giảm n_estimators")

        elif train_impr < 0.01 and val_impr < 0.01:

            print("\n[DIAGNOSTIC] Underfitting detected")
            print(" - Model học chưa đủ")
            print(" - Gợi ý:")
            print("   + tăng model capacity")
            print("   + train lâu hơn")
            print("   + cải thiện feature")

        else:
            print("\n[DIAGNOSTIC] Loss curves look normal")

        # =====================================================
        # OSCILLATION CHECK
        # =====================================================

        window = min(20, len(train_loss))

        if window >= 3:

            def norm_std(arr):
                seg = arr[-window:]
                mean = float(np.mean(seg))
                sd = float(np.std(seg))

                return sd / mean if mean != 0 else 0.0

            train_var = norm_std(train_loss)
            val_var = norm_std(val_loss)

            if train_var > 0.05 or val_var > 0.05:

                print("\n[DIAGNOSTIC] Loss oscillation detected")
                print(" - Có thể learning_rate quá cao")
                print(" - Thử:")
                print("   + giảm learning_rate")
                print("   + tăng subsample")

    else:
        print("Không tìm thấy lịch sử loss.")

except Exception as e:
    print(f"Could not plot loss curves: {e}")

# =========================================================
# 9. PREDICT
# =========================================================

y_proba = model.predict_proba(X_test)

y_pred = model.predict(X_test)

# =========================================================
# 10. EVALUATION
# =========================================================

print("\n" + "=" * 50)
print("KẾT QUẢ TEST")
print("=" * 50)

# =========================
# FIX AUC ERROR
# =========================

try:

    if num_classes == 2:

        # Binary classification
        auc = roc_auc_score(
            y_test,
            y_proba[:, 1]
        )

    else:

        # Multi-class classification
        auc = roc_auc_score(
            y_test,
            y_proba,
            multi_class='ovr'
        )

    print(f"AUC: {auc:.4f}")

except Exception as e:
    print(f"AUC calculation failed: {e}")

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        digits=4
    )
)

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred))

# =========================================================
# 11. FEATURE IMPORTANCE
# =========================================================

print("\n" + "=" * 50)
print("TOP 10 FEATURES")
print("=" * 50)

feature_names = df.drop(columns=[LABEL_COL]).columns

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

for i in range(min(10, len(feature_names))):

    print(
        f"{i+1}. "
        f"{feature_names[indices[i]]}: "
        f"{importances[indices[i]]:.4f}"
    )