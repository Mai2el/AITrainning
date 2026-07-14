"""
start.py
--------
Mô hình DDoS Hybrid kết hợp: XGBoost (Bộ lọc Nhị phân) + PyTorch MHA (Bộ phân loại Attack chuyên sâu)
Đã được cấu hình hóa toàn bộ dưới dạng SIÊU THAM SỐ (Hyperparameters) ở đầu file.
"""
import mlflow 
import mlflow.pytorch 
import mlflow.xgboost 
import matplotlib.pyplot as plt 
import seaborn as sns 
import json
import shutil
from pathlib import Path

import os
import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import random
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Hybrid_DDoS_XGB_MHA")
mlflow.autolog(disable=True)

# =========================================================================
# ⚙️ BỘ SIÊU THAM SỐ CẤU HÌNH HỆ THỐNG (HYPERPARAMETERS CONFIGURATION)
# =========================================================================
CONFIG = {
    # --- Cấu hình chung ---
    "seed": 42,
    "test_size_total": 0.30,      # Tỷ lệ chia tập (Validation + Test) từ ban đầu
    "val_test_split": 0.50,       # Tỷ lệ chia đôi tập Temp thành Validation và Test (0.50 của 30% = 15% mỗi tập)
    
    # --- Siêu tham số STAGE 1: XGBoost Gác Cổng ---
    "xgb_n_estimators": 200,
    "xgb_learning_rate": 0.05,
    "xgb_max_depth": 6,
    "xgb_early_stopping_rounds": 20,
    "xgb_tree_method": "hist",
    
    # --- Siêu tham số STAGE 2: Kiến trúc mô hình PyTorch MHA ---
    "mha_value_dim": 192,         # Chiều của cấu trúc nhúng giá trị số (Value Embedding)
    "mha_feat_dim": 64,           # Chiều của cấu trúc nhúng đặc trưng vị trí (Feature Embedding)
    "mha_n_segments": 192,        # Số phân đoạn áp dụng cho cấu trúc PLE Embedding
    "mha_num_layers": 4,          # Số tầng Parallel Transformer Blocks
    "mha_num_branches": 4,        # Số nhánh Attention song song trong một Block
    "mha_num_heads_per_branch": 4,# Số Head Attention trên từng nhánh độc lập
    "mha_mlp_dropout": 0.15,      # Tỷ lệ Dropout trong các lớp MLP ẩn
    "mha_value_embed": "ple",     # Phương thức nhúng giá trị: "ple" hoặc "simple"
    
    # --- Siêu tham số Huấn luyện MHA (Training Strategy) ---
    "mha_batch_size": 256,
    "mha_epochs": 50,
    "mha_patience": 12,           # Số lượng epoch chờ cải thiện trước khi Early Stopping
    "mha_lr": 1e-3,               # Learning rate cực đại (Max LR của OneCycleLR)
    "mha_weight_decay": 1e-2,     # Trừng phạt trọng số nhằm chống Overfitting
    "focal_loss_gamma": 2.0,      # Tham số Gamma điều chỉnh tiêu điểm lỗi của Focal Loss
    
    # --- Siêu tham số Suy luận kết hợp (Hybrid Inference) ---
    "inference_batch_size": 256   # Kích thước Batch xử lý dữ liệu nghi ngờ qua MHA
}


class FocalLoss(nn.Module):
    """
    Focal Loss đa lớp tiêu chuẩn giúp tăng cường học các lớp Attack thiểu số.
    """
    def __init__(self, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma     = gamma
        self.reduction = reduction
        self.ce        = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        ce_loss    = self.ce(inputs, targets)
        pt         = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


# ==========================================
# 1. TABULAR EMBEDDING & MULTI-HEAD ATTENTION COMPONENTS
# ==========================================

class PLEValueEmbedding(nn.Module):
    def __init__(self, num_bins: int, value_dim: int, n_segments: int = None):
        super().__init__()
        self.num_bins   = max(num_bins, 1)
        self.value_dim  = value_dim
        raw_segments    = n_segments if n_segments is not None else value_dim
        self.n_segments = min(raw_segments, value_dim)
        self.proj = nn.Linear(self.n_segments, value_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _ple_encode(self, t: torch.Tensor) -> torch.Tensor:
        T    = self.n_segments
        w    = 1.0 / T
        left = torch.arange(T, dtype=torch.float32, device=t.device) * w
        fill = ((t.unsqueeze(-1) - left) / w).clamp(0.0, 1.0)
        return fill

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        t   = token_ids.float() / (self.num_bins - 1)
        ple = self._ple_encode(t)
        return self.proj(ple)

class SimpleValueEmbedding(nn.Module):
    def __init__(self, num_bins: int, value_dim: int, n_segments: int = None):
        super().__init__()
        self.num_bins = max(num_bins, 1)
        self.net = nn.Sequential(
            nn.Linear(1, value_dim),
            nn.GELU(),
            nn.Linear(value_dim, value_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        t = (token_ids.float() / (self.num_bins - 1)).unsqueeze(-1)
        return self.net(t)

class MonotonicTabularEmbedding(nn.Module):
    def __init__(self, num_features: int, num_bins: int,
                 value_dim: int, feat_dim: int, n_segments: int = None,
                 value_embed: str = "ple"):
        super().__init__()
        self.num_features = num_features
        enc = PLEValueEmbedding if value_embed == "ple" else SimpleValueEmbedding
        self.value_embedding   = enc(num_bins, value_dim, n_segments)
        self.feature_embedding = nn.Embedding(num_features, feat_dim)
        nn.init.trunc_normal_(self.feature_embedding.weight, std=0.02)

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        val_emb  = self.value_embedding(x_tokens)
        feat_idx = torch.arange(self.num_features, device=x_tokens.device)
        feat_emb = self.feature_embedding(feat_idx).unsqueeze(0)
        feat_emb = feat_emb.expand(x_tokens.size(0), -1, -1)
        return torch.cat([val_emb, feat_emb], dim=-1)

class ScaledDotProductMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k       = embed_dim // num_heads
        self.d_model   = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        for w in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(w.weight)

    def forward(self, x):
        N, S, _ = x.size()
        H, dk   = self.num_heads, self.d_k
        Q = self.W_q(x).view(N, S, H, dk).transpose(1, 2)
        K = self.W_k(x).view(N, S, H, dk).transpose(1, 2)
        V = self.W_v(x).view(N, S, H, dk).transpose(1, 2)
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        weights = self.attn_drop(torch.softmax(scores, dim=-1))
        ctx     = torch.matmul(weights, V)
        ctx = ctx.transpose(1, 2).contiguous().view(N, S, self.d_model)
        return self.W_o(ctx)

class ParallelBranchAttention(nn.Module):
    def __init__(self, embed_dim, num_branches=4, num_heads_per_branch=4, dropout=0.1):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList([
            ScaledDotProductMultiHeadAttention(embed_dim, num_heads_per_branch, dropout)
            for _ in range(num_branches)
        ])
    def forward(self, x):
        out = sum(branch(x) for branch in self.branches)
        return out / self.num_branches

class ParallelBranchTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_branches, num_heads_per_branch, mlp_dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attention = ParallelBranchAttention(
            embed_dim, num_branches=num_branches,
            num_heads_per_branch=num_heads_per_branch, dropout=0.1)
        self.ffn_norm  = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ==========================================
# 2. STANDALONE PYTORCH MHA MODEL FOR ATTACK CLASSIFICATION
# ==========================================

class Deep_MHA_Tabular(nn.Module):
    def __init__(self, num_numeric_features, num_bins,
                 value_dim=224, feat_dim=64, n_segments=None,
                 num_layers=4, num_branches=4, num_heads_per_branch=4,
                 num_classes=5, mlp_dropout=0.4, value_embed="ple"):
        super().__init__()
        embed_dim = value_dim + feat_dim
        
        self.feature_embedder = MonotonicTabularEmbedding(
            num_features=num_numeric_features, num_bins=num_bins,
            value_dim=value_dim, feat_dim=feat_dim, n_segments=n_segments, value_embed=value_embed,
        )
        
        self.input_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            ParallelBranchTransformerBlock(embed_dim, num_branches, num_heads_per_branch, mlp_dropout)
            for _ in range(num_layers)
        ])
        self.final_norm  = nn.LayerNorm(embed_dim)
        self.attn_pool_w = nn.Linear(embed_dim, 1)

        self.attack_classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(mlp_dropout / 2),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, tokenized_x):
        x = self.feature_embedder(tokenized_x)
        x = self.input_norm(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)

        attn_w        = torch.softmax(self.attn_pool_w(x), dim=1)
        attn_pooled   = (x * attn_w).sum(dim=1)
        max_pooled, _ = x.max(dim=1)
        mean_pooled   = x.mean(dim=1)

        pooled = torch.cat([attn_pooled, max_pooled, mean_pooled], dim=1)
        logits = self.attack_classifier(pooled)
        return logits


# ==========================================
# 3. MHA EXPERT TRAINER COMPONENT
# ==========================================
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, scheduler_per_batch=False, grad_clip=1.0):
        self.model               = model
        self.criterion           = criterion
        self.optimizer           = optimizer
        self.scheduler           = scheduler
        self.device              = device
        self.scheduler_per_batch = scheduler_per_batch
        self.grad_clip           = grad_clip
        self.best_val_f1         = 0.0
        self.best_model_state    = None
        self.scaler = torch.amp.GradScaler('cuda') if getattr(device, 'type', 'cpu') == 'cuda' else None
        # Lịch sử để vẽ đường huấn luyện
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [],
                        "val_acc": [], "train_f1m": [], "val_f1m": []}

    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss = 0.0
        preds, trues = [], []
        ctx = torch.enable_grad() if train else torch.no_grad()

        pbar = tqdm(loader, total=len(loader), desc="MHA-Train" if train else "MHA-Eval", dynamic_ncols=True, leave=True)

        with ctx:
            for step, (num_x, lbl) in enumerate(pbar):
                num_x = num_x.to(torch.long).to(self.device)
                lbl   = lbl.to(torch.long).to(self.device)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                device_type = getattr(self.device, 'type', 'cpu')
                with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                    logits = self.model(num_x)
                    loss   = self.criterion(logits, lbl)

                if train:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        if self.grad_clip:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                    if self.scheduler and self.scheduler_per_batch:
                        self.scheduler.step()

                loss_val = loss.item()
                total_loss += loss_val

                preds.extend(logits.argmax(dim=1).cpu().numpy())
                trues.extend(lbl.cpu().numpy())

                if step % 10 == 0:
                    current_acc = accuracy_score(trues, preds)
                    postfix = {'loss': f'{loss_val:.4f}', 'acc': f'{current_acc:.4f}'}
                    if train:
                        postfix['lr'] = f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    pbar.set_postfix(postfix)

        avg_loss = total_loss / len(loader)
        acc  = accuracy_score(trues, preds)
        f1_w = f1_score(trues, preds, average='weighted', zero_division=0)
        f1_m = f1_score(trues, preds, average='macro',    zero_division=0)
        return avg_loss, acc, f1_w, f1_m, preds, trues

    def fit(self, train_loader, val_loader, epochs=50, patience=15):
        print(f"\n{'='*80}\n{'BẮT ĐẦU HUẤN LUYỆN CHUYÊN GIA MHA ATTACK CLASSIFIER':^80}\n{'='*80}\n")
        no_improve = 0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, tr_f1w, tr_f1m, _, _ = self._run_epoch(train_loader, train=True)
            va_loss, va_acc, va_f1w, va_f1m, _, _ = self._run_epoch(val_loader,   train=False)

            if self.scheduler and not self.scheduler_per_batch:
                self.scheduler.step(va_loss)

            mark = " ✦" if va_f1m > self.best_val_f1 else ""
            print(f"Epoch {epoch:03d}/{epochs} | Tr Loss: {tr_loss:.4f} | Tr Acc: {tr_acc:.4f} | "
                  f"Va Loss: {va_loss:.4f} | Va Acc: {va_acc:.4f} | Va F1-m: {va_f1m:.4f} | {time.time()-t0:.1f}s{mark}")

            # Lưu lịch sử để vẽ đường huấn luyện cuối
            self.history["train_loss"].append(tr_loss); self.history["val_loss"].append(va_loss)
            self.history["train_acc"].append(tr_acc);   self.history["val_acc"].append(va_acc)
            self.history["train_f1m"].append(tr_f1m);   self.history["val_f1m"].append(va_f1m)

            # Log đường cong huấn luyện MHA theo từng epoch (prefix mha_ để khỏi lẫn metric tổng kết)
            if mlflow.active_run() is not None:
                mlflow.log_metrics({
                    "mha_train_loss": tr_loss, "mha_train_acc": tr_acc, "mha_train_f1m": tr_f1m,
                    "mha_val_loss":   va_loss, "mha_val_acc":   va_acc,
                    "mha_val_f1w":    va_f1w,  "mha_val_f1m":   va_f1m,
                    "mha_lr":         self.optimizer.param_groups[0]["lr"],
                }, step=epoch)

            if va_f1m > self.best_val_f1:
                self.best_val_f1      = va_f1m
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n⚠️ Early stopping tại epoch {epoch}")
                    break

        print(f"\n✓ Nạp lại best checkpoint MHA (Macro-F1 = {self.best_val_f1:.4f})")
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)


def export_hybrid_misclassified_to_txt(xgb_model, mha_model, X_test, y_test, target_names, benign_idx, device, filepath="misclassified_samples.txt"):
    """
    Xuất các mẫu bị toàn hệ thống Hybrid dự đoán sai ra file text để phân tích.
    """
    print(f"\nĐang trích xuất mẫu dự đoán sai của hệ thống Hybrid ra file: {filepath} ...")
    
    # Giai đoạn 1: Chạy XGBoost gác cổng trên Test set
    test_bin_preds = xgb_model.predict(X_test)
    final_preds    = np.full(len(X_test), benign_idx)
    
    # Giai đoạn 2: Lấy các mẫu XGBoost nghi ngờ là Attack chuyển qua MHA
    attack_indices = np.where(test_bin_preds == 1)[0]
    if len(attack_indices) > 0:
        X_test_suspects = X_test[attack_indices]
        mha_model.eval()
        mha_preds_list = []
        batch_size = CONFIG["inference_batch_size"]
        with torch.no_grad():
            for i in range(0, len(X_test_suspects), batch_size):
                batch_x = torch.tensor(X_test_suspects[i:i+batch_size], dtype=torch.long).to(device)
                logits  = mha_model(batch_x)
                logits[:, benign_idx] = -float('inf')  # Ép MHA không chọn nhãn Benign
                preds_batch = logits.argmax(dim=1).cpu().numpy()
                mha_preds_list.extend(preds_batch)
        final_preds[attack_indices] = mha_preds_list
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("True_Label\tPredicted_Label\tFeatures_Tokenized\n")
        for i in range(len(final_preds)):
            if final_preds[i] != y_test[i]:
                f.write(f"{target_names[y_test[i]]}\t{target_names[final_preds[i]]}\t{','.join(map(str, X_test[i]))}\n")
    print("✓ Hoàn tất xuất file mẫu lỗi.")




def plot_mha_training_curves(history, path="mha_training_curves.png"):
    """Vẽ đường huấn luyện MHA: Loss / Accuracy / Macro-F1 (train vs val) theo epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    if len(history["train_loss"]) == 0:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [("Loss", "train_loss", "val_loss"),
              ("Accuracy", "train_acc", "val_acc"),
              ("Macro-F1", "train_f1m", "val_f1m")]
    for ax, (title, tr_k, va_k) in zip(axes, panels):
        ax.plot(epochs, history[tr_k], "o-", label="train")
        ax.plot(epochs, history[va_k], "s-", label="val")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.set_title(f"MHA {title}"); ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("MHA Attack-Classifier — Training Curves")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_xgb_training_curve(xgb_model, path="xgb_training_curve.png", metric="logloss"):
    """Vẽ đường học XGBoost (logloss train vs val) theo số cây boosting."""
    try:
        res = xgb_model.evals_result()
    except Exception:
        return None
    if not res:
        return None
    keys = list(res.keys())                       # validation_0 (train), validation_1 (val)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = {0: "train", 1: "val"}
    for i, k in enumerate(keys):
        if metric in res[k]:
            ax.plot(range(1, len(res[k][metric]) + 1), res[k][metric],
                    label=labels.get(i, k))
    best_it = getattr(xgb_model, "best_iteration", None)
    if best_it is not None:
        ax.axvline(best_it + 1, color="red", ls="--", alpha=0.6,
                   label=f"best_iter={best_it}")
    ax.set_xlabel("Boosting round"); ax.set_ylabel(metric)
    ax.set_title("XGBoost Gatekeeper — Learning Curve")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def log_and_persist_run_artifacts(
    xgb_model,
    mha_model,
    trainer,
    y_test,
    final_hybrid_preds,
    target_names,
    device,
    misclassified_path="misclassified_samples.txt",
    source_path=None,
):
    """Log final metrics, reports, confusion matrix, and models to MLflow."""
    overall_acc = accuracy_score(y_test, final_hybrid_preds)
    macro_f1 = f1_score(y_test, final_hybrid_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, final_hybrid_preds, average="weighted", zero_division=0)

    mlflow.log_metrics({
        "hybrid_test_accuracy": overall_acc,
        "hybrid_test_macro_f1": macro_f1,
        "hybrid_test_weighted_f1": weighted_f1,
        "best_mha_val_macro_f1": trainer.best_val_f1,
    })

    report_text = classification_report(
        y_test,
        final_hybrid_preds,
        target_names=target_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        final_hybrid_preds,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    with open("classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)

    cm = confusion_matrix(y_test, final_hybrid_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Hybrid Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    with open("target_names.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_names": target_names,
                "num_classes": len(target_names),
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    if source_path is None:
        source_path = Path(globals().get("__file__", "Main.py"))
    source_path = Path(source_path)
    if source_path.exists():
        try:
            shutil.copy2(source_path, "source_code_copy.py")
        except Exception:
            pass

    xgb_model.save_model("xgb_gatekeeper.json")
    torch.save(mha_model.state_dict(), "best_hybrid_mha_expert.pt")

    for artifact in [
        "classification_report.txt",
        "classification_report.json",
        "confusion_matrix.png",
        "target_names.json",
        "xgb_gatekeeper.json",
        "best_hybrid_mha_expert.pt",
        misclassified_path,
        "source_code_copy.py",
    ]:
        if Path(artifact).exists():
            mlflow.log_artifact(artifact)

    # mlflow 3.x: dùng `name=` (artifact_path đã deprecated)
    mlflow.xgboost.log_model(xgb_model, name="xgb_binary_filter")
    mlflow.pytorch.log_model(mha_model, name="mha_expert")

    # Backup thư mục mlruns (chỉ trên Kaggle) — bỏ qua nếu không tồn tại để không crash local
    try:
        if os.path.isdir("/kaggle/working/mlruns"):
            shutil.make_archive("/kaggle/working/mlruns_backup", "zip",
                                "/kaggle/working/mlruns")
    except Exception as e:
        print(f"⚠️ Bỏ qua backup mlruns: {e}")

# ==========================================
# 4. RUN PIPELINE
# ==========================================
if __name__ == "__main__":
    with mlflow.start_run(run_name="Hybrid_XGB_MHA"):
        mlflow.log_params(CONFIG)
        mlflow.set_tags({"project": "Hybrid_DDoS", "framework": "XGBoost+MHA", "run_type": "kaggle"})
    
        def seed_everything(seed=42):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False

        # Sử dụng seed từ CONFIG (seed đã nằm trong log_params(CONFIG) → không log lại)
        seed_everything(CONFIG["seed"])
        mlflow.log_param("tracking_uri", mlflow.get_tracking_uri())

        print("1. Đang tải dữ liệu từ 'dataset_tokenized.txt'...")
        _env_data = os.environ.get("DATA_FILE")
        _CANDIDATE_PATHS = ([_env_data] if _env_data else []) + [
            "/kaggle/input/datasets/zerefnako/datafinel/dataset_tokenized (1).txt",
            "dataset_tokenized.txt",
        ]
        df = None
        for _p in _CANDIDATE_PATHS:
            if os.path.exists(_p):
                df = pd.read_csv(_p, sep='\t', low_memory=False)
                print(f"   -> Đọc thành công từ PATH: {_p}")
                break
        if df is None:
            print("❌ LỖI: Không tìm thấy file dataset_tokenized.txt!")
            exit(1)

        LABEL_COL = 'activity'
        df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
        df = df[~df[LABEL_COL].str.lower().eq('suspicious')].copy()
        df[LABEL_COL] = df[LABEL_COL].apply(lambda x: 'Benign' if x.lower().startswith('benign') else x)

        features  = [c for c in df.columns if c != LABEL_COL]
        X         = df[features].values
        y_raw     = df[LABEL_COL].values

        le           = LabelEncoder()
        y            = le.fit_transform(y_raw)
        target_names = [str(cls) for cls in le.classes_]
    
        # LƯU Ý: nhãn benign trong dataset là dạng SỐ '17'. LabelEncoder mã hóa theo thứ tự
        # chuỗi nên '17' KHÔNG map về encoded 17 (mà ra 9). Fallback cũ `else 17` là encoded
        # index SAI -> cổng XGBoost bị đấu nhầm lớp. Xác định benign theo nhãn gốc + kiểm tra.
        if 'Benign' in target_names:
            BENIGN_INDEX = target_names.index('Benign')
        elif '17' in target_names:
            BENIGN_INDEX = target_names.index('17')
        else:
            BENIGN_INDEX = int(np.bincount(y).argmax())   # fallback: lớp đa số
        # Benign phải là lớp đa số trong dataset này -> chốt chặn chống đấu nhầm index
        _majority = int(np.bincount(y).argmax())
        assert BENIGN_INDEX == _majority, (
            f"BENIGN_INDEX={BENIGN_INDEX} (nhãn gốc='{target_names[BENIGN_INDEX]}') không phải "
            f"lớp đa số ({_majority}='{target_names[_majority]}'). Kiểm tra lại nhãn Benign!")
        print(f"   -> Đã mã hóa {len(target_names)} nhãn. "
              f"[Index của Benign = {BENIGN_INDEX} (nhãn gốc='{target_names[BENIGN_INDEX]}')]")

        # Tạo nhãn nhị phân cho XGBoost (0: Benign, 1: Toàn bộ các loại Attack)
        y_binary = (y != BENIGN_INDEX).astype(int)

        print("\n   -> Phân phối nhãn thực tế toàn dataset:")
        for cls, ratio in pd.Series(y_raw).value_counts(normalize=True).items():
            print(f"      {str(cls):35s}: {ratio*100:.2f}%")

        NUM_NUMERIC_FEATURES = len(features)
        NUM_CLASSES          = len(target_names)
        NUM_BINS             = int(df[features].max().max()) + 1

        # Chia tập dữ liệu dựa trên Siêu tham số cấu hình
        X_train, X_temp, y_train, y_temp, y_train_bin, y_temp_bin = train_test_split(
            X, y, y_binary, test_size=CONFIG["test_size_total"], stratify=y, random_state=CONFIG["seed"]
        )
        X_val, X_test, y_val, y_test, y_val_bin, y_test_bin = train_test_split(
            X_temp, y_temp, y_temp_bin, test_size=CONFIG["val_test_split"], stratify=y_temp, random_state=CONFIG["seed"]
        )
        mlflow.log_params({
            "num_features": len(features),
            "num_classes": len(target_names),
            "num_bins": NUM_BINS,
        })
        mlflow.log_metrics({
            "dataset_size": float(len(df)),
            "train_size": float(len(X_train)),
            "val_size": float(len(X_val)),
            "test_size": float(len(X_test)),
        })

        # =========================================================================
        # [STAGE 1] HUẤN LUYỆN XGBOOST BINARY FILTER (BỘ LỌC GÁC CỔNG)
        # =========================================================================
        print(f"\n{'='*80}\n[STAGE 1] HUẤN LUYỆN XGBOOST BINARY FILTER\n{'='*80}")
    
        xgb_model = xgb.XGBClassifier(
            n_estimators=CONFIG["xgb_n_estimators"],
            learning_rate=CONFIG["xgb_learning_rate"],
            max_depth=CONFIG["xgb_max_depth"],
            tree_method=CONFIG["xgb_tree_method"],
            random_state=CONFIG["seed"],
            eval_metric='logloss',
            early_stopping_rounds=CONFIG["xgb_early_stopping_rounds"]
        )
    
        t0 = time.time()
        xgb_model.fit(
            X_train, y_train_bin,
            eval_set=[(X_train, y_train_bin), (X_val, y_val_bin)],  # train + val → 2 đường học
            verbose=10
        )
        print(f"✅ XGBoost Filter hoàn tất huấn luyện sau {time.time() - t0:.1f}s")

        # Đường học XGBoost → PNG + log MLflow
        _xgb_curve = plot_xgb_training_curve(xgb_model, "xgb_training_curve.png")
        if _xgb_curve and mlflow.active_run() is not None:
            mlflow.log_artifact(_xgb_curve)
    
        # Kiểm tra nhanh hiệu năng độc lập của XGBoost trên tập Validation
        xgb_val_preds = xgb_model.predict(X_val)
        print(f"   -> XGBoost Validation Accuracy (Nhị phân): {accuracy_score(y_val_bin, xgb_val_preds):.4f}")
        xgb_val_acc = accuracy_score(y_val_bin, xgb_val_preds)
        xgb_val_f1 = f1_score(y_val_bin, xgb_val_preds, zero_division=0)
        mlflow.log_metrics({
            "xgb_val_accuracy": xgb_val_acc,
            "xgb_val_f1": xgb_val_f1,
        })

        # =========================================================================
        # [STAGE 2] HUẤN LUYỆN MHA ATTACK CLASSIFIER (BỘ CHUYÊN GIA)
        # =========================================================================
        print(f"\n{'='*80}\n[STAGE 2] HUẤN LUYỆN MHA ATTACK SPECIALIST\n{'='*80}")
    
        # LỌC BỎ HOÀN TOÀN MẪU BENIGN: MHA chỉ học cách bóc tách giữa các dạng Attack với nhau
        train_atk_mask = (y_train != BENIGN_INDEX)
        X_train_atk    = X_train[train_atk_mask]
        y_train_atk    = y_train[train_atk_mask]

        val_atk_mask   = (y_val != BENIGN_INDEX)
        X_val_atk      = X_val[val_atk_mask]
        y_val_atk      = y_val[val_atk_mask]
    
        print(f"   -> Số mẫu đưa vào MHA: Train_Atk = {len(X_train_atk)}, Val_Atk = {len(X_val_atk)}")

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train_atk, dtype=torch.long), torch.tensor(y_train_atk, dtype=torch.long)), batch_size=CONFIG["mha_batch_size"], shuffle=True)
        val_loader   = DataLoader(TensorDataset(torch.tensor(X_val_atk, dtype=torch.long), torch.tensor(y_val_atk, dtype=torch.long)), batch_size=CONFIG["mha_batch_size"], shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        mha_model = Deep_MHA_Tabular(
            num_numeric_features = NUM_NUMERIC_FEATURES,
            num_bins             = NUM_BINS,
            value_dim            = CONFIG["mha_value_dim"],
            feat_dim             = CONFIG["mha_feat_dim"],
            n_segments           = CONFIG["mha_n_segments"],
            num_layers           = CONFIG["mha_num_layers"],
            num_branches         = CONFIG["mha_num_branches"],
            num_heads_per_branch = CONFIG["mha_num_heads_per_branch"],
            num_classes          = NUM_CLASSES,
            mlp_dropout          = CONFIG["mha_mlp_dropout"],
            value_embed          = CONFIG["mha_value_embed"]
        ).to(device)

        criterion = FocalLoss(gamma=CONFIG["focal_loss_gamma"])
        optimizer = torch.optim.AdamW(mha_model.parameters(), lr=CONFIG["mha_lr"], weight_decay=CONFIG["mha_weight_decay"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG["mha_lr"], epochs=CONFIG["mha_epochs"], steps_per_epoch=len(train_loader))

        trainer = Trainer(mha_model, criterion, optimizer, scheduler, device, scheduler_per_batch=True)
        trainer.fit(train_loader, val_loader, epochs=CONFIG["mha_epochs"], patience=CONFIG["mha_patience"])
        mlflow.log_metric("best_mha_val_macro_f1", trainer.best_val_f1)

        # Đường huấn luyện MHA → PNG + log MLflow
        _mha_curve = plot_mha_training_curves(trainer.history, "mha_training_curves.png")
        if _mha_curve and mlflow.active_run() is not None:
            mlflow.log_artifact(_mha_curve)

        # =========================================================================
        # [STAGE 3] HYBRID GATED INFERENCE (SUY LUẬN KẾT HỢP TRÊN TẬP TEST ĐỘC LẬP)
        # =========================================================================
        print(f"\n{'='*80}\n[STAGE 3] ĐÁNH GIÁ CHẤT LƯỢNG HỆ THỐNG HYBRID (TEST SET)\n{'='*80}")
    
        print("-> Bước 3.1: Quét toàn bộ traffic tập Test bằng XGBoost Filter...")
        test_bin_preds = xgb_model.predict(X_test)
    
        # Khởi tạo mảng lưu kết quả cuối cùng (mặc định toàn bộ là Benign)
        final_hybrid_preds = np.full(len(X_test), BENIGN_INDEX)
    
        # Tìm các sample bị XGBoost đánh dấu là Attack (nhãn 1)
        attack_indices = np.where(test_bin_preds == 1)[0]
        print(f"-> Bước 3.2: Phát hiện {len(attack_indices)}/{len(X_test)} mẫu nghi ngờ Tấn công.")
    
        if len(attack_indices) > 0:
            print("             Chuyển các mẫu này qua lớp MHA Expert để bóc tách loại Attack...")
            X_test_suspects = X_test[attack_indices]
        
            mha_model.eval()
            mha_preds_list = []
            batch_size = CONFIG["inference_batch_size"]
        
            with torch.no_grad():
                for i in range(0, len(X_test_suspects), batch_size):
                    batch_x = torch.tensor(X_test_suspects[i:i+batch_size], dtype=torch.long).to(device)
                    logits  = mha_model(batch_x)
                
                    # CHẶN TUYỆT ĐỐI: Ép MHA không được chọn lại nhãn Benign bằng cách hạ logit xuống -inf
                    logits[:, BENIGN_INDEX] = -float('inf')
                
                    preds_batch = logits.argmax(dim=1).cpu().numpy()
                    mha_preds_list.extend(preds_batch)
                
            # Ghi đè các kết quả dự đoán chi tiết của MHA vào mảng kết quả cuối
            final_hybrid_preds[attack_indices] = mha_preds_list

        # =========================================================================
        # ĐÁNH GIÁ BÁO CÁO CUỐI CÙNG
        # =========================================================================
        print("\n" + "#"*40 + "\n[KẾT QUẢ CUỐI CÙNG CỦA HỆ THỐNG HYBRID (XGBOOST + MHA)]\n" + "#"*40)
    
        # ---- TÍNH TOÁN VÀ HIỂN THỊ CÔNG THỨC ACCURACY CUỐI HỆ THỐNG ----
        overall_hybrid_acc = accuracy_score(y_test, final_hybrid_preds)
        total_samples = len(y_test)
        correct_samples = np.sum(final_hybrid_preds == y_test)
    
        print(f"\n🎯 ĐỘ CHÍNH XÁC TỔNG THỂ (Overall Accuracy): {overall_hybrid_acc * 100:.4f}%")
        print(f"📊 Chi tiết số mẫu: Đúng {correct_samples} / Tổng số {total_samples} mẫu tập Test.")
        print(f"📝 Công thức logic hệ thống:")
        print(f"   Accuracy = (Số mẫu XGBoost đoán đúng Benign + Số mẫu MHA đoán đúng loại Attack) / Tổng số mẫu")
        print(f"            = {correct_samples} / {total_samples} = {overall_hybrid_acc:.6f}")
        print("-" * 80)

        print("\nClassification Report:")
        report_text = classification_report(y_test, final_hybrid_preds, target_names=target_names, zero_division=0)
        report_dict = classification_report(y_test, final_hybrid_preds, target_names=target_names, zero_division=0, output_dict=True)
        print(report_text)

        with open("classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        with open("classification_report.json", "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=4, ensure_ascii=False)
    
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, final_hybrid_preds)
        print(cm)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Hybrid Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Lưu lại trọng số của mô hình MHA chuyên gia
        torch.save(mha_model.state_dict(), "best_hybrid_mha_expert.pt")
        xgb_model.save_model("xgb_gatekeeper.json")
        print("\n✅ Đã lưu trọng số MHA Expert tại 'best_hybrid_mha_expert.pt'")

        # Xuất các mẫu đoán sai ra file text để debug
        export_hybrid_misclassified_to_txt(
            xgb_model, mha_model, X_test, y_test, target_names, BENIGN_INDEX, device, "misclassified_samples.txt"
        )
        log_and_persist_run_artifacts(
            xgb_model=xgb_model,
            mha_model=mha_model,
            trainer=trainer,
            y_test=y_test,
            final_hybrid_preds=final_hybrid_preds,
            target_names=target_names,
            device=device,
            misclassified_path="misclassified_samples.txt",
            source_path=None,   # None → tự lấy __file__ của script này
        )
