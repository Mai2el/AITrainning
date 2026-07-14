"""
start.py
--------
Huấn luyện mô hình DDoS Detection (Parallel Multi-Branch Attention - 4 nhánh × 4 head = 16 head).

Xử lý bảng như ngôn ngữ tự nhiên: mỗi dòng network flow là 1 sequence, mỗi cột
là 1 token. Pipeline khớp với tabular_tokenizer.py:
  Pre-processing → Feature Profiling → Grouping → Scaling (căn 2/3) → Discrete tokens
  Embedding mỗi token:
    v_value   = PLE(token_id)         [piecewise-linear value embedding, đơn điệu]
    v_feature = Embedding(feature_id) [feature/column identity embedding]
    v_out     = concat(v_value, v_feature)
  Mỗi dòng → ma trận (num_features, embed_dim) → Transformer MHA

Kiến trúc:
  4 Transformer Layer × Multi-Head Attention (16 head) · Pre-LayerNorm
  Pooling 3 chiều (attention + max + mean) → MLP classifier

Ghi chú thiết kế:
  - N_SEGMENTS <= VALUE_DIM để tránh nén thông tin PLE
  - Pooling gồm cả mean_pooled (attn + max + mean)
  - FocalLoss không nhân class_weights (tránh double-counting với (1-pt)^gamma)
  - NUM_BINS clamp tối đa 1024
"""

import os
import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, TensorDataset
import time
import random
from tqdm.auto import tqdm

# MLflow — tuỳ chọn: log params/metrics/artifacts. Không cài cũng chạy bình thường.
try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False
    print("ℹ️  Không có mlflow — bỏ qua logging (pip install mlflow để bật).")


class FocalLoss(nn.Module):
    """
    Focal Loss — không dùng class_weights trực tiếp để tránh double-counting
    với focal modulation factor (1-pt)^gamma.
    Class imbalance được xử lý qua sample_weight bên ngoài nếu cần.
    """
    def __init__(self, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma     = gamma
        self.reduction = reduction
        # Không nhân class_weights ở đây — focal đã điều chỉnh theo pt
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

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
    """
    Piecewise Linear Encoding (PLE) — Monotonic value embedding.
    """
    def __init__(self, num_bins: int, value_dim: int, n_segments: int = None):
        super().__init__()
        self.num_bins   = max(num_bins, 1)
        self.value_dim  = value_dim
        # Clamp n_segments <= value_dim để tránh nén thông tin PLE
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
    """
    Embedding số ĐƠN GIẢN (baseline so với PLE): t∈[0,1] → MLP 2 lớp → value_dim.
    Không có cấu trúc piecewise; chỉ là MLP nhỏ trên giá trị chuẩn hoá.
    """
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
        assert embed_dim % num_heads == 0, (
            f"embed_dim={embed_dim} phải chia hết cho num_heads={num_heads}"
        )
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
    """
    Parallel Multi-Branch Attention (KHÔNG phải MoE — không có router).
      - num_branches nhánh xử lý SONG SONG, mỗi nhánh là 1 Multi-Head Attention
        độc lập với num_heads_per_branch head (W_q/W_k/W_v/W_o riêng).
      - Mặc định 4 nhánh × 4 head = 16 head.
      - Output = trung bình của các nhánh.
    """
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
    """Khối Transformer Pre-LayerNorm dùng Parallel Multi-Branch Attention + FFN."""
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


class Deep_MHA_Tabular(nn.Module):
    """
    Mô hình chính: nhúng PLE + feature-identity → N tầng Transformer MHA →
    pooling 3 chiều (attention + max + mean) → MLP classifier.
    """
    def __init__(self, num_numeric_features, num_bins,
                 value_dim=224, feat_dim=64, n_segments=None,
                 num_layers=4, num_branches=4, num_heads_per_branch=4,
                 num_classes=5, mlp_dropout=0.4, value_embed="ple"):
        super().__init__()
        embed_dim = value_dim + feat_dim
        assert embed_dim % num_heads_per_branch == 0, (
            f"embed_dim={embed_dim} (value_dim={value_dim} + feat_dim={feat_dim}) "
            f"phải chia hết cho num_heads_per_branch={num_heads_per_branch}"
        )

        self.feature_embedder = MonotonicTabularEmbedding(
            num_features=num_numeric_features,
            num_bins=num_bins,
            value_dim=value_dim,
            feat_dim=feat_dim,
            n_segments=n_segments,
            value_embed=value_embed,
        )
        self.input_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            ParallelBranchTransformerBlock(embed_dim, num_branches,
                                           num_heads_per_branch, mlp_dropout)
            for _ in range(num_layers)
        ])
        self.final_norm  = nn.LayerNorm(embed_dim)
        self.attn_pool_w = nn.Linear(embed_dim, 1)

        # Classifier nhận embed_dim * 3 vì pooling gồm attn + max + mean
        self.classifier = nn.Sequential(
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

        # Pooling 3 chiều: attention + max + mean
        attn_w        = torch.softmax(self.attn_pool_w(x), dim=1)
        attn_pooled   = (x * attn_w).sum(dim=1)
        max_pooled, _ = x.max(dim=1)
        mean_pooled   = x.mean(dim=1)

        pooled = torch.cat([attn_pooled, max_pooled, mean_pooled], dim=1)
        return self.classifier(pooled)


# ==========================================
# 3. TRAINER COMPONENTS
# ==========================================
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler=None, device="cpu",
                 scheduler_per_batch=False, grad_clip=1.0):
        self.model               = model
        self.criterion           = criterion
        self.optimizer           = optimizer
        self.scheduler           = scheduler
        self.device              = device
        self.scheduler_per_batch = scheduler_per_batch
        self.grad_clip           = grad_clip
        self.best_val_f1         = 0.0
        self.best_model_state    = None
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if getattr(device, 'type', 'cpu') == 'cuda'
            else None
        )

    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss = 0.0
        preds, trues = [], []
        correct = 0
        total = 0
        ctx = torch.enable_grad() if train else torch.no_grad()

        pbar = tqdm(
            loader, total=len(loader),
            desc="Train" if train else "Eval",
            dynamic_ncols=True, leave=True, ascii=True
        )

        with ctx:
            for step, (num_x, lbl) in enumerate(pbar):
                num_x = num_x.to(torch.long, non_blocking=True).to(self.device)
                lbl   = lbl.to(torch.long,   non_blocking=True).to(self.device)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                device_type = getattr(self.device, 'type', 'cpu')
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.float16,
                    enabled=(device_type == 'cuda')
                ):
                    logits = self.model(num_x)
                    loss   = self.criterion(logits, lbl)

                if train:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        if self.grad_clip is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_norm=self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_norm=self.grad_clip)
                        self.optimizer.step()

                    if self.scheduler is not None and self.scheduler_per_batch:
                        self.scheduler.step()

                loss_val = loss.item()
                total_loss += loss_val
                pred = logits.argmax(1)
                correct += (pred == lbl).sum().item()
                total += lbl.size(0)
                preds.extend(pred.cpu().numpy())
                trues.extend(lbl.cpu().numpy())

                if step % 10 == 0:
                    current_acc = correct / max(total, 1)
                    if train:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        pbar.set_postfix({
                            'loss': f'{loss_val:.4f}',
                            'acc':  f'{current_acc:.4f}',
                            'lr':   f'{current_lr:.2e}'
                        })
                    else:
                        pbar.set_postfix({
                            'loss': f'{loss_val:.4f}',
                            'acc':  f'{current_acc:.4f}'
                        })

        avg_loss = total_loss / len(loader)
        acc  = accuracy_score(trues, preds)
        f1_w = f1_score(trues, preds, average='weighted', zero_division=0)
        f1_m = f1_score(trues, preds, average='macro',    zero_division=0)
        return avg_loss, acc, f1_w, f1_m, preds, trues

    def fit(self, train_loader, val_loader, epochs=50, patience=18):
        print(f"\n{'='*80}")
        print(f"{'BẮT ĐẦU HUẤN LUYỆN':^80}")
        print(f"{'='*80}\n")

        no_improve = 0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, tr_f1w, tr_f1m, _, _ = self._run_epoch(train_loader, train=True)
            va_loss, va_acc, va_f1w, va_f1m, _, _ = self._run_epoch(val_loader,   train=False)

            if self.scheduler and not self.scheduler_per_batch:
                self.scheduler.step(va_loss)

            elapsed = time.time() - t0
            mark    = " ✦" if va_f1m > self.best_val_f1 else ""
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
                f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc:.4f} | "
                f"Val F1-w: {va_f1w:.4f} | Val F1-m: {va_f1m:.4f} | "
                f"{elapsed:.1f}s{mark}"
            )

            if _HAS_MLFLOW and mlflow.active_run() is not None:
                mlflow.log_metrics({
                    "train_loss": tr_loss, "train_acc": tr_acc, "train_f1m": tr_f1m,
                    "val_loss":   va_loss, "val_acc":   va_acc,
                    "val_f1w":    va_f1w,  "val_f1m":   va_f1m,
                    "lr":         self.optimizer.param_groups[0]["lr"],
                }, step=epoch)

            if va_f1m > self.best_val_f1:
                self.best_val_f1      = va_f1m
                self.best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n⚠️ Early stopping tại epoch {epoch}")
                    break

        print(f"\n✓ Nạp lại best checkpoint (Macro-F1 = {self.best_val_f1:.4f})")
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

    def evaluate(self, val_loader, target_names=None):
        print("\nĐang đánh giá mô hình cuối cùng...")
        _, acc, f1w, f1m, preds, trues = self._run_epoch(val_loader, train=False)
        print("\nClassification Report:")
        print(classification_report(trues, preds, target_names=target_names, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(trues, preds))
        print(f"\nAccuracy: {acc:.4f} | Weighted F1: {f1w:.4f} | Macro F1: {f1m:.4f}")
        return acc, f1m

    def export_misclassified_to_txt(self, loader, target_names, filepath="misclassified_samples.txt"):
        print(f"\nĐang trích xuất các mẫu dự đoán sai ra file: {filepath} ...")
        self.model.eval()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("True_Label\tPredicted_Label\tFeatures_Tokenized\n")
            with torch.no_grad():
                for num_x, lbl in loader:
                    num_x_device = num_x.to(torch.long).to(self.device)
                    lbl_device   = lbl.to(torch.long).to(self.device)
                    logits       = self.model(num_x_device)
                    preds        = logits.argmax(1).cpu().numpy()
                    trues        = lbl_device.cpu().numpy()
                    features     = num_x.numpy()
                    for i in range(len(preds)):
                        if preds[i] != trues[i]:
                            true_str = target_names[trues[i]]
                            pred_str = target_names[preds[i]]
                            feat_str = ",".join(map(str, features[i]))
                            f.write(f"{true_str}\t{pred_str}\t{feat_str}\n")
        print(f"✅ Đã ghi xong danh sách dự đoán sai vào: {filepath}")


# ==========================================
# 4. RUN PIPELINE
# ==========================================
if __name__ == "__main__":
    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        _ = check_random_state(seed)

    seed_everything(42)

    # ------------------------------------------------------------------
    # Kiểm tra class distribution trước khi train
    # ------------------------------------------------------------------
    print("1. Đang tải dữ liệu từ 'dataset_tokenized.txt'...")
    _env_data = os.environ.get("DATA_FILE")
    _CANDIDATE_PATHS = ([_env_data] if _env_data else []) + [
        "dataset_tokenized.txt",          # file token chính
        "dataset_bin_tokenized.txt",
        "/kaggle/working/dataset_tokenized.txt",
    ]
    df = None
    for _p in _CANDIDATE_PATHS:
        if os.path.exists(_p):
            df = pd.read_csv(_p, sep='\t', low_memory=False)
            print(f"   -> Đọc từ: {_p}")
            break
    if df is None:
        print("❌ LỖI: Cần chạy file 'tabular_tokenizer.py' trước để tạo file TXT!")
        exit(1)

    LABEL_COL = 'activity'

    # ------------------------------------------------------------------
    # [DATA-STRATEGY] Làm sạch nhãn
    # ------------------------------------------------------------------
    DROP_SUSPICIOUS = True
    MERGE_BENIGN    = True
    BINARY_MODE     = False   # tối ưu phân loại nhị phân Benign/Attack bằng MHA

    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    n0 = len(df)

    if DROP_SUSPICIOUS:
        df = df[~df[LABEL_COL].str.lower().eq('suspicious')].copy()
        print(f"   -> [DROP] Đã bỏ 'Suspicious': {n0:,} → {len(df):,} dòng "
              f"(-{n0 - len(df):,})")

    if MERGE_BENIGN:
        df[LABEL_COL] = df[LABEL_COL].apply(
            lambda x: 'Benign' if x.lower().startswith('benign') else x)
        print(f"   -> [MERGE] Đã gộp mọi 'Benign*' về 1 lớp 'Benign'")

    if BINARY_MODE:
        df[LABEL_COL] = df[LABEL_COL].apply(
            lambda x: 'Benign' if x.lower().startswith('benign') else 'Attack')
        print(f"   -> [BINARY] Gộp mọi nhãn không-Benign → 'Attack' (phân loại 2 lớp)")

    features  = [c for c in df.columns if c != LABEL_COL]
    X         = df[features].values
    y_raw     = df[LABEL_COL].values

    le           = LabelEncoder()
    y            = le.fit_transform(y_raw)
    target_names = [str(cls) for cls in le.classes_]
    print(f"   -> Đã mã hóa {len(target_names)} nhãn: "
          f"{dict(zip(target_names, range(len(target_names))))}")

    print("\n   -> Phân phối nhãn (toàn bộ dataset):")
    dist = pd.Series(y_raw).value_counts(normalize=True).sort_index()
    for cls, ratio in dist.items():
        print(f"      {str(cls):35s}: {ratio*100:.2f}%")

    NUM_NUMERIC_FEATURES = len(features)
    NUM_CLASSES          = len(target_names)

    # NUM_BINS = vocab thật (KHÔNG clamp). PLE không phải bảng tra theo token nên
    # NUM_BINS lớn KHÔNG tạo embedding lớn — nó chỉ chuẩn hoá token → [0,1].
    # Clamp xuống 1024 sẽ khiến mọi token > 1023 bão hoà (t=token/1023 >1 → clamp).
    NUM_BINS = int(df[features].max().max()) + 1
    print(f"\n   -> Vocabulary size (NUM_BINS): {NUM_BINS:,}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long))
    val_dataset   = TensorDataset(
        torch.tensor(X_val,   dtype=torch.long),
        torch.tensor(y_val,   dtype=torch.long))
    test_dataset  = TensorDataset(
        torch.tensor(X_test,  dtype=torch.long),
        torch.tensor(y_test,  dtype=torch.long))

    BATCH_SIZE = 256
    pin_mem    = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=pin_mem, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=pin_mem, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=pin_mem, num_workers=0)

    config = {
        # Nhúng + rời rạc: value_dim & n_segments lớn → PLE phân giải cao
        # (embed_dim = 192 + 64 = 256, chia hết cho NUM_HEADS=16)
        "VALUE_DIM":             192,
        "FEAT_DIM":               64,
        "N_SEGMENTS":            192,   # = VALUE_DIM → PLE 192 đoạn tuyến tính

        "NUM_LAYERS":              4,
        "NUM_BRANCHES":            4,   
        "NUM_HEADS_PER_BRANCH":    4,  
        "VALUE_EMBED":  os.environ.get("VALUE_EMBED", "ple"),  
        "BATCH_SIZE":        BATCH_SIZE,
        "EPOCHS":      int(os.environ.get("EPOCHS", "50")),
        "PATIENCE":              12,
        "MAX_LR":               1e-3,
        "WEIGHT_DECAY":         1e-2,
        "MLP_DROPOUT":          0.15,
        "NUM_BINS":          NUM_BINS,
        "LABEL_SMOOTHING":      0.05,
        "FOCAL_GAMMA":           1.5,
        "GRAD_CLIP":             1.0,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n   -> Sử dụng thiết bị: {device}")

    model = Deep_MHA_Tabular(
        num_numeric_features = NUM_NUMERIC_FEATURES,
        num_bins             = config["NUM_BINS"],
        value_dim            = config["VALUE_DIM"],
        feat_dim             = config["FEAT_DIM"],
        n_segments           = config["N_SEGMENTS"],
        num_layers           = config["NUM_LAYERS"],
        num_branches         = config["NUM_BRANCHES"],
        num_heads_per_branch = config["NUM_HEADS_PER_BRANCH"],
        num_classes          = NUM_CLASSES,
        mlp_dropout          = config["MLP_DROPOUT"],
        value_embed          = config["VALUE_EMBED"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _nb, _nh = config["NUM_BRANCHES"], config["NUM_HEADS_PER_BRANCH"]
    print(f"   -> Kiến trúc: {_nb} nhánh × {_nh} head = {_nb*_nh} head | "
          f"value_embed={config['VALUE_EMBED']} | Tổng tham số: {total_params:,}")

    # FocalLoss không dùng class_weights trực tiếp
    # → tránh double-counting với focal modulation factor
    criterion = FocalLoss(
        gamma=config["FOCAL_GAMMA"],
        label_smoothing=config["LABEL_SMOOTHING"]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["MAX_LR"],
        weight_decay=config["WEIGHT_DECAY"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["MAX_LR"],
        epochs=config["EPOCHS"],
        steps_per_epoch=len(train_loader),
        pct_start=0.20,
        anneal_strategy="cos",
        div_factor=20.0,
        final_div_factor=250.0,
    )

    trainer = Trainer(
        model, criterion, optimizer, scheduler, device,
        scheduler_per_batch=True,
        grad_clip=config["GRAD_CLIP"]
    )

    # ── MLflow: mở run + log siêu tham số ──
    if _HAS_MLFLOW:
        if os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "ddos-mha"))
        mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME"))
        mlflow.log_params({
            **config,
            "NUM_NUMERIC_FEATURES": NUM_NUMERIC_FEATURES,
            "NUM_CLASSES":          NUM_CLASSES,
            "TOTAL_PARAMS":         total_params,
            "BINARY_MODE":          BINARY_MODE,
            "DROP_SUSPICIOUS":      DROP_SUSPICIOUS,
            "MERGE_BENIGN":         MERGE_BENIGN,
            "DATA_FILE":            _p,
        })
        print(f"   -> MLflow run: {mlflow.active_run().info.run_id}")

    trainer.fit(train_loader, val_loader,
                epochs=config["EPOCHS"],
                patience=config["PATIENCE"])

    print("\n--- Validation set ---")
    val_acc, val_f1m = trainer.evaluate(val_loader, target_names=target_names)

    print("\n--- Test set (độc lập) ---")
    test_acc, test_f1m = trainer.evaluate(test_loader, target_names=target_names)

    misclassified_filename = f"misclassified_test_{time.strftime('%Y%m%d_%H%M')}.txt"
    trainer.export_misclassified_to_txt(test_loader, target_names, filepath=misclassified_filename)

    torch.save(model.state_dict(), "best_mha_model.pt")
    print("\n✅ HOÀN TẤT ĐÀO TẠO VÀ LƯU CHECKPOINT TẠI 'best_mha_model.pt'!")
    print("=" * 80)

    # ── MLflow: log metric cuối + artifacts, đóng run ──
    if _HAS_MLFLOW and mlflow.active_run() is not None:
        mlflow.log_metrics({
            "best_val_f1m": trainer.best_val_f1,
            "final_val_acc": val_acc, "final_val_f1m": val_f1m,
            "test_acc": test_acc, "test_f1m": test_f1m,
        })
        for _art in (misclassified_filename, "best_mha_model.pt"):
            if os.path.exists(_art):
                mlflow.log_artifact(_art)
        mlflow.end_run()
        print("   -> MLflow: đã log metric + artifacts, đóng run.")

    del train_loader, val_loader, test_loader
    torch.cuda.empty_cache()