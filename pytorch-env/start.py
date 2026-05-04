"""
start.py
--------
Huấn luyện mô hình DDoS Detection.

Pipeline (khớp với tabular_tokenizer.py):
  Pre-processing → Feature Profiling → Grouping → Scaling [0,1] → Discrete tokens
  Embedding:
    v_value   = Embedding(token_id   → R^d)   [value embedding]
    v_feature = Embedding(feature_id → R^k)   [feature/column embedding]
    v_out     = concat(v_value, v_feature)     [per-feature token vector]
  Each row → matrix of (num_features, 2d) → MoE Attention model

Config:
  4 Transformer Layers × MoE Attention (4 Experts, 4 Heads/Expert)
  Pre-LayerNorm Architecture
  MLflow Tracking
"""

import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, TensorDataset
import time
import mlflow
import mlflow.pytorch
import random


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9992):
        self.decay = decay
        self.shadow = {}
        self.buffers = {}
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()
                
        for name, b in model.named_buffers():
            self.buffers[name] = b.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)
                
        for name, b in model.named_buffers():
            if name in self.buffers:
                self.buffers[name].copy_(b.detach())

    @torch.no_grad()
    def copy_to_model(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])
                
        for name, b in model.named_buffers():
            if name in self.buffers:
                b.data.copy_(self.buffers[name])

    def state_dict(self):
        state = {k: v.clone() for k, v in self.shadow.items()}
        state.update({'buffer_' + k: v.clone() for k, v in self.buffers.items()})
        return state

    def load_state_dict(self, state: dict):
        for k in self.shadow:
            if k in state:
                self.shadow[k].copy_(state[k])
        for k in self.buffers:
            buf_key = 'buffer_' + k
            if buf_key in state:
                self.buffers[k].copy_(state[buf_key])


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight 
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)          # single CE (with label_smoothing)
        pt = torch.exp(-ce_loss)                     # consistent pt
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.weight is not None:
            weights = self.weight[targets]
            focal_loss = focal_loss * weights
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


# ==========================================
# 1. TABULAR EMBEDDING & MOE ATTENTION COMPONENTS
# ==========================================

class PLEValueEmbedding(nn.Module):
    """
    Piecewise Linear Encoding (PLE) — Monotonic value embedding.
    """
    def __init__(self, num_bins: int, value_dim: int, n_segments: int = None):
        super().__init__()
        self.num_bins   = max(num_bins, 1)
        self.value_dim  = value_dim
        self.n_segments = n_segments if n_segments is not None else value_dim
        self.proj = nn.Linear(self.n_segments, value_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _ple_encode(self, t: torch.Tensor) -> torch.Tensor:
        T   = self.n_segments
        w   = 1.0 / T
        left = torch.arange(T, dtype=torch.float32, device=t.device) * w
        fill = ((t.unsqueeze(-1) - left) / w).clamp(0.0, 1.0)
        return fill

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        t   = token_ids.float() / (self.num_bins - 1)
        ple = self._ple_encode(t)
        return self.proj(ple)


class MonotonicTabularEmbedding(nn.Module):
    def __init__(self, num_features: int, num_bins: int,
                 value_dim: int, feat_dim: int, n_segments: int = None):
        super().__init__()
        self.num_features = num_features
        self.value_embedding = PLEValueEmbedding(num_bins, value_dim, n_segments)
        self.feature_embedding = nn.Embedding(num_features, feat_dim)
        nn.init.trunc_normal_(self.feature_embedding.weight, std=0.02)

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        val_emb  = self.value_embedding(x_tokens)
        feat_idx = torch.arange(self.num_features, device=x_tokens.device)
        feat_emb = self.feature_embedding(feat_idx).unsqueeze(0)
        feat_emb = feat_emb.expand(x_tokens.size(0), -1, -1)
        return torch.cat([val_emb, feat_emb], dim=-1)


class MultiHeadAttentionGrid(nn.Module):
    """
    Parameterized Multi-Head Attention:

        head_i = softmax( Q Aᵢ Qᵀ / √d_k ) V

    Thay vì dùng W^Q_i và W^K_i riêng biệt mỗi head (chuẩn MHA),
    ở đây dùng:
      - 1 projection Q chung: Q = X W^Q          shape (N, H, S, d_k)
      - Ma trận Aᵢ học được per-head:            shape (H, d_k, d_k)
      - scores_i = Q_i @ Aᵢ @ Q_iᵀ / √d_k      shape (N, H, S, S)
      - V = X W^V vẫn giữ nguyên

    Lợi ích: Aᵢ học cách hai token tương tác mà không cần key riêng,
    giảm tham số (bỏ W^K) và cho phép các heads học các "interaction basis"
    khác nhau qua Aᵢ.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k       = embed_dim // num_heads
        self.d_model   = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.A   = nn.Parameter(torch.empty(num_heads, self.d_k, self.d_k))
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W_q.weight); nn.init.zeros_(self.W_q.bias)
        nn.init.xavier_uniform_(self.W_v.weight); nn.init.zeros_(self.W_v.bias)
        nn.init.xavier_uniform_(self.W_o.weight); nn.init.zeros_(self.W_o.bias)
        for i in range(num_heads):
            nn.init.eye_(self.A[i])

    def forward(self, x):
        N, S, _ = x.size()
        H, dk   = self.num_heads, self.d_k
        Q = self.W_q(x).view(N, S, H, dk).transpose(1, 2)   
        V = self.W_v(x).view(N, S, H, dk).transpose(1, 2)   
        QA     = torch.einsum('nhsd,hde->nhse', Q, self.A)   
        scores = torch.matmul(QA, Q.transpose(-2, -1)) / math.sqrt(dk) 
        weights = self.attn_drop(torch.softmax(scores, dim=-1))
        ctx     = torch.matmul(weights, V) 
        ctx     = ctx.transpose(1, 2).contiguous().view(N, S, self.d_model)
        return self.W_o(ctx)



class MoEAttention(nn.Module):
    def __init__(self, embed_dim, num_experts=4, num_heads_per_expert=4, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            MultiHeadAttentionGrid(embed_dim, num_heads_per_expert, dropout)
            for _ in range(num_experts)
        ])

        self.router_norm = nn.LayerNorm(embed_dim)
        self.router      = nn.Linear(embed_dim, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

        self.routing_probs = None
        self.router_z_loss = None

    def forward(self, x):
        B, F, D = x.size()
        router_input = self.router_norm(x.mean(dim=1))           
        clean_logits = self.router(router_input)                   
        self.routing_probs = torch.softmax(clean_logits, dim=-1)           
        self.router_z_loss = clean_logits.logsumexp(dim=-1).pow(2).mean() 
        router_logits = clean_logits
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * 0.1
        top_k_weights, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  
        top_k_weights = torch.softmax(top_k_weights, dim=-1)                          
        final_output = torch.zeros_like(x)                         
        for i in range(self.num_experts):
            mask = (top_k_indices == i).any(dim=-1)               
            if not mask.any():
                continue
            expert_out  = self.experts[i](x[mask])                 
            weight_mask = (top_k_indices[mask] == i)               
            expert_w    = (top_k_weights[mask] * weight_mask).sum(dim=-1) 
            final_output[mask] += expert_out * expert_w.unsqueeze(-1).unsqueeze(-1)

        return final_output


# ==========================================
# 2. KIẾN TRÚC MÔ HÌNH CHÍNH (4 LAYERS)
# ==========================================
class MoETransformerBlock(nn.Module):
    """Khối Transformer hoàn chỉnh sử dụng Pre-LayerNorm"""
    def __init__(self, embed_dim, num_experts, num_heads_per_expert, top_k, mlp_dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attention = MoEAttention(
            embed_dim, 
            num_experts=num_experts, 
            num_heads_per_expert=num_heads_per_expert, 
            top_k=top_k, 
            dropout=0.1
        )
        
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # Pre-LN: Chuẩn hóa trước khi đưa vào sub-layer
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Deep_MoE_Tabular(nn.Module):
    def __init__(self, num_numeric_features, num_bins,
                 value_dim=224, feat_dim=64, n_segments=None,
                 num_layers=4, num_experts=4, num_heads_per_expert=4, top_k=2,
                 num_classes=5, mlp_dropout=0.4):
        super().__init__()
        embed_dim = value_dim + feat_dim 
        self.feature_embedder = MonotonicTabularEmbedding(
            num_features = num_numeric_features,
            num_bins     = num_bins,
            value_dim    = value_dim,
            feat_dim     = feat_dim,
            n_segments   = n_segments,
        )
        self.input_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            MoETransformerBlock(embed_dim, num_experts, num_heads_per_expert, top_k, mlp_dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.attn_pool_w = nn.Linear(embed_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(mlp_dropout / 2),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def get_lb_loss(self, lb_coeff=0.01, z_coeff=0.001):
        lb = torch.tensor(0.0, device=next(self.parameters()).device)
        zl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            attn = layer.attention
            if attn.routing_probs is not None:
                mean_probs = attn.routing_probs.mean(dim=0)
                E = attn.num_experts
                lb = lb + E * (mean_probs * mean_probs).sum()
            if attn.router_z_loss is not None:
                zl = zl + attn.router_z_loss
        return lb_coeff * lb + z_coeff * zl

    def forward(self, tokenized_x):
        x = self.feature_embedder(tokenized_x)
        x = self.input_norm(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        attn_w = torch.softmax(self.attn_pool_w(x), dim=1)
        attn_pooled = (x * attn_w).sum(dim=1)
        max_pooled, _ = x.max(dim=1)
        pooled = torch.cat([attn_pooled, max_pooled], dim=1)

        return self.classifier(pooled)


# ==========================================
# 3. TRAINER & MLFLOW LOGGING
# ==========================================
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler=None, device="cpu",
                 scheduler_per_batch=False, ema=None, grad_clip=1.0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scheduler_per_batch = scheduler_per_batch
        self.ema = ema
        self.grad_clip = grad_clip
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.best_ema_state = None
        self.scaler = torch.amp.GradScaler('cuda') if getattr(device, 'type', 'cpu') == 'cuda' else None

    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss, preds, trues = 0.0, [], []
        ctx = torch.enable_grad() if train else torch.no_grad()
        
        with ctx:
            for num_x, lbl in loader:
                num_x = num_x.to(torch.long, non_blocking=True).to(self.device)
                lbl = lbl.to(torch.long, non_blocking=True).to(self.device)
                
                if train: 
                    self.optimizer.zero_grad(set_to_none=True)
                device_type = getattr(self.device, 'type', 'cpu')
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=device_type=='cuda'):
                    logits = self.model(num_x)
                    loss = self.criterion(logits, lbl)
                    # Load-balancing auxiliary loss (encourages uniform expert usage)
                    if train and hasattr(self.model, 'get_lb_loss'):
                        lb_coeff = getattr(self, '_lb_coeff', 0.01)
                        z_coeff  = getattr(self, '_z_coeff',  0.001)
                        loss = loss + self.model.get_lb_loss(lb_coeff=lb_coeff, z_coeff=z_coeff)

                if train:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        if self.grad_clip is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                        self.optimizer.step()
                        
                    if self.scheduler is not None and self.scheduler_per_batch:
                        self.scheduler.step()
                    if self.ema is not None:
                         self.ema.update(self.model)

                total_loss += loss.item()
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(lbl.cpu().numpy())

        avg_loss = total_loss / len(loader)
        acc  = accuracy_score(trues, preds)
        f1_w = f1_score(trues, preds, average='weighted', zero_division=0)
        f1_m = f1_score(trues, preds, average='macro',    zero_division=0)
        return avg_loss, acc, f1_w, f1_m, preds, trues

    def fit(self, train_loader, val_loader, epochs=50, patience=7):
        print(f"\n{'='*80}\n{'BẮT ĐẦU HUẤN LUYỆN':^80}\n{'='*80}\n")
        no_improve = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, tr_f1w, tr_f1m, _, _ = self._run_epoch(train_loader, train=True)
            backup = None
            backup_buf = None
            if self.ema is not None:
                backup = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
                backup_buf = {n: b.detach().clone() for n, b in self.model.named_buffers()}
                self.ema.copy_to_model(self.model)
            va_loss, va_acc, va_f1w, va_f1m, _, _ = self._run_epoch(val_loader, train=False)
            if backup is not None:
                for name, p in self.model.named_parameters():
                    if name in backup:
                        p.data.copy_(backup[name])
                for name, b in self.model.named_buffers():
                    if name in backup_buf:
                        b.data.copy_(backup_buf[name])

            if self.scheduler and not self.scheduler_per_batch:
                self.scheduler.step(va_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            mlflow.log_metrics({
                "train_loss": tr_loss,  "train_acc": tr_acc,
                "train_f1_weighted": tr_f1w, "train_f1_macro": tr_f1m,
                "val_loss": va_loss,    "val_acc": va_acc,
                "val_f1_weighted": va_f1w,   "val_f1_macro": va_f1m,
                "learning_rate": current_lr,
            }, step=epoch)

            elapsed = time.time() - t0
            # ✅ Dùng macro F1 làm tiêu chí early-stopping
            # Weighted F1 bị dominant bởi class đa số → bỏ sót minority DDoS classes
            mark = " ✦" if va_f1m > self.best_val_f1 else ""
            print(f"Epoch {epoch:03d}/{epochs} | "
                  f"Train Loss: {tr_loss:.4f} | "
                  f"Val Loss: {va_loss:.4f}  Val F1-w: {va_f1w:.4f}  Val F1-m: {va_f1m:.4f} | "
                  f"{elapsed:.1f}s{mark}")

            if va_f1m > self.best_val_f1:
                self.best_val_f1 = va_f1m
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                if self.ema is not None:
                    self.best_ema_state = self.ema.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n⚠️ Early stopping tại epoch {epoch} (patience={patience}, metric=macro_F1)")
                    break

        print(f"\n✓ Nạp lại best checkpoint (val Macro-F1 = {self.best_val_f1:.4f})")
        if self.ema is not None and self.best_ema_state is not None:
            self.ema.load_state_dict(self.best_ema_state)
            self.ema.copy_to_model(self.model)
            print("   -> Đã áp dụng trọng số EMA tương ứng best val Macro-F1.")
        elif self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def evaluate(self, val_loader, target_names=None):
        print("\nĐang đánh giá mô hình cuối cùng...")
        _, acc, f1w, f1m, preds, trues = self._run_epoch(val_loader, train=False)
        print("\nClassification Report:")
        report = classification_report(
            trues, preds, target_names=target_names,
            zero_division=0, output_dict=True
        )
        print(classification_report(trues, preds, target_names=target_names, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(trues, preds))
        print(f"\n   Accuracy: {acc:.4f} | Weighted F1: {f1w:.4f} | Macro F1: {f1m:.4f}")
        # Log per-class F1 vào MLflow để dễ debug minority class
        if target_names:
            for cls in target_names:
                if cls in report:
                    mlflow.log_metric(f"f1_{cls}", round(report[cls]['f1-score'], 4))
        return acc, f1m
        
    def export_misclassified_to_txt(self, loader, target_names, filepath="misclassified_samples.txt"):
        print(f"\nĐang trích xuất các mẫu dự đoán sai ra file: {filepath} ...")
        self.model.eval()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("True_Label\tPredicted_Label\tFeatures_Tokenized\n")
            with torch.no_grad():
                for num_x, lbl in loader:
                    num_x_device = num_x.to(torch.long).to(self.device)
                    lbl_device = lbl.to(torch.long).to(self.device)
                    logits = self.model(num_x_device)
                    preds = logits.argmax(1).cpu().numpy()
                    trues = lbl_device.cpu().numpy()
                    features = num_x.numpy() 
                    for i in range(len(preds)):
                        if preds[i] != trues[i]:
                            true_str = target_names[trues[i]]
                            pred_str = target_names[preds[i]]
                            feat_str = ",".join(map(str, features[i]))
                            f.write(f"{true_str}\t{pred_str}\t{feat_str}\n")
        print(f"✅ Đã ghi xong danh sách dự đoán sai vào: {filepath}")


# ==========================================
# 4. CHẠY PIPELINE
# ==========================================
if __name__ == "__main__":
    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        _ = check_random_state(seed)

    seed_everything(42)

    print("1. Đang tải dữ liệu từ 'dataset_tokenized.txt'...")
    try:
        df = pd.read_csv("dataset_tokenized.txt", sep='\t', low_memory=False)
    except FileNotFoundError:
        print("❌ LỖI: Cần chạy file 'tabular_tokenizer.py' trước để tạo file TXT!")
        exit(1)
        
    LABEL_COL = 'activity'
    features = [c for c in df.columns if c != LABEL_COL]
    X = df[features].values
    y_raw = df[LABEL_COL].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    target_names = [str(cls) for cls in le.classes_]
    print(f"   -> Đã mã hóa {len(target_names)} nhãn: {dict(zip(target_names, range(len(target_names))))}")

    NUM_NUMERIC_FEATURES = len(features)
    NUM_CLASSES = len(target_names)
    # Auto-detect vocabulary size from tokenized data
    # (tokenizer may produce token IDs up to 20 000 depending on feature groups)
    NUM_BINS = int(df[features].max().max()) + 1
    print(f"   -> Vocabulary size (NUM_BINS): {NUM_BINS:,}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))

    BATCH_SIZE = 512
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_mem, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_mem, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_mem, num_workers=0)

    # ---------------------------------------------------------
    # BƯỚC 3: CẤU HÌNH MLFLOW VÀ MODEL DEEP 4-LAYER
    # ---------------------------------------------------------
    config = {

        "VALUE_DIM":  224,     
        "FEAT_DIM":    64,     
        "N_SEGMENTS": 224,
        "NUM_LAYERS": 4,               
        "NUM_EXPERTS": 4,              
        "NUM_HEADS_PER_EXPERT": 4,     
        "TOP_K_ROUTING": 2,            
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": 50,
        "PATIENCE": 12,          # tăng từ 8 → cho model thêm thời gian thoát plateau
        "MAX_LR": 1e-3,
        "WEIGHT_DECAY": 1e-2,
        "MLP_DROPOUT": 0.2,      # giảm từ 0.3 → giảm underfitting minority classes
        "NUM_BINS": NUM_BINS,          
        "LABEL_SMOOTHING": 0.0,
        "EMA_DECAY": 0.9992,
        "FOCAL_GAMMA": 2.0,
        "GRAD_CLIP": 1.0,
        "LB_LOSS_COEFF": 0.01,
        "Z_LOSS_COEFF":  0.001,
    }

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("DDoS_Deep4Layer_MoE")
    
    run_name = f"Run_{time.strftime('%Y%m%d_%H%M')}"
    print(f"\n2. Khởi tạo phiên làm việc MLflow: {run_name}")
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   -> Sử dụng thiết bị: {device}")
        
        model = Deep_MoE_Tabular(
            num_numeric_features = NUM_NUMERIC_FEATURES,
            num_bins             = config["NUM_BINS"],
            value_dim            = config["VALUE_DIM"],
            feat_dim             = config["FEAT_DIM"],
            n_segments           = config["N_SEGMENTS"],
            num_layers           = config["NUM_LAYERS"],
            num_experts          = config["NUM_EXPERTS"],
            num_heads_per_expert = config["NUM_HEADS_PER_EXPERT"],
            top_k                = config["TOP_K_ROUTING"],
            num_classes          = NUM_CLASSES,
            mlp_dropout          = config["MLP_DROPOUT"]
        ).to(device)

        samples_per_class = np.bincount(y_train).astype(float)
        # inv-sqrt weighting + soft cap (max 15× min)
        # - inv-sqrt: smoothơn inv-freq, tđ minority có lợi thế nhưng không quá cực đoan
        # - soft cap 15×: tránh ratio 480:1 của inv-freq thuần mà không clip cưứng 0.5
        class_w = 1.0 / np.sqrt(samples_per_class + 1e-8)
        class_w = class_w / class_w.sum() * len(class_w)          # normalize mean=1
        cap     = class_w.min() * 15.0                             # soft cap: max 15× min
        class_w = np.minimum(class_w, cap)
        class_w = class_w / class_w.sum() * len(class_w)          # renormalize
        class_weights = torch.FloatTensor(class_w).to(device)
        print(f"   -> Class weights (inv-sqrt + soft-cap 15×):")
        for tn, w in zip(target_names, class_w):
            print(f"      {tn:35s}: {w:.3f}")
        
        criterion = FocalLoss(
            weight=class_weights,
            gamma=config["FOCAL_GAMMA"],
            label_smoothing=config["LABEL_SMOOTHING"]
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["MAX_LR"], weight_decay=config["WEIGHT_DECAY"])
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

        ema = EMA(model, decay=config["EMA_DECAY"])

        trainer = Trainer(
            model, criterion, optimizer, scheduler, device,
            scheduler_per_batch=True, ema=ema, grad_clip=config["GRAD_CLIP"]
        )
        # Wire MoE auxiliary loss coefficients into Trainer
        trainer._lb_coeff = config["LB_LOSS_COEFF"]
        trainer._z_coeff  = config["Z_LOSS_COEFF"]
        trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"], patience=config["PATIENCE"])

        print("\n--- Validation set ---")
        val_acc, val_f1m = trainer.evaluate(val_loader, target_names=target_names)
        mlflow.log_metric("final_val_acc", val_acc)
        mlflow.log_metric("final_val_macro_f1", val_f1m)

        print("\n--- Test set (độc lập) ---")
        test_acc, test_f1m = trainer.evaluate(test_loader, target_names=target_names)
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_macro_f1", test_f1m)
        mlflow.log_metric("best_val_macro_f1", trainer.best_val_f1)
        
        model.cpu()
        mlflow.pytorch.log_model(model, "best_model")
        
        misclassified_filename = f"misclassified_test_{time.strftime('%Y%m%d_%H%M')}.txt"
        model.to(device)
        trainer.export_misclassified_to_txt(test_loader, target_names, filepath=misclassified_filename)
        mlflow.log_artifact(misclassified_filename)
        
        print("\n HOÀN TẤT ĐÀO TẠO VÀ LƯU MODEL!")
        print("="*80)
        print("Để xem đồ thị Training và quản lý Model trực quan, hãy mở Terminal mới và gõ lệnh sau:")
        print("    mlflow ui --backend-store-uri sqlite:///mlruns.db")
        print("Sau đó truy cập: http://127.0.0.1:5000")
        print("="*80)

        del train_loader, val_loader, test_loader
        torch.cuda.empty_cache()