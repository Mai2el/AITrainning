"""
start.py
--------
Huấn luyện mô hình DDoS Detection với Tabular Embedding, Standard MHA (chuẩn bài báo) và MLflow.
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
    """Đã sửa lỗi: Cập nhật thêm việc đồng bộ Buffers (cho BatchNorm)."""
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
    """Giải quyết mất cân bằng dữ liệu"""
    def __init__(self, weight=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight 
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        ce_loss_raw = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss_raw)
        ce_loss = self.ce(inputs, targets)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.weight is not None:
            weights = self.weight[targets]
            focal_loss = focal_loss * weights
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


# ==========================================
# 1. TABULAR EMBEDDING & STANDARD MULTI-HEAD ATTENTION
# ==========================================
class TabularFeatureEmbedding(nn.Module):
    """
    Chuẩn hóa theo cấu trúc Additive Value & Feature Embedding:
    Tách biệt việc nhúng giá trị (bin) và nhúng định danh cột.
    """
    def __init__(self, num_features, num_bins, embed_dim):
        super().__init__()
        self.num_features = num_features      
        self.value_embedding = nn.Embedding(num_bins, embed_dim)
        
        self.feature_embedding = nn.Embedding(num_features, embed_dim)
        
        nn.init.trunc_normal_(self.value_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.feature_embedding.weight, std=0.02)

    def forward(self, x_categorical):
        val_emb = self.value_embedding(x_categorical)
        feat_idx = torch.arange(self.num_features, device=x_categorical.device)
        feat_emb = self.feature_embedding(feat_idx).unsqueeze(0)
        return val_emb + feat_emb


class StandardMultiHeadAttention(nn.Module):
    """Attention."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim phải chia hết cho num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)  
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x): 
        B, T, D = x.size()
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)    
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = self.attn_drop(torch.softmax(scores, dim=-1)) 
        Z_heads = torch.matmul(attn_weights, V)
        Z_concat = Z_heads.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(Z_concat)


# ==========================================
# 2. KIẾN TRÚC MÔ HÌNH CHÍNH
# ==========================================
class ResidualCNNBlock(nn.Module):
    def __init__(self, channels, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class CNN1D_Attention_Tabular(nn.Module):
    def __init__(self, num_numeric_features, num_bins, embed_dim, num_heads,
                 num_classes, num_cnn_layers=2, mlp_dropout=0.4):
        super().__init__()    
        self.feature_embedder = TabularFeatureEmbedding(num_numeric_features, num_bins, embed_dim)  
        self.input_norm = nn.LayerNorm(embed_dim)
        self.cnn_layers = nn.Sequential(*[ResidualCNNBlock(embed_dim) for _ in range(num_cnn_layers)])
        self.attention = StandardMultiHeadAttention(embed_dim, num_heads, dropout=0.1)
        self.attn_norm = nn.LayerNorm(embed_dim)  
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
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
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        mean_pooled = x.mean(dim=1)
        max_pooled, _ = x.max(dim=1)
        pooled = torch.cat([mean_pooled, max_pooled], dim=1)
        return self.classifier(pooled)


# ==========================================
# 3. TRAINER & MLFLOW LOGGING
# ==========================================
class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler=None,
        device="cpu",
        scheduler_per_batch=False,
        ema=None,
        grad_clip=1.0 
    ):
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
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average='weighted', zero_division=0)
        return avg_loss, acc, f1, preds, trues

    def fit(self, train_loader, val_loader, epochs=50, patience=7):
        print(f"\n{'='*80}\n{'BẮT ĐẦU HUẤN LUYỆN':^80}\n{'='*80}\n")
        no_improve = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, tr_f1, _, _ = self._run_epoch(train_loader, train=True)
            backup = None
            backup_buf = None
            if self.ema is not None:
                backup = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
                backup_buf = {n: b.detach().clone() for n, b in self.model.named_buffers()}
                self.ema.copy_to_model(self.model)
            va_loss, va_acc, va_f1, _, _ = self._run_epoch(val_loader, train=False)
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
                "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
                "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1,
                "learning_rate": current_lr
            }, step=epoch)

            elapsed = time.time() - t0
            mark = " ✦" if va_f1 > self.best_val_f1 else ""
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} Val F1: {va_f1:.4f} | {elapsed:.1f}s{mark}")

            if va_f1 > self.best_val_f1:
                self.best_val_f1 = va_f1
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                if self.ema is not None:
                    self.best_ema_state = self.ema.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n⚠️ Early stopping tại epoch {epoch} (patience={patience})")
                    break

        print(f"\n✓ Nạp lại best checkpoint (val F1 = {self.best_val_f1:.4f})")
        if self.ema is not None and self.best_ema_state is not None:
            self.ema.load_state_dict(self.best_ema_state)
            self.ema.copy_to_model(self.model)
            print("   -> Đã áp dụng trọng số EMA tương ứng best val F1.")
        elif self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def evaluate(self, val_loader, target_names=None):
        print("\nĐang đánh giá mô hình cuối cùng...")
        _, acc, f1, preds, trues = self._run_epoch(val_loader, train=False)
        print("\nClassification Report:")
        print(classification_report(trues, preds, target_names=target_names, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(trues, preds))
        print(f"\n   Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}")
        return acc, f1
        
    def export_misclassified_to_txt(self, loader, target_names, filepath="misclassified_samples.txt"):
        """
        Quét qua dataloader, tìm các mẫu đoán sai và ghi ra file txt.
        """
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
    NUM_BINS = 256  

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
    # BƯỚC 3: CẤU HÌNH MLFLOW VÀ MODEL
    # ---------------------------------------------------------
    config = {
        "EMBED_DIM": 288,
        "NUM_HEADS": 8,
        "NUM_CNN_LAYERS": 1,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": 30, 
        "PATIENCE": 3,
        "MAX_LR": 1e-3,           
        "WEIGHT_DECAY": 1e-2,
        "MLP_DROPOUT": 0.3,
        "NUM_BINS": NUM_BINS,
        "LABEL_SMOOTHING": 0.05,
        "EMA_DECAY": 0.9992,
        "FOCAL_GAMMA": 1.0,
        "GRAD_CLIP": 1.0 
    }

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("DDoS_StandardMHA_TabularEmbedding")
    
    run_name = f"Run_{time.strftime('%Y%m%d_%H%M')}"
    print(f"\n2. Khởi tạo phiên làm việc MLflow: {run_name}")
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   -> Sử dụng thiết bị: {device}")
        
        model = CNN1D_Attention_Tabular(
            num_numeric_features=NUM_NUMERIC_FEATURES, 
            num_bins=config["NUM_BINS"], 
            embed_dim=config["EMBED_DIM"], 
            num_heads=config["NUM_HEADS"], 
            num_classes=NUM_CLASSES, 
            num_cnn_layers=config["NUM_CNN_LAYERS"], 
            mlp_dropout=config["MLP_DROPOUT"]
        ).to(device)

        class_w = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_w = np.power(class_w, 0.5)
        class_weights = torch.FloatTensor(class_w).to(device)
        
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
            pct_start=0.12,
            anneal_strategy="cos",
            div_factor=20.0,
            final_div_factor=250.0,
        )

        ema = EMA(model, decay=config["EMA_DECAY"])

        trainer = Trainer(
            model,
            criterion,
            optimizer,
            scheduler,
            device,
            scheduler_per_batch=True,
            ema=ema,
            grad_clip=config["GRAD_CLIP"]
        )
        trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"], patience=config["PATIENCE"])

        print("\n--- Validation set ---")
        val_acc, _ = trainer.evaluate(val_loader, target_names=target_names)
        mlflow.log_metric("final_val_acc", val_acc)

        print("\n--- Test set (độc lập) ---")
        test_acc, test_f1 = trainer.evaluate(test_loader, target_names=target_names)
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_weighted_f1", test_f1)
        mlflow.log_metric("best_val_f1", trainer.best_val_f1)
        
        model.cpu()
        mlflow.pytorch.log_model(model, "best_model")
        
        # ---------------------------------------------------------
        # XUẤT CÁC MẪU SAI RA FILE TXT
        # ---------------------------------------------------------
        misclassified_filename = f"misclassified_test_{time.strftime('%Y%m%d_%H%M')}.txt"
        
        # Trả model về lại GPU để hàm predict chạy nhanh và không bị lỗi device mismatch
        model.to(device)
        trainer.export_misclassified_to_txt(test_loader, target_names, filepath=misclassified_filename)
        
        # Log luôn file này vào mlflow để dễ quản lý
        mlflow.log_artifact(misclassified_filename)
        
        print("\n HOÀN TẤT ĐÀO TẠO VÀ LƯU MODEL!")
        print("="*80)
        print("Để xem đồ thị Training và quản lý Model trực quan, hãy mở Terminal mới và gõ lệnh sau:")
        print("    mlflow ui --backend-store-uri sqlite:///mlruns.db")
        print("Sau đó truy cập: http://127.0.0.1:5000")
        print("="*80)

        # 2. Xóa DataLoaders và giải phóng bộ nhớ để thoát script an toàn
        del train_loader
        del val_loader
        del test_loader
        torch.cuda.empty_cache()