import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import time
from feature_selection import FeatureSelector
from config_manager import ConfigManager, TrainingHistoryManager, create_config_dict
# ==========================================
# 1. MULTI-HEAD ATTENTION
# ==========================================
class CustomFeatureAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.q_proj    = nn.Linear(embed_dim, embed_dim)
        self.k_proj    = nn.Linear(embed_dim, embed_dim)
        self.v_proj    = nn.Linear(embed_dim, embed_dim)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x):
        B, T, D = x.size()
        def project(proj, t):
            return proj(t).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = project(self.q_proj, x), project(self.k_proj, x), project(self.v_proj, x)
        scores   = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights  = self.attn_drop(torch.softmax(scores, dim=-1))
        out      = torch.matmul(weights, V)
        out      = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ==========================================
#    - Multi-layer CNN với BatchNorm & Residual
#    - Transformer-style Feed-Forward sau Attention
# ==========================================
class ResidualCNNBlock(nn.Module):
    """Conv1d + BN + ReLU + Residual connection."""
    def __init__(self, channels, kernel_size=3):
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
        return self.relu(x + self.block(x))   # residual


class CNN1D_Attention_Tabular(nn.Module):
    def __init__(self, num_numeric_features, embed_dim, num_heads,
                 num_classes, num_cnn_layers=2, mlp_dropout=0.4):
        super().__init__()

        # --- Tokenizer ---
        self.num_tokenizer   = nn.Linear(1, embed_dim)
        self.port_embedding  = nn.Embedding(num_embeddings=65536, embedding_dim=embed_dim)

        # Input projection 
        total_tokens = num_numeric_features + 2          # +2 cho src/dst port
        self.input_norm = nn.LayerNorm(embed_dim)

        # --- Multi-layer Residual CNN ---
        self.cnn_layers = nn.Sequential(
            *[ResidualCNNBlock(embed_dim) for _ in range(num_cnn_layers)]
        )

        # --- Attention + Feed-Forward  ---
        self.attention  = CustomFeatureAttention(embed_dim, num_heads, attn_dropout=0.1)
        self.attn_norm  = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(mlp_dropout / 2),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, numerical_x, src_port, dst_port):
        # Tokenize features
        num_tokens  = self.num_tokenizer(numerical_x.unsqueeze(-1))   # (B, F, D)
        src_token   = self.port_embedding(src_port).unsqueeze(1)       # (B, 1, D)
        dst_token   = self.port_embedding(dst_port).unsqueeze(1)       # (B, 1, D)
        x = torch.cat([src_token, dst_token, num_tokens], dim=1)       # (B, T, D)
        x = self.input_norm(x)

        # CNN
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)         # (B, T, D)

        # Attention block (pre-norm residual)
        x = x + self.attention(self.attn_norm(x))

        # FFN block (pre-norm residual)
        x = x + self.ffn(self.ffn_norm(x))

        # Global average pooling + classify
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


# ==========================================
# 3. DATA PROCESSOR (BỔ SUNG STRATIFIED SPLIT)
# ==========================================
class DataProcessor:
    def __init__(self, csv_path=None, numeric_cols=None,
                 port_cols=None, label_col='label'):
        self.csv_path     = csv_path
        self.numeric_cols = numeric_cols
        self.port_cols    = port_cols or ['src_port', 'dst_port']
        self.label_col    = label_col
        self.df           = None
        self.numerical_data = None
        self.port_data      = None
        self.labels         = None
        self.label_encoder  = {}
        self.class_weights  = None   

    def load_data(self, csv_path=None):
        path = csv_path or self.csv_path
        if path is None:
            raise ValueError("Cần cung cấp đường dẫn CSV")
        self.df = pd.read_csv(path)
        self.df.columns = self.df.columns.str.strip()
        print(f"✓ Loaded: {self.df.shape}")
        return self.df

    def filter_suspicious_rows(self, suspicious_col='label', suspicious_value='Suspicious'):
        if suspicious_col in self.df.columns:
            before = len(self.df)
            self.df = self.df[self.df[suspicious_col] != suspicious_value].reset_index(drop=True)
            print(f"✓ Filtered {before - len(self.df)} suspicious rows → {len(self.df)} remaining")

    def preprocess_features(self, drop_cols=None):
        if drop_cols:
            drop_cols = [c for c in drop_cols if c in self.df.columns]
            if drop_cols:
                self.df = self.df.drop(columns=drop_cols)
                print(f"✓ Dropped columns: {drop_cols}")

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        if self.numeric_cols is None:
            num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.numeric_cols = [c for c in num_cols
                                 if c not in self.port_cols + [self.label_col]]

        self.df = self.df.fillna(self.df[self.numeric_cols].mean())

        self.numerical_data = self.df[self.numeric_cols].values

        port_list = [self.df[c].values for c in self.port_cols if c in self.df.columns]
        self.port_data = np.array(port_list) if port_list else None

        if self.label_col in self.df.columns:
            self.labels = self.df[self.label_col].values
            self._encode_labels()
            self._compute_class_weights()

        print(f"✓ Numeric features: {len(self.numeric_cols)} | Classes: {len(self.label_encoder)}")
        return self.numerical_data, self.port_data, self.labels

    def _encode_labels(self):
        if self.labels is None:
            return
        if isinstance(self.labels[0], str):
            unique = np.unique(self.labels)
            self.label_encoder = {l: i for i, l in enumerate(unique)}
            self.labels = np.array([self.label_encoder[l] for l in self.labels])
            print("✓ Label encoding:")
            for k, v in self.label_encoder.items():
                print(f"   {k} → {v}")
        else:
            self.labels = self.labels.astype(np.int64)

    def _compute_class_weights(self):
        """Tính class weights để xử lý imbalanced dataset."""
        counts = np.bincount(self.labels)
        total  = counts.sum()
        n_cls  = len(counts)
        # Balanced weighting: w_c = total / (n_classes * count_c)
        weights = total / (n_cls * counts.astype(float))
        self.class_weights = torch.FloatTensor(weights)
        print("✓ Class weights (imbalance handling):")
        inv = {v: k for k, v in self.label_encoder.items()} if self.label_encoder else {}
        for i, w in enumerate(weights):
            label_name = inv.get(i, str(i))
            print(f"   Class {i} ({label_name}): {counts[i]:,} samples → weight {w:.4f}")

    def get_dataloader(self, batch_size=256, test_split=0.2, random_state=42):
        # Stratified split để giữ phân phối class
        idx = np.arange(len(self.numerical_data))
        train_idx, val_idx = train_test_split(
            idx, test_size=test_split,
            stratify=self.labels, random_state=random_state
        )

        # Fit scaler chỉ trên train
        scaler = StandardScaler()
        train_num = scaler.fit_transform(self.numerical_data[train_idx])
        val_num   = scaler.transform(self.numerical_data[val_idx])

        def get_ports(idx_arr):
            if self.port_data is not None:
                return self.port_data[0][idx_arr], self.port_data[1][idx_arr]
            z = np.zeros(len(idx_arr), dtype=np.int64)
            return z, z.copy()

        tr_src, tr_dst = get_ports(train_idx)
        va_src, va_dst = get_ports(val_idx)

        def make_loader(num, src, dst, lbl, shuffle):
            ds = TensorDataset(
                torch.FloatTensor(num),
                torch.LongTensor(src),
                torch.LongTensor(dst),
                torch.LongTensor(lbl),
            )
            return DataLoader(ds, batch_size=batch_size,
                              shuffle=shuffle, num_workers=4, pin_memory=True)

        train_loader = make_loader(train_num, tr_src, tr_dst, self.labels[train_idx], True)
        val_loader   = make_loader(val_num,   va_src, va_dst, self.labels[val_idx],   False)
        return train_loader, val_loader


# ==========================================
# 4. TRAINER (BỔ SUNG SCHEDULER + GRADIENT CLIPPING + FINAL REPORT + HISTORY LOGGING)
# ==========================================
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler=None, device='cpu', history_manager=None):
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.device     = device
        self.best_val_f1     = 0.0
        self.best_model_state = None
        self.history = {k: [] for k in
                        ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1']}
        self.history_manager = history_manager

    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss, preds, trues = 0.0, [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for num_x, src, dst, lbl in loader:
                num_x, src, dst, lbl = (
                    num_x.to(self.device), src.to(self.device),
                    dst.to(self.device),   lbl.to(self.device)
                )
                if train:
                    self.optimizer.zero_grad()

                logits = self.model(num_x, src, dst)
                loss   = self.criterion(logits, lbl)

                if train:
                    loss.backward()
                    #Gradient clipping chống exploding gradient
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(lbl.cpu().numpy())

        avg_loss = total_loss / len(loader)
        acc      = accuracy_score(trues, preds)
        f1       = f1_score(trues, preds, average='weighted', zero_division=0)
        return avg_loss, acc, f1, preds, trues

    def fit(self, train_loader, val_loader, epochs=50, patience=7):
        print(f"\n{'='*80}")
        print(f"{'BẮT ĐẦU HUẤN LUYỆN':^80}")
        print(f"{'='*80}\n")
        no_improve = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, tr_f1, _, _ = self._run_epoch(train_loader, train=True)
            va_loss, va_acc, va_f1, _, _ = self._run_epoch(val_loader,   train=False)

            # ✅ Scheduler step dựa trên val_loss
            if self.scheduler:
                self.scheduler.step(va_loss)

            for k, v in zip(
                ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1'],
                [tr_loss, tr_acc, va_loss, va_acc, va_f1]
            ):
                self.history[k].append(v)

            elapsed = time.time() - t0
            mark = " ✦" if va_f1 > self.best_val_f1 else ""
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
                f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} F1: {va_f1:.4f} | "
                f"{elapsed:.1f}s{mark}"
            )

            # ✅ Log epoch vào file nếu có history_manager
            if self.history_manager:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history_manager.log_epoch(
                    epoch=epoch,
                    train_loss=tr_loss,
                    train_acc=tr_acc,
                    train_f1=tr_f1,
                    val_loss=va_loss,
                    val_acc=va_acc,
                    val_f1=va_f1,
                    learning_rate=current_lr,
                    elapsed_sec=elapsed
                )

            # ✅ Early stopping theo val F1 (metric ý nghĩa hơn loss cho imbalanced)
            if va_f1 > self.best_val_f1:
                self.best_val_f1   = va_f1
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n⚠️  Early stopping tại epoch {epoch} (patience={patience})")
                    break

        print(f"\n✓ Nạp lại best model (val F1 = {self.best_val_f1:.4f})")
        self.model.load_state_dict(self.best_model_state)

    def evaluate(self, val_loader, label_encoder=None):
        """In classification report chi tiết sau khi training xong."""
        _, _, _, preds, trues = self._run_epoch(val_loader, train=False)

        # Map index → tên class nếu có
        target_names = None
        if label_encoder:
            inv_enc = {v: k for k, v in label_encoder.items()}
            target_names = [inv_enc[i] for i in range(len(inv_enc))]

        print(f"\n{'='*80}")
        print(f"{'KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG':^80}")
        print(f"{'='*80}")
        print(classification_report(trues, preds, target_names=target_names, zero_division=0))

        cm = confusion_matrix(trues, preds)
        print("Confusion Matrix:")
        print(cm)
        return preds, trues


# ==========================================
# 5. KHỞI TẠO VÀ CHẠY
# ==========================================
if __name__ == "__main__":
    # ── Hyperparameters ──────────────────────────────────────────────────────
    EMBED_DIM     = 64   
    NUM_HEADS     = 4
    NUM_CNN_LAYERS = 2    # 2 residual CNN blocks
    BATCH_SIZE    = 1024
    EPOCHS        = 50
    PATIENCE      = 7     
    LR            = 3e-4  
    WEIGHT_DECAY  = 1e-4  
    MLP_DROPOUT   = 0.35  

    COLS_TO_DROP  = ['activity']
    # ─────────────────────────────────────────────────────────────────────────

    # ✅ Tạo cấu hình
    config = create_config_dict(
        EMBED_DIM=EMBED_DIM,
        NUM_HEADS=NUM_HEADS,
        NUM_CNN_LAYERS=NUM_CNN_LAYERS,
        BATCH_SIZE=BATCH_SIZE,
        EPOCHS=EPOCHS,
        PATIENCE=PATIENCE,
        LR=LR,
        WEIGHT_DECAY=WEIGHT_DECAY,
        MLP_DROPOUT=MLP_DROPOUT,
        COLS_TO_DROP=COLS_TO_DROP,
        TEST_SPLIT=0.2,
        VARIANCE_THRESHOLD=0.01,
        CORR_THRESHOLD=0.95,
        TOP_K=80
    )

    # Lưu cấu hình
    config_manager = ConfigManager()
    config_path = config_manager.save_config(config, 
        config_name=f"ddos_model_{time.strftime('%Y%m%d_%H%M%S')}")

    # Khởi tạo logger lịch sử huấn luyện
    history_manager = TrainingHistoryManager()
    csv_history, json_history = history_manager.init_logger(
        experiment_name=f"ddos_experiment_{time.strftime('%Y%m%d_%H%M%S')}")

    processor = DataProcessor(csv_path="output.csv", label_col='label')
    processor.load_data()
    processor.filter_suspicious_rows(suspicious_col='label', suspicious_value='Suspicious')
    processor.preprocess_features(drop_cols=COLS_TO_DROP)
    fs = FeatureSelector(
    variance_threshold=0.01,
    corr_threshold=0.95,
    top_k=80
    )

    X_new, selected_cols = fs.fit_transform(
    processor.numerical_data,
    processor.labels,
    processor.numeric_cols
    )

# Update lại processor
    processor.numerical_data = X_new
    processor.numeric_cols = selected_cols   
    NUM_NUMERIC_FEATURES = processor.numerical_data.shape[1]
    NUM_CLASSES = len(processor.label_encoder) if processor.label_encoder else 2

    model = CNN1D_Attention_Tabular(
        num_numeric_features=NUM_NUMERIC_FEATURES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
        num_cnn_layers=NUM_CNN_LAYERS,
        mlp_dropout=MLP_DROPOUT,
    )

    train_loader, val_loader = processor.get_dataloader(
        batch_size=BATCH_SIZE, test_split=0.2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
    print(f"\n✓ Sử dụng device: {device}")
    model = model.to(device)

    # Weighted CrossEntropyLoss cho imbalanced data
    class_weights = processor.class_weights.to(device) if processor.class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ReduceLROnPlateau: giảm LR khi val_loss không cải thiện
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    # Truyền history_manager vào Trainer
    trainer = Trainer(
        model=model, criterion=criterion,
        optimizer=optimizer, scheduler=scheduler, 
        device=device, history_manager=history_manager
    )

    trainer.fit(train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE)

    # Đánh giá chi tiết sau training
    preds, trues = trainer.evaluate(val_loader, label_encoder=processor.label_encoder)

    # Tính toán metrics cuối cùng
    final_metrics = {
        'best_val_f1': float(trainer.best_val_f1),
        'final_train_loss': float(trainer.history['train_loss'][-1]) if trainer.history['train_loss'] else 0,
        'final_val_loss': float(trainer.history['val_loss'][-1]) if trainer.history['val_loss'] else 0,
        'final_val_acc': float(trainer.history['val_acc'][-1]) if trainer.history['val_acc'] else 0,
        'num_epochs_completed': len(trainer.history['train_loss']),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

    # Lưu tóm tắt lịch sử
    history_manager.save_summary(
        trainer_history=trainer.history,
        final_metrics=final_metrics,
        config=config
    )

    # Lưu model với metadata
    model_checkpoint = {
        'model_state_dict': model.state_dict(),
        'label_encoder':    processor.label_encoder,
        'numeric_cols':     processor.numeric_cols,
        'embed_dim':        EMBED_DIM,
        'num_heads':        NUM_HEADS,
        'num_classes':      NUM_CLASSES,
        'config': config,
        'training_history': trainer.history,
        'final_metrics': final_metrics,
        'selected_features': selected_cols
    }
    torch.save(model_checkpoint, 'model/best_model.pth')
    print("\n✓ Đã lưu model tại model/best_model.pth")
    print(f"✓ Đã lưu cấu hình tại {config_path}")
    print(f"✓ Đã lưu lịch sử tại {csv_history} và {json_history}")
    print(f" Tóm tắt huấn luyện:")
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.6f}" if value < 1 else f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")