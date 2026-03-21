"""
config_manager.py
─────────────────
Module quản lý cấu hình và lịch sử huấn luyện.
- Lưu config hyperparameters vào JSON
- Lưu lịch sử huấn luyện vào CSV
- Đọc & khôi phục cấu hình từ file
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path


class ConfigManager:
    """Quản lý lưu/đọc cấu hình model."""
    
    def __init__(self, config_dir="configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def save_config(self, config_dict, config_name=None):
        """
        Lưu cấu hình vào file JSON.
        
        Args:
            config_dict: dict chứa hyperparameters
            config_name: tên file (mặc định: timestamp)
        
        Returns:
            Đường dẫn file đã lưu
        """
        if config_name is None:
            config_name = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config_path = self.config_dir / f"{config_name}.json"
        
        # Thêm thông tin metadata
        config_with_meta = {
            "timestamp": datetime.now().isoformat(),
            "config_name": config_name,
            **config_dict
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_with_meta, f, indent=4)
        
        print(f"✓ Đã lưu cấu hình tại: {config_path}")
        return str(config_path)
    
    def load_config(self, config_path):
        """
        Đọc cấu hình từ file JSON.
        
        Args:
            config_path: đường dẫn file config
        
        Returns:
            dict cấu hình
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✓ Đã tải cấu hình từ: {config_path}")
        return config
    
    def get_latest_config(self):
        """Lấy cấu hình mới nhất đã lưu."""
        configs = list(self.config_dir.glob("config_*.json"))
        if not configs:
            raise FileNotFoundError("Không tìm thấy cấu hình nào")
        
        latest = max(configs, key=os.path.getctime)
        return self.load_config(latest)
    
    def list_configs(self):
        """Liệt kê tất cả cấu hình đã lưu."""
        configs = list(self.config_dir.glob("config_*.json"))
        configs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print("\n📋 Danh sách cấu hình:")
        for i, cfg in enumerate(configs, 1):
            print(f"  {i}. {cfg.name}")
        
        return [str(c) for c in configs]


class TrainingHistoryManager:
    """Quản lý lưu lịch sử huấn luyện."""
    
    def __init__(self, history_dir="training_history"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        self.csv_path = None
        self.json_path = None
        
    def init_logger(self, experiment_name=None):
        """
        Khởi tạo logger cho một lần huấn luyện.
        
        Args:
            experiment_name: tên thí nghiệm (mặc định: timestamp)
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.csv_path = self.history_dir / f"{experiment_name}_history.csv"
        self.json_path = self.history_dir / f"{experiment_name}_history.json"
        
        # Khởi tạo file CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_acc', 'train_f1',
                'val_loss', 'val_acc', 'val_f1', 'lr', 'elapsed_sec'
            ])
        
        print(f"✓ Khởi tạo logger lịch sử: {experiment_name}")
        return str(self.csv_path), str(self.json_path)
    
    def log_epoch(self, epoch, train_loss, train_acc, train_f1,
                  val_loss, val_acc, val_f1, learning_rate, elapsed_sec):
        """
        Ghi lại một epoch vào lịch sử.
        
        Args:
            epoch: số epoch (bắt đầu từ 1)
            train_loss, train_acc, train_f1: metrics training
            val_loss, val_acc, val_f1: metrics validation
            learning_rate: learning rate hiện tại
            elapsed_sec: thời gian thực hiện epoch (giây)
        """
        if self.csv_path is None:
            raise RuntimeError("Chưa khởi tạo logger. Gọi init_logger() trước.")
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{train_f1:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}",
                f"{learning_rate:.8f}", f"{elapsed_sec:.2f}"
            ])
    
    def save_summary(self, trainer_history, final_metrics, config=None):
        """
        Lưu tóm tắt lịch sử và kết quả cuối cùng.
        
        Args:
            trainer_history: dict chứa lịch sử từ Trainer
            final_metrics: dict chứa metrics cuối cùng
            config: dict chứa cấu hình (tuỳ chọn)
        """
        if self.json_path is None:
            raise RuntimeError("Chưa khởi tạo logger. Gọi init_logger() trước.")
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "training_history": trainer_history,
            "final_metrics": final_metrics,
            "config": config or {}
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"✓ Đã lưu tóm tắt lịch sử: {self.json_path}")
    
    def load_history(self, history_path):
        """Đọc lịch sử từ file JSON."""
        with open(history_path, 'r') as f:
            history = json.load(f)
        return history
    
    def get_latest_history(self):
        """Lấy lịch sử huấn luyện mới nhất."""
        histories = list(self.history_dir.glob("*_history.json"))
        if not histories:
            raise FileNotFoundError("Không tìm thấy lịch sử huấn luyện nào")
        
        latest = max(histories, key=os.path.getctime)
        return self.load_history(latest)


def create_config_dict(**kwargs):
    """
    Helper function để tạo config dict từ hyperparameters.
    
    Returns:
        dict cấu hình
    """
    return kwargs


if __name__ == "__main__":
    # Example usage
    print("Testing ConfigManager...")
    cm = ConfigManager()
    
    test_config = create_config_dict(
        EMBED_DIM=64,
        NUM_HEADS=4,
        NUM_CNN_LAYERS=2,
        BATCH_SIZE=1024,
        EPOCHS=50,
        PATIENCE=7,
        LR=3e-4
    )
    
    # Lưu config
    cm.save_config(test_config, "ddos_model_v1")
    
    # Liệt kê configs
    cm.list_configs()
    
    # Tải config
    loaded = cm.load_config("configs/ddos_model_v1.json")
    print("\nLoaded config:", loaded)
    
    print("\n" + "="*60)
    print("Testing TrainingHistoryManager...")
    
    thm = TrainingHistoryManager()
    csv_file, json_file = thm.init_logger("test_experiment")
    
    # Mô phỏng logging
    for epoch in range(1, 4):
        thm.log_epoch(
            epoch=epoch,
            train_loss=0.5 - epoch*0.05,
            train_acc=0.6 + epoch*0.1,
            train_f1=0.65 + epoch*0.1,
            val_loss=0.4 - epoch*0.04,
            val_acc=0.65 + epoch*0.08,
            val_f1=0.70 + epoch*0.08,
            learning_rate=3e-4,
            elapsed_sec=45.5
        )
    
    # Lưu tóm tắt
    thm.save_summary(
        {"train_loss": [0.5, 0.45], "val_acc": [0.65, 0.73]},
        {"best_val_f1": 0.78, "total_epochs": 3},
        test_config
    )
    
    print("✓ Test hoàn tất!")
