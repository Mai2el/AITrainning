"""
analyze_training.py
──────────────────
Tiện ích phân tích lịch sử huấn luyện.
- Tải lịch sử từ file
- Vẽ biểu đồ
- In báo cáo tóm tắt
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config_manager import TrainingHistoryManager, ConfigManager
import argparse


def load_latest_history():
    """Tải lịch sử huấn luyện mới nhất."""
    thm = TrainingHistoryManager()
    history = thm.get_latest_history()
    return history


def load_history_by_name(experiment_name):
    """Tải lịch sử theo tên thí nghiệm."""
    history_path = Path(f"training_history/{experiment_name}_history.json")
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history


def plot_training_curves(history, save_path="training_curves.png"):
    """Vẽ biểu đồ training curves."""
    train_hist = history['training_history']
    
    # Chuẩn bị dữ liệu
    epochs = list(range(1, len(train_hist['train_loss']) + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, train_hist['train_loss'], marker='o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, train_hist['val_loss'], marker='s', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, train_hist['train_acc'], marker='o', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, train_hist['val_acc'], marker='s', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(epochs, train_hist['val_f1'], marker='o', color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    max_f1_idx = train_hist['val_f1'].index(max(train_hist['val_f1']))
    axes[1, 0].axvline(x=max_f1_idx+1, color='red', linestyle='--', alpha=0.5, label=f'Best (epoch {max_f1_idx+1})')
    axes[1, 0].legend()
    
    # Metrics summary
    axes[1, 1].axis('off')
    metrics = history['final_metrics']
    summary_text = "📊 FINAL METRICS\n" + "="*30 + "\n"
    for key, val in metrics.items():
        if isinstance(val, float):
            summary_text += f"• {key}:\n  {val:.6f}\n" if val < 1 else f"• {key}:\n  {val:.2f}\n"
        else:
            summary_text += f"• {key}:\n  {val}\n"
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ tại: {save_path}")
    return fig


def plot_csv_history(csv_path, save_path="training_curves.png"):
    """Vẽ biểu đồ từ file CSV."""
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], marker='o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], marker='s', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(df['epoch'], df['train_acc'], marker='o', label='Train Acc', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val_acc'], marker='s', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    best_f1_epoch = df.loc[df['val_f1'].idxmax(), 'epoch']
    axes[1, 0].plot(df['epoch'], df['val_f1'], marker='o', color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=best_f1_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best (epoch {int(best_f1_epoch)})')
    axes[1, 0].legend()
    
    # Summary
    axes[1, 1].axis('off')
    summary_text = f"""📊 TRAINING SUMMARY
{'='*30}
Total Epochs: {len(df)}
Best Epoch: {int(best_f1_epoch)}
Best Val F1: {df['val_f1'].max():.6f}
Final Val Acc: {df['val_acc'].iloc[-1]:.6f}
Final Val Loss: {df['val_loss'].iloc[-1]:.6f}
Best LR: {df['lr'].min():.8f}
Avg Epoch Time: {df['elapsed_sec'].mean():.2f}s
    """
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ tại: {save_path}")
    return fig


def print_summary(history):
    """In tóm tắt lịch sử."""
    print("\n" + "="*80)
    print(f"{'📊 TRAINING SUMMARY':^80}")
    print("="*80)
    
    metrics = history['final_metrics']
    config = history.get('config', {})
    
    print("\n🎯 FINAL METRICS:")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  • {key:.<40} {val:.6f}" if val < 1 else f"  • {key:.<40} {val:,.0f}")
        else:
            print(f"  • {key:.<40} {val:,.0f}" if isinstance(val, int) else f"  • {key:.<40} {val}")
    
    if config:
        print("\n⚙️  CONFIGURATION USED:")
        for key, val in sorted(config.items()):
            if key not in ['timestamp', 'config_name']:
                print(f"  • {key:.<40} {val}")
    
    train_hist = history['training_history']
    print("\n📈 TRAINING STATISTICS:")
    if 'val_f1' in train_hist and train_hist['val_f1']:
        print(f"  • Best Val F1: {max(train_hist['val_f1']):.6f} (Epoch {train_hist['val_f1'].index(max(train_hist['val_f1']))+1})")
    if 'val_acc' in train_hist and train_hist['val_acc']:
        print(f"  • Best Val Acc: {max(train_hist['val_acc']):.6f}")
    if 'train_loss' in train_hist and train_hist['train_loss']:
        print(f"  • Min Train Loss: {min(train_hist['train_loss']):.6f}")
    if 'val_loss' in train_hist and train_hist['val_loss']:
        print(f"  • Min Val Loss: {min(train_hist['val_loss']):.6f}")
    
    print("\n" + "="*80 + "\n")


def compare_experiments(experiment_names):
    """So sánh nhiều thí nghiệm."""
    histories = {}
    for name in experiment_names:
        try:
            histories[name] = load_history_by_name(name)
        except FileNotFoundError:
            print(f"⚠️  Không tìm thấy {name}")
            continue
    
    if not histories:
        print("❌ Không tìm thấy thí nghiệm nào")
        return
    
    print("\n" + "="*80)
    print(f"{'🔬 EXPERIMENT COMPARISON':^80}")
    print("="*80)
    
    # Tìm metrics tốt nhất
    results = []
    for name, history in histories.items():
        metrics = history['final_metrics']
        results.append({
            'experiment': name,
            'best_val_f1': metrics.get('best_val_f1', 0),
            'final_val_acc': metrics.get('final_val_acc', 0),
            'num_epochs': metrics.get('num_epochs_completed', 0),
            'total_params': metrics.get('total_parameters', 0)
        })
    
    df_compare = pd.DataFrame(results)
    print("\n")
    print(df_compare.to_string(index=False))
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phân tích lịch sử huấn luyện DDoS Model")
    parser.add_argument('--latest', action='store_true', help='Phân tích thí nghiệm mới nhất')
    parser.add_argument('--name', type=str, help='Tên thí nghiệm (ví dụ: ddos_experiment_20240315_120000)')
    parser.add_argument('--csv', type=str, help='Đường dẫn file CSV')
    parser.add_argument('--compare', nargs='+', help='So sánh nhiều thí nghiệm')
    parser.add_argument('--output', type=str, default='training_curves.png', help='Lưu biểu đồ tại đây')
    parser.add_argument('--no-plot', action='store_true', help='Không vẽ biểu đồ, chỉ in tóm tắt')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_experiments(args.compare)
    elif args.name:
        history = load_history_by_name(args.name)
        print_summary(history)
        if not args.no_plot:
            plot_training_curves(history, save_path=args.output)
    elif args.csv:
        print_summary_from_csv(args.csv)
        if not args.no_plot:
            plot_csv_history(args.csv, save_path=args.output)
    elif args.latest:
        history = load_latest_history()
        print_summary(history)
        if not args.no_plot:
            plot_training_curves(history, save_path=args.output)
    else:
        # Mặc định: tải mới nhất nếu có, nếu không liệt kê các tùy chọn
        try:
            history = load_latest_history()
            print("✓ Tải lịch sử mới nhất...")
            print_summary(history)
            if not args.no_plot:
                plot_training_curves(history, save_path=args.output)
        except FileNotFoundError:
            print("ℹ️  Chưa có lịch sử huấn luyện nào. Sử dụng:")
            print("   python analyze_training.py --latest    # Thí nghiệm mới nhất")
            print("   python analyze_training.py --name <name>  # Thí nghiệm cụ thể")
            print("   python analyze_training.py --help      # Xem tất cả tùy chọn")
