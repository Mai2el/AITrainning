"""
count_multilabel_activities.py
------------------------------
Đếm activities trong file dataset_ready_multilabel.csv
"""

import pandas as pd
import os
import json
from datetime import datetime

def count_multilabel_activities(file_path='dataset_ready_multilabel.csv', label_col='activity'):
    """
    Đếm activities trong file multilabel
    """
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file {file_path}")
        return None
    
    try:
        print(f"📥 Đang đọc file: {file_path}...")
        df = pd.read_csv(file_path, low_memory=False)
        
        print(f"✅ Đã tải {len(df):,} records")
        print(f"📋 Tổng cột: {len(df.columns)}")
        print(f"\nCác cột có sẵn:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        # Chuẩn hóa tên cột
        df.columns = df.columns.str.strip()
        
        if label_col not in df.columns:
            print(f"\n⚠️  Không tìm thấy cột '{label_col}'")
            # Tìm cột tương tự
            similar = [col for col in df.columns if 'activity' in col.lower() or 'label' in col.lower()]
            if similar:
                print(f"   Gợi ý: {similar}")
                label_col = similar[0]
                print(f"   Sử dụng cột: {label_col}")
            else:
                return None
        
        # Chuẩn hóa labels
        df[label_col] = df[label_col].astype(str).str.strip()
        
        # Đếm activities
        print(f"\n📊 Đang phân tích cột '{label_col}'...")
        activity_counts = df[label_col].value_counts()
        total_records = len(df)
        
        print(f"\n{'='*70}")
        print(f"📈 KẾT QUẢ ĐẾM ACTIVITIES")
        print(f"{'='*70}")
        print(f"\n✅ Tổng records: {total_records:,}")
        print(f"📌 Tổng loại activity: {len(activity_counts)}")
        print(f"\n{'Activity':<45} | {'Số lượng':>12} | {'Phần trăm':>10}")
        print("-" * 72)
        
        for activity, count in activity_counts.items():
            percentage = (count / total_records) * 100
            print(f"{activity:<45} | {count:>12,} | {percentage:>9.2f}%")
        
        # Tạo stats dict
        stats = {
            'file': os.path.basename(file_path),
            'timestamp': datetime.now().isoformat(),
            'total_records': total_records,
            'unique_activities': len(activity_counts),
            'activities': activity_counts.to_dict(),
            'percentages': {activity: round(count / total_records * 100, 2) 
                          for activity, count in activity_counts.items()}
        }
        
        return stats
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_report(stats, output_file='multilabel_activity_report.json'):
    """
    Lưu báo cáo thống kê
    """
    if not stats:
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Báo cáo đã được lưu: {output_file}")

def main():
    """
    Main function
    """
    file_path = 'dataset_ready_multilabel.csv'
    
    # Đếm activities
    stats = count_multilabel_activities(file_path)
    
    if stats:
        # Lưu báo cáo
        save_report(stats)
        print(f"\n✅ Hoàn thành!")
    else:
        print(f"\n❌ Không thể xử lý file")

if __name__ == "__main__":
    main()
