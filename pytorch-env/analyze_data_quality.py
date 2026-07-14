"""
analyze_data_quality.py
-----------------------
Thống kê các cột có giá trị inf (infinite) và missing (NaN/None)
Phân tích chất lượng dữ liệu trước khi huấn luyện mô hình
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List


def analyze_inf_missing_values(filepath: str, verbose: bool = True) -> Dict:
    """
    Phân tích các cột có giá trị inf và missing trong file CSV.
    
    Args:
        filepath (str): Đường dẫn tới file CSV
        verbose (bool): In chi tiết kết quả phân tích
        
    Returns:
        Dict: Kết quả phân tích gồm:
            - inf_columns: Dict chứa cột có giá trị inf và số lượng
            - missing_columns: Dict chứa cột có giá trị missing và số lượng
            - summary: Tóm tắt phân tích
    """
    
    print(f"\n{'='*80}")
    print(f"📊 PHÂN TÍCH CHẤT LƯỢNG DỮ LIỆU: {os.path.basename(filepath)}")
    print(f"{'='*80}\n")
    
    # Đọc file CSV
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Đã tải file: {filepath}")
        print(f"   Kích thước: {df.shape[0]} dòng × {df.shape[1]} cột\n")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")
        return {}
    
    results = {
        'inf_columns': {},
        'missing_columns': {},
        'summary': {}
    }
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 1. PHÂN TÍCH CÁC CỘT CÓ GIÁ TRỊ INF (INFINITE)
    # ═══════════════════════════════════════════════════════════════════════════════
    print("1️⃣ PHÂN TÍCH GIÁ TRỊ INFINITE (INF):")
    print("-" * 80)
    
    inf_found = False
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            inf_count = np.isinf(df[col]).sum()
            pos_inf = (df[col] == np.inf).sum()
            neg_inf = (df[col] == -np.inf).sum()
            
            if inf_count > 0:
                inf_found = True
                results['inf_columns'][col] = {
                    'total_inf': int(inf_count),
                    'positive_inf': int(pos_inf),
                    'negative_inf': int(neg_inf),
                    'percentage': round(inf_count / len(df) * 100, 2)
                }
                
                if verbose:
                    print(f"   ⚠️  {col}:")
                    print(f"       • Tổng cộng inf: {inf_count} ({results['inf_columns'][col]['percentage']}%)")
                    print(f"       • Positive inf (+∞): {pos_inf}")
                    print(f"       • Negative inf (-∞): {neg_inf}\n")
    
    if not inf_found:
        print("   ✅ Không tìm thấy giá trị inf trong bất kỳ cột nào!\n")
    else:
        print(f"   🔴 Tìm thấy {len(results['inf_columns'])} cột có giá trị inf\n")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 2. PHÂN TÍCH CÁC CỘT CÓ GIÁ TRỊ MISSING (NaN/None)
    # ═══════════════════════════════════════════════════════════════════════════════
    print("2️⃣ PHÂN TÍCH GIÁ TRỊ MISSING (NaN/None):")
    print("-" * 80)
    
    missing_found = False
    for col in df.columns:
        missing_count = df[col].isna().sum()
        
        if missing_count > 0:
            missing_found = True
            results['missing_columns'][col] = {
                'total_missing': int(missing_count),
                'percentage': round(missing_count / len(df) * 100, 2)
            }
            
            if verbose:
                print(f"   ⚠️  {col}:")
                print(f"       • Tổng missing: {missing_count} ({results['missing_columns'][col]['percentage']}%)")
                print(f"       • Non-missing: {len(df) - missing_count}\n")
    
    if not missing_found:
        print("   ✅ Không tìm thấy giá trị missing trong bất kỳ cột nào!\n")
    else:
        print(f"   🔴 Tìm thấy {len(results['missing_columns'])} cột có giá trị missing\n")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 3. PHÂN TÍCH COMBINE (INF + MISSING)
    # ═══════════════════════════════════════════════════════════════════════════════
    print("3️⃣ PHÂN TÍCH KẾT HỢP (INF + MISSING):")
    print("-" * 80)
    
    problem_columns = []
    for col in df.columns:
        inf_count = np.isinf(df[col]).sum() if df[col].dtype in ['float64', 'float32', 'int64', 'int32'] else 0
        missing_count = df[col].isna().sum()
        total_problems = inf_count + missing_count
        
        if total_problems > 0:
            problem_columns.append({
                'column': col,
                'inf': int(inf_count),
                'missing': int(missing_count),
                'total': int(total_problems),
                'percentage': round(total_problems / len(df) * 100, 2)
            })
    
    if not problem_columns:
        print("   ✅ Dữ liệu sạch! Không tìm thấy vấn đề về inf hoặc missing.\n")
    else:
        problem_columns.sort(key=lambda x: x['total'], reverse=True)
        print(f"   Tìm thấy {len(problem_columns)} cột có vấn đề:\n")
        
        for item in problem_columns:
            print(f"   {item['column']}:")
            print(f"      • Inf: {item['inf']}, Missing: {item['missing']}, "
                  f"Tổng cộng: {item['total']} ({item['percentage']}%)")
        print()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 4. TÓM TẮT CHUNG
    # ═══════════════════════════════════════════════════════════════════════════════
    print("4️⃣ TÓM TẮT CHUNG:")
    print("-" * 80)
    
    total_inf = sum(v['total_inf'] for v in results['inf_columns'].values())
    total_missing = sum(v['total_missing'] for v in results['missing_columns'].values())
    total_cells = df.shape[0] * df.shape[1]
    total_problems = total_inf + total_missing
    
    results['summary'] = {
        'file': filepath,
        'rows': df.shape[0],
        'columns': df.shape[1],
        'total_cells': total_cells,
        'columns_with_inf': len(results['inf_columns']),
        'total_inf_values': int(total_inf),
        'columns_with_missing': len(results['missing_columns']),
        'total_missing_values': int(total_missing),
        'total_problems': int(total_problems),
        'problem_percentage': round(total_problems / total_cells * 100, 2),
        'data_quality': '✅ Tốt' if total_problems == 0 else '⚠️  Cần xử lý' if total_problems < total_cells * 0.01 else '🔴 Kém'
    }
    
    print(f"   📈 Tổng dòng: {results['summary']['rows']}")
    print(f"   📊 Tổng cột: {results['summary']['columns']}")
    print(f"   🧮 Tổng ô: {results['summary']['total_cells']}")
    print(f"   🔴 Cột có inf: {results['summary']['columns_with_inf']}")
    print(f"   🔴 Tổng giá trị inf: {results['summary']['total_inf_values']}")
    print(f"   🔴 Cột có missing: {results['summary']['columns_with_missing']}")
    print(f"   🔴 Tổng giá trị missing: {results['summary']['total_missing_values']}")
    print(f"   🔴 Tổng vấn đề: {results['summary']['total_problems']} "
          f"({results['summary']['problem_percentage']}%)")
    print(f"   💾 Chất lượng dữ liệu: {results['summary']['data_quality']}")
    print(f"\n{'='*80}\n")
    
    return results


def create_quality_report(results: Dict, output_file: str = 'data_quality_report.txt') -> None:
    """
    Tạo báo cáo chất lượng dữ liệu và lưu vào file.
    
    Args:
        results (Dict): Kết quả từ analyze_inf_missing_values()
        output_file (str): Tên file xuất ra
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BÁO CÁO CHẤT LƯỢNG DỮ LIỆU\n")
        f.write("=" * 80 + "\n\n")
        
        summary = results.get('summary', {})
        f.write(f"File: {summary.get('file', 'N/A')}\n")
        f.write(f"Dòng: {summary.get('rows', 0)}, Cột: {summary.get('columns', 0)}\n")
        f.write(f"Tổng ô: {summary.get('total_cells', 0)}\n\n")
        
        f.write("CHẤT LƯỢNG: " + summary.get('data_quality', 'N/A') + "\n\n")
        
        f.write("CÁC CỘT CÓ GIÁ TRỊ INF:\n")
        f.write("-" * 80 + "\n")
        if results['inf_columns']:
            for col, stats in results['inf_columns'].items():
                f.write(f"{col}: {stats['total_inf']} ({stats['percentage']}%)\n")
                f.write(f"  - Positive inf: {stats['positive_inf']}\n")
                f.write(f"  - Negative inf: {stats['negative_inf']}\n")
        else:
            f.write("Không có cột với giá trị inf.\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("CÁC CỘT CÓ GIÁ TRỊ MISSING:\n")
        f.write("-" * 80 + "\n")
        if results['missing_columns']:
            for col, stats in results['missing_columns'].items():
                f.write(f"{col}: {stats['total_missing']} ({stats['percentage']}%)\n")
        else:
            f.write("Không có cột với giá trị missing.\n")
    
    print(f"✅ Báo cáo đã lưu vào: {output_file}\n")


def display_inf_rows_with_labels(filepath: str, label_column: str = None, 
                                 max_rows: int = 50) -> pd.DataFrame:
    """
    Hiển thị các hàng có giá trị inf cùng với nhãn của chúng.
    (Sử dụng vectorized operations cho hiệu suất cao trên dữ liệu lớn)
    
    Args:
        filepath (str): Đường dẫn tới file CSV
        label_column (str): Tên cột chứa nhãn. Nếu None, sẽ tìm cột cuối cùng hoặc cột có tên 'label'
        max_rows (int): Số hàng tối đa để hiển thị (0 = hiển thị tất cả)
        
    Returns:
        pd.DataFrame: DataFrame chứa các hàng có inf
    """
    
    print(f"\n{'='*80}")
    print(f"🔍 XEM CÁC HÀNG CÓ GIÁ TRỊ INF CÙNG NHÃN: {os.path.basename(filepath)}")
    print(f"{'='*80}\n")
    
    # Đọc file CSV
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Đã tải file: {filepath}")
        print(f"   Kích thước: {df.shape[0]} dòng × {df.shape[1]} cột\n")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")
        return pd.DataFrame()
    
    # Xác định cột nhãn
    if label_column is None:
        # Tìm cột có tên 'label', 'Label', 'labels' hoặc lấy cột cuối cùng
        possible_names = ['label', 'Label', 'labels', 'Labels', 'CLASS', 'class', 'target', 'Target']
        label_column = None
        
        for name in possible_names:
            if name in df.columns:
                label_column = name
                break
        
        if label_column is None:
            label_column = df.columns[-1]
    
    if label_column not in df.columns:
        print(f"❌ Không tìm thấy cột nhãn: {label_column}")
        return pd.DataFrame()
    
    print(f"📌 Cột nhãn được sử dụng: '{label_column}'\n")
    print("⏳ Đang quét dữ liệu để tìm giá trị inf... (sử dụng vectorized operations)")
    
    # Sử dụng vectorized operations để tìm inf (nhanh hơn vòng lặp)
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
    
    # Tạo boolean mask cho các hàng có chứa inf
    mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    for col in numeric_cols:
        mask[col] = np.isinf(df[col])
    
    # Hàng nào có inf
    rows_with_inf_mask = mask.any(axis=1)
    rows_with_inf = df[rows_with_inf_mask].copy()
    
    if len(rows_with_inf) == 0:
        print("✅ Không tìm thấy hàng nào có giá trị inf!")
        return pd.DataFrame()
    
    print(f"🔴 Tìm thấy {len(rows_with_inf)} hàng có giá trị inf\n")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # HIỂN THỊ CHI TIẾT CÁC HÀNG CÓ INF
    # ═══════════════════════════════════════════════════════════════════════════════
    print("📊 CHI TIẾT CÁC HÀNG CÓ GIÁ TRỊ INF:")
    print("-" * 80)
    
    display_count = len(rows_with_inf) if max_rows == 0 else min(max_rows, len(rows_with_inf))
    
    # Nhóm theo nhãn
    label_inf_counts = rows_with_inf[label_column].value_counts().to_dict()
    
    print(f"\n1️⃣ THỐNG KÊ INF THEO NHÃN:")
    print("-" * 80)
    for label, count in sorted(label_inf_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = round(count / len(rows_with_inf) * 100, 2)
        print(f"   Nhãn '{label}': {count} hàng ({percentage}%)")
    
    # Hiển thị chi tiết từng hàng
    print(f"\n2️⃣ CHI TIẾT {display_count} HÀNG ĐẦU TIÊN:")
    print("-" * 80)
    
    for i, idx in enumerate(rows_with_inf.index[:display_count]):
        label = rows_with_inf.loc[idx, label_column]
        
        # Tìm cột nào có inf trong hàng này
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(rows_with_inf.loc[idx, col]):
                inf_cols.append(col)
        
        if inf_cols:
            print(f"\n   📍 Hàng {idx + 1} (Nhãn: '{label}'):")
            print(f"      Cột(s) có inf: {', '.join(inf_cols)}")
            print(f"      Giá trị inf:")
            
            for col in inf_cols:
                value = rows_with_inf.loc[idx, col]
                value_type = "Positive (+∞)" if value == np.inf else "Negative (-∞)"
                print(f"         • {col} = {value} ({value_type})")
    
    if display_count < len(rows_with_inf):
        print(f"\n   ... và {len(rows_with_inf) - display_count} hàng khác ...")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # TÓM TẮT THỐNG KÊ
    # ═══════════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("3️⃣ THỐNG KÊ CHUNG:")
    print("-" * 80)
    print(f"   Tổng hàng có inf: {len(rows_with_inf)}")
    print(f"   Tỷ lệ: {round(len(rows_with_inf) / len(df) * 100, 4)}% của tổng dữ liệu")
    print(f"   Số nhãn bị ảnh hưởng: {len(label_inf_counts)}")
    print(f"\n{'='*80}\n")
    
    return rows_with_inf


def save_inf_rows_report(filepath: str, label_column: str = None, 
                         output_file: str = 'inf_rows_report.csv') -> None:
    """
    Lưu các hàng có inf vào file CSV.
    (Sử dụng vectorized operations cho hiệu suất cao)
    
    Args:
        filepath (str): Đường dẫn tới file CSV
        label_column (str): Tên cột chứa nhãn
        output_file (str): Tên file xuất ra
    """
    
    print(f"\n💾 LƯỚI CÁC HÀNG CÓ INF VÀO FILE...")
    print("-" * 80)
    
    df = pd.read_csv(filepath)
    
    if label_column is None:
        possible_names = ['label', 'Label', 'labels', 'Labels', 'CLASS', 'class', 'target', 'Target']
        label_column = None
        
        for name in possible_names:
            if name in df.columns:
                label_column = name
                break
        
        if label_column is None:
            label_column = df.columns[-1]
    
    # Tìm hàng có inf (vectorized)
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
    
    mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    for col in numeric_cols:
        mask[col] = np.isinf(df[col])
    
    rows_with_inf_mask = mask.any(axis=1)
    
    if rows_with_inf_mask.any():
        inf_df = df[rows_with_inf_mask].copy()
        inf_df.to_csv(output_file, index=False)
        print(f"✅ Đã lưu {len(inf_df)} hàng có inf vào: {output_file}\n")
    else:
        print(f"✅ Không có hàng nào với inf để lưu.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN - Ví dụ sử dụng
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Phân tích file CSV bất kỳ
    # Ví dụ: dataset_optimized.csv, dataset_ready_for_model.csv, vv.
    
    csv_file = "dataset_bin60.csv"  # Thay tên file tại đây
    
    if os.path.exists(csv_file):
        # 1. Phân tích chung
        print("\n" + "="*80)
        print("BƯỚC 1: PHÂN TÍCH CHUNG")
        print("="*80)
        results = analyze_inf_missing_values(csv_file, verbose=True)
        create_quality_report(results)
        
        # 2. Xem chi tiết hàng có inf (nếu có)
        if results['inf_columns']:
            print("\n" + "="*80)
            print("BƯỚC 2: CHI TIẾT CÁC HÀNG CÓ INF")
            print("="*80)
            # Thay 'label' bằng tên cột nhãn của bạn (nếu khác)
            inf_rows = display_inf_rows_with_labels(csv_file, label_column=None, max_rows=50)
            
            # 3. Lưu các hàng có inf vào file riêng
            print("\n" + "="*80)
            print("BƯỚC 3: LƯU CÁC HÀNG CÓ INF")
            print("="*80)
            save_inf_rows_report(csv_file, label_column=None, output_file='inf_rows_found.csv')
    else:
        print(f"❌ Không tìm thấy file: {csv_file}")
        print("\nCác file có sẵn trong thư mục:")
        csv_files = [f for f in os.listdir() if f.endswith('.csv')]
        for f in csv_files:
            print(f"   - {f}")
