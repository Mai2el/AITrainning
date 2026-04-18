import pandas as pd
from ydata_profiling import ProfileReport

def create_html_report(csv_path: str, output_html: str):
    print(f"Đang đọc dữ liệu từ: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {csv_path}.")
        return

    print("Đang phân tích dữ liệu và tạo báo cáo... (Có thể mất vài phút tùy kích thước dataset)")
    
    # Tạo profile report
    # hãy đổi thành: ProfileReport(df, title="...", minimal=True)
    profile = ProfileReport(
        df, 
        title="Báo Cáo Phân Tích Dữ Liệu Mạng (Network Traffic EDA)",
        explorative=True,  
        minimal=True,  )

    # Xuất ra file HTML
    profile.to_file(output_html)
    print(f"✓ Tuyệt vời! Đã lưu báo cáo tại: {output_html}")
    print("→ Hãy click đúp vào file HTML đó để mở bằng trình duyệt web (Chrome/Edge/Firefox).")

# ==========================================
if __name__ == "__main__":
    INPUT_CSV = "dataset_optimized.csv" 
    OUTPUT_HTML = "scaled.html"
    
    create_html_report(INPUT_CSV, OUTPUT_HTML)