import os
import sys
import time
import threading
import webbrowser
import uvicorn

def open_browser():
    """Đợi 2 giây cho server khởi động rồi tự động mở trình duyệt"""
    time.sleep(2)
    
    # Lấy đường dẫn tuyệt đối của file app.html
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_file = os.path.join(current_dir, "frontend", "app.html")
    
    # Chuyển đổi thành định dạng URI cho trình duyệt
    file_uri = f"file:///{html_file.replace(os.path.sep, '/')}"
    
    print(f"\n[+] Đang tự động mở giao diện tại: {file_uri}\n")
    webbrowser.open(file_uri)

if __name__ == "__main__":
    # Tránh việc mở nhiều tab nếu uvicorn tự động reload
    if os.environ.get("GREEN_CLEARANCE_UI_OPENED") != "1":
        os.environ["GREEN_CLEARANCE_UI_OPENED"] = "1"
        
        # Khởi chạy luồng mở trình duyệt ngầm để không block server
        threading.Thread(target=open_browser, daemon=True).start()

    print("==================================================")
    print("      KHỞI ĐỘNG HỆ THỐNG GREEN CLEARANCE          ")
    print("==================================================")
    
    # Khởi động Backend (FastAPI) bằng Uvicorn
    # Chạy ở host 0.0.0.0 để có thể truy cập từ máy khác trong mạng LAN
    uvicorn.run(
        "backend.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False  # Đặt False để môi trường production chạy ổn định, không chiếm dụng file watcher
    )
