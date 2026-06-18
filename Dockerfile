# Sử dụng image chính thức của Python làm base
FROM python:3.9-slim

# Thiết lập biến môi trường để không tạo ra file .pyc và log không bị buffer
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và PyQt5
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libqt5gui5 \
    libqt5core5a \
    libqt5widgets5 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Copy file requirements và cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Chạy ứng dụng (Lưu ý: Chạy app GUI trong Docker cần cấu hình DISPLAY)
CMD ["python", "main.py"]
