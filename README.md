# 🤖 Face Recognition AI

Hệ thống nhận diện khuôn mặt real-time với **FastAPI**, **YOLO**, **ArcFace**, **FAISS** — có giao diện web truy cập từ mọi thiết bị qua **ngrok**.

## 📐 Kiến trúc

```
[Camera/RTSP] → [YOLO Detector] → [ArcFace Embedder] → [FAISS Search]
                                                              ↓
                                               [FastAPI Server :5000]
                                                              ↓
                                                      [ngrok Tunnel]
                                                              ↓
                                              [Browser: Phone / PC]
```

## 🗂 Cấu trúc thư mục

```
face_recognition_app/
├── app_web.py          # FastAPI server (Web mode)
├── main.py             # PyQt5 Desktop app
├── web/index.html      # Giao diện web responsive
├── core/pipeline/      # AI pipeline (camera, processor, tracker)
├── models/             # Detector, Embedder, Searcher
├── database/           # SQLite + FAISS
├── utils/              # Config, logger
├── weights/            # Model weights (không push GitHub)
├── data/               # DB + face photos (không push GitHub)
├── Dockerfile          # Desktop mode
├── Dockerfile.web      # Web/API mode ← dùng cái này
├── docker-compose.yml
├── requirements.txt
├── requirements-web.txt
├── deploy.sh
└── ngrok.yml
```

---

## 🚀 Chạy Local (Không Docker)

### 1. Cài dependencies
```bash
pip install -r requirements.txt -r requirements-web.txt
```

### 2. Chạy FastAPI server
```bash
python app_web.py
# Hoặc:
uvicorn app_web:app --host 0.0.0.0 --port 5000 --reload
```

### 3. Mở trình duyệt
```
http://localhost:5000
```

### 4. Xem API docs
```
http://localhost:5000/docs
```

---

## 🐳 Chạy với Docker

### Build image
```bash
docker build -f Dockerfile.web -t facerecognize-web .
```

### Chạy container
```bash
# Chạy web service
docker-compose up -d face-web

# Xem log
docker-compose logs -f face-web

# Dừng
docker-compose down
```

### Webcam trong Docker (Linux)
Uncomment dòng `devices` trong `docker-compose.yml`:
```yaml
devices:
  - /dev/video0:/dev/video0
```

---

## 📡 Expose qua Ngrok (truy cập từ điện thoại)

### Cách 1: Ngrok standalone (khuyến nghị)

```bash
# Cài ngrok
# Windows: https://ngrok.com/download
# Mac: brew install ngrok
# Linux: snap install ngrok

# Đăng nhập (lấy authtoken tại ngrok.com)
ngrok config add-authtoken YOUR_AUTHTOKEN

# Tunnel port 5000
ngrok http 5000
```
Sau đó mở URL hiển thị (vd: `https://abc123.ngrok-free.app`) trên điện thoại.

### Cách 2: Ngrok qua Docker Compose

```bash
# Thêm NGROK_AUTHTOKEN vào .env
echo "NGROK_AUTHTOKEN=your_token_here" >> .env

# Chạy web + ngrok cùng lúc
docker-compose --profile ngrok up -d

# Xem URL ngrok
docker-compose logs ngrok
# Hoặc mở: http://localhost:4040
```

---

## 🌐 Deploy lên SSH Server

### Yêu cầu server
- Ubuntu 20.04+
- Docker + Docker Compose đã cài
- Port 5000 mở trong firewall

### Bước 1: Push lên GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/face_recognition_app.git
git branch -M main
git push -u origin main
```

### Bước 2: Clone và chạy trên server
```bash
# SSH vào server
ssh ubuntu@YOUR_SERVER_IP

# Clone project
git clone https://github.com/YOUR_USERNAME/face_recognition_app.git
cd face_recognition_app

# Copy model weights (dùng scp từ máy local)
# Local: scp -r weights/ ubuntu@YOUR_SERVER_IP:/opt/face_recognition_app/

# Tạo .env trên server
cp .env.example .env   # hoặc tạo thủ công
nano .env

# Build & run
docker-compose build face-web
docker-compose up -d face-web
```

### Bước 3: Script tự động (từ máy local)
```bash
# Sửa SERVER và REPO_URL trong deploy.sh
chmod +x deploy.sh
./deploy.sh ubuntu@YOUR_SERVER_IP
```

### Bước 4: Chạy ngrok trên server
```bash
ngrok http 5000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Web UI |
| GET | `/video_feed` | MJPEG live stream |
| GET | `/api/status` | Trạng thái hệ thống |
| GET | `/api/recognize` | Kết quả nhận diện (JSON) |
| GET | `/api/persons` | Danh sách người |
| POST | `/api/persons` | Thêm người mới + ảnh |
| PUT | `/api/persons/{id}` | Cập nhật thông tin |
| DELETE | `/api/persons/{id}` | Xoá người |
| GET | `/api/snapshot` | Chụp ảnh frame hiện tại |
| GET | `/docs` | Swagger UI |

---

## ⚙️ Cấu hình (.env)

```env
# Camera
CAMERA_INDEX=0          # 0=webcam, hoặc rtsp://... cho IP camera
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30

# AI Thresholds
DETECTION_THRESHOLD=0.5
RECOGNITION_THRESHOLD=0.45

# Ngrok
NGROK_AUTHTOKEN=your_token_here
```

---

## 📱 Truy cập từ điện thoại

1. Chạy `ngrok http 5000` trên máy/server
2. Lấy URL dạng `https://xxxx.ngrok-free.app`
3. Mở URL đó trên điện thoại
4. Giao diện tự động responsive cho mobile

---

## 🛠 Tech Stack

- **Backend**: FastAPI + Uvicorn
- **AI**: YOLOv8 (face detection), ArcFace/InsightFace (embedding), FAISS (search)
- **Anti-spoof**: MiniFASNet
- **DB**: SQLite + FAISS index
- **Stream**: MJPEG over HTTP
- **Tunnel**: ngrok
- **Container**: Docker + Docker Compose
