#!/bin/bash
# deploy.sh — Script tự động deploy lên SSH server
# Sử dụng: bash deploy.sh <user@server_ip>
# Ví dụ:   bash deploy.sh ubuntu@123.456.789.0

set -e

SERVER="${1:-ubuntu@your-server-ip}"
APP_DIR="/opt/face_recognition_app"
REPO_URL="https://github.com/YOUR_USERNAME/face_recognition_app.git"

echo "================================================"
echo "  Face Recognition AI — Auto Deploy"
echo "  Target: $SERVER"
echo "================================================"

# Bước 1: Push code lên GitHub (chạy local trước)
echo ""
echo "▶ [1/5] Commit & Push lên GitHub..."
git add -A
git commit -m "deploy: $(date '+%Y-%m-%d %H:%M')" || echo "Không có thay đổi mới."
git push origin main
echo "✅ GitHub updated."

# Bước 2: SSH vào server và deploy
echo ""
echo "▶ [2/5] SSH vào server $SERVER..."
ssh "$SERVER" << 'REMOTE_SCRIPT'
set -e

APP_DIR="/opt/face_recognition_app"
REPO_URL="https://github.com/YOUR_USERNAME/face_recognition_app.git"

echo "▶ [3/5] Pull code mới nhất..."
if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR"
    git pull origin main
else
    sudo mkdir -p "$APP_DIR"
    sudo chown $USER:$USER "$APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

echo "▶ [4/5] Download model weights (nếu chưa có)..."
mkdir -p weights data/faces
# Thêm lệnh download weights của bạn ở đây, ví dụ:
# wget -O weights/yolov8n-face.pt "https://your-storage/yolov8n-face.pt"
# wget -O weights/MiniFASNetV2.pth "https://your-storage/MiniFASNetV2.pth"
echo "  ⚠️  Nhớ copy model weights thủ công vào thư mục weights/"

echo "▶ [5/5] Docker build & restart..."
docker-compose pull 2>/dev/null || true
docker-compose build face-web
docker-compose up -d face-web
docker-compose ps

echo ""
echo "✅ Deploy xong! Ứng dụng đang chạy tại: http://$(hostname -I | awk '{print $1}'):5000"
REMOTE_SCRIPT

echo ""
echo "================================================"
echo "  ✅ Deploy hoàn tất!"
echo "  Mở trình duyệt: http://$(ssh $SERVER 'hostname -I | awk "{print \$1}"' 2>/dev/null):5000"
echo "================================================"
