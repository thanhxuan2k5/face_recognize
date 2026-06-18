"""
app_web.py — FastAPI Web Server cho Face Recognition App
Chạy độc lập với PyQt5 GUI, dùng chung AI core.
"""
from __future__ import annotations

import asyncio
import base64
import io
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, FRAME_QUEUE_MAXSIZE
from utils.logger import get_logger
from database.db_manager import DBManager, Person
from database.feature_store import FeatureStore
from database.cache import EmbeddingCache

log = get_logger("app_web")

# ── Global AI Pipeline State ───────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.db: Optional[DBManager] = None
        self.feature_store: Optional[FeatureStore] = None
        self.detector = None
        self.aligner = None
        self.embedder = None
        self.tracker = None
        self.anti_spoof = None
        self.embed_cache: Optional[EmbeddingCache] = None

        self.camera_thread: Optional[threading.Thread] = None
        self.processor_thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._lock = threading.Lock()

        # Latest processed frame (BGR numpy array with overlays)
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_faces: List[Dict] = []
        self.fps: float = 0.0
        self._frame_lock = threading.Lock()

state = AppState()


# ── Camera + AI Thread (non-Qt) ────────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
    """Vẽ bounding box và label lên frame."""
    out = frame.copy()
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.get("bbox", [0, 0, 0, 0])]
        pid = f.get("person_id")
        sim = f.get("similarity", 0.0) or 0.0
        is_real = f.get("is_real", True)
        name = f.get("name", "Unknown")

        # Box color
        if pid and is_real:
            color = (0, 220, 100)      # green — known & real
        elif pid and not is_real:
            color = (0, 140, 255)      # orange — known but fake
        elif is_real:
            color = (255, 200, 0)      # blue — unknown but real
        else:
            color = (0, 0, 220)        # red — spoof

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{name} ({sim:.0%})" if pid else ("Real" if is_real else "FAKE")
        liveness = " [SPOOF]" if not is_real else ""
        full_label = label + liveness
        (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, full_label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # FPS overlay
    cv2.putText(out, f"FPS: {state.fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2, cv2.LINE_AA)
    return out


def _camera_loop():
    """Vòng lặp đọc frame từ camera (thread riêng)."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        log.error("Không thể mở camera index %d", CAMERA_INDEX)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    state._cap = cap

    frame_q: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
    state._frame_q = frame_q
    log.info("Camera started (index=%d, %dx%d @ %dfps)", CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)

    while state._running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        if frame_q.full():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put_nowait(frame)

    cap.release()
    log.info("Camera stopped.")


def _process_loop():
    """Vòng lặp xử lý AI trên frame (thread riêng)."""
    from core.pipeline.tracker import FaceTracker
    from core.pipeline.anti_spoof import AntiSpoof

    frame_skip = 0
    skip_counter = 0
    last_dets = []
    last_time = 0.0
    fps_ema = 0.0
    identity_cache: Dict[int, tuple] = {}
    liveness_cache: Dict[int, tuple] = {}

    while state._running:
        if not hasattr(state, '_frame_q'):
            time.sleep(0.1)
            continue

        try:
            frame = state._frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        t0 = time.perf_counter()

        # Detection
        skip_counter = (skip_counter + 1) % (frame_skip + 1)
        if skip_counter == 0 and state.detector:
            raw_dets = state.detector.detect(frame)
            last_dets = raw_dets
        else:
            raw_dets = last_dets

        # Track
        tracks = state.tracker.update(raw_dets) if state.tracker else []

        # Active track IDs
        active_tids = {t["track_id"] for t in tracks}
        for tid in list(identity_cache.keys()):
            if tid not in active_tids:
                del identity_cache[tid]
        for tid in list(liveness_cache.keys()):
            if tid not in active_tids:
                del liveness_cache[tid]

        faces_data = []
        db_names: Dict[int, str] = {}

        for t in tracks:
            tid = t["track_id"]
            bbox = t["bbox"]
            conf = t["conf"]
            kps = t.get("kps")

            aligned = state.aligner.align(frame, kps=kps, bbox=bbox) if state.aligner else None

            # Embed
            emb = None
            if aligned is not None and state.embedder:
                cached_emb = state.embed_cache.get(tid) if state.embed_cache else None
                if cached_emb is None:
                    emb = state.embedder.embed(aligned)
                    if emb is not None and state.embed_cache:
                        state.embed_cache.set(tid, emb, conf)
                else:
                    emb = cached_emb

            # Search identity
            person_id, similarity = None, None
            if tid in identity_cache:
                person_id, similarity = identity_cache[tid]
            elif emb is not None and state.feature_store:
                match = state.feature_store.searcher.search(emb)
                if match:
                    person_id, similarity = match
                identity_cache[tid] = (person_id, similarity)

            # Liveness
            is_real = True
            if tid in liveness_cache:
                is_real, _ = liveness_cache[tid]
            elif aligned is not None and state.anti_spoof:
                if state.anti_spoof._model is not None:
                    is_real, score = state.anti_spoof.is_real(aligned)
                    liveness_cache[tid] = (is_real, score)

            # Fetch name
            name = "Unknown"
            if person_id:
                if person_id not in db_names and state.db:
                    p = state.db.get_person(person_id)
                    db_names[person_id] = p.full_name if p else "Unknown"
                name = db_names.get(person_id, "Unknown")

            faces_data.append({
                "track_id": tid,
                "bbox": list(bbox),
                "conf": float(conf),
                "person_id": person_id,
                "similarity": float(similarity) if similarity else None,
                "is_real": is_real,
                "name": name,
            })

        # FPS
        now = time.perf_counter()
        if last_time > 0:
            elapsed = now - last_time
            curr = 1.0 / elapsed if elapsed > 0 else 0
            fps_ema = 0.9 * fps_ema + 0.1 * curr
        last_time = now

        # Draw and store
        rendered = _draw_overlay(frame, faces_data)
        with state._frame_lock:
            state.latest_frame = rendered
            state.latest_faces = faces_data
            state.fps = fps_ema

    log.info("Processor thread stopped.")


# ── Lifespan (startup/shutdown) ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo AI pipeline khi server start."""
    log.info("Khởi tạo AI pipeline...")
    state.db = DBManager()
    state.feature_store = FeatureStore()
    state.embed_cache = EmbeddingCache()

    try:
        from models.detector import FaceDetector
        from models.aligner import FaceAligner
        from models.embedder import FaceEmbedder
        from core.pipeline.tracker import FaceTracker
        from core.pipeline.anti_spoof import AntiSpoof

        state.aligner = FaceAligner()
        state.tracker = FaceTracker()
        state.anti_spoof = AntiSpoof()
        state.detector = FaceDetector()
        state.embedder = FaceEmbedder()
        log.info("Tải mô hình AI thành công.")
    except Exception as exc:
        log.error("Lỗi tải mô hình: %s", exc)

    # Sync FAISS vs SQLite
    try:
        db_ids = {p.id for p in state.db.list_persons()}
        faiss_ids = set(state.feature_store.searcher._id_map)
        for gid in faiss_ids - db_ids:
            state.feature_store.unregister(gid)
    except Exception as exc:
        log.warning("Sync FAISS lỗi: %s", exc)

    # Start camera + processor threads
    state._running = True
    state.camera_thread = threading.Thread(target=_camera_loop, daemon=True, name="camera")
    state.processor_thread = threading.Thread(target=_process_loop, daemon=True, name="processor")
    state.camera_thread.start()
    state.processor_thread.start()
    log.info("Server sẵn sàng.")

    yield

    # Shutdown
    log.info("Dọn dẹp...")
    state._running = False
    if state.feature_store:
        state.feature_store.searcher.save()
    if state.db:
        state.db.close()


# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Face Recognition API",
    description="API nhận diện khuôn mặt real-time với camera, FAISS và ArcFace.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static web files
web_dir = Path(__file__).parent / "web"
web_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


# ── MJPEG Video Stream ─────────────────────────────────────────────────────────

def _generate_mjpeg():
    """Generator trả về MJPEG frames liên tục."""
    while True:
        with state._frame_lock:
            frame = state.latest_frame

        if frame is None:
            # Placeholder khi chưa có frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for camera...", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
            frame = placeholder

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        time.sleep(1.0 / 30)  # ~30 FPS stream


@app.get("/video_feed", tags=["Stream"])
def video_feed():
    """MJPEG live stream — nhúng vào thẻ <img> trong HTML."""
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── REST API ───────────────────────────────────────────────────────────────────

class PersonCreate(BaseModel):
    full_name: str
    dob: Optional[str] = None
    id_number: Optional[str] = None
    gender: Optional[str] = None
    phone: Optional[str] = None


class PersonUpdate(PersonCreate):
    pass


@app.get("/api/status", tags=["System"])
def get_status():
    """Trạng thái hệ thống."""
    return {
        "status": "running",
        "camera_active": state._running and state.latest_frame is not None,
        "fps": round(state.fps, 1),
        "models_loaded": state.detector is not None,
        "total_persons": len(state.db.list_persons()) if state.db else 0,
    }


@app.get("/api/recognize", tags=["Recognition"])
def get_recognition():
    """Kết quả nhận diện khuôn mặt của frame mới nhất."""
    with state._frame_lock:
        faces = state.latest_faces.copy()
        fps = state.fps
    return {"fps": round(fps, 1), "faces": faces}


@app.get("/api/persons", tags=["Persons"])
def list_persons(search: str = ""):
    """Danh sách tất cả người đã đăng ký."""
    if not state.db:
        raise HTTPException(503, "DB chưa sẵn sàng")
    persons = state.db.list_persons(search=search)
    return [p.to_dict() for p in persons]


@app.get("/api/persons/{person_id}", tags=["Persons"])
def get_person(person_id: int):
    """Thông tin một người."""
    if not state.db:
        raise HTTPException(503, "DB chưa sẵn sàng")
    p = state.db.get_person(person_id)
    if not p:
        raise HTTPException(404, f"Không tìm thấy person_id={person_id}")
    return p.to_dict()


@app.post("/api/persons", tags=["Persons"])
async def create_person(
    full_name: str = Form(...),
    dob: Optional[str] = Form(None),
    id_number: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
):
    """Thêm người mới và đăng ký khuôn mặt."""
    if not state.db:
        raise HTTPException(503, "DB chưa sẵn sàng")

    person = Person(
        id=None,
        full_name=full_name,
        dob=dob,
        id_number=id_number,
        gender=gender,
        phone=phone,
    )

    # Xử lý ảnh
    face_img = None
    embeddings = []

    if photo:
        content = await photo.read()
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Ảnh không hợp lệ")

        # Detect & align
        if state.detector and state.aligner:
            dets = state.detector.detect(img)
            if not dets:
                raise HTTPException(400, "Không phát hiện khuôn mặt trong ảnh")
            d = dets[0]
            aligned = state.aligner.align(img, kps=d.get("kps"), bbox=d["bbox"])
            face_img = aligned
            if state.embedder and aligned is not None:
                emb = state.embedder.embed(aligned)
                if emb is not None:
                    embeddings.append(emb)

    # Save photo
    photo_path_str = None
    if face_img is not None:
        from utils.config import FACES_DIR
        FACES_DIR.mkdir(parents=True, exist_ok=True)
        tmp_id = int(time.time() * 1000)
        fname = f"face_{tmp_id}.jpg"
        fpath = FACES_DIR / fname
        cv2.imwrite(str(fpath), face_img)
        photo_path_str = fname
        person.photo_path = photo_path_str

    pid = state.db.add_person(person)

    # Register embeddings
    if embeddings and state.feature_store:
        state.feature_store.register(pid, embeddings)

    return {"success": True, "person_id": pid, "embeddings_registered": len(embeddings)}


@app.put("/api/persons/{person_id}", tags=["Persons"])
def update_person(person_id: int, data: PersonUpdate):
    """Cập nhật thông tin người."""
    if not state.db:
        raise HTTPException(503, "DB chưa sẵn sàng")
    existing = state.db.get_person(person_id)
    if not existing:
        raise HTTPException(404, f"Không tìm thấy person_id={person_id}")
    existing.full_name = data.full_name
    existing.dob = data.dob
    existing.id_number = data.id_number
    existing.gender = data.gender
    existing.phone = data.phone
    state.db.update_person(existing)
    return {"success": True}


@app.delete("/api/persons/{person_id}", tags=["Persons"])
def delete_person(person_id: int):
    """Xoá người và embedding."""
    if not state.db or not state.feature_store:
        raise HTTPException(503, "DB chưa sẵn sàng")
    existing = state.db.get_person(person_id)
    if not existing:
        raise HTTPException(404, f"Không tìm thấy person_id={person_id}")
    # Remove photo
    if existing.photo_path:
        from utils.config import FACES_DIR
        p = FACES_DIR / existing.photo_path
        if p.exists():
            p.unlink()
    state.feature_store.unregister(person_id)
    state.db.delete_person(person_id)
    return {"success": True}


@app.get("/api/snapshot", tags=["Stream"])
def get_snapshot():
    """Chụp ảnh frame hiện tại, trả về base64 JPEG."""
    with state._frame_lock:
        frame = state.latest_frame

    if frame is None:
        raise HTTPException(503, "Camera chưa sẵn sàng")

    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        raise HTTPException(500, "Encode ảnh thất bại")
    b64 = base64.b64encode(buf.tobytes()).decode()
    return {"image": f"data:image/jpeg;base64,{b64}"}


@app.get("/api/persons/{person_id}/photo", tags=["Persons"])
def get_person_photo(person_id: int):
    """Trả về ảnh khuôn mặt của người."""
    from fastapi.responses import FileResponse
    from utils.config import FACES_DIR
    if not state.db:
        raise HTTPException(503, "DB chưa sẵn sàng")
    p = state.db.get_person(person_id)
    if not p or not p.photo_path:
        raise HTTPException(404, "Không có ảnh")
    fpath = FACES_DIR / p.photo_path
    if not fpath.exists():
        raise HTTPException(404, "File ảnh không tồn tại")
    return FileResponse(str(fpath), media_type="image/jpeg")


# ── Serve HTML UI ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    """Serve giao diện web chính."""
    html_path = web_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_web:app", host="0.0.0.0", port=5000, reload=False, log_level="info")
