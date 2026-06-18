"""
app_web.py — FastAPI Web Server (Industry Standard Edge-Cloud Architecture)
Server đóng vai trò nhận ảnh từ các Client (Web/Thiết bị nhúng) gửi lên, xử lý và trả kết quả JSON.
"""
from __future__ import annotations

import base64
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

        # Caching identity for performance across stateless requests (based on track_id if available)
        self.identity_cache: Dict[int, tuple] = {}
        self.liveness_cache: Dict[int, tuple] = {}
        self.db_names: Dict[int, str] = {}

state = AppState()

# ── Core AI Processing Logic ──────────────────────────────────────────────────

def process_frame_stateless(frame: np.ndarray, client_id: str = "default") -> dict:
    """Xử lý 1 bức ảnh và trả về danh sách khuôn mặt (Dùng chung cho API và WebSocket)"""
    t0 = time.perf_counter()
    
    if state.detector is None:
        return {"error": "AI models not loaded"}

    # Detection
    raw_dets = state.detector.detect(frame)
    if not raw_dets:
        return {"faces": [], "process_time_ms": int((time.perf_counter() - t0) * 1000)}

    # Track (mỗi client_id có thể cần tracker riêng để không bị lẫn lộn track_id)
    # Tuy nhiên vì làm đơn giản, ta dùng tracker chung hoặc bỏ qua tracker nếu gửi từ nhiều máy.
    # Trong phiên bản này, ta vẫn dùng tracker để ổn định ID.
    tracks = state.tracker.update(raw_dets) if state.tracker else []
    if not tracks:
        # Fallback to detections if tracker is missing
        tracks = raw_dets

    faces_data = []

    for t in tracks:
        tid = t.get("track_id", hash(str(t["bbox"]))) # Fallback ID nếu không có tracker
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
        if tid in state.identity_cache:
            person_id, similarity = state.identity_cache[tid]
        elif emb is not None and state.feature_store:
            match = state.feature_store.searcher.search(emb)
            if match:
                person_id, similarity = match
            state.identity_cache[tid] = (person_id, similarity)

        # Liveness
        is_real = True
        spoof_score = 1.0
        if tid in state.liveness_cache:
            is_real, spoof_score = state.liveness_cache[tid]
        elif aligned is not None and state.anti_spoof:
            if state.anti_spoof._model is not None:
                is_real, spoof_score = state.anti_spoof.is_real(aligned)
                state.liveness_cache[tid] = (is_real, spoof_score)

        # Fetch name
        name = "Unknown"
        if person_id:
            if person_id not in state.db_names and state.db:
                p = state.db.get_person(person_id)
                state.db_names[person_id] = p.full_name if p else "Unknown"
            name = state.db_names.get(person_id, "Unknown")

        faces_data.append({
            "track_id": tid,
            "bbox": list(bbox),
            "conf": float(conf),
            "person_id": person_id,
            "similarity": float(similarity) if similarity else None,
            "is_real": is_real,
            "name": name,
        })

    process_time_ms = int((time.perf_counter() - t0) * 1000)
    return {"faces": faces_data, "process_time_ms": process_time_ms}


# ── Lifespan (startup/shutdown) ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo AI pipeline khi server start."""
    log.info("Initializing AI pipeline...")
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
        log.info("AI models loaded successfully.")
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

    log.info("Server ready. (Waiting for client connections)")
    yield

    log.info("Shutting down...")
    if state.feature_store:
        state.feature_store.searcher.save()
    if state.db:
        state.db.close()


# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Face Recognition API (Edge-Cloud)",
    description="API nhận diện khuôn mặt xử lý ảnh từ Client gửi lên (Stateless).",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

web_dir = Path(__file__).parent / "web"
web_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

# ── REST API Endpoints ─────────────────────────────────────────────────────────

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
        "models_loaded": state.detector is not None,
        "total_persons": len(state.db.list_persons()) if state.db else 0,
    }


@app.post("/api/recognize_image", tags=["Recognition"])
async def recognize_image(file: UploadFile = File(...)):
    """(Dành cho IoT/Camera) Gửi 1 file ảnh lên để nhận diện."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = process_frame_stateless(img, client_id="rest_api")
    return JSONResponse(content=result)


@app.get("/api/persons", tags=["Persons"])
def list_persons(search: str = ""):
    if not state.db: raise HTTPException(503, "DB chưa sẵn sàng")
    return [p.to_dict() for p in state.db.list_persons(search=search)]

@app.get("/api/persons/{person_id}", tags=["Persons"])
def get_person(person_id: int):
    if not state.db: raise HTTPException(503, "DB chưa sẵn sàng")
    p = state.db.get_person(person_id)
    if not p: raise HTTPException(404, "Không tìm thấy người dùng")
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
    if not state.db: raise HTTPException(503, "DB chưa sẵn sàng")

    person = Person(id=None, full_name=full_name, dob=dob, id_number=id_number, gender=gender, phone=phone)
    face_img = None
    embeddings = []

    if photo:
        content = await photo.read()
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: raise HTTPException(400, "Ảnh không hợp lệ")

        if state.detector and state.aligner:
            dets = state.detector.detect(img)
            if not dets: raise HTTPException(400, "Không phát hiện khuôn mặt")
            aligned = state.aligner.align(img, kps=dets[0].get("kps"), bbox=dets[0]["bbox"])
            face_img = aligned
            if state.embedder and aligned is not None:
                emb = state.embedder.embed(aligned)
                if emb is not None: embeddings.append(emb)

    photo_path_str = None
    if face_img is not None:
        from utils.config import FACES_DIR
        FACES_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"face_{int(time.time() * 1000)}.jpg"
        cv2.imwrite(str(FACES_DIR / fname), face_img)
        person.photo_path = fname

    pid = state.db.add_person(person)
    if embeddings and state.feature_store:
        state.feature_store.register(pid, embeddings)

    # Xoá cache identity vì có người mới
    state.identity_cache.clear()
    state.db_names.clear()

    return {"success": True, "person_id": pid}

@app.delete("/api/persons/{person_id}", tags=["Persons"])
def delete_person(person_id: int):
    if not state.db or not state.feature_store: raise HTTPException(503, "DB chưa sẵn sàng")
    existing = state.db.get_person(person_id)
    if not existing: raise HTTPException(404, "Không tìm thấy")
    if existing.photo_path:
        from utils.config import FACES_DIR
        p = FACES_DIR / existing.photo_path
        if p.exists(): p.unlink()
    state.feature_store.unregister(person_id)
    state.db.delete_person(person_id)
    
    state.identity_cache.clear()
    state.db_names.clear()
    return {"success": True}

@app.get("/api/persons/{person_id}/photo", tags=["Persons"])
def get_person_photo(person_id: int):
    from fastapi.responses import FileResponse
    from utils.config import FACES_DIR
    p = state.db.get_person(person_id) if state.db else None
    if not p or not p.photo_path: raise HTTPException(404, "Không có ảnh")
    fpath = FACES_DIR / p.photo_path
    if not fpath.exists(): raise HTTPException(404, "File không tồn tại")
    return FileResponse(str(fpath), media_type="image/jpeg")


# ── WebSocket Endpoint (Dành cho WebRTC / Browser Streaming) ───────────────────

@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    """
    Client (Trình duyệt) gửi frames liên tục dưới dạng Base64 hoặc bytes.
    Server phân tích và gửi lại JSON kết quả (bounding box, tên).
    """
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "unknown"
    log.info(f"WebSocket Client connected: {client_ip}")

    try:
        while True:
            # Nhận dữ liệu text (base64) hoặc bytes. Ưu tiên nhận bytes để nhanh.
            data = await websocket.receive_bytes()
            
            # Giải mã JPEG
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Xử lý frame
                result = process_frame_stateless(frame, client_id=client_ip)
                
                # Trả JSON về cho Client để tự vẽ lên Canvas
                await websocket.send_json(result)
            else:
                await websocket.send_json({"error": "Decode failed"})

    except WebSocketDisconnect:
        log.info(f"WebSocket Client disconnected: {client_ip}")
    except Exception as e:
        log.error(f"WebSocket Error ({client_ip}): {e}")


# ── Serve HTML UI ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    html_path = web_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_web:app", host="0.0.0.0", port=5000, reload=False, log_level="info")
