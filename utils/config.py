
import os
from pathlib import Path
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")

def _int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _path(key: str, default: str) -> Path:
    return ROOT_DIR / os.getenv(key, default)


# ── Camera
CAMERA_INDEX: int = _int("CAMERA_INDEX", 0)
CAMERA_WIDTH: int = _int("CAMERA_WIDTH", 1280)
CAMERA_HEIGHT: int = _int("CAMERA_HEIGHT", 720)
CAMERA_FPS: int = _int("CAMERA_FPS", 30)

# ── Detection / Recognition
DETECTION_THRESHOLD: float = _float("DETECTION_THRESHOLD", 0.5)
RECOGNITION_THRESHOLD: float = _float("RECOGNITION_THRESHOLD", 0.4)
ANTI_SPOOF_THRESHOLD: float = _float("ANTI_SPOOF_THRESHOLD", 0.8)

# ── Frame Processing
FRAME_SKIP: int = _int("FRAME_SKIP", 0)
FRAME_QUEUE_MAXSIZE: int = _int("FRAME_QUEUE_MAXSIZE", 1)

# ── Model Weights
YOLO_WEIGHT: Path = _path("YOLO_WEIGHT", "weights/yolov8n-face.pt")
ARCFACE_WEIGHT: Path = _path("ARCFACE_WEIGHT", "weights/arcface_r100.pth")
# Updated to match the actual file name in weights/
SILENT_FACE_WEIGHT: Path = _path("SILENT_FACE_WEIGHT", "weights/MiniFASNetV2.pth")

# ── Storage
DB_PATH: Path = _path("DB_PATH", "data/persons.db")
FAISS_INDEX_PATH: Path = _path("FAISS_INDEX_PATH", "data/faiss.index")
FACES_DIR: Path = _path("FACES_DIR", "data/faces")
FACES_DIR.mkdir(parents=True, exist_ok=True)

# ── UI
UI_TARGET_FPS: int = _int("UI_TARGET_FPS", 60)
UI_THEME: str = _str("UI_THEME", "dark")

# ── Embedding
EMBEDDING_DIM: int = 512
FACE_ALIGN_SIZE: int = 112
