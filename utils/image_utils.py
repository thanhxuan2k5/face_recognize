
import cv2
import numpy as np
from typing import Tuple


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def crop_face(img: np.ndarray, bbox: Tuple[int, int, int, int], margin: float = 0.2) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    return img[y1:y2, x1:x2]


def resize_to(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def normalize_face(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img


def draw_bbox(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (34, 197, 94),
    thickness: int = 2,
) -> np.ndarray:

    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        # Background for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, th = 0.55, 1
        (tw, th_px), _ = cv2.getTextSize(label, font, scale, th)
        cv2.rectangle(img, (x1, y1 - th_px - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), font, scale, (0, 0, 0), th, cv2.LINE_AA)
    return img


def qimage_from_bgr(img: np.ndarray):
    from PyQt5.QtGui import QImage
    rgb = bgr_to_rgb(img)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
