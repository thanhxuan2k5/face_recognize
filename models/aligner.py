"""
models/aligner.py — Face alignment via 5-point landmark + affine warp to 112×112.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from utils.config import FACE_ALIGN_SIZE
from utils.logger import get_logger

log = get_logger(__name__)

# ArcFace reference 5-point landmarks (112×112 space)
_ARCFACE_REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceAligner:

    def __init__(self, size: int = FACE_ALIGN_SIZE) -> None:
        self.size = size
        ref = _ARCFACE_REF_LANDMARKS.copy()
        if size != 112:
            ref = ref * (size / 112.0)
        self._ref = ref

    def align(
        self,
        frame: np.ndarray,
        kps: Optional[np.ndarray] = None,
        bbox: Optional[tuple] = None,
    ) -> np.ndarray:
        if kps is not None and len(kps) == 5:
            return self._warp(frame, kps)
        if bbox is not None:
            return self._crop_resize(frame, bbox)
        log.warning("No kps or bbox provided to aligner.")
        return cv2.resize(frame, (self.size, self.size))

    def _warp(self, frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
        src = kps.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, self._ref, method=cv2.LMEDS)
        if M is None:
            # Fallback: use bounding box of kps
            x1, y1 = kps.min(axis=0).astype(int)
            x2, y2 = kps.max(axis=0).astype(int)
            return self._crop_resize(frame, (x1, y1, x2, y2))
        aligned = cv2.warpAffine(frame, M, (self.size, self.size), flags=cv2.INTER_LINEAR)
        return aligned

    def _crop_resize(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((self.size, self.size, 3), dtype=np.uint8)
        return cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
