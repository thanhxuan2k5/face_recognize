from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Dict, Optional

from utils.config import DETECTION_THRESHOLD
from utils.logger import get_logger

log = get_logger(__name__)


class FaceDetector:

    def __init__(
        self,
        threshold: float = DETECTION_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        self.mp_face_mesh = mp.solutions.face_mesh
        self._model = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=self.threshold,
            min_tracking_confidence=self.threshold
        )
        log.info("MediaPipe FaceMesh loaded.")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        if self._model is None or frame is None:
            return []

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._model.process(rgb)

        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = []
                for lm in face_landmarks.landmark:
                    points.append([int(lm.x * w), int(lm.y * h)])
                
                points = np.array(points)
                
                # Bounding box from mesh
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0)
                
                # Expand bounding box slightly (FaceMesh is tight)
                bw = x_max - x_min
                bh = y_max - y_min
                x1 = max(0, int(x_min - 0.1 * bw))
                y1 = max(0, int(y_min - 0.2 * bh))
                x2 = min(w, int(x_max + 0.1 * bw))
                y2 = min(h, int(y_max + 0.1 * bh))

                # Compute custom confidence based on frontal symmetry & resolution
                # Landmark 33: outer corner of right eye (left in image)
                # Landmark 263: outer corner of left eye (right in image)
                # Landmark 1: nose tip
                conf = 0.5  # default fallback
                if len(points) >= 264:
                    d_left = abs(points[1][0] - points[33][0])
                    d_right = abs(points[263][0] - points[1][0])
                    symmetry = 1.0 - abs(d_left - d_right) / max(d_left + d_right, 1e-5)
                    # Normalize box width relative to a target 200px size
                    size_factor = min(bw, 200) / 200.0
                    # 70% symmetry, 30% resolution size
                    conf = float(symmetry * 0.7 + size_factor * 0.3)

                # 5 keypoints for ArcFace Aligner
                # 468: right eye pupil (left in image), 473: left eye pupil (right in image)
                # 1: nose tip
                # 61: right mouth corner (left in image), 291: left mouth corner (right in image)
                if len(points) >= 474:
                    kps = np.array([
                        points[468],  # Left eye (image)
                        points[473],  # Right eye (image)
                        points[1],    # Nose
                        points[61],   # Left mouth (image)
                        points[291]   # Right mouth (image)
                    ])
                else:
                    kps = None
                
                faces.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "kps": kps,
                    "mesh_points": points
                })
                
        return faces
