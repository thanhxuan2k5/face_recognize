
import os
import time
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from utils.logger import get_logger

log = get_logger(__name__)



@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    conf: float
    age: int = 0
    hits: int = 1
    kps: np.ndarray = field(default=None)


def _iou(a: Tuple, b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """
    Simple IoU-based tracker with track ID persistence.
    Drop-in fallback when ByteTrack / OCSort is unavailable.
    """

    IOU_THRESHOLD = 0.2  # lower threshold to allow matching across larger movement
    MAX_AGE = 200  # increased to keep IDs longer for re-identification identity persistence

    def __init__(self) -> None:
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._last_time = 0.0
        self._fps = 0.0
        self._try_load_bytetrack()

    def _try_load_bytetrack(self) -> None:
        try:
            from bytetracker import BYTETracker  # type: ignore
            class _Args:
                track_thresh = 0.45
                track_buffer = 30
                match_thresh = 0.8
                mot20 = False
            self._byte = BYTETracker(_Args())
            self._use_byte = True
            log.info("ByteTracker loaded.")
        except ImportError:
            self._use_byte = False
            log.info("ByteTrack not installed — using IoU tracker fallback.")

    def update(self, detections: List[Dict]) -> List[Dict]:
        if self._use_byte:
            return self._update_byte(detections)
        return self._update_iou(detections)

    # ── ByteTrack path ────────────────────────────────────────────────────────

    def _update_byte(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []
        dets = np.array([[*d["bbox"], d["conf"]] for d in detections], dtype=np.float32)
        online_targets = self._byte.update(dets, [9999, 9999], [9999, 9999])
        results = []
        for t in online_targets:
            x1, y1, w, h = t.tlwh
            tid = int(t.track_id)
            # match back to original detection for kps
            tb = (int(x1), int(y1), int(x1 + w), int(y1 + h))
            kps = None
            mesh_points = None
            for d in detections:
                if _iou(d["bbox"], tb) > 0.5:
                    kps = d.get("kps")
                    mesh_points = d.get("mesh_points")
                    break
            results.append({"track_id": tid, "bbox": tb, "conf": float(t.score), "kps": kps, "mesh_points": mesh_points})
        return results

    # ── IoU fallback ──────────────────────────────────────────────────────────

    def _update_iou(self, detections: List[Dict]) -> List[Dict]:
        # Age all existing tracks
        for t in self._tracks.values():
            t.age += 1

        matched_track_ids = set()
        results = []

        for det in detections:
            bbox = det["bbox"]
            best_tid, best_iou = None, self.IOU_THRESHOLD
            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                iou = _iou(bbox, track.bbox)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_tid is not None:
                self._tracks[best_tid].bbox = bbox
                self._tracks[best_tid].conf = det["conf"]
                self._tracks[best_tid].kps = det.get("kps")
                self._tracks[best_tid].age = 0
                self._tracks[best_tid].hits += 1
                matched_track_ids.add(best_tid)
                results.append({"track_id": best_tid, **det})
            else:

                # New track
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = Track(
                    track_id=tid,
                    bbox=bbox,
                    conf=det["conf"],
                    kps=det.get("kps"),
                )
                results.append({"track_id": tid, **det})

        # Prune old tracks
        dead = [tid for tid, t in self._tracks.items() if t.age > self.MAX_AGE]
        for tid in dead:
            del self._tracks[tid]

        return results
