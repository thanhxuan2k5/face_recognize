from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
# from PyQt5.QtCore import QThread, pyqtSignal  # Removed for headless server

from utils.config import FRAME_SKIP, UI_TARGET_FPS
from utils.logger import get_logger
from models.detector import FaceDetector
from models.aligner import FaceAligner
from models.embedder import FaceEmbedder
from models.searcher import FaceSearcher
from core.pipeline.tracker import FaceTracker
from core.pipeline.anti_spoof import AntiSpoof
from database.cache import EmbeddingCache

log = get_logger(__name__)


@dataclass
class FaceResult:
    track_id: int
    bbox: tuple          # (x1, y1, x2, y2)
    conf: float          # detection confidence
    person_id: Optional[int] = None
    similarity: Optional[float] = None
    is_real: bool = True
    spoof_score: float = 1.0
    aligned_face: Optional[np.ndarray] = None  # (112,112,3) BGR


@dataclass
class FrameResult:
    frame: np.ndarray
    faces: List[FaceResult] = field(default_factory=list)
    fps: float = 0.0
    ts: float = field(default_factory=time.time)


class FrameProcessor(QThread):

    result_ready: pyqtSignal = pyqtSignal(object)   # emits FrameResult
    error_signal: pyqtSignal = pyqtSignal(str)

    def __init__(
        self,
        frame_queue: queue.Queue,
        detector: FaceDetector,
        aligner: FaceAligner,
        embedder: FaceEmbedder,
        searcher: FaceSearcher,
        tracker: FaceTracker,
        anti_spoof: AntiSpoof,
        embed_cache: EmbeddingCache,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._q = frame_queue
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.searcher = searcher
        self.tracker = tracker
        self.anti_spoof = anti_spoof
        self.embed_cache = embed_cache

        self._running = False
        self._skip_counter = 0
        self._last_detections: List[Dict] = []  # carry-over when skipping

        # FPS tracking (EMA)
        self._fps = 0.0
        self._last_time = 0.0

        # Blink Detection State
        self._blink_history: Dict[int, List[float]] = {}
        self._real_tracks: set = set()
        self._identity_cache: Dict[int, tuple] = {}
        self._liveness_cache: Dict[int, tuple] = {}

    # ── Thread lifecycle ──────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        log.info("FrameProcessor started.")
        target_delay = 1.0 / UI_TARGET_FPS

        while self._running:
            t0 = time.perf_counter()
            try:
                frame = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            result = self._process(frame)
            self.result_ready.emit(result)

            # Pace to target FPS
            elapsed = time.perf_counter() - t0
            sleep = target_delay - elapsed
            if sleep > 0:
                time.sleep(sleep)

        log.info("FrameProcessor stopped.")

    def stop(self) -> None:
        self._running = False
        self.wait()

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _process(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()

        self._skip_counter = (self._skip_counter + 1) % (FRAME_SKIP + 1)
        run_detection = self._skip_counter == 0

        if run_detection:
            raw_dets = self.detector.detect(frame)
            self._last_detections = raw_dets
        else:
            raw_dets = self._last_detections  # reuse

        # Track
        tracks = self.tracker.update(raw_dets)

        face_results: List[FaceResult] = []
        for t in tracks:
            fr = self._process_face(frame, t)
            face_results.append(fr)

        # Cleanup inactive tracks from blink history
        active_tids = {t["track_id"] for t in tracks}
        for tid in list(self._blink_history.keys()):
            if tid not in active_tids:
                del self._blink_history[tid]
        for tid in list(self._real_tracks):
            if tid not in active_tids:
                self._real_tracks.remove(tid)
        for tid in list(self._identity_cache.keys()):
            if tid not in active_tids:
                del self._identity_cache[tid]
        for tid in list(self._liveness_cache.keys()):
            if tid not in active_tids:
                del self._liveness_cache[tid]

        fps = self._compute_fps()
        return FrameResult(frame=frame, faces=face_results, fps=fps)

    def _process_face(self, frame: np.ndarray, track: Dict) -> FaceResult:
        tid = track["track_id"]
        bbox = track["bbox"]
        conf = track["conf"]
        kps = track.get("kps")

        # Align
        aligned = self.aligner.align(frame, kps=kps, bbox=bbox)

        # Embed (with cache)
        emb = self.embed_cache.get(tid)
        cached_conf = self.embed_cache.get_conf(tid)
        
        run_search = False
        if emb is None or conf > cached_conf + 0.05:
            new_emb = self.embedder.embed(aligned)
            if new_emb is not None:
                emb = new_emb
                self.embed_cache.set(tid, emb, conf)
                run_search = True

        # Search identity (with cache)
        person_id, similarity = None, None
        if tid in self._identity_cache and not run_search:
            person_id, similarity = self._identity_cache[tid]
        else:
            if emb is not None:
                match = self.searcher.search(emb)
                if match:
                    person_id, similarity = match
                self._identity_cache[tid] = (person_id, similarity)

        # Anti-spoof / Liveness detection (with cache)
        is_real = False
        spoof_score = 1.0
        
        if tid in self._liveness_cache:
            is_real, spoof_score = self._liveness_cache[tid]
        else:
            if self.anti_spoof._model is not None:
                is_real, spoof_score = self.anti_spoof.is_real(aligned)
                self._liveness_cache[tid] = (is_real, spoof_score)
            else:
                # Fallback to blink detection if MiniFASNet model is not loaded
                if tid in self._real_tracks:
                    is_real = True
                    spoof_score = 1.0
                    self._liveness_cache[tid] = (is_real, spoof_score)
                elif track.get("mesh_points") is not None:
                    pts = track["mesh_points"]
                    
                    def eye_aspect_ratio(eye_pts):
                        A = np.linalg.norm(eye_pts[1] - eye_pts[5])
                        B = np.linalg.norm(eye_pts[2] - eye_pts[4])
                        C = np.linalg.norm(eye_pts[0] - eye_pts[3])
                        return (A + B) / (2.0 * C) if C > 0 else 0
                    
                    if len(pts) >= 474:
                        right_eye_pts = np.array([pts[33], pts[160], pts[158], pts[133], pts[153], pts[144]])
                        left_eye_pts = np.array([pts[362], pts[385], pts[387], pts[263], pts[373], pts[380]])
                        
                        ear_right = eye_aspect_ratio(right_eye_pts)
                        ear_left = eye_aspect_ratio(left_eye_pts)
                        ear = (ear_right + ear_left) / 2.0
                        
                        if tid not in self._blink_history:
                            self._blink_history[tid] = []
                        
                        self._blink_history[tid].append(ear)
                        if len(self._blink_history[tid]) > 30:
                            self._blink_history[tid].pop(0)
                        
                        hist = self._blink_history[tid]
                        
                        # Apply moving average (window=3) to reduce jitter on static photos
                        if len(hist) >= 3:
                            smoothed_hist = [sum(hist[i:i+3])/3.0 for i in range(len(hist)-2)]
                            
                            if len(smoothed_hist) >= 10:
                                min_ear = min(smoothed_hist)
                                max_ear = max(smoothed_hist)
                                
                                # Requires a significant drop and rise (valley)
                                if min_ear < 0.21 and max_ear > 0.27 and (max_ear - min_ear) > 0.07:
                                    min_idx = smoothed_hist.index(min_ear)
                                    
                                    # The minimum must not be at the very edges (must open -> close -> open)
                                    if 2 <= min_idx <= len(smoothed_hist) - 3:
                                        before_max = max(smoothed_hist[:min_idx])
                                        after_max = max(smoothed_hist[min_idx+1:])
                                        
                                        if before_max > 0.26 and after_max > 0.26:
                                            self._real_tracks.add(tid)
                                            is_real = True
                                            self._liveness_cache[tid] = (is_real, spoof_score)

        spoof_score = 1.0 if is_real else 0.0

        return FaceResult(
            track_id=tid,
            bbox=bbox,
            conf=conf,
            person_id=person_id,
            similarity=similarity,
            is_real=is_real,
            spoof_score=spoof_score,
            aligned_face=aligned,
        )

    def _compute_fps(self) -> float:
        now = time.perf_counter()
        if self._last_time == 0.0:
            self._last_time = now
            return 0.0
        elapsed = now - self._last_time
        self._last_time = now
        if elapsed > 0:
            current_fps = 1.0 / elapsed
            if self._fps == 0.0:
                self._fps = current_fps
            else:
                self._fps = 0.9 * self._fps + 0.1 * current_fps
        return self._fps
