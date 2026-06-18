from __future__ import annotations
import cv2
import queue
import time
from typing import Optional

# from PyQt5.QtCore import QThread, pyqtSignal  # Removed for headless server

# PyQt5 import removed – not needed in headless server

# Camera config imports retained for possible future use
from utils.logger import get_logger

log = get_logger(__name__)


class CameraThread(QThread):

    error_signal: pyqtSignal = pyqtSignal(str)

    def __init__(
        self,
        cam_index: int = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_queue: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None


    def run(self) -> None:
        self._running = True
        self._cap = cv2.VideoCapture(self.cam_index)
        if not self._cap.isOpened():
            msg = f"Không thể mở camera index {self.cam_index}"
            log.error(msg)
            self.error_signal.emit(msg)
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        log.info("Camera thread started (index=%d, %dx%d @ %dfps).", self.cam_index, self.width, self.height, self.fps)

        delay = 1.0 / self.fps
        while self._running:
            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            if not ret:
                log.warning("Frame read failed; retrying…")
                time.sleep(0.1)
                continue


            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put_nowait(frame)

            elapsed = time.perf_counter() - t0
            sleep = delay - elapsed
            if sleep > 0:
                time.sleep(sleep)

        if self._cap:
            self._cap.release()
        log.info("Camera thread stopped.")

    def stop(self) -> None:
        self._running = False
        self.wait()

    @property
    def actual_fps(self) -> float:
        if self._cap and self._cap.isOpened():
            return self._cap.get(cv2.CAP_PROP_FPS)
        return 0.0

    @property
    def resolution(self) -> tuple:
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0
