
from __future__ import annotations

import cv2
import numpy as np
from typing import List, Dict

from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import Qt, pyqtSlot

from core.pipeline.frame_processor import FrameResult, FaceResult
from utils.image_utils import bgr_to_rgb
from database.db_manager import DBManager


# Recognition box colors (BGR for cv2)
_COLOR_KNOWN = (34, 197, 94)    # green
_COLOR_UNKNOWN = (100, 116, 139)  # slate
_COLOR_SPOOF = (239, 68, 68)    # red


class LiveviewWidget(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._fps = 0.0
        self._cam_info = ""
        
        # Database for name lookup
        self._db = DBManager()
        self._name_cache: Dict[int, str] = {} # person_id -> full_name

        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #050709; border-radius: 8px;")

    def set_cam_info(self, info: str) -> None:
        self._cam_info = info

    @pyqtSlot(object)
    def on_result(self, result: FrameResult) -> None:
        self._fps = result.fps
        frame = result.frame.copy()
        self._annotate(frame, result.faces)
        self._show_frame(frame)

    def _get_name(self, person_id: int) -> str:
        """Lấy tên người từ DB hoặc Cache."""
        if person_id in self._name_cache:
            return self._name_cache[person_id]
        
        person = self._db.get_person(person_id)
        name = person.full_name if person else f"ID:{person_id}"
        self._name_cache[person_id] = name
        return name

    def _annotate(self, frame: np.ndarray, faces: List[FaceResult]) -> None:
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]

            if not f.is_real:
                color = _COLOR_SPOOF
                if f.person_id is not None:
                    name = self._get_name(f.person_id)
                    pct = int((f.similarity or 0) * 100)
                    label = f"FAKE - {name} ({pct}%)"
                else:
                    label = "FAKE"
            elif f.person_id is not None:
                color = _COLOR_KNOWN
                name = self._get_name(f.person_id)
                pct = int((f.similarity or 0) * 100)
                label = f"{name} ({pct}%)"
            else:
                color = _COLOR_UNKNOWN
                label = "Unknown"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background (vẽ chữ Tiếng Việt có dấu bằng OpenCV có thể bị lỗi font, 
            # nhưng ta sẽ dùng phương pháp vẽ label đơn giản nhất)
            font = cv2.FONT_HERSHEY_DUPLEX # Dùng font chữ đẹp hơn chút
            scale, th = 0.5, 1
            (tw, th_px), _ = cv2.getTextSize(label, font, scale, th)
            
            # Đảm bảo label không bị văng ra ngoài khung ảnh
            label_y = max(y1, th_px + 10)
            cv2.rectangle(frame, (x1, label_y - th_px - 10), (x1 + tw + 6, label_y), color, -1)
            cv2.putText(frame, label, (x1 + 3, label_y - 4), font, scale, (255, 255, 255), th, cv2.LINE_AA)

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = bgr_to_rgb(frame)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event) -> None:
        if self._pixmap is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        scaled = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
    
    def __del__(self):
        # Đóng kết nối DB khi widget bị hủy
        if hasattr(self, '_db'):
            self._db.close()
