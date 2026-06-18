
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QStatusBar, QSizePolicy, QFrame, QButtonGroup, QStackedWidget, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QFont

from core.pipeline.camera_thread import CameraThread
from core.pipeline.frame_processor import FrameProcessor, FrameResult, FaceResult
from core.pipeline.tracker import FaceTracker
from core.pipeline.anti_spoof import AntiSpoof
from models.detector import FaceDetector
from models.aligner import FaceAligner
from models.embedder import FaceEmbedder
from database.db_manager import DBManager
from database.feature_store import FeatureStore
from database.cache import EmbeddingCache
from ui.liveview_widget import LiveviewWidget
from ui.person_panel import PersonPanel
from ui.manage_widget import ManageWidget
from utils.logger import get_logger
from utils.config import CAMERA_INDEX, CAMERA_FPS

log = get_logger(__name__)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Camera AI — Nhận diện khuôn mặt")
        self.resize(1100, 720)

        # ── Shared components ─────────────────────────────────────────────────
        self._db = DBManager()
        self._feature_store = FeatureStore()

        # AI models (lazy-loaded to avoid blocking UI startup)
        self._detector: Optional[FaceDetector] = None
        self._aligner = FaceAligner()
        self._embedder: Optional[FaceEmbedder] = None
        self._tracker = FaceTracker()
        self._anti_spoof = AntiSpoof()
        self._embed_cache = EmbeddingCache()

        # Threads
        self._cam_thread: Optional[CameraThread] = None
        self._processor: Optional[FrameProcessor] = None
        
        # State for UI persistence
        self._last_shown_id: Optional[int] = None
        self._last_seen_time: float = 0

        # ── UI ────────────────────────────────────────────────────────────────
        self._build_ui()
        self._setup_status_bar()

        # Deferred model loading + pipeline start
        QTimer.singleShot(200, self._init_pipeline)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(15)

        # Main Title
        main_title = QLabel("Desktop App – Nhận diện khuôn mặt (Wireframe)")
        main_title.setAlignment(Qt.AlignCenter)
        main_title.setStyleSheet("font-size: 22px; color: #1e293b; font-weight: 500;")
        root_layout.addWidget(main_title)

        # Container
        self.container_frame = QFrame()
        self.container_frame.setObjectName("MainContainer")
        self.container_layout = QVBoxLayout(self.container_frame)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(0)
        root_layout.addWidget(self.container_frame, 1)

        # Header Label
        self.header_label = QLabel()
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setObjectName("MainHeader")
        self.header_label.setFixedHeight(40)
        self.container_layout.addWidget(self.header_label)

        # Content area
        content_area = QWidget()
        content_area.setStyleSheet("background-color: #ffffff; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px;")
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(15, 15, 15, 15)
        self.container_layout.addWidget(content_area, 1)

        # Custom Tab Buttons
        tab_row = QHBoxLayout()
        self.btn_tab_live = QPushButton("Liveview")
        self.btn_tab_manage = QPushButton("Quản lý người")
        self.btn_tab_live.setCheckable(True)
        self.btn_tab_manage.setCheckable(True)
        self.btn_tab_live.setObjectName("btn_tab_live")
        self.btn_tab_manage.setObjectName("btn_tab_manage")
        self.btn_tab_live.setCursor(Qt.PointingHandCursor)
        self.btn_tab_manage.setCursor(Qt.PointingHandCursor)
        
        self.tab_group = QButtonGroup(self)
        self.tab_group.addButton(self.btn_tab_live, 0)
        self.tab_group.addButton(self.btn_tab_manage, 1)
        self.tab_group.buttonClicked[int].connect(self._switch_tab)

        tab_row.addWidget(self.btn_tab_live)
        tab_row.addWidget(self.btn_tab_manage)
        tab_row.addStretch()
        content_layout.addLayout(tab_row)

        # Stacked Widget
        self.stacked_widget = QStackedWidget()
        content_layout.addWidget(self.stacked_widget, 1)

        # Tab 1: Liveview
        self._liveview_tab = QWidget()
        self._build_liveview_tab()
        self.stacked_widget.addWidget(self._liveview_tab)

        # Tab 2: Manage
        self._manage_widget = ManageWidget(
            db=self._db,
            feature_store=self._feature_store,
            embedder=None,
            detector=None,
            aligner=self._aligner,
        )
        self.stacked_widget.addWidget(self._manage_widget)
        
        # Init state
        self.btn_tab_live.setChecked(True)
        self._switch_tab(0)

    def _switch_tab(self, index: int) -> None:
        self.stacked_widget.setCurrentIndex(index)
        if index == 0:
            self.header_label.setText("Camera AI — Nhận diện khuôn mặt (Liveview)")
            self.container_frame.setStyleSheet("QFrame#MainContainer { border: 1px solid #93c5fd; border-radius: 8px; background-color: #ffffff; }")
            self.header_label.setStyleSheet("QLabel#MainHeader { background-color: #e0f2fe; color: #1e293b; font-size: 15px; border-bottom: 1px solid #93c5fd; border-top-left-radius: 7px; border-top-right-radius: 7px; }")
        else:
            self.header_label.setText("Camera AI — Quản lý người (CRUD)")
            self.container_frame.setStyleSheet("QFrame#MainContainer { border: 1px solid #c084fc; border-radius: 8px; background-color: #ffffff; }")
            self.header_label.setStyleSheet("QLabel#MainHeader { background-color: #f3e8ff; color: #1e293b; font-size: 15px; border-bottom: 1px solid #c084fc; border-top-left-radius: 7px; border-top-right-radius: 7px; }")

    def _build_liveview_tab(self) -> None:
        layout = QHBoxLayout(self._liveview_tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(15)

        # Camera feed
        feed_container = QWidget()
        feed_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        feed_container.setStyleSheet("background-color: #1e293b; border-radius: 8px;")
        feed_layout = QVBoxLayout(feed_container)
        feed_layout.setContentsMargins(12, 12, 12, 12)

        # LIVE badge row
        badge_row = QHBoxLayout()
        self._live_badge = QLabel("● LIVE")
        self._live_badge.setObjectName("lbl_live_badge")
        
        self.btn_toggle_cam = QPushButton("Tắt Camera")
        self.btn_toggle_cam.setCheckable(True)
        self.btn_toggle_cam.setStyleSheet("""
            QPushButton {
                background-color: #ef4444; color: white; border-radius: 4px; padding: 4px 10px; font-weight: bold; font-size: 11px;
            }
            QPushButton:checked {
                background-color: #22c55e;
            }
        """)
        self.btn_toggle_cam.clicked.connect(self._toggle_camera)

        badge_row.addWidget(self._live_badge)
        badge_row.addWidget(self.btn_toggle_cam)
        badge_row.addStretch()
        feed_layout.addLayout(badge_row)

        self._liveview = LiveviewWidget()
        self._liveview.setStyleSheet("background-color: transparent;")
        feed_layout.addWidget(self._liveview, 1)
        
        # FPS row at bottom
        fps_row = QHBoxLayout()
        self._fps_label = QLabel("—")
        self._fps_label.setStyleSheet("color: #94a3b8; font-size: 12px; background: transparent;")
        fps_row.addWidget(self._fps_label)
        fps_row.addStretch()
        feed_layout.addLayout(fps_row)

        layout.addWidget(feed_container, 1)

        # Right panel
        self._person_panel = PersonPanel()
        layout.addWidget(self._person_panel)

    def _setup_status_bar(self) -> None:
        sb = QStatusBar()
        sb.setStyleSheet("color: #64748b; font-size: 12px; background: #0a0d14;")
        self.setStatusBar(sb)
        self._status_bar = sb
        self._status_bar.showMessage("Đang khởi động…")

    # ── Pipeline init ─────────────────────────────────────────────────────────

    def _init_pipeline(self) -> None:
        self._status_bar.showMessage("Đang tải mô hình AI…")
        try:
            self._detector = FaceDetector()
            self._embedder = FaceEmbedder()
            self._manage_widget._embedder = self._embedder
            self._manage_widget._detector = self._detector
        except Exception as exc:
            log.error("Model loading error: %s", exc)
            self._status_bar.showMessage(f"Lỗi tải mô hình: {exc}")

        # Sync FAISS index with SQLite (remove ghost IDs)
        try:
            db_ids = {p.id for p in self._db.list_persons()}
            faiss_ids = set(self._feature_store.searcher._id_map)
            ghost_ids = faiss_ids - db_ids
            if ghost_ids:
                log.warning("Found ghost IDs in FAISS index: %s. Cleaning up...", ghost_ids)
                for gid in ghost_ids:
                    self._feature_store.unregister(gid)
        except Exception as exc:
            log.error("Failed to sync FAISS and SQLite: %s", exc)

        self._start_camera()

    def _start_camera(self) -> None:
        self._cam_thread = CameraThread()
        self._cam_thread.error_signal.connect(self._on_camera_error)
        self._cam_thread.start()

        self._processor = FrameProcessor(
            frame_queue=self._cam_thread.frame_queue,
            detector=self._detector,
            aligner=self._aligner,
            embedder=self._embedder,
            searcher=self._feature_store.searcher,
            tracker=self._tracker,
            anti_spoof=self._anti_spoof,
            embed_cache=self._embed_cache,
        )
        self._processor.result_ready.connect(self._on_frame_result)
        self._processor.start()

        cam_info = f"Webcam {CAMERA_INDEX}  ·  {CAMERA_FPS} FPS"
        self._liveview.set_cam_info(cam_info)
        self._status_bar.showMessage(f"Camera {CAMERA_INDEX} đang chạy")
        log.info("Pipeline started.")

    def _toggle_camera(self) -> None:
        if self.btn_toggle_cam.isChecked():
            # Turn OFF
            self.btn_toggle_cam.setText("Bật Camera")
            self._live_badge.setStyleSheet("color: #64748b;") # grey out
            if self._processor:
                self._processor.stop()
            if self._cam_thread:
                self._cam_thread.stop()
            self._status_bar.showMessage("Camera đã tắt")
        else:
            # Turn ON
            self.btn_toggle_cam.setText("Tắt Camera")
            self._live_badge.setStyleSheet("") # reset
            self._start_camera()

    # ── Slots ─────────────────────────────────────────────────────────────────

    @pyqtSlot(object)
    def _on_frame_result(self, result: FrameResult) -> None:
        self._liveview.on_result(result)
        self._fps_label.setText(f"Webcam {CAMERA_INDEX} · {result.fps:.0f} FPS")

        # Update person panel with highest-confidence known face
        best: Optional[FaceResult] = None
        for f in result.faces:
            if f.person_id is not None:
                if best is None or (f.similarity or 0) > (best.similarity or 0):
                    best = f

        if best and best.person_id is not None:
            # DEBUG: See what's coming from AI
            log.debug("Found face with person_id=%s, sim=%.2f", best.person_id, best.similarity or 0)
            
            # Fetch from DB
            person = self._db.get_person(best.person_id)
            if person:
                ts = datetime.now().strftime("%H:%M:%S  %d/%m")
                self._person_panel.show_person(person, best.similarity or 0, ts, best.is_real)
                self._last_shown_id = best.person_id
                self._last_seen_time = time.time()
            else:
                log.warning("AI found person_id=%d, but it does NOT exist in SQLite database!", best.person_id)
        else:
            # If no known face in this frame, wait 3 seconds before clearing
            if self._last_shown_id is not None:
                if time.time() - self._last_seen_time > 3.0:
                    self._person_panel.clear()
                    self._last_shown_id = None

    @pyqtSlot(str)
    def _on_camera_error(self, msg: str) -> None:
        self._status_bar.showMessage(f"Lỗi camera: {msg}")
        log.error("Camera error: %s", msg)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        log.info("Shutting down…")
        if self._processor:
            self._processor.stop()
        if self._cam_thread:
            self._cam_thread.stop()
        self._db.close()
        self._feature_store.searcher.save()
        event.accept()
