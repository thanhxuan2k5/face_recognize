
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout, QComboBox, QSplitter, QDateEdit,
    QAbstractItemView, QHeaderView, QSizePolicy,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QDate, pyqtSignal

from database.db_manager import DBManager, Person
from database.feature_store import FeatureStore
from utils.config import FACES_DIR
from utils.logger import get_logger

log = get_logger(__name__)


class ManageWidget(QWidget):
    """Full CRUD widget for person management."""

    persons_changed = pyqtSignal()   # emitted after add/edit/delete

    def __init__(
        self,
        db: DBManager,
        feature_store: FeatureStore,
        embedder=None,   # FaceEmbedder | None
        detector=None,   # FaceDetector | None
        aligner=None,    # FaceAligner | None
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._db = db
        self._fs = feature_store
        self._embedder = embedder
        self._detector = detector
        self._aligner = aligner
        self._editing_id: Optional[int] = None
        self._selected_photo: Optional[Path] = None
        self._build_ui()
        self.refresh_table()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)

        # Left: table + search
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        search_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Tìm theo tên / CMND…")
        self._search_edit.textChanged.connect(self.refresh_table)
        btn_add = QPushButton("+ Thêm người")
        btn_add.setProperty("cssClass", "success")
        btn_add.clicked.connect(self._new_form)
        search_row.addWidget(self._search_edit)
        search_row.addWidget(btn_add)
        left_layout.addLayout(search_row)

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Ảnh", "Họ tên", "Ngày sinh", "CMND", "Thao tác"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.setColumnWidth(0, 48)
        self._table.setColumnWidth(2, 100)
        self._table.setColumnWidth(3, 120)
        self._table.setColumnWidth(4, 100)
        self._table.setRowHeight(0, 46)
        left_layout.addWidget(self._table)
        splitter.addWidget(left)

        # Right: form
        right = QGroupBox("Thêm / Sửa người")
        right.setFixedWidth(280)
        form_layout = QVBoxLayout(right)
        form_layout.setSpacing(10)

        # Photo picker
        photo_row = QHBoxLayout()
        self._photo_preview = QLabel("+ Ảnh")
        self._photo_preview.setFixedSize(84, 84)
        self._photo_preview.setAlignment(Qt.AlignCenter)
        self._photo_preview.setStyleSheet(
            "background:#e2e8f0; border:1px dashed #94a3b8; border-radius:8px; color: #475569;"
        )
        self._photo_preview.mousePressEvent = lambda _: self._pick_photo()
        photo_row.addStretch()
        photo_row.addWidget(self._photo_preview)
        photo_row.addStretch()
        form_layout.addLayout(photo_row)

        # Form fields
        fields_form = QFormLayout()
        fields_form.setSpacing(8)

        self._f_name = QLineEdit()
        self._f_name.setPlaceholderText("Họ tên")
        fields_form.addRow("Họ tên *", self._f_name)

        self._f_dob = QDateEdit()
        self._f_dob.setDisplayFormat("dd/MM/yyyy")
        self._f_dob.setDate(QDate(1990, 1, 1))
        self._f_dob.setCalendarPopup(True)
        fields_form.addRow("Ngày sinh", self._f_dob)

        self._f_id = QLineEdit()
        self._f_id.setPlaceholderText("CMND / CCCD")
        fields_form.addRow("CMND/CCCD", self._f_id)

        self._f_gender = QComboBox()
        self._f_gender.addItems(["Nam", "Nữ", "Khác"])
        fields_form.addRow("Giới tính", self._f_gender)

        self._f_phone = QLineEdit()
        self._f_phone.setPlaceholderText("Số điện thoại")
        fields_form.addRow("Điện thoại", self._f_phone)

        form_layout.addLayout(fields_form)

        btn_row = QHBoxLayout()
        self._btn_save = QPushButton("Lưu")
        self._btn_save.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                color: white;
                border-radius: 8px;
                padding: 10px;
                font-weight: 500;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #16a34a; }
        """)
        self._btn_save.clicked.connect(self._save_form)
        
        btn_cancel = QPushButton("Hủy")
        btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #e2e8f0;
                color: #475569;
                border-radius: 8px;
                padding: 10px;
                font-weight: 500;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #cbd5e1; }
        """)
        btn_cancel.clicked.connect(self._clear_form)
        btn_row.addWidget(self._btn_save)
        btn_row.addWidget(btn_cancel)
        form_layout.addLayout(btn_row)
        form_layout.addStretch()
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        root.addWidget(splitter)

    # ── Table ─────────────────────────────────────────────────────────────────

    def refresh_table(self) -> None:
        search = self._search_edit.text().strip()
        persons = self._db.list_persons(search)
        self._table.setRowCount(len(persons))
        for row, p in enumerate(persons):
            self._table.setRowHeight(row, 46)

            # Photo thumbnail
            thumb = QLabel()
            thumb.setAlignment(Qt.AlignCenter)
            if p.photo_path:
                pp = FACES_DIR / p.photo_path
                if pp.exists():
                    pix = QPixmap(str(pp)).scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    thumb.setPixmap(pix)
                else:
                    thumb.setText("👤")
            else:
                thumb.setText("👤")
            self._table.setCellWidget(row, 0, thumb)

            self._table.setItem(row, 1, QTableWidgetItem(p.full_name))
            self._table.setItem(row, 2, QTableWidgetItem(p.dob or ""))
            self._table.setItem(row, 3, QTableWidgetItem(p.id_number or ""))

            # Action buttons
            actions = QWidget()
            act_layout = QHBoxLayout(actions)
            act_layout.setContentsMargins(4, 4, 4, 4)
            act_layout.setSpacing(4)

            btn_edit = QPushButton("Sửa")
            btn_edit.setFixedHeight(26)
            btn_edit.setProperty("cssClass", "")
            btn_edit.setStyleSheet(
                "background:#3b82f6; color:#ffffff; border:none;"
                " border-radius:5px; font-size:12px; padding:0 8px;"
            )
            btn_edit.clicked.connect(lambda _, pid=p.id: self._load_form(pid))

            btn_del = QPushButton("Xóa")
            btn_del.setFixedHeight(26)
            btn_del.setStyleSheet(
                "background:#ef4444; color:#ffffff; border:none;"
                " border-radius:5px; font-size:12px; padding:0 8px;"
            )
            btn_del.clicked.connect(lambda _, pid=p.id: self._delete(pid))

            act_layout.addWidget(btn_edit)
            act_layout.addWidget(btn_del)
            self._table.setCellWidget(row, 4, actions)

    # ── Form actions ──────────────────────────────────────────────────────────

    def _new_form(self) -> None:
        self._editing_id = None
        self._clear_form()

    def _clear_form(self) -> None:
        self._editing_id = None
        self._selected_photo = None
        self._f_name.clear()
        self._f_dob.setDate(QDate(1990, 1, 1))
        self._f_id.clear()
        self._f_gender.setCurrentIndex(0)
        self._f_phone.clear()
        self._photo_preview.clear()
        self._photo_preview.setText("+ Ảnh")

    def _load_form(self, person_id: int) -> None:
        p = self._db.get_person(person_id)
        if not p:
            return
        self._editing_id = person_id
        self._f_name.setText(p.full_name)
        if p.dob:
            try:
                d = QDate.fromString(p.dob, "yyyy-MM-dd")
                self._f_dob.setDate(d)
            except Exception:
                pass
        self._f_id.setText(p.id_number or "")
        idx = self._f_gender.findText(p.gender or "Nam")
        self._f_gender.setCurrentIndex(max(0, idx))
        self._f_phone.setText(p.phone or "")
        if p.photo_path:
            pp = FACES_DIR / p.photo_path
            if pp.exists():
                pix = QPixmap(str(pp)).scaled(84, 84, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._photo_preview.setPixmap(pix)
                self._selected_photo = pp

    def _pick_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return
        self._selected_photo = Path(path)
        pix = QPixmap(path).scaled(84, 84, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self._photo_preview.setPixmap(pix)

    def _save_form(self) -> None:
        name = self._f_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập họ tên.")
            return

        dob_str = self._f_dob.date().toString("yyyy-MM-dd")
        person = Person(
            id=self._editing_id,
            full_name=name,
            dob=dob_str,
            id_number=self._f_id.text().strip() or None,
            gender=self._f_gender.currentText(),
            phone=self._f_phone.text().strip() or None,
        )

        if self._editing_id is None:
            pid = self._db.add_person(person)
            person.id = pid
        else:
            pid = self._editing_id

        # Save photo
        if self._selected_photo and self._selected_photo.exists():
            dest_name = f"{pid}{self._selected_photo.suffix}"
            dest = FACES_DIR / dest_name
            if self._selected_photo != dest:
                shutil.copy2(str(self._selected_photo), str(dest))
            person.photo_path = dest_name
            self._db.update_person(person)

        # Register embedding
        if self._embedder and self._selected_photo:
            success = self._register_embedding(pid, self._selected_photo)
            if not success:
                QMessageBox.warning(self, "Cảnh báo", "Không tìm thấy khuôn mặt trong ảnh đã chọn. Vui lòng chọn ảnh khác.")
                # Optional: delete the person if first time
                return

        self._clear_form()
        self.refresh_table()
        self.persons_changed.emit()
        log.info("Person saved: id=%d name=%s", pid, name)

    def _delete(self, person_id: int) -> None:
        p = self._db.get_person(person_id)
        if not p:
            return
        reply = QMessageBox.question(
            self, "Xác nhận", f"Xóa người '{p.full_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self._db.delete_person(person_id)
        self._fs.unregister(person_id)
        self.refresh_table()
        self.persons_changed.emit()

    # ── Embedding registration ────────────────────────────────────────────────

    def _register_embedding(self, person_id: int, photo_path: Path) -> bool:
        """
        Detects face, aligns it properly, and registers the embedding.
        Returns True if successful.
        """
        try:
            import cv2
            img = cv2.imread(str(photo_path))
            if img is None:
                return False

            # 1. Detect face
            faces = []
            if self._detector:
                faces = self._detector.detect(img)
            
            if not faces:
                log.warning("No face detected in %s", photo_path)
                return False

            # Take the largest face
            best_face = max(faces, key=lambda f: (f["bbox"][2]-f["bbox"][0]) * (f["bbox"][3]-f["bbox"][1]))
            bbox = best_face["bbox"]
            kps = best_face.get("kps")

            # 2. Align face
            if self._aligner:
                aligned = self._aligner.align(img, kps=kps, bbox=bbox)
            else:
                # Fallback to simple crop
                x1, y1, x2, y2 = bbox
                aligned = cv2.resize(img[y1:y2, x1:x2], (112, 112))

            # 3. Embed and register
            emb = self._embedder.embed(aligned)
            if emb is not None:
                self._fs.register(person_id, [emb])
                log.info("Embedding registered for person_id=%d", person_id)
                return True
            
            return False
        except Exception as exc:
            log.error("Failed to register embedding: %s", exc)
            return False
