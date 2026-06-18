
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QSizePolicy, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, pyqtSlot

from database.db_manager import Person
from utils.config import FACES_DIR


class PersonPanel(QWidget):


    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(280)
        self._build_ui()
        self.clear()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.frame = QFrame()
        self.frame.setObjectName("PersonPanelFrame")
        main_layout.addWidget(self.frame)

        layout = QVBoxLayout(self.frame)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Title
        title = QLabel("Thông tin người nhận diện")
        title.setStyleSheet("font-size: 14px; font-weight: 600; color: #334155; padding-bottom: 6px;")
        layout.addWidget(title)

        # Photo
        self._photo_lbl = QLabel()
        self._photo_lbl.setFixedSize(120, 120)
        self._photo_lbl.setAlignment(Qt.AlignCenter)
        self._photo_lbl.setStyleSheet(
            "background-color: #e2e8f0; border: 1px solid #cbd5e1; border-radius: 10px; color: #475569;"
        )
        photo_row = QHBoxLayout()
        photo_row.addStretch()
        photo_row.addWidget(self._photo_lbl)
        photo_row.addStretch()
        layout.addLayout(photo_row)

        # Info fields
        self._fields: dict[str, QLabel] = {}
        for key, display in [
            ("full_name", "Họ tên"),
            ("dob", "Ngày sinh"),
            ("id_number", "CMND/CCCD"),
            ("gender", "Giới tính"),
            ("phone", "Điện thoại"),
            ("time", "Thời gian"),
            ("similarity", "Độ tin cậy"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl_key = QLabel(f"{display}:")
            lbl_key.setFixedWidth(90)
            lbl_key.setStyleSheet("color: #475569; font-size: 12px;")
            lbl_val = QLabel("—")
            lbl_val.setStyleSheet("color: #0f172a; font-size: 12px; font-weight: 500;")
            lbl_val.setWordWrap(True)
            row.addWidget(lbl_key)
            row.addWidget(lbl_val, 1)
            layout.addLayout(row)
            self._fields[key] = lbl_val

        # Status badge
        self._status_lbl = QLabel("Chưa nhận diện")
        self._status_lbl.setObjectName("lbl_status_unknown")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_lbl)
        layout.addStretch()


    def show_person(self, person: Person, similarity: float, timestamp: str, is_real: bool) -> None:
        if not is_real:
            self._status_lbl.setText("⚠ Ảnh giả mạo")
            self._status_lbl.setObjectName("lbl_status_spoof")
            self._status_lbl.setStyle(self._status_lbl.style())
            return

        self._set_field("full_name", person.full_name)
        self._set_field("dob", person.dob or "—")
        self._set_field("id_number", person.id_number or "—")
        self._set_field("gender", person.gender or "—")
        self._set_field("phone", person.phone or "—")
        self._set_field("time", timestamp)
        self._set_field("similarity", f"{int(similarity * 100)}%")

        # Photo
        if person.photo_path:
            photo_path = FACES_DIR / person.photo_path
            if photo_path.exists():
                pix = QPixmap(str(photo_path)).scaled(
                    120, 120, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
                )
                self._photo_lbl.setPixmap(pix)
            else:
                self._photo_lbl.setText("Ảnh")
        else:
            self._photo_lbl.setText("Ảnh")

        self._status_lbl.setText("✓ Đã nhận diện")
        self._status_lbl.setObjectName("lbl_status_ok")
        self._status_lbl.setStyle(self._status_lbl.style())

    def clear(self) -> None:
        for lbl in self._fields.values():
            lbl.setText("—")
        self._photo_lbl.clear()
        self._photo_lbl.setText("Ảnh")
        self._status_lbl.setText("Chưa nhận diện")
        self._status_lbl.setObjectName("lbl_status_unknown")
        self._status_lbl.setStyle(self._status_lbl.style())

    def _set_field(self, key: str, value: str) -> None:
        if key in self._fields:
            self._fields[key].setText(value)
