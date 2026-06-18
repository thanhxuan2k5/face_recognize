"""
ui/styles.py — Light QSS stylesheet for the face recognition app.
"""

LIGHT_STYLE = """
/* ── Global ──────────────────────────────────────────────────────────────── */
QWidget {
    background-color: #ffffff;
    color: #1e293b;
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

/* ── Main Window ─────────────────────────────────────────────────────────── */
QMainWindow {
    background-color: #ffffff;
}

/* ── Panels / GroupBox / Frame ────────────────────────────────────────────── */
QGroupBox, QFrame#PersonPanelFrame {
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background-color: #f8fafc;
}
QGroupBox {
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
    color: #334155;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #475569;
    font-size: 12px;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
QPushButton {
    background-color: #f1f5f9;
    color: #334155;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 7px 18px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #e2e8f0;
    border-color: #94a3b8;
}
QPushButton:pressed {
    background-color: #cbd5e1;
}
QPushButton:disabled {
    color: #94a3b8;
    background-color: #f8fafc;
    border-color: #e2e8f0;
}

/* Primary / action button */
QPushButton#btn_primary, QPushButton[cssClass="primary"] {
    background-color: #3b82f6;
    color: #fff;
    border: none;
    font-weight: 600;
}
QPushButton#btn_primary:hover, QPushButton[cssClass="primary"]:hover {
    background-color: #60a5fa;
}

/* Danger button */
QPushButton#btn_danger, QPushButton[cssClass="danger"] {
    background-color: #ef4444;
    color: #fff;
    border: none;
    font-weight: 600;
}
QPushButton#btn_danger:hover, QPushButton[cssClass="danger"]:hover {
    background-color: #f87171;
}

/* Success button */
QPushButton#btn_success, QPushButton[cssClass="success"] {
    background-color: #22c55e;
    color: #fff;
    border: none;
    font-weight: 600;
}
QPushButton#btn_success:hover, QPushButton[cssClass="success"]:hover {
    background-color: #4ade80;
}

/* ── Line Edits ──────────────────────────────────────────────────────────── */
QLineEdit {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 7px 10px;
    color: #1e293b;
    selection-background-color: #bfdbfe;
}
QLineEdit:focus {
    border-color: #3b82f6;
    background-color: #f8fafc;
}

/* ── ComboBox ────────────────────────────────────────────────────────────── */
QComboBox {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 10px;
    color: #1e293b;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    selection-background-color: #bfdbfe;
    color: #1e293b;
}

/* ── Table ───────────────────────────────────────────────────────────────── */
QTableWidget {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    gridline-color: #e2e8f0;
    selection-background-color: #eff6ff;
    alternate-background-color: #f8fafc;
}
QTableWidget::item {
    padding: 6px 10px;
    border: none;
    color: #334155;
}
QTableWidget::item:selected {
    background-color: #eff6ff;
    color: #1e40af;
}
QHeaderView::section {
    background-color: #f1f5f9;
    color: #475569;
    border: none;
    border-bottom: 1px solid #e2e8f0;
    padding: 8px 10px;
    font-size: 11px;
    font-weight: 600;
}

/* ── Scroll Bars ─────────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #f8fafc;
    width: 8px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

/* ── Labels ──────────────────────────────────────────────────────────────── */
QLabel#lbl_live_badge {
    background-color: transparent;
    color: #ef4444;
    font-size: 12px;
    font-weight: 700;
}
QLabel#lbl_status_ok {
    background-color: #dcfce7;
    color: #166534;
    border: 1px solid #22c55e;
    border-radius: 6px;
    padding: 5px 14px;
    font-weight: 600;
}
QLabel#lbl_status_unknown {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 14px;
    font-weight: 600;
}
QLabel#lbl_status_spoof {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #ef4444;
    border-radius: 6px;
    padding: 5px 14px;
    font-weight: 600;
}

/* ── Custom Tab Buttons ──────────────────────────────────────────────────── */
QPushButton#btn_tab_live, QPushButton#btn_tab_manage {
    background-color: white;
    color: #4b5563;
    border: 1px solid #d1d5db;
    border-radius: 14px;
    padding: 6px 30px;
    font-weight: 600;
}
QPushButton#btn_tab_live:checked {
    background-color: #93c5fd;
    border-color: #3b82f6;
    color: #1e3a8a;
}
QPushButton#btn_tab_manage:checked {
    background-color: #d8b4fe;
    border-color: #a855f7;
    color: #4c1d95;
}

/* ── Date Edit ───────────────────────────────────────────────────────────── */
QDateEdit {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 10px;
    color: #1e293b;
}

/* ── Message Box ─────────────────────────────────────────────────────────── */
QMessageBox {
    background-color: #ffffff;
}
"""
