
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from utils.config import DB_PATH
from utils.logger import get_logger

log = get_logger(__name__)

_CREATE_PERSONS = """
CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name   TEXT    NOT NULL,
    dob         TEXT,           -- YYYY-MM-DD
    id_number   TEXT,           -- CMND / CCCD
    gender      TEXT,           -- Nam / Nữ / Khác
    phone       TEXT,
    photo_path  TEXT,           -- relative to FACES_DIR
    created_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
);
"""


@dataclass
class Person:
    id: Optional[int]
    full_name: str
    dob: Optional[str] = None
    id_number: Optional[str] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    photo_path: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Person":
        return cls(**{k: row[k] for k in row.keys()})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "full_name": self.full_name,
            "dob": self.dob,
            "id_number": self.id_number,
            "gender": self.gender,
            "phone": self.phone,
            "photo_path": self.photo_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class DBManager:

    def __init__(self, db_path: Path = DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()
        log.info("SQLite database ready at %s", db_path)

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(_CREATE_PERSONS)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add_person(self, p: Person) -> int:
        """Insert a new person; returns new row id."""
        sql = """
            INSERT INTO persons (full_name, dob, id_number, gender, phone, photo_path)
            VALUES (:full_name, :dob, :id_number, :gender, :phone, :photo_path)
        """
        with self._conn:
            cur = self._conn.execute(sql, p.to_dict())
        pid = cur.lastrowid
        log.info("Added person id=%d name=%s", pid, p.full_name)
        return pid

    def get_person(self, person_id: int) -> Optional[Person]:
        row = self._conn.execute("SELECT * FROM persons WHERE id=?", (person_id,)).fetchone()
        return Person.from_row(row) if row else None

    def list_persons(self, search: str = "") -> List[Person]:
        if search:
            q = f"%{search}%"
            rows = self._conn.execute(
                "SELECT * FROM persons WHERE full_name LIKE ? OR id_number LIKE ? ORDER BY full_name",
                (q, q),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM persons ORDER BY full_name").fetchall()
        return [Person.from_row(r) for r in rows]

    def update_person(self, p: Person) -> None:
        sql = """
            UPDATE persons
            SET full_name=:full_name, dob=:dob, id_number=:id_number,
                gender=:gender, phone=:phone, photo_path=:photo_path,
                updated_at=datetime('now','localtime')
            WHERE id=:id
        """
        with self._conn:
            self._conn.execute(sql, p.to_dict())
        log.info("Updated person id=%d", p.id)

    def delete_person(self, person_id: int) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
        log.info("Deleted person id=%d", person_id)

    def close(self) -> None:
        self._conn.close()
