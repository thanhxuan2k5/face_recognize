"""
database/feature_store.py — High-level façade around FaceSearcher for person management.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import numpy as np

from models.searcher import FaceSearcher
from utils.config import FAISS_INDEX_PATH, RECOGNITION_THRESHOLD
from utils.logger import get_logger

log = get_logger(__name__)


class FeatureStore:

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        threshold: float = RECOGNITION_THRESHOLD,
    ) -> None:
        self._searcher = FaceSearcher(index_path=index_path, threshold=threshold)

    # ── Delegation ────────────────────────────────────────────────────────────

    @property
    def searcher(self) -> FaceSearcher:
        return self._searcher

    def register(self, person_id: int, embeddings: List[np.ndarray]) -> None:
        """Add multiple embeddings for a new or existing person."""
        for emb in embeddings:
            self._searcher.add(person_id, emb)
        self._searcher.save()
        log.info("Registered %d embeddings for person_id=%d.", len(embeddings), person_id)

    def unregister(self, person_id: int) -> None:
        """Remove all embeddings for a person."""
        self._searcher.remove(person_id)
        self._searcher.save()

    def rebuild(self, person_embeddings: dict) -> None:
        """
        Full rebuild: person_embeddings = {person_id: [emb, ...]}
        """
        import faiss
        self._searcher._index = faiss.IndexFlatIP(self._searcher.dim)
        self._searcher._id_map = []
        for pid, embs in person_embeddings.items():
            for emb in embs:
                self._searcher.add(pid, emb)
        self._searcher.save()
        log.info("FAISS index rebuilt with %d total vectors.", self._searcher.total)
