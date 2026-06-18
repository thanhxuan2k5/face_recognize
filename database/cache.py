"""
database/cache.py — In-memory LRU embedding cache (per track_id).
Avoids re-computing ArcFace embeddings every frame for the same tracked face.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


class EmbeddingCache:
    """LRU cache keyed by track_id."""

    def __init__(self, max_size: int = 128) -> None:
        self._cache: OrderedDict[int, Tuple[np.ndarray, float]] = OrderedDict()
        self._max = max_size

    def get(self, track_id: int) -> Optional[np.ndarray]:
        if track_id not in self._cache:
            return None
        self._cache.move_to_end(track_id)
        return self._cache[track_id][0]

    def get_conf(self, track_id: int) -> float:
        if track_id not in self._cache:
            return 0.0
        return self._cache[track_id][1]

    def set(self, track_id: int, embedding: np.ndarray, conf: float = 0.0) -> None:
        if track_id in self._cache:
            self._cache.move_to_end(track_id)
        self._cache[track_id] = (embedding, conf)
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)

    def invalidate(self, track_id: int) -> None:
        self._cache.pop(track_id, None)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
