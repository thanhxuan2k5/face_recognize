
from __future__ import annotations

import numpy as np
import faiss
from pathlib import Path
from typing import Optional, Tuple, List

from utils.config import FAISS_INDEX_PATH, EMBEDDING_DIM, RECOGNITION_THRESHOLD
from utils.logger import get_logger

log = get_logger(__name__)


class FaceSearcher:

    def __init__(self,index_path: Path = FAISS_INDEX_PATH,threshold: float = RECOGNITION_THRESHOLD,dim: int = EMBEDDING_DIM,) -> None:
        self.index_path = index_path
        self.threshold = threshold
        self.dim = dim
        self._index = None
        self._id_map: List[int] = []  # position → person_id
        self._load_or_create()


    def _load_or_create(self) -> None:
        if self.index_path.exists():
            try:
                self._index = faiss.read_index(str(self.index_path))
                id_map_path = self.index_path.with_suffix(".ids.npy")
                if id_map_path.exists():
                    loaded_ids = np.load(str(id_map_path))
                    self._id_map = [int(x) for x in loaded_ids]
                log.info("FAISS index loaded — %d vectors.", self._index.ntotal)
                return
            except Exception as exc:
                log.warning("Could not load FAISS index (%s); creating new.", exc)
        self._index = faiss.IndexFlatIP(self.dim)
        self._id_map = []
        log.info("New FAISS index created (dim=%d).", self.dim)


    def save(self) -> None:
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))
            # Save as int32 for compatibility
            np.save(str(self.index_path.with_suffix(".ids.npy")), np.array(self._id_map, dtype=np.int32))
            log.info("FAISS index saved (%d vectors).", self._index.ntotal)
        except Exception as exc:
            log.error("Failed to save FAISS index: %s", exc)


    def add(self, person_id: int, embedding: np.ndarray) -> None:
        vec = embedding.astype(np.float32).reshape(1, -1)
        self._index.add(vec)
        # Force conversion to Python int
        self._id_map.append(int(person_id))

    def remove(self, person_id: int) -> None:
        import faiss
        if not self._id_map:
            return
        # Collect all remaining vectors
        n = self._index.ntotal
        all_vecs = np.zeros((n, self.dim), dtype=np.float32)
        for i in range(n):
            vec = np.zeros((1, self.dim), dtype=np.float32)
            self._index.reconstruct(i, vec[0])
            all_vecs[i] = vec[0]

        new_ids = []
        new_vecs = []
        for i, pid in enumerate(self._id_map):
            if pid != person_id:
                new_ids.append(pid)
                new_vecs.append(all_vecs[i])

        self._index = faiss.IndexFlatIP(self.dim)
        self._id_map = new_ids
        if new_vecs:
            self._index.add(np.array(new_vecs, dtype=np.float32))
        log.info("Removed person_id=%d from index.", person_id)

    def update(self, person_id: int, embeddings: List[np.ndarray]) -> None:
        self.remove(person_id)
        for emb in embeddings:
            self.add(person_id, emb)

    def search(self, embedding: np.ndarray, top_k: int = 1) -> Optional[Tuple[int, float]]:
        if self._index is None or self._index.ntotal == 0:
            return None
        vec = embedding.astype(np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(vec, k)
        
        best_dist = float(distances[0][0])
        best_idx = int(indices[0][0])
        
        if best_idx < 0 or best_dist < self.threshold:
            return None
            
        # Ensure we return a standard Python int
        person_id = int(self._id_map[best_idx])
        return person_id, best_dist

    @property
    def total(self) -> int:
        return self._index.ntotal if self._index else 0
