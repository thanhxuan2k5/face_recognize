
from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from typing import Optional

from utils.config import ARCFACE_WEIGHT, EMBEDDING_DIM, FACE_ALIGN_SIZE
from utils.logger import get_logger

log = get_logger(__name__)


def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


class FaceEmbedder:

    def __init__(self, weight: Path = ARCFACE_WEIGHT) -> None:
        self._model = None
        self._backend = None
        self._load(weight)

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load(self, weight: Path) -> None:
        try:
            self._load_insightface()
        except Exception as exc:
            log.warning("insightface embedder failed (%s); trying PyTorch.", exc)
            try:
                self._load_torch(weight)
            except Exception as exc2:
                log.error("PyTorch embedder also failed: %s", exc2)

    def _load_insightface(self) -> None:
        from insightface.model_zoo import get_model
        # buffalo_l ships with w600k_r50 recognition model
        import insightface
        handler = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
        )
        handler.prepare(ctx_id=-1)
        self._rec_model = handler.models.get("recognition")
        if self._rec_model is None:
            raise RuntimeError("No recognition model in buffalo_l pack.")
        self._backend = "insightface"
        log.info("ArcFace (insightface/buffalo_l) loaded.")

    def _load_torch(self, weight: Path) -> None:
        import torch
        from torchvision import models as tvm

        if not weight.exists():
            raise FileNotFoundError(f"ArcFace weight not found: {weight}")
        # Simple wrapper — expects the weight to be a state dict for iresnet100
        try:
            from models._arcface_net import iresnet100
            net = iresnet100(pretrained=False)
        except ImportError:
            import torchvision.models as tvm
            net = tvm.resnet50(pretrained=False)
            net.fc = torch.nn.Linear(2048, EMBEDDING_DIM)

        state = torch.load(str(weight), map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state, strict=False)
        net.eval()
        self._model = net
        self._backend = "torch"
        log.info("ArcFace (PyTorch) loaded from %s", weight)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        if self._backend is None:
            log.error("No embedder available.")
            return None
        try:
            if self._backend == "insightface":
                return self._embed_insightface(aligned_face)
            return self._embed_torch(aligned_face)
        except Exception as exc:
            log.error("Embedding failed: %s", exc)
            return None

    def _embed_insightface(self, face: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        emb = self._rec_model.get_feat(rgb).flatten()
        return _l2_norm(emb.astype(np.float32))

    def _embed_torch(self, face: np.ndarray) -> np.ndarray:
        import torch
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,112,112)
        with torch.no_grad():
            emb = self._model(tensor).squeeze().numpy()
        return _l2_norm(emb.astype(np.float32))
