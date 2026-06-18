
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

from utils.config import SILENT_FACE_WEIGHT, ANTI_SPOOF_THRESHOLD
from utils.logger import get_logger

log = get_logger(__name__)


class AntiSpoof:

    def __init__(self, weight: Path = SILENT_FACE_WEIGHT, threshold: float = ANTI_SPOOF_THRESHOLD) -> None:
        self.threshold = threshold
        self._model = None
        self._load(weight)

    def _load(self, weight: Path) -> None:
        if not weight.exists():
            log.warning("Anti-spoof weight not found at %s — running pass-through (always REAL).", weight)
            return
        try:
            import torch
            # MiniFASNet is a simple CNN; load state dict
            from core.pipeline._minifas import MiniFASNetV2
            net = MiniFASNetV2()
            state = torch.load(str(weight), map_location="cpu")

            # Handle different state_dict formats
            if isinstance(state, dict):
                if "state_dict" in state:
                    state = state["state_dict"]
                # Remove 'module.' prefix if it exists (from DataParallel)
                new_state = {}
                for k, v in state.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state[name] = v
                state = new_state

            net.load_state_dict(state, strict=True)
            net.eval()
            self._model = net
            log.info("MiniFASNet loaded successfully from %s", weight)
        except Exception as exc:
            log.warning("Could not load anti-spoof model: %s — pass-through active.", exc)

    def is_real(self, face_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Args:
            face_crop: BGR uint8 aligned face (any size, will be resized)
        Returns:
            (is_real: bool, real_score: float 0..1)
        """
        if self._model is None:
            return True, 1.0  # pass-through

        try:
            return self._infer(face_crop)
        except Exception as exc:
            log.error("Anti-spoof inference failed: %s", exc)
            return True, 1.0

    def _infer(self, face: np.ndarray) -> Tuple[bool, float]:
        import torch
        import torch.nn.functional as F

        # MiniFASNet expects 80x80 for V2
        img = cv2.resize(face, (80, 80)).astype(np.float32) / 255.0
        # Normalize (standard torchvision ToTensor scale)
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1)
            # MiniFASNet typically has 3 classes.
            # Class 1 is usually "Real", Class 0 and 2 are different types of spoof.
            real_score = float(probs[0][1])

        is_real = real_score >= self.threshold
        return is_real, real_score
