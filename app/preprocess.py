from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessedFace:
    bgr: np.ndarray
    display_bgr: np.ndarray
    rgb: np.ndarray
    hsv: np.ndarray
    lab: np.ndarray
    gray: np.ndarray
    focus_mask: np.ndarray
    skin_mask: np.ndarray


def _build_focus_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.ellipse(
        mask,
        (w // 2, int(h * 0.58)),
        (int(w * 0.28), int(h * 0.36)),
        0,
        0,
        360,
        255,
        -1,
    )

    # Keep forehead
    mask[: int(h * 0.10), :] = 0

    # Keep mouth/chin
    mask[int(h * 0.82):, :] = 0

    return mask


def preprocess_face(face_bgr: np.ndarray, output_size: int = 420) -> PreprocessedFace:
    img_bgr = cv2.resize(face_bgr, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

    lab0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a0, b0 = cv2.split(lab0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced_bgr = cv2.cvtColor(cv2.merge([l, a0, b0]), cv2.COLOR_LAB2BGR)

    display_bgr = enhanced_bgr.copy()

    analysis_bgr = cv2.GaussianBlur(enhanced_bgr, (3, 3), 0.4)

    rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2GRAY)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    skin_mask = (((h < 25) & (s > 20) & (v > 50)).astype(np.uint8) * 255)
    focus_mask = _build_focus_mask(gray.shape)

    return PreprocessedFace(
        bgr=analysis_bgr,
        display_bgr=display_bgr,
        rgb=rgb,
        hsv=hsv,
        lab=lab,
        gray=gray,
        focus_mask=focus_mask,
        skin_mask=skin_mask,
    )