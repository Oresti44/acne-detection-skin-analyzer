from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessedFace:
    bgr: np.ndarray
    hsv: np.ndarray
    lab: np.ndarray
    gray: np.ndarray
    analysis_mask: np.ndarray



def central_face_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    top = int(h * 0.12)
    bottom = int(h * 0.92)
    left = int(w * 0.12)
    right = int(w * 0.88)
    cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)

    # Remove rough eye / lip / edge zones for v1.
    eye_band_top = int(h * 0.18)
    eye_band_bottom = int(h * 0.38)
    cv2.rectangle(mask, (int(w * 0.16), eye_band_top), (int(w * 0.84), eye_band_bottom), 0, -1)

    mouth_top = int(h * 0.72)
    mouth_bottom = int(h * 0.92)
    cv2.rectangle(mask, (int(w * 0.24), mouth_top), (int(w * 0.76), mouth_bottom), 0, -1)

    return mask



def preprocess_face(face_bgr: np.ndarray, output_size: int = 512) -> PreprocessedFace:
    resized = cv2.resize(face_bgr, (output_size, output_size), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    normalized_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    denoised = cv2.GaussianBlur(normalized_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    lab_out = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    mask = central_face_mask(gray.shape)

    return PreprocessedFace(
        bgr=denoised,
        hsv=hsv,
        lab=lab_out,
        gray=gray,
        analysis_mask=mask,
    )
