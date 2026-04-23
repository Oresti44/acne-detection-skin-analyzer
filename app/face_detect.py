from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class FaceDetectionResult:
    bbox: Tuple[int, int, int, int]
    cropped_face: np.ndarray
    face_count: int
    warning: str | None = None


class FaceDetectionError(RuntimeError):
    """Raised when a face cannot be reliably detected."""


def _load_cascade(filename: str) -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + filename
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        
        raise RuntimeError(f"Failed to load cascade: {filename}")
    return cascade


def _preprocess_gray(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def _deduplicate_boxes(boxes: List[Tuple[int, int, int, int]], iou_threshold: float = 0.3):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []

    def iou(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter_area

        return inter_area / union if union > 0 else 0.0

    for box in boxes:
        if all(iou(box, kept_box) < iou_threshold for kept_box in kept):
            kept.append(box)

    return kept


def detect_faces(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = _preprocess_gray(image_bgr)

    frontal = _load_cascade("haarcascade_frontalface_default.xml")
    profile = _load_cascade("haarcascade_profileface.xml")

    all_boxes = []

    frontal_settings = [
        {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (60, 60)},
        {"scaleFactor": 1.05, "minNeighbors": 4, "minSize": (50, 50)},
        {"scaleFactor": 1.03, "minNeighbors": 3, "minSize": (40, 40)},
    ]

    for params in frontal_settings:
        faces = frontal.detectMultiScale(gray, **params)
        all_boxes.extend([tuple(map(int, face)) for face in faces])
        if all_boxes:
            break

    faces_profile = profile.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(50, 50),
    )
    all_boxes.extend([tuple(map(int, face)) for face in faces_profile])

    gray_flipped = cv2.flip(gray, 1)
    faces_profile_flipped = profile.detectMultiScale(
        gray_flipped,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(50, 50),
    )

    img_w = gray.shape[1]
    for (x, y, w, h) in faces_profile_flipped:
        corrected_x = img_w - (x + w)
        all_boxes.append((int(corrected_x), int(y), int(w), int(h)))

    all_boxes = _deduplicate_boxes(all_boxes)
    return sorted(all_boxes, key=lambda b: b[2] * b[3], reverse=True)


def detect_main_face(image_bgr: np.ndarray, padding_ratio: float = 0.12) -> FaceDetectionResult:
    faces = detect_faces(image_bgr)
    if not faces:
        raise FaceDetectionError(
            "No face detected. Try a clearer image, bring the face closer, and avoid heavy side angles."
        )

    x, y, w, h = faces[0]
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(image_bgr.shape[1], x + w + pad_w)
    y1 = min(image_bgr.shape[0], y + h + pad_h)

    cropped = image_bgr[y0:y1, x0:x1].copy()

    warning = None
    if len(faces) > 1:
        warning = f"Multiple faces found ({len(faces)}). Using the largest face only."

    return FaceDetectionResult(
        bbox=(x0, y0, x1 - x0, y1 - y0),
        cropped_face=cropped,
        face_count=len(faces),
        warning=warning,
    )