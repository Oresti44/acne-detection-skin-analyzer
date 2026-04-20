from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

ALLOWED_SUFFIXES = {'.jpg', '.jpeg', '.png'}


class ImageValidationError(ValueError):
    """Raised when an uploaded image is not suitable for analysis."""



def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def is_supported_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_SUFFIXES



def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError('The uploaded file could not be decoded as an image.')
    return image



def load_image(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found: {image_path}')
    if image_path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ImageValidationError('Unsupported file type. Use JPG or PNG.')

    image = cv2.imread(str(image_path))
    if image is None:
        raise ImageValidationError('Image is corrupted or unreadable.')
    return image



def validate_image_quality(image_bgr: np.ndarray, min_size: int = 256) -> None:
    h, w = image_bgr.shape[:2]
    if h < min_size or w < min_size:
        raise ImageValidationError(
            f'Image is too small ({w}x{h}). Upload at least {min_size}x{min_size}.'
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < 40:
        raise ImageValidationError('Image is too dark. Use a brighter, front-facing photo.')
    if blur_score < 40:
        raise ImageValidationError('Image is too blurry. Upload a sharper photo.')



def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)



def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay[mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    return blended



def draw_bbox(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    output = image_bgr.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return output



def save_image(path: str | Path, image_bgr: np.ndarray) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image_bgr)
    return path
