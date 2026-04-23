import cv2
import numpy as np
from dataclasses import dataclass

from app.preprocess import PreprocessedFace


@dataclass
class AcneFeatureResult:
    features: dict
    red_mask: np.ndarray
    redness_map: np.ndarray
    dog_map: np.ndarray
    lap_map: np.ndarray
    candidate_mask: np.ndarray

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return (arr * 255).astype(np.uint8)

def _inside_ellipse(cx, cy, ex, ey, ax, ay):
    if ax <= 0 or ay <= 0:
        return False
    return ((cx - ex) ** 2) / (ax ** 2) + ((cy - ey) ** 2) / (ay ** 2) <= 1.0


def extract_features(face: PreprocessedFace) -> AcneFeatureResult:
    hsv = face.hsv
    lab = face.lab
    gray = face.gray
    skin_mask = face.skin_mask
    focus_mask = face.focus_mask

    a = lab[:, :, 1].astype(np.float32)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    blur_a = cv2.GaussianBlur(a, (0, 0), 5)
    redness = a - blur_a

    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

    g1 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1)
    g2 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
    dog = np.abs(g1 - g2)

    candidate = (
        (redness > 2.4)
        & (dog > 3.2)
        & (lap > 5.5)
        & (skin_mask > 0)
        & (focus_mask > 0)
        & (s > 18)
        & (v > 45)
    ).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    mask = np.zeros_like(candidate)

    kept_areas = []
    kept_redness = []

    h_img, w_img = gray.shape

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])

        if h == 0:
            continue

        ar = w / h
        cx = x + w / 2.0
        cy = y + h / 2.0

        if not (4 < area < 85 and 0.4 < ar < 2.3):
            continue

        if _inside_ellipse(
            cx, cy,
            ex=w_img * 0.28, ey=h_img * 0.58,
            ax=w_img * 0.12, ay=h_img * 0.14
        ):
            continue

        if (h_img * 0.35 < cy < h_img * 0.48) and (w_img * 0.2 < cx < w_img * 0.8):
            continue

        component_pixels = labels == i
        component_redness = float(np.mean(redness[component_pixels]))

        mask[component_pixels] = 255
        kept_areas.append(area)
        kept_redness.append(component_redness)

    vis_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    vis_mask = cv2.dilate(mask, vis_kernel, iterations=1)

    spots = max(cv2.connectedComponents(mask)[0] - 1, 0)
    area_pixels = int(np.sum(mask > 0))
    total_focus = max(int(np.sum(focus_mask > 0)), 1)
    percent = 100.0 * area_pixels / total_focus

    features = {
        "spot_count": int(spots),
        "red_area_percent": float(percent),
        "avg_spot_size": float(np.mean(kept_areas)) if kept_areas else 0.0,
        "mean_redness": float(np.mean(kept_redness)) if kept_redness else 0.0,
        "texture_score": float(np.mean(lap[mask > 0])) if area_pixels > 0 else 0.0,
    }

    return AcneFeatureResult(
        features=features,
        red_mask=vis_mask,
        redness_map=_normalize_to_uint8(redness),
        dog_map=_normalize_to_uint8(dog),
        lap_map=_normalize_to_uint8(lap),
        candidate_mask=candidate,
    )