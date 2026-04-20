from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.preprocess import PreprocessedFace


@dataclass
class AcneFeatureResult:
    features: dict
    red_mask: np.ndarray
    texture_map: np.ndarray



def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned



def detect_red_regions(face: PreprocessedFace) -> tuple[np.ndarray, dict]:
    hsv = face.hsv
    lab = face.lab
    analysis_mask = face.analysis_mask

    lower_red1 = np.array([0, 30, 40], dtype=np.uint8)
    upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 30, 40], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    hsv_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    a_channel = lab[:, :, 1]
    redness_boost = cv2.threshold(a_channel, 145, 255, cv2.THRESH_BINARY)[1]
    combined = cv2.bitwise_and(hsv_red, redness_boost)
    combined = cv2.bitwise_and(combined, analysis_mask)
    cleaned = _clean_mask(combined)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    areas = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 8 <= area <= 5000:
            areas.append(area)
        else:
            cleaned[labels == i] = 0

    analysis_pixels = max(int(np.count_nonzero(analysis_mask)), 1)
    red_pixels = int(np.count_nonzero(cleaned))
    red_area_percent = 100.0 * red_pixels / analysis_pixels

    mean_redness = float(np.mean(a_channel[cleaned > 0])) if red_pixels else 0.0
    feature_stats = {
        'spot_count': len(areas),
        'avg_spot_size': float(np.mean(areas)) if areas else 0.0,
        'red_area_percent': float(red_area_percent),
        'mean_redness': mean_redness,
    }
    return cleaned, feature_stats



def texture_irregularity(face: PreprocessedFace) -> tuple[np.ndarray, dict]:
    gray = face.gray
    analysis_mask = face.analysis_mask

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_abs = cv2.convertScaleAbs(lap)
    lap_masked = cv2.bitwise_and(lap_abs, lap_abs, mask=analysis_mask)

    kernel = np.ones((7, 7), np.float32) / 49.0
    mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    sq_mean = cv2.filter2D((gray.astype(np.float32) ** 2), -1, kernel)
    variance = np.maximum(sq_mean - mean**2, 0)
    variance = variance * (analysis_mask > 0)

    analysis_pixels = max(int(np.count_nonzero(analysis_mask)), 1)
    edge_density = float(np.count_nonzero(lap_masked > 20) / analysis_pixels)
    texture_score = float(np.mean(variance[analysis_mask > 0]) / 100.0)

    normalized_var = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return normalized_var, {
        'texture_score': texture_score,
        'edge_density': edge_density,
    }



def extract_features(face: PreprocessedFace) -> AcneFeatureResult:
    red_mask, red_stats = detect_red_regions(face)
    texture_map, texture_stats = texture_irregularity(face)

    features = {**red_stats, **texture_stats}
    return AcneFeatureResult(features=features, red_mask=red_mask, texture_map=texture_map)
