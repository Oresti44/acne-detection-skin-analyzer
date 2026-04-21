from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class SeverityResult:
    score: float
    label: str
    explanation: str


def calculate_severity(features: Dict[str, float]) -> SeverityResult:
    spots = float(features.get("spot_count", 0.0))
    percent = float(features.get("red_area_percent", 0.0))

    norm_percent = min(percent / 15.0, 1.0)   # 15% = very severe
    norm_spots = min(spots / 40.0, 1.0)       # 40 spots = severe

    score = (0.6 * norm_percent + 0.4 * norm_spots) * 100.0
    score = round(score, 2)

    if score < 35:
        label = "Mild"
    elif score < 70:
        label = "Moderate"
    else:
        label = "Severe"

    if spots > 20 or percent > 8:
        explanation = "many visible acne-like spots and a larger affected area"
    elif spots > 8 or percent > 3:
        explanation = "a moderate number of acne-like spots in the focused region"
    else:
        explanation = "limited visible acne-like spots in the focused region"

    return SeverityResult(score=score, label=label, explanation=explanation)