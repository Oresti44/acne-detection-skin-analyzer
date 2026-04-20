from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SeverityResult:
    score: float
    label: str
    explanation: str



def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))



def calculate_severity(features: Dict[str, float]) -> SeverityResult:
    red_area = _clip(features.get('red_area_percent', 0.0) * 3.0, 0, 40)
    spot_component = _clip(features.get('spot_count', 0.0) * 1.8, 0, 30)
    texture_component = _clip(features.get('texture_score', 0.0) * 25.0, 0, 20)
    edge_component = _clip(features.get('edge_density', 0.0) * 200.0, 0, 10)

    score = round(red_area + spot_component + texture_component + edge_component, 1)

    if score <= 30:
        label = 'Mild'
    elif score <= 60:
        label = 'Moderate'
    else:
        label = 'Severe'

    explanation_points: List[str] = []
    if features.get('red_area_percent', 0) > 8:
        explanation_points.append('noticeable inflamed red regions')
    elif features.get('red_area_percent', 0) > 3:
        explanation_points.append('a small-to-moderate amount of redness')

    if features.get('spot_count', 0) > 18:
        explanation_points.append('many suspicious red spots')
    elif features.get('spot_count', 0) > 8:
        explanation_points.append('a moderate number of suspicious spots')

    if features.get('texture_score', 0) > 1.2:
        explanation_points.append('high texture irregularity')
    elif features.get('texture_score', 0) > 0.6:
        explanation_points.append('moderate texture irregularity')

    explanation = '; '.join(explanation_points) if explanation_points else 'limited visible acne-like signals in the analyzed region'
    return SeverityResult(score=score, label=label, explanation=explanation)
