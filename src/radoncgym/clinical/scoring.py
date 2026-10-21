

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radoncgym.clinical.preferences import ClinicalPreferenceProfile
from radoncgym.clinical.protocol import CLINICAL_CRITERIA
from radoncgym.core.models import MetricVector


_OBJECTIVE_TO_PROTOCOL_METRIC = {
    "PTV56_coverage": "D_99:PTV56",
    "PTV63_coverage": "D_99:PTV63",
    "PTV70_coverage": "D_99:PTV70",
    "Brainstem_sparing": "D_0.1_cc:Brainstem",
    "SpinalCord_sparing": "D_0.1_cc:SpinalCord",
    "RightParotid_sparing": "mean:RightParotid",
    "LeftParotid_sparing": "mean:LeftParotid",
    "Esophagus_sparing": "mean:Esophagus",
    "Larynx_sparing": "mean:Larynx",
    "Mandible_sparing": "D_0.1_cc:Mandible",
}
_CRITERIA_BY_KEY = {criterion.key: criterion for criterion in CLINICAL_CRITERIA}


@dataclass(frozen=True, slots=True)
class PreferenceCompliance:
    pass

    score: float
    components_evaluated: int
    component_scores: MetricVector


def calculate_preference_compliance(
    protocol_metrics: MetricVector,
    profile: ClinicalPreferenceProfile,
) -> PreferenceCompliance:
    pass

    metrics = protocol_metrics.to_dict()
    emphasis = profile.emphasis_dict()
    values: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for objective_name, metric_name in _OBJECTIVE_TO_PROTOCOL_METRIC.items():
        if metric_name not in metrics:
            continue
        criterion = _CRITERIA_BY_KEY[metric_name]
        value = metrics[metric_name]
        if criterion.relation == "at_least":
            component = float(np.clip(value / criterion.threshold_gy, 0.0, 1.0))
        else:
            component = float(
                np.clip(
                    (criterion.threshold_gy - value) / criterion.threshold_gy,
                    0.0,
                    1.0,
                )
            )
        values[f"preference_component:{objective_name}"] = component
        weight = emphasis[objective_name]
        weighted_sum += weight * component
        total_weight += weight
    score = 0.0 if total_weight == 0.0 else weighted_sum / total_weight
    return PreferenceCompliance(
        score=float(score),
        components_evaluated=len(values),
        component_scores=MetricVector.from_mapping(values),
    )
