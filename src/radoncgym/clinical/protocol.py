

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ClinicalCriterion:
    metric_name: str
    roi_name: str
    relation: Literal["at_most", "at_least"]
    threshold_gy: float

    @property
    def key(self) -> str:
        return f"{self.metric_name}:{self.roi_name}"


CLINICAL_CRITERIA = (
    ClinicalCriterion("D_0.1_cc", "Brainstem", "at_most", 50.0),
    ClinicalCriterion("D_0.1_cc", "SpinalCord", "at_most", 45.0),
    ClinicalCriterion("mean", "RightParotid", "at_most", 26.0),
    ClinicalCriterion("mean", "LeftParotid", "at_most", 26.0),
    ClinicalCriterion("mean", "Esophagus", "at_most", 45.0),
    ClinicalCriterion("D_0.1_cc", "Mandible", "at_most", 73.5),
    ClinicalCriterion("mean", "Larynx", "at_most", 45.0),
    ClinicalCriterion("D_99", "PTV56", "at_least", 53.2),
    ClinicalCriterion("D_99", "PTV63", "at_least", 59.85),
    ClinicalCriterion("D_99", "PTV70", "at_least", 66.5),
)

TARGET_PRESCRIPTIONS_GY = {
    "PTV56": 56.0,
    "PTV63": 63.0,
    "PTV70": 70.0,
}
