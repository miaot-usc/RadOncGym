

from typing import Final

OBJECTIVE_NAMES: Final[tuple[str, ...]] = (
    "PTV56_coverage",
    "PTV63_coverage",
    "PTV70_coverage",
    "Brainstem_sparing",
    "SpinalCord_sparing",
    "RightParotid_sparing",
    "LeftParotid_sparing",
    "Esophagus_sparing",
    "Larynx_sparing",
    "Mandible_sparing",
    "ring_falloff",
    "posterior_neck_sparing",
)

MIN_OBJECTIVE_WEIGHT: Final[float] = 0.25
MAX_OBJECTIVE_WEIGHT: Final[float] = 4.0
PRIMARY_FORMULATION: Final[str] = "absolute_mean"
CORE_SCHEMA_VERSION: Final[str] = "1.0.0-draft"
PREDICTED_CLINICAL_IMPROVEMENT_METRIC: Final[str] = (
    "predicted_clinical_improvement_from_all_ones"
)
FULL_CLINICAL_IMPROVEMENT_METRIC: Final[str] = (
    "clinical_improvement_from_all_ones"
)
