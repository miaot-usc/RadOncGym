

from __future__ import annotations

import numpy as np

from radoncgym.core.errors import ContractError
from radoncgym.core.constants import (
    FULL_CLINICAL_IMPROVEMENT_METRIC,
    PREDICTED_CLINICAL_IMPROVEMENT_METRIC,
)
from radoncgym.core.models import EvaluationResult, MetricVector, SolverStatus, StepOutcome


PUBLIC_PLAN_METRIC_SCHEMA_VERSION = "public-plan-metrics-0.4.0"
PUBLIC_CONSTRAINT_SCHEMA_VERSION = "public-constraint-metrics-0.1.0"

_OARS = (
    "Brainstem",
    "SpinalCord",
    "RightParotid",
    "LeftParotid",
    "Esophagus",
    "Larynx",
    "Mandible",
)
_TARGETS = ("PTV56", "PTV63", "PTV70")

PUBLIC_PLAN_METRIC_NAMES = (
    "objective",
    "mean_protocol_violation_gy",
    "preference_compliance_score",
    "preference_components_evaluated",
    PREDICTED_CLINICAL_IMPROVEMENT_METRIC,
    FULL_CLINICAL_IMPROVEMENT_METRIC,
    "ensemble_std",
    *(f"D_0.1_cc:{roi}" for roi in _OARS),
    *(f"mean:{roi}" for roi in _OARS),
    *(f"D_99:{roi}" for roi in _TARGETS),
    *(f"D_95:{roi}" for roi in _TARGETS),
    *(f"D_1:{roi}" for roi in _TARGETS),
    "plan_max_dose_gy",
    "plan_max_dose_percent_of_PTV70_prescription",
    "high_dose_spill_cc_above_73_5_outside_PTV70",
    *(f"D_2:{roi}" for roi in _TARGETS),
    *(f"D_98:{roi}" for roi in _TARGETS),
    *(f"homogeneity_index:{roi}" for roi in _TARGETS),
    *(f"conformity_ratio:{roi}" for roi in _TARGETS),
    *(f"falloff_mean_gy:{shell}" for shell in ("0_5_mm", "5_10_mm", "10_20_mm")),
    *(f"falloff_p95_gy:{shell}" for shell in ("0_5_mm", "5_10_mm", "10_20_mm")),
)

PUBLIC_CONSTRAINT_METRIC_NAMES = (
    "D_0.1_cc:Brainstem",
    "D_0.1_cc:SpinalCord",
    "D_0.1_cc:Mandible",
    "mean:RightParotid",
    "mean:LeftParotid",
    "mean:Esophagus",
    "mean:Larynx",
    "D_99:PTV56",
    "D_99:PTV63",
    "D_99:PTV70",

    "PTV56_D99_shortfall",
    "PTV63_D99_shortfall",
    "PTV70_D99_shortfall",
    "Brainstem_D0.1cc_excess",
    "SpinalCord_D0.1cc_excess",
)


def vectorize_public_metrics(
    metrics: MetricVector,
    *,
    names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    pass

    by_name = metrics.to_dict()
    unknown = tuple(sorted(set(by_name) - set(names)))
    if unknown:
        raise ContractError(f"public vector metric schema is missing fields: {unknown}")
    values = np.zeros(len(names), dtype=np.float32)
    mask = np.zeros(len(names), dtype=np.int8)
    for index, name in enumerate(names):
        if name in by_name:
            values[index] = np.float32(by_name[name])
            mask[index] = 1
    return values, mask


def clinical_improvement(result: EvaluationResult) -> float | None:
    for name in (
        FULL_CLINICAL_IMPROVEMENT_METRIC,
        PREDICTED_CLINICAL_IMPROVEMENT_METRIC,
    ):
        value = result.metrics.get(name)
        if value is not None:
            return float(value)
    return None


def returned_clinical_improvement(outcome: StepOutcome) -> float | None:
    result = outcome.full if outcome.full is not None else outcome.cheap
    valid_statuses = (
        {SolverStatus.SOLVED}
        if outcome.full is not None
        else {SolverStatus.SOLVED, SolverStatus.NOT_APPLICABLE}
    )
    if result.status not in valid_statuses:
        return None
    return clinical_improvement(result)


def absolute_clinical_objective(result: EvaluationResult) -> float | None:
    value = result.metrics.get("mean_protocol_violation_gy")
    if value is None:
        value = result.metrics.get("objective")
    return None if value is None else float(value)
