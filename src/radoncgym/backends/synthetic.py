

from __future__ import annotations

import hashlib
import math

from radoncgym.core.models import (
    ComputeCost,
    EvaluationResult,
    FeedbackSource,
    MetricVector,
    ObjectiveWeights,
    PlanningContext,
    SolverStatus,
)
from radoncgym.core.constants import PREDICTED_CLINICAL_IMPROVEMENT_METRIC


def _context_offset(context: PlanningContext) -> float:
    payload = "|".join(
        (
            context.case_id,
            context.patient_id,
            context.prediction_set_id,
            context.planning_formulation,
        )
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return (value % 1000) / 10_000.0


def _target(index: int) -> float:
    return 0.75 + 0.05 * (index % 7)


def _objective(context: PlanningContext, weights: ObjectiveWeights) -> float:
    squared_error = math.fsum(
        (value - _target(index)) ** 2 for index, value in enumerate(weights.values)
    )
    return squared_error + _context_offset(context)


def _constraints(weights: ObjectiveWeights) -> MetricVector:
    values = {
        "PTV56_D99_shortfall": max(0.0, 0.90 - weights.values[0]),
        "PTV63_D99_shortfall": max(0.0, 0.90 - weights.values[1]),
        "PTV70_D99_shortfall": max(0.0, 1.00 - weights.values[2]),
        "Brainstem_D0.1cc_excess": max(0.0, 0.70 - weights.values[3]),
        "SpinalCord_D0.1cc_excess": max(0.0, 0.75 - weights.values[4]),
    }
    return MetricVector.from_mapping(values)


class SyntheticCheapBackend:
    pass

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self, context: PlanningContext, weights: ObjectiveWeights
    ) -> EvaluationResult:
        self.calls += 1
        full_objective = _objective(context, weights)
        bias = 0.02 * math.fsum(weights.values) / len(weights.values)
        baseline = ObjectiveWeights.all_ones()
        baseline_prediction = _objective(context, baseline) + 0.02
        return EvaluationResult(
            source=FeedbackSource.CHEAP_LEARNED_SURROGATE,
            status=SolverStatus.NOT_APPLICABLE,
            metrics=MetricVector.from_mapping(
                {
                    PREDICTED_CLINICAL_IMPROVEMENT_METRIC: (
                        baseline_prediction - (full_objective + bias)
                    )
                }
            ),
            constraints=MetricVector(),
            cost=ComputeCost(wall_seconds=0.001, cheap_calls=1),
            uncertainty=0.05,
        )


class SyntheticFullBackend:
    pass

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self, context: PlanningContext, weights: ObjectiveWeights
    ) -> EvaluationResult:
        self.calls += 1
        return EvaluationResult(
            source=FeedbackSource.FULL_LIVE_OPTIMIZER,
            status=SolverStatus.SOLVED,
            metrics=MetricVector.from_mapping({"objective": _objective(context, weights)}),
            constraints=_constraints(weights),
            cost=ComputeCost(
                wall_seconds=1.0,
                solver_seconds=0.9,
                full_calls=1,
            ),
            uncertainty=0.0,
        )
