

from typing import Protocol

from radoncgym.core.models import EvaluationResult, ObjectiveWeights, PlanningContext


class CheapBackend(Protocol):
    pass

    def evaluate(
        self, context: PlanningContext, weights: ObjectiveWeights
    ) -> EvaluationResult:
        pass


class FullBackend(Protocol):
    pass

    def evaluate(
        self, context: PlanningContext, weights: ObjectiveWeights
    ) -> EvaluationResult:
        pass

