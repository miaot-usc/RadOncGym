

from radoncgym.core.errors import ContractError
from radoncgym.core.models import EvaluationResult, ObjectiveWeights, PlanningContext


class DisabledFullBackend:
    pass

    def evaluate(
        self, context: PlanningContext, weights: ObjectiveWeights
    ) -> EvaluationResult:
        del context, weights
        raise ContractError(
            "full evaluation is disabled for this benchmark execution profile"
        )
