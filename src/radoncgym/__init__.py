

from gymnasium.envs.registration import register, registry

from radoncgym.core.engine import RadOncEngine
from radoncgym.core.models import (
    BudgetConfig,
    CandidateAction,
    ObjectiveWeights,
    PlanningContext,
    PublicContextFeatures,
)
from radoncgym.protocol import (
    FinalSelection,
    PaperProtocol,
    select_final_full_step,
    select_final_weights,
)

__all__ = [
    "BudgetConfig",
    "CandidateAction",
    "ObjectiveWeights",
    "PlanningContext",
    "PublicContextFeatures",
    "RadOncEngine",
    "FinalSelection",
    "PaperProtocol",
    "select_final_full_step",
    "select_final_weights",
]

__version__ = "0.1.0rc1"


if "RadOncGym/SyntheticMultiFidelity-v0" not in registry:
    register(
        id="RadOncGym/SyntheticMultiFidelity-v0",
        entry_point="radoncgym.envs:make_synthetic_env",
    )

if "RadOncGym/SyntheticNormalized-v0" not in registry:
    register(
        id="RadOncGym/SyntheticNormalized-v0",
        entry_point="radoncgym.envs:make_normalized_synthetic_env",
    )
