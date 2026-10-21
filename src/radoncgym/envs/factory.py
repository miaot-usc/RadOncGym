

from __future__ import annotations

from pathlib import Path

from radoncgym.adapters.structured import vectorize_public_case
from radoncgym.backends.base import CheapBackend, FullBackend
from radoncgym.core.engine import RadOncEngine
from radoncgym.core.models import BudgetConfig, PlanningContext, TaskConfig
from radoncgym.data.openkbp import OpenKBPPublicCaseLoader
from radoncgym.rewards.base import RewardFunction


def make_openkbp_engine(
    *,
    context: PlanningContext,
    budget: BudgetConfig,
    task_config: TaskConfig,
    cheap_backend: CheapBackend,
    full_backend: FullBackend,
    public_case_loader: OpenKBPPublicCaseLoader | None = None,
    openkbp_root: str | Path = "datasets/open-kbp-opt",
    reward_function: RewardFunction | None = None,
    all_ones_clinical_objective: float | None,
) -> RadOncEngine:
    pass

    loader = public_case_loader or OpenKBPPublicCaseLoader(openkbp_root)
    public_case = loader.load(context)
    public_context = vectorize_public_case(
        public_case,
        include_prediction_identity=True,
    ).to_public_context()
    return RadOncEngine(
        context=context,
        public_context=public_context,
        budget=budget,
        cheap_backend=cheap_backend,
        full_backend=full_backend,
        reward_function=reward_function,
        task_config=task_config,
        all_ones_clinical_objective=all_ones_clinical_objective,
    )
