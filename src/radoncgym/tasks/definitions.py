

from __future__ import annotations

from radoncgym.core.errors import ContractError
from radoncgym.core.models import (
    FidelityControl,
    FullIncumbentRule,
    TaskConfig,
    TaskKind,
)


TASK_CONFIGS: dict[str, TaskConfig] = {
    "SingleShot-v0": TaskConfig(
        task_id="SingleShot-v0",
        kind=TaskKind.SINGLE_SHOT,
        fidelity_control=FidelityControl.FULL_ONLY,
    ),
    "SequentialFull-v0": TaskConfig(
        task_id="SequentialFull-v0",
        kind=TaskKind.SEQUENTIAL_FULL,
        fidelity_control=FidelityControl.FULL_ONLY,
    ),
    "SequentialFullFixed16-v0": TaskConfig(
        task_id="SequentialFullFixed16-v0",
        kind=TaskKind.SEQUENTIAL_FULL,
        fidelity_control=FidelityControl.FULL_ONLY,
        stop_enabled=False,
    ),
    "SequentialFull14-v0": TaskConfig(
        task_id="SequentialFull14-v0",
        kind=TaskKind.SEQUENTIAL_FULL,
        fidelity_control=FidelityControl.FULL_ONLY,
        minimum_evaluated_candidates_before_stop=14,
    ),
    "BudgetedMultiFidelity-v0": TaskConfig(
        task_id="BudgetedMultiFidelity-v0",
        kind=TaskKind.BUDGETED_MULTI_FIDELITY,
        fidelity_control=FidelityControl.AGENT_CHOICE,
    ),
    "FixedClinicalOnly-v0": TaskConfig(
        task_id="FixedClinicalOnly-v0",
        kind=TaskKind.BUDGETED_MULTI_FIDELITY,
        fidelity_control=FidelityControl.AGENT_CHOICE,
        stop_enabled=False,
        evaluate_cheap_on_full_request=False,
    ),
    "ClinicallyConstrained-v0": TaskConfig(
        task_id="ClinicallyConstrained-v0",
        kind=TaskKind.CLINICALLY_CONSTRAINED,
        fidelity_control=FidelityControl.AGENT_CHOICE,
        full_incumbent_rule=FullIncumbentRule.FEASIBILITY_THEN_QUALITY,
    ),
    "PriorityConstrained-v0": TaskConfig(
        task_id="PriorityConstrained-v0",
        kind=TaskKind.CLINICALLY_CONSTRAINED,
        fidelity_control=FidelityControl.AGENT_CHOICE,
        full_incumbent_rule=FullIncumbentRule.PRIORITY_LEXICOGRAPHIC,
    ),
    "OfflineTrajectory-v0": TaskConfig(
        task_id="OfflineTrajectory-v0",
        kind=TaskKind.OFFLINE_TRAJECTORY,
        fidelity_control=FidelityControl.DATASET_ONLY,
    ),
}


def get_task_config(task_id: str) -> TaskConfig:
    pass

    try:
        return TASK_CONFIGS[task_id]
    except KeyError as error:
        raise ContractError(f"unknown task_id: {task_id!r}") from error
