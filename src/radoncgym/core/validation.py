

from __future__ import annotations

from dataclasses import dataclass

from radoncgym.core.constants import MAX_OBJECTIVE_WEIGHT, MIN_OBJECTIVE_WEIGHT
from radoncgym.core.models import (
    BudgetConfig,
    BudgetState,
    CandidateAction,
    FidelityControl,
    HistoryEntry,
    TaskConfig,
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    pass

    accepted: bool
    error: str | None = None
    duplicate_of_step: int | None = None


def validate_task_action(
    action: CandidateAction,
    task: TaskConfig,
    history: tuple[HistoryEntry, ...],
) -> ValidationResult:
    pass

    if task.fidelity_control is FidelityControl.DATASET_ONLY:
        return ValidationResult(False, "offline_task_is_not_interactive")

    accepted_candidates = sum(
        entry.accepted and entry.outcome is not None for entry in history
    )
    if action.stop:
        if not task.stop_enabled:
            return ValidationResult(False, "stop_disabled_for_task")
        required_candidates = (
            0
            if task.allow_stop_before_candidate
            else task.minimum_evaluated_candidates_before_stop
        )
        if accepted_candidates < required_candidates:
            if required_candidates == 1:
                error = "stop_requires_evaluated_candidate"
            else:
                error = (
                    "stop_requires_minimum_evaluated_candidates:"
                    f"{required_candidates}"
                )
            return ValidationResult(False, error)
        return ValidationResult(True)

    if (
        task.fidelity_control is FidelityControl.FULL_ONLY
        and not action.request_full_verifier
    ):
        return ValidationResult(False, "task_requires_full_evaluation")
    return ValidationResult(True)


def validate_action(
    action: CandidateAction,
    budget_config: BudgetConfig,
    budget_state: BudgetState,
    history: tuple[HistoryEntry, ...],
) -> ValidationResult:
    pass

    if action.stop:
        if action.request_full_verifier:
            return ValidationResult(False, "stop_action_cannot_request_full")
        return ValidationResult(True)

    if action.weights is None:
        return ValidationResult(False, "candidate_weights_required")

    if not action.weights.in_bounds(MIN_OBJECTIVE_WEIGHT, MAX_OBJECTIVE_WEIGHT):
        return ValidationResult(
            False,
            f"weights_out_of_bounds:[{MIN_OBJECTIVE_WEIGHT},{MAX_OBJECTIVE_WEIGHT}]",
        )

    if (
        budget_config.max_cheap_calls is not None
        and budget_state.cheap_calls_used >= budget_config.max_cheap_calls
    ):
        return ValidationResult(False, "cheap_budget_exhausted")

    if (
        action.request_full_verifier
        and budget_state.full_calls_attempted >= budget_config.max_full_calls
    ):
        return ValidationResult(False, "full_budget_exhausted")

    duplicate_of_step: int | None = None
    for entry in history:
        if (
            entry.accepted
            and entry.outcome is not None
            and entry.outcome.executed_weights == action.weights
        ):
            duplicate_of_step = entry.step_index
            break

    return ValidationResult(True, duplicate_of_step=duplicate_of_step)
