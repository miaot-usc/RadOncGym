

from __future__ import annotations

from typing import Any, Mapping

from radoncgym.core.errors import ContractError
from radoncgym.core.models import (
    BudgetConfig,
    BudgetState,
    CandidateAction,
    ComputeCost,
    EndReason,
    EvaluationResult,
    FeedbackSource,
    HistoryEntry,
    Incumbents,
    MetricVector,
    ObjectiveWeights,
    Observation,
    PlanningContext,
    PublicContextFeatures,
    SolverStatus,
    StepOutcome,
    TaskConfig,
    TaskKind,
    FidelityControl,
    FullIncumbentRule,
)


JsonObject = dict[str, Any]


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping")
    return value


def _weights_to_dict(weights: ObjectiveWeights | None) -> JsonObject | None:
    if weights is None:
        return None
    return weights.to_dict()


def _weights_from_dict(value: Any) -> ObjectiveWeights | None:
    if value is None:
        return None
    mapping = _require_mapping(value, "weights")
    return ObjectiveWeights.from_mapping(
        {str(name): float(weight) for name, weight in mapping.items()}
    )


def _metric_vector_to_dict(vector: MetricVector) -> list[JsonObject]:
    return [{"name": name, "value": value} for name, value in vector.items]


def _metric_vector_from_dict(value: Any, name: str) -> MetricVector:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be an ordered list")
    items: list[tuple[str, float]] = []
    for index, item in enumerate(value):
        mapping = _require_mapping(item, f"{name}[{index}]")
        items.append((str(mapping["name"]), float(mapping["value"])))
    return MetricVector(tuple(items))


def _cost_to_dict(cost: ComputeCost) -> JsonObject:
    return {
        "wall_seconds": cost.wall_seconds,
        "solver_seconds": cost.solver_seconds,
        "cheap_calls": cost.cheap_calls,
        "full_calls": cost.full_calls,
    }


def _cost_from_dict(value: Any) -> ComputeCost:
    mapping = _require_mapping(value, "cost")
    return ComputeCost(
        wall_seconds=float(mapping["wall_seconds"]),
        solver_seconds=float(mapping["solver_seconds"]),
        cheap_calls=int(mapping["cheap_calls"]),
        full_calls=int(mapping["full_calls"]),
    )


def _evaluation_to_dict(result: EvaluationResult) -> JsonObject:
    return {
        "source": result.source.value,
        "status": result.status.value,
        "metrics": _metric_vector_to_dict(result.metrics),
        "constraints": _metric_vector_to_dict(result.constraints),
        "cost": _cost_to_dict(result.cost),
        "uncertainty": result.uncertainty,
        "message": result.message,
    }


def _evaluation_from_dict(value: Any) -> EvaluationResult:
    mapping = _require_mapping(value, "evaluation")
    return EvaluationResult(
        source=FeedbackSource(str(mapping["source"])),
        status=SolverStatus(str(mapping["status"])),
        metrics=_metric_vector_from_dict(mapping["metrics"], "metrics"),
        constraints=_metric_vector_from_dict(mapping["constraints"], "constraints"),
        cost=_cost_from_dict(mapping["cost"]),
        uncertainty=(
            None if mapping.get("uncertainty") is None else float(mapping["uncertainty"])
        ),
        message=None if mapping.get("message") is None else str(mapping["message"]),
    )


def _action_to_dict(action: CandidateAction) -> JsonObject:
    return {
        "weights": _weights_to_dict(action.weights),
        "request_full_verifier": action.request_full_verifier,
        "stop": action.stop,
    }


def _action_from_dict(value: Any) -> CandidateAction:
    mapping = _require_mapping(value, "action")
    return CandidateAction(
        weights=_weights_from_dict(mapping.get("weights")),
        request_full_verifier=bool(mapping["request_full_verifier"]),
        stop=bool(mapping["stop"]),
    )


def _outcome_to_dict(outcome: StepOutcome | None) -> JsonObject | None:
    if outcome is None:
        return None
    return {
        "executed_weights": _weights_to_dict(outcome.executed_weights),
        "cheap": _evaluation_to_dict(outcome.cheap),
        "full": None if outcome.full is None else _evaluation_to_dict(outcome.full),
    }


def _outcome_from_dict(value: Any) -> StepOutcome | None:
    if value is None:
        return None
    mapping = _require_mapping(value, "outcome")
    weights = _weights_from_dict(mapping["executed_weights"])
    if weights is None:
        raise ContractError("executed_weights cannot be null")
    return StepOutcome(
        executed_weights=weights,
        cheap=_evaluation_from_dict(mapping["cheap"]),
        full=(
            None
            if mapping.get("full") is None
            else _evaluation_from_dict(mapping["full"])
        ),
    )


def history_entry_to_dict(entry: HistoryEntry) -> JsonObject:
    pass

    return {
        "step_index": entry.step_index,
        "action": _action_to_dict(entry.action),
        "accepted": entry.accepted,
        "validation_error": entry.validation_error,
        "duplicate_of_step": entry.duplicate_of_step,
        "outcome": _outcome_to_dict(entry.outcome),
        "terminated": entry.terminated,
        "truncated": entry.truncated,
        "end_reason": None if entry.end_reason is None else entry.end_reason.value,
    }


def _history_from_dict(value: Any) -> HistoryEntry:
    mapping = _require_mapping(value, "history_entry")
    return HistoryEntry(
        step_index=int(mapping["step_index"]),
        action=_action_from_dict(mapping["action"]),
        accepted=bool(mapping["accepted"]),
        validation_error=(
            None
            if mapping.get("validation_error") is None
            else str(mapping["validation_error"])
        ),
        duplicate_of_step=(
            None
            if mapping.get("duplicate_of_step") is None
            else int(mapping["duplicate_of_step"])
        ),
        outcome=_outcome_from_dict(mapping.get("outcome")),
        terminated=bool(mapping["terminated"]),
        truncated=bool(mapping["truncated"]),
        end_reason=(
            None
            if mapping.get("end_reason") is None
            else EndReason(str(mapping["end_reason"]))
        ),
    )


def observation_to_dict(observation: Observation) -> JsonObject:
    pass

    context = observation.context
    config = observation.budget_config
    state = observation.budget_state
    return {
        "context": {
            "case_id": context.case_id,
            "patient_id": context.patient_id,
            "prediction_set_id": context.prediction_set_id,
            "split_id": context.split_id,
            "clinical_profile_id": context.clinical_profile_id,
            "planning_formulation": context.planning_formulation,
            "environment_version": context.environment_version,
        },
        "public_context": {
            "schema_version": observation.public_context.schema_version,
            "features": _metric_vector_to_dict(
                MetricVector(observation.public_context.items)
            ),
        },
        "task_config": {
            "task_id": observation.task_config.task_id,
            "kind": observation.task_config.kind.value,
            "fidelity_control": observation.task_config.fidelity_control.value,
            "full_incumbent_rule": observation.task_config.full_incumbent_rule.value,
            "stop_enabled": observation.task_config.stop_enabled,
            "evaluate_cheap_on_full_request": (
                observation.task_config.evaluate_cheap_on_full_request
            ),
            "allow_stop_before_candidate": (
                observation.task_config.allow_stop_before_candidate
            ),
            "minimum_evaluated_candidates_before_stop": (
                observation.task_config.minimum_evaluated_candidates_before_stop
            ),
        },
        "budget_config": {
            "max_steps": config.max_steps,
            "max_full_calls": config.max_full_calls,
            "max_cheap_calls": config.max_cheap_calls,
            "max_wall_seconds": config.max_wall_seconds,
            "invalid_action_limit": config.invalid_action_limit,
        },
        "budget_state": {
            "steps_used": state.steps_used,
            "cheap_calls_used": state.cheap_calls_used,
            "full_calls_attempted": state.full_calls_attempted,
            "full_calls_solved": state.full_calls_solved,
            "full_calls_failed": state.full_calls_failed,
            "invalid_actions": state.invalid_actions,
            "wall_seconds": state.wall_seconds,
            "solver_seconds": state.solver_seconds,
        },
        "history": [history_entry_to_dict(entry) for entry in observation.history],
        "incumbents": {
            "best_cheap_step": observation.incumbents.best_cheap_step,
            "best_full_step": observation.incumbents.best_full_step,
        },
    }


def observation_from_dict(value: Any) -> Observation:
    pass

    root = _require_mapping(value, "observation")
    context_data = _require_mapping(root["context"], "context")
    public_context_data = _require_mapping(
        root["public_context"], "public_context"
    )
    task_data = _require_mapping(root["task_config"], "task_config")
    config_data = _require_mapping(root["budget_config"], "budget_config")
    state_data = _require_mapping(root["budget_state"], "budget_state")
    incumbents_data = _require_mapping(root["incumbents"], "incumbents")
    history_data = root["history"]
    if not isinstance(history_data, list):
        raise ContractError("history must be a list")

    return Observation(
        context=PlanningContext(
            case_id=str(context_data["case_id"]),
            patient_id=str(context_data["patient_id"]),
            prediction_set_id=str(context_data["prediction_set_id"]),
            split_id=str(context_data["split_id"]),
            clinical_profile_id=str(context_data["clinical_profile_id"]),
            planning_formulation=str(context_data["planning_formulation"]),
            environment_version=str(context_data["environment_version"]),
        ),
        public_context=PublicContextFeatures(
            schema_version=str(public_context_data["schema_version"]),
            items=_metric_vector_from_dict(
                public_context_data["features"], "public_context.features"
            ).items,
        ),
        task_config=TaskConfig(
            task_id=str(task_data["task_id"]),
            kind=TaskKind(str(task_data["kind"])),
            fidelity_control=FidelityControl(str(task_data["fidelity_control"])),
            full_incumbent_rule=FullIncumbentRule(
                str(task_data["full_incumbent_rule"])
            ),
            stop_enabled=bool(task_data.get("stop_enabled", True)),
            evaluate_cheap_on_full_request=bool(
                task_data.get("evaluate_cheap_on_full_request", True)
            ),
            allow_stop_before_candidate=bool(
                task_data["allow_stop_before_candidate"]
            ),
            minimum_evaluated_candidates_before_stop=int(
                task_data.get("minimum_evaluated_candidates_before_stop", 1)
            ),
        ),
        budget_config=BudgetConfig(
            max_steps=int(config_data["max_steps"]),
            max_full_calls=int(config_data["max_full_calls"]),
            max_cheap_calls=(
                None
                if config_data.get("max_cheap_calls") is None
                else int(config_data["max_cheap_calls"])
            ),
            max_wall_seconds=(
                None
                if config_data.get("max_wall_seconds") is None
                else float(config_data["max_wall_seconds"])
            ),
            invalid_action_limit=int(config_data["invalid_action_limit"]),
        ),
        budget_state=BudgetState(
            steps_used=int(state_data["steps_used"]),
            cheap_calls_used=int(state_data["cheap_calls_used"]),
            full_calls_attempted=int(state_data["full_calls_attempted"]),
            full_calls_solved=int(state_data["full_calls_solved"]),
            full_calls_failed=int(state_data["full_calls_failed"]),
            invalid_actions=int(state_data["invalid_actions"]),
            wall_seconds=float(state_data["wall_seconds"]),
            solver_seconds=float(state_data["solver_seconds"]),
        ),
        history=tuple(_history_from_dict(entry) for entry in history_data),
        incumbents=Incumbents(
            best_cheap_step=(
                None
                if incumbents_data.get("best_cheap_step") is None
                else int(incumbents_data["best_cheap_step"])
            ),
            best_full_step=(
                None
                if incumbents_data.get("best_full_step") is None
                else int(incumbents_data["best_full_step"])
            ),
        ),
    )
