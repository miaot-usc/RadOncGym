

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Iterable

from radoncgym.core.incumbents import constraint_priority_sums, full_incumbent_key
from radoncgym.core.metric_schema import clinical_improvement
from radoncgym.core.models import (
    FullIncumbentRule,
    HistoryEntry,
    SolverStatus,
)
from radoncgym.trajectories.schema import TrajectoryEpisode


@dataclass(frozen=True, slots=True)
class AnytimePoint:
    pass

    full_call_index: int
    best_step: int | None
    best_objective: float | None
    best_constraint_sum: float | None
    feasible: bool | None
    best_target_constraint_sum: float | None = None
    best_serial_constraint_sum: float | None = None
    best_parallel_constraint_sum: float | None = None

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return {
            "full_call_index": self.full_call_index,
            "best_step": self.best_step,
            "best_objective": self.best_objective,
            "best_constraint_sum": self.best_constraint_sum,
            "feasible": self.feasible,
            "best_target_constraint_sum": self.best_target_constraint_sum,
            "best_serial_constraint_sum": self.best_serial_constraint_sum,
            "best_parallel_constraint_sum": self.best_parallel_constraint_sum,
        }


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    pass

    episode_id: str
    task_id: str
    patient_id: str
    prediction_set_id: str
    split_id: str
    policy_family: str
    policy_name: str
    policy_version: str
    seed: int
    max_steps: int
    max_full_calls: int
    max_cheap_calls: int | None
    steps: int
    accepted_candidates: int
    invalid_actions: int
    invalid_action_rate: float
    cheap_calls: int
    full_calls_attempted: int
    full_calls_solved: int
    full_calls_failed: int
    wall_seconds: float
    solver_seconds: float
    best_cheap_step: int | None
    best_cheap_improvement: float | None
    best_full_step: int | None
    best_full_objective: float | None
    best_full_constraint_sum: float | None
    best_full_feasible: bool | None
    terminal_reason: str | None
    anytime_full: tuple[AnytimePoint, ...]
    best_full_target_constraint_sum: float | None = None
    best_full_serial_constraint_sum: float | None = None
    best_full_parallel_constraint_sum: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != "anytime_full"
        }
        result["anytime_full"] = [point.to_dict() for point in self.anytime_full]
        return result


def _entry_objective(entry: HistoryEntry, *, full: bool) -> float | None:
    if entry.outcome is None:
        return None
    result = entry.outcome.full if full else entry.outcome.cheap
    if result is None:
        return None
    if full:
        return result.metrics.get("objective")
    return clinical_improvement(result)


def _full_key(
    entry: HistoryEntry,
    rule: FullIncumbentRule,
) -> tuple[float, ...] | None:
    if entry.outcome is None or entry.outcome.full is None:
        return None
    result = entry.outcome.full
    if result.status is not SolverStatus.SOLVED:
        return None
    objective = result.metrics.get("objective")
    if objective is None:
        return None
    return full_incumbent_key(
        rule=rule,
        constraints=result.constraints,
        objective=objective,
        step_index=entry.step_index,
    )


def _anytime_curve(episode: TrajectoryEpisode) -> tuple[AnytimePoint, ...]:
    rule = episode.final_observation.task_config.full_incumbent_rule
    solved_entries: list[HistoryEntry] = []
    points = []
    full_call_index = 0
    for entry in episode.final_observation.history:
        if not entry.action.request_full_verifier or entry.action.stop:
            continue
        full_call_index += 1
        if (
            entry.outcome is not None
            and entry.outcome.full is not None
            and entry.outcome.full.status is SolverStatus.SOLVED
            and entry.outcome.full.metrics.get("objective") is not None
        ):
            solved_entries.append(entry)
        keyed = [
            (key, candidate)
            for candidate in solved_entries
            if (key := _full_key(candidate, rule)) is not None
        ]
        if not keyed:
            points.append(AnytimePoint(full_call_index, None, None, None, None))
            continue
        _, best = min(keyed, key=lambda item: item[0])
        assert best.outcome is not None and best.outcome.full is not None
        constraint_sum = best.outcome.full.constraints.sum()
        target_sum, serial_sum, parallel_sum = constraint_priority_sums(
            best.outcome.full.constraints
        )
        points.append(
            AnytimePoint(
                full_call_index=full_call_index,
                best_step=best.step_index,
                best_objective=best.outcome.full.metrics.get("objective"),
                best_constraint_sum=constraint_sum,
                feasible=constraint_sum <= 0.0,
                best_target_constraint_sum=target_sum,
                best_serial_constraint_sum=serial_sum,
                best_parallel_constraint_sum=parallel_sum,
            )
        )
    return tuple(points)


def summarize_episode(episode: TrajectoryEpisode) -> EpisodeSummary:
    pass

    observation = episode.final_observation
    history = observation.history
    accepted_candidates = sum(
        entry.accepted and entry.outcome is not None for entry in history
    )
    best_cheap_entry = (
        None
        if observation.incumbents.best_cheap_step is None
        else history[observation.incumbents.best_cheap_step]
    )
    best_full_entry = (
        None
        if observation.incumbents.best_full_step is None
        else history[observation.incumbents.best_full_step]
    )
    best_full = (
        None
        if best_full_entry is None or best_full_entry.outcome is None
        else best_full_entry.outcome.full
    )
    constraint_sum = None if best_full is None else best_full.constraints.sum()
    priority_sums = (
        (None, None, None)
        if best_full is None
        else constraint_priority_sums(best_full.constraints)
    )
    terminal_reason = (
        None
        if not history or history[-1].end_reason is None
        else history[-1].end_reason.value
    )
    state = observation.budget_state
    return EpisodeSummary(
        episode_id=episode.episode_id,
        task_id=episode.task_id,
        patient_id=observation.context.patient_id,
        prediction_set_id=observation.context.prediction_set_id,
        split_id=observation.context.split_id,
        policy_family=episode.policy.family,
        policy_name=episode.policy.name,
        policy_version=episode.policy.version,
        seed=episode.seed,
        max_steps=observation.budget_config.max_steps,
        max_full_calls=observation.budget_config.max_full_calls,
        max_cheap_calls=observation.budget_config.max_cheap_calls,
        steps=state.steps_used,
        accepted_candidates=accepted_candidates,
        invalid_actions=state.invalid_actions,
        invalid_action_rate=(state.invalid_actions / state.steps_used if state.steps_used else 0.0),
        cheap_calls=state.cheap_calls_used,
        full_calls_attempted=state.full_calls_attempted,
        full_calls_solved=state.full_calls_solved,
        full_calls_failed=state.full_calls_failed,
        wall_seconds=state.wall_seconds,
        solver_seconds=state.solver_seconds,
        best_cheap_step=observation.incumbents.best_cheap_step,
        best_cheap_improvement=(
            None if best_cheap_entry is None else _entry_objective(best_cheap_entry, full=False)
        ),
        best_full_step=observation.incumbents.best_full_step,
        best_full_objective=(
            None if best_full_entry is None else _entry_objective(best_full_entry, full=True)
        ),
        best_full_constraint_sum=constraint_sum,
        best_full_feasible=(None if constraint_sum is None else constraint_sum <= 0.0),
        terminal_reason=terminal_reason,
        anytime_full=_anytime_curve(episode),
        best_full_target_constraint_sum=priority_sums[0],
        best_full_serial_constraint_sum=priority_sums[1],
        best_full_parallel_constraint_sum=priority_sums[2],
    )


def aggregate_episode_summaries(
    summaries: Iterable[EpisodeSummary],
) -> dict[str, Any]:
    pass

    values = tuple(summaries)
    if not values:
        raise ValueError("at least one episode summary is required")
    solved = [item for item in values if item.best_full_objective is not None]
    cheap_improvements = [
        float(item.best_cheap_improvement)
        for item in values
        if item.best_cheap_improvement is not None
    ]
    objectives = [float(item.best_full_objective) for item in solved]
    constraint_sums = [
        float(item.best_full_constraint_sum)
        for item in solved
        if item.best_full_constraint_sum is not None
    ]
    target_constraint_sums = [
        float(item.best_full_target_constraint_sum)
        for item in solved
        if item.best_full_target_constraint_sum is not None
    ]
    serial_constraint_sums = [
        float(item.best_full_serial_constraint_sum)
        for item in solved
        if item.best_full_serial_constraint_sum is not None
    ]
    parallel_constraint_sums = [
        float(item.best_full_parallel_constraint_sum)
        for item in solved
        if item.best_full_parallel_constraint_sum is not None
    ]
    aggregate = {
        "episodes": len(values),
        "unique_patients": len({item.patient_id for item in values}),
        "unique_contexts": len(
            {(item.patient_id, item.prediction_set_id) for item in values}
        ),
        "full_solution_rate": len(solved) / len(values),
        "cheap_solution_rate": len(cheap_improvements) / len(values),
        "mean_best_cheap_improvement": (
            mean(cheap_improvements) if cheap_improvements else None
        ),
        "median_best_cheap_improvement": (
            median(cheap_improvements) if cheap_improvements else None
        ),
        "feasible_full_rate": (
            sum(item.best_full_feasible is True for item in solved) / len(solved)
            if solved
            else None
        ),
        "mean_best_full_objective": mean(objectives) if objectives else None,
        "median_best_full_objective": median(objectives) if objectives else None,
        "mean_best_full_constraint_sum": (
            mean(constraint_sums) if constraint_sums else None
        ),
        "mean_best_full_target_constraint_sum": (
            mean(target_constraint_sums) if target_constraint_sums else None
        ),
        "mean_best_full_serial_constraint_sum": (
            mean(serial_constraint_sums) if serial_constraint_sums else None
        ),
        "mean_best_full_parallel_constraint_sum": (
            mean(parallel_constraint_sums) if parallel_constraint_sums else None
        ),
        "mean_full_calls_attempted": mean(
            item.full_calls_attempted for item in values
        ),
        "mean_cheap_calls": mean(item.cheap_calls for item in values),
        "mean_solver_seconds": mean(item.solver_seconds for item in values),
        "mean_invalid_action_rate": mean(
            item.invalid_action_rate for item in values
        ),
    }
    return aggregate
