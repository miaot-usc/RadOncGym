

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Iterable, Mapping

from radoncgym.core.constants import OBJECTIVE_NAMES
from radoncgym.core.errors import ContractError
from radoncgym.core.models import FeedbackSource, SolverStatus
from radoncgym.trajectories.schema import TrajectoryEpisode


GENERATED_PLAN_TRANSITION_SCHEMA_VERSION = "generated-plan-transition-0.2.0"

_AUTHORITATIVE_FULL_SOURCES = {
    FeedbackSource.FULL_LIVE_OPTIMIZER,
    FeedbackSource.FULL_EXACT_CACHE,
}


class GeneratedPlanRewardMode(str, Enum):
    pass

    STEP_DELTA = "step_delta"
    INCUMBENT_DELTA = "incumbent_delta"


@dataclass(frozen=True, slots=True)
class GeneratedPlanTransition:
    pass

    episode_id: str
    source_episode_sha256: str
    patient_id: str
    prediction_set_id: str
    split_id: str
    task_id: str
    policy_family: str
    policy_name: str
    seed: int
    step_index: int
    full_call_index: int
    objective_names: tuple[str, ...]
    weights: tuple[float, ...]
    generated_objective: float
    positive_constraint_sum: float
    protocol_feasible: bool
    previous_generated_objective: float | None
    previous_incumbent_objective: float | None
    step_objective_delta: float
    incumbent_objective_delta: float
    solver_seconds: float
    wall_seconds: float
    source: str
    public_context_items: tuple[tuple[str, float], ...] = ()
    state_weights: tuple[float, ...] | None = None
    state_generated_metrics: tuple[tuple[str, float], ...] = ()
    state_generated_constraints: tuple[tuple[str, float], ...] = ()
    paired_cheap_metrics: tuple[tuple[str, float], ...] = ()
    paired_cheap_constraints: tuple[tuple[str, float], ...] = ()
    paired_cheap_source: str | None = None
    next_generated_metrics: tuple[tuple[str, float], ...] = ()
    next_generated_constraints: tuple[tuple[str, float], ...] = ()
    schema_version: str = GENERATED_PLAN_TRANSITION_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> GeneratedPlanTransition:
        pass

        version = str(payload.get("schema_version", ""))
        if version != GENERATED_PLAN_TRANSITION_SCHEMA_VERSION:
            raise ContractError(
                f"unsupported generated-plan transition schema: {version!r}"
            )

        def pairs(name: str) -> tuple[tuple[str, float], ...]:
            raw = payload.get(name, {})
            if not isinstance(raw, Mapping):
                raise ContractError(f"generated-plan field {name!r} must be a mapping")
            return tuple((str(key), float(value)) for key, value in raw.items())

        state_weights_raw = payload.get("state_weights")
        state_weights = (
            None
            if state_weights_raw is None
            else tuple(float(value) for value in state_weights_raw)
        )
        return cls(
            episode_id=str(payload["episode_id"]),
            source_episode_sha256=str(payload["source_episode_sha256"]),
            patient_id=str(payload["patient_id"]),
            prediction_set_id=str(payload["prediction_set_id"]),
            split_id=str(payload["split_id"]),
            task_id=str(payload["task_id"]),
            policy_family=str(payload["policy_family"]),
            policy_name=str(payload["policy_name"]),
            seed=int(payload["seed"]),
            step_index=int(payload["step_index"]),
            full_call_index=int(payload["full_call_index"]),
            objective_names=tuple(str(value) for value in payload["objective_names"]),
            weights=tuple(float(value) for value in payload["weights"]),
            generated_objective=float(payload["generated_objective"]),
            positive_constraint_sum=float(payload["positive_constraint_sum"]),
            protocol_feasible=bool(payload["protocol_feasible"]),
            previous_generated_objective=(
                None
                if payload.get("previous_generated_objective") is None
                else float(payload["previous_generated_objective"])
            ),
            previous_incumbent_objective=(
                None
                if payload.get("previous_incumbent_objective") is None
                else float(payload["previous_incumbent_objective"])
            ),
            step_objective_delta=float(payload["step_objective_delta"]),
            incumbent_objective_delta=float(payload["incumbent_objective_delta"]),
            solver_seconds=float(payload["solver_seconds"]),
            wall_seconds=float(payload["wall_seconds"]),
            source=str(payload["source"]),
            public_context_items=pairs("public_context"),
            state_weights=state_weights,
            state_generated_metrics=pairs("state_generated_metrics"),
            state_generated_constraints=pairs("state_generated_constraints"),
            paired_cheap_metrics=pairs("paired_cheap_metrics"),
            paired_cheap_constraints=pairs("paired_cheap_constraints"),
            paired_cheap_source=(
                None
                if payload.get("paired_cheap_source") is None
                else str(payload["paired_cheap_source"])
            ),
            next_generated_metrics=pairs("next_generated_metrics"),
            next_generated_constraints=pairs("next_generated_constraints"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "source_episode_sha256": self.source_episode_sha256,
            "patient_id": self.patient_id,
            "prediction_set_id": self.prediction_set_id,
            "split_id": self.split_id,
            "task_id": self.task_id,
            "policy_family": self.policy_family,
            "policy_name": self.policy_name,
            "seed": self.seed,
            "step_index": self.step_index,
            "full_call_index": self.full_call_index,
            "objective_names": list(self.objective_names),
            "weights": list(self.weights),
            "generated_objective": self.generated_objective,
            "positive_constraint_sum": self.positive_constraint_sum,
            "protocol_feasible": self.protocol_feasible,
            "previous_generated_objective": self.previous_generated_objective,
            "previous_incumbent_objective": self.previous_incumbent_objective,
            "step_objective_delta": self.step_objective_delta,
            "incumbent_objective_delta": self.incumbent_objective_delta,
            "solver_seconds": self.solver_seconds,
            "wall_seconds": self.wall_seconds,
            "source": self.source,
            "public_context": dict(self.public_context_items),
            "state_weights": (
                None if self.state_weights is None else list(self.state_weights)
            ),
            "state_generated_metrics": dict(self.state_generated_metrics),
            "state_generated_constraints": dict(
                self.state_generated_constraints
            ),
            "paired_cheap_metrics": dict(self.paired_cheap_metrics),
            "paired_cheap_constraints": dict(self.paired_cheap_constraints),
            "paired_cheap_source": self.paired_cheap_source,
            "next_generated_metrics": dict(self.next_generated_metrics),
            "next_generated_constraints": dict(
                self.next_generated_constraints
            ),
        }


def extract_generated_plan_transitions(
    episode: TrajectoryEpisode,
    *,
    feasibility_tolerance: float = 1e-8,
) -> tuple[GeneratedPlanTransition, ...]:
    pass

    if feasibility_tolerance < 0 or not math.isfinite(feasibility_tolerance):
        raise ContractError("feasibility_tolerance must be finite and nonnegative")
    source_sha256 = episode.sha256()
    context = episode.final_observation.context
    previous: float | None = None
    incumbent: float | None = None
    previous_weights: tuple[float, ...] | None = None
    previous_metrics: tuple[tuple[str, float], ...] = ()
    previous_constraints: tuple[tuple[str, float], ...] = ()
    records: list[GeneratedPlanTransition] = []
    for entry in episode.final_observation.history:
        if not entry.accepted or entry.outcome is None or entry.outcome.full is None:
            continue
        full = entry.outcome.full
        if full.status is not SolverStatus.SOLVED:
            continue
        if full.source not in _AUTHORITATIVE_FULL_SOURCES:
            raise ContractError(
                "generated-plan transition requires live or exact-cache full source"
            )
        objective = full.metrics.get("objective")
        if objective is None or not math.isfinite(objective):
            raise ContractError(
                "solved generated-plan transition requires finite objective"
            )
        objective = float(objective)
        constraint_sum = float(full.constraints.sum())
        if constraint_sum < 0 or not math.isfinite(constraint_sum):
            raise ContractError(
                "generated-plan constraint sum must be finite and nonnegative"
            )
        step_delta = 0.0 if previous is None else previous - objective
        incumbent_delta = (
            0.0 if incumbent is None else incumbent - min(incumbent, objective)
        )
        cheap = entry.outcome.cheap
        records.append(
            GeneratedPlanTransition(
                episode_id=episode.episode_id,
                source_episode_sha256=source_sha256,
                patient_id=context.patient_id,
                prediction_set_id=context.prediction_set_id,
                split_id=context.split_id,
                task_id=episode.task_id,
                policy_family=episode.policy.family,
                policy_name=episode.policy.name,
                seed=episode.seed,
                step_index=entry.step_index,
                full_call_index=len(records) + 1,
                objective_names=OBJECTIVE_NAMES,
                weights=entry.outcome.executed_weights.values,
                generated_objective=objective,
                positive_constraint_sum=constraint_sum,
                protocol_feasible=constraint_sum <= feasibility_tolerance,
                previous_generated_objective=previous,
                previous_incumbent_objective=incumbent,
                step_objective_delta=step_delta,
                incumbent_objective_delta=incumbent_delta,
                solver_seconds=full.cost.solver_seconds,
                wall_seconds=full.cost.wall_seconds,
                source=full.source.value,
                public_context_items=episode.final_observation.public_context.items,
                state_weights=previous_weights,
                state_generated_metrics=previous_metrics,
                state_generated_constraints=previous_constraints,
                paired_cheap_metrics=cheap.metrics.items,
                paired_cheap_constraints=cheap.constraints.items,
                paired_cheap_source=cheap.source.value,
                next_generated_metrics=full.metrics.items,
                next_generated_constraints=full.constraints.items,
            )
        )
        previous = objective
        incumbent = objective if incumbent is None else min(incumbent, objective)
        previous_weights = entry.outcome.executed_weights.values
        previous_metrics = full.metrics.items
        previous_constraints = full.constraints.items
    if not records:
        raise ContractError("trajectory contains no solved authoritative full outcome")
    return tuple(records)


def relabel_generated_plan_rewards(
    episode: TrajectoryEpisode,
    *,
    mode: GeneratedPlanRewardMode | str,
) -> TrajectoryEpisode:
    pass

    reward_mode = GeneratedPlanRewardMode(mode)
    transitions = extract_generated_plan_transitions(episode)
    reward_by_step = {
        transition.step_index: (
            transition.step_objective_delta
            if reward_mode is GeneratedPlanRewardMode.STEP_DELTA
            else transition.incumbent_objective_delta
        )
        for transition in transitions
    }
    source_sha256 = episode.sha256()
    metadata = dict(episode.metadata)
    for key in ("reward_relabel_source_sha256", "reward_relabel_mode"):
        if key in metadata:
            raise ContractError(f"trajectory metadata already contains {key!r}")
    metadata.update(
        {
            "reward_relabel_source_sha256": source_sha256,
            "reward_relabel_mode": reward_mode.value,
        }
    )
    return replace(
        episode,
        reward_version=(
            "GeneratedFullObjectiveImprovementReward-v0:" + reward_mode.value
        ),
        rewards=tuple(
            float(reward_by_step.get(entry.step_index, 0.0))
            for entry in episode.final_observation.history
        ),
        metadata=tuple(sorted(metadata.items())),
    )


def collect_generated_plan_transitions(
    episodes: Iterable[TrajectoryEpisode],
) -> tuple[GeneratedPlanTransition, ...]:
    pass

    records: list[GeneratedPlanTransition] = []
    seen: set[str] = set()
    for episode in episodes:
        if episode.episode_id in seen:
            raise ContractError(f"duplicate generated-plan episode: {episode.episode_id}")
        seen.add(episode.episode_id)
        records.extend(extract_generated_plan_transitions(episode))
    if not records:
        raise ContractError("generated-plan transition collection cannot be empty")
    return tuple(records)
