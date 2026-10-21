

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Mapping

from radoncgym.core.constants import (
    CORE_SCHEMA_VERSION,
    MAX_OBJECTIVE_WEIGHT,
    MIN_OBJECTIVE_WEIGHT,
    OBJECTIVE_NAMES,
    PRIMARY_FORMULATION,
)
from radoncgym.core.errors import ContractError


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FeedbackSource(str, Enum):
    pass

    CONTEXT = "context"
    CHEAP_LIVE_EVALUATOR = "cheap_live_evaluator"
    CHEAP_EXACT_CACHE = "cheap_exact_cache"
    CHEAP_LEARNED_SURROGATE = "cheap_learned_surrogate"
    FULL_LIVE_OPTIMIZER = "full_live_optimizer"
    FULL_EXACT_CACHE = "full_exact_cache"
    APPROXIMATE_CACHE_SURROGATE = "approximate_cache_surrogate"
    ENVIRONMENT_BOOKKEEPING = "environment_bookkeeping"


class SolverStatus(str, Enum):
    pass

    SOLVED = "solved"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NUMERICAL_ERROR = "numerical_error"
    ERROR = "error"
    NOT_APPLICABLE = "not_applicable"


class EndReason(str, Enum):
    pass

    AGENT_STOP = "agent_stop"
    INVALID_ACTION_LIMIT = "invalid_action_limit"
    UNRECOVERABLE_BACKEND = "unrecoverable_backend"
    MAX_STEPS = "max_steps"
    CHEAP_BUDGET_EXHAUSTED = "cheap_budget_exhausted"
    WALL_TIME_BUDGET_EXHAUSTED = "wall_time_budget_exhausted"


class TaskKind(str, Enum):
    pass

    SINGLE_SHOT = "single_shot"
    SEQUENTIAL_FULL = "sequential_full"
    BUDGETED_MULTI_FIDELITY = "budgeted_multi_fidelity"
    CLINICALLY_CONSTRAINED = "clinically_constrained"
    OFFLINE_TRAJECTORY = "offline_trajectory"


class FidelityControl(str, Enum):
    pass

    FULL_ONLY = "full_only"
    AGENT_CHOICE = "agent_choice"
    DATASET_ONLY = "dataset_only"


class FullIncumbentRule(str, Enum):
    pass

    QUALITY_ONLY = "quality_only"
    FEASIBILITY_THEN_QUALITY = "feasibility_then_quality"
    PRIORITY_LEXICOGRAPHIC = "priority_lexicographic"


@dataclass(frozen=True, slots=True)
class TaskConfig:
    pass

    task_id: str
    kind: TaskKind
    fidelity_control: FidelityControl
    full_incumbent_rule: FullIncumbentRule = FullIncumbentRule.QUALITY_ONLY
    stop_enabled: bool = True
    evaluate_cheap_on_full_request: bool = True
    allow_stop_before_candidate: bool = False
    minimum_evaluated_candidates_before_stop: int = 1

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ContractError("task_id must be non-empty")
        if self.minimum_evaluated_candidates_before_stop < 0:
            raise ContractError(
                "minimum_evaluated_candidates_before_stop cannot be negative"
            )
        if (
            self.kind is TaskKind.OFFLINE_TRAJECTORY
            and self.fidelity_control is not FidelityControl.DATASET_ONLY
        ):
            raise ContractError("offline trajectory tasks must be dataset-only")
        if (
            self.kind is not TaskKind.OFFLINE_TRAJECTORY
            and self.fidelity_control is FidelityControl.DATASET_ONLY
        ):
            raise ContractError("dataset-only fidelity is reserved for offline tasks")


@dataclass(frozen=True, slots=True)
class MetricVector:
    pass

    items: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        names: set[str] = set()
        for name, value in self.items:
            if not name:
                raise ContractError("Metric names must be non-empty")
            if name in names:
                raise ContractError(f"Duplicate metric name: {name}")
            if not math.isfinite(value):
                raise ContractError(f"Metric {name} must be finite, got {value!r}")
            names.add(name)

    @classmethod
    def from_mapping(cls, values: Mapping[str, float] | None) -> MetricVector:
        if not values:
            return cls()
        return cls(tuple((str(name), float(value)) for name, value in values.items()))

    def to_dict(self) -> dict[str, float]:
        return dict(self.items)

    def get(self, name: str, default: float | None = None) -> float | None:
        for metric_name, value in self.items:
            if metric_name == name:
                return value
        return default

    def sum(self) -> float:
        return math.fsum(value for _, value in self.items)


@dataclass(frozen=True, slots=True)
class PublicContextFeatures:
    pass

    schema_version: str
    items: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ContractError("public context schema_version is required")
        names: set[str] = set()
        forbidden = ("reference", "hidden", "oracle", "ground_truth")
        for name, value in self.items:
            normalized = name.lower()
            if not name:
                raise ContractError("public context feature names must be non-empty")
            if name in names:
                raise ContractError("duplicate public context feature")
            if any(token in normalized for token in forbidden):
                raise ContractError(
                    "forbidden private-derived public context feature"
                )
            if not math.isfinite(value):
                raise ContractError(
                    f"public context feature {name} must be finite"
                )
            names.add(name)

    @classmethod
    def empty(cls) -> PublicContextFeatures:
        return cls(schema_version="public-context-empty-0.1.0")

    def to_dict(self) -> dict[str, float]:
        return dict(self.items)


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    pass

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(OBJECTIVE_NAMES):
            raise ContractError(
                f"Expected {len(OBJECTIVE_NAMES)} weights, got {len(self.values)}"
            )
        for name, value in zip(OBJECTIVE_NAMES, self.values, strict=True):
            if not math.isfinite(value):
                raise ContractError(f"Weight {name} must be finite, got {value!r}")

    @classmethod
    def all_ones(cls) -> ObjectiveWeights:
        return cls((1.0,) * len(OBJECTIVE_NAMES))

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> ObjectiveWeights:
        expected = set(OBJECTIVE_NAMES)
        actual = set(values)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ContractError("Invalid objective keys: " + ", ".join(details))
        return cls(tuple(float(values[name]) for name in OBJECTIVE_NAMES))

    def to_dict(self) -> dict[str, float]:
        return dict(zip(OBJECTIVE_NAMES, self.values, strict=True))

    def in_bounds(
        self,
        minimum: float = MIN_OBJECTIVE_WEIGHT,
        maximum: float = MAX_OBJECTIVE_WEIGHT,
    ) -> bool:
        return all(minimum <= value <= maximum for value in self.values)


@dataclass(frozen=True, slots=True)
class PlanningContext:
    pass

    case_id: str
    patient_id: str
    prediction_set_id: str
    split_id: str
    clinical_profile_id: str = "openkbp_hn_v1"
    planning_formulation: str = PRIMARY_FORMULATION
    environment_version: str = CORE_SCHEMA_VERSION
    evaluator_request_id: str | None = None
    evaluator_request_key_sha256: str | None = None

    def __post_init__(self) -> None:
        required = {
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "prediction_set_id": self.prediction_set_id,
            "split_id": self.split_id,
            "clinical_profile_id": self.clinical_profile_id,
            "planning_formulation": self.planning_formulation,
            "environment_version": self.environment_version,
        }
        for name, value in required.items():
            if not value:
                raise ContractError(f"{name} must be non-empty")
        if self.evaluator_request_id is not None and (
            not isinstance(self.evaluator_request_id, str)
            or not self.evaluator_request_id
        ):
            raise ContractError(
                "evaluator_request_id must be non-empty when provided"
            )
        if (
            self.evaluator_request_key_sha256 is not None
            and not _SHA256_RE.fullmatch(self.evaluator_request_key_sha256)
        ):
            raise ContractError(
                "evaluator_request_key_sha256 must be lowercase hexadecimal"
            )


@dataclass(frozen=True, slots=True)
class CandidateAction:
    pass

    weights: ObjectiveWeights | None = None
    request_full_verifier: bool = False
    stop: bool = False


@dataclass(frozen=True, slots=True)
class ComputeCost:
    pass

    wall_seconds: float = 0.0
    solver_seconds: float = 0.0
    cheap_calls: int = 0
    full_calls: int = 0

    def __post_init__(self) -> None:
        if self.wall_seconds < 0 or not math.isfinite(self.wall_seconds):
            raise ContractError("wall_seconds must be finite and nonnegative")
        if self.solver_seconds < 0 or not math.isfinite(self.solver_seconds):
            raise ContractError("solver_seconds must be finite and nonnegative")
        if self.cheap_calls < 0 or self.full_calls < 0:
            raise ContractError("call counts must be nonnegative")

    def __add__(self, other: ComputeCost) -> ComputeCost:
        return ComputeCost(
            wall_seconds=self.wall_seconds + other.wall_seconds,
            solver_seconds=self.solver_seconds + other.solver_seconds,
            cheap_calls=self.cheap_calls + other.cheap_calls,
            full_calls=self.full_calls + other.full_calls,
        )


@dataclass(frozen=True, slots=True)
class EvaluationBackendIdentity:
    pass

    evaluator_id: str
    evaluator_version: str
    evaluator_profile_id: str
    solver_id: str | None = None
    solver_version: str | None = None
    solver_profile_id: str | None = None
    cache_namespace: str | None = None
    cache_record_sha256: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("evaluator_id", self.evaluator_id),
            ("evaluator_version", self.evaluator_version),
            ("evaluator_profile_id", self.evaluator_profile_id),
        ):
            if not isinstance(value, str) or not value:
                raise ContractError(f"{name} must be non-empty")
        solver_values = (
            self.solver_id,
            self.solver_version,
            self.solver_profile_id,
        )
        if any(value is None for value in solver_values) and any(
            value is not None for value in solver_values
        ):
            raise ContractError(
                "solver_id, solver_version, and solver_profile_id "
                "must be declared together"
            )
        if any(
            value is not None
            and (not isinstance(value, str) or not value)
            for value in solver_values
        ):
            raise ContractError("solver identity values must be non-empty")
        cache_values = (self.cache_namespace, self.cache_record_sha256)
        if any(value is None for value in cache_values) and any(
            value is not None for value in cache_values
        ):
            raise ContractError(
                "cache_namespace and cache_record_sha256 must be declared together"
            )
        if self.cache_namespace is not None and (
            not isinstance(self.cache_namespace, str)
            or not self.cache_namespace
        ):
            raise ContractError("cache_namespace must be non-empty")
        if (
            self.cache_record_sha256 is not None
            and not _SHA256_RE.fullmatch(self.cache_record_sha256)
        ):
            raise ContractError(
                "cache_record_sha256 must be lowercase hexadecimal"
            )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    pass

    source: FeedbackSource
    status: SolverStatus
    metrics: MetricVector = field(default_factory=MetricVector)
    constraints: MetricVector = field(default_factory=MetricVector)
    cost: ComputeCost = field(default_factory=ComputeCost)
    uncertainty: float | None = None
    message: str | None = None
    evaluator_request_id: str | None = None
    evaluator_request_key_sha256: str | None = None
    backend_identity: EvaluationBackendIdentity | None = None

    def __post_init__(self) -> None:
        if self.uncertainty is not None and (
            self.uncertainty < 0 or not math.isfinite(self.uncertainty)
        ):
            raise ContractError("uncertainty must be finite and nonnegative")
        if self.evaluator_request_id is not None and (
            not isinstance(self.evaluator_request_id, str)
            or not self.evaluator_request_id
        ):
            raise ContractError(
                "evaluator_request_id must be non-empty when provided"
            )
        if (
            self.evaluator_request_key_sha256 is not None
            and not _SHA256_RE.fullmatch(self.evaluator_request_key_sha256)
        ):
            raise ContractError(
                "evaluator_request_key_sha256 must be lowercase hexadecimal"
            )


@dataclass(frozen=True, slots=True)
class StepOutcome:
    pass

    executed_weights: ObjectiveWeights
    cheap: EvaluationResult
    full: EvaluationResult | None

    @property
    def total_cost(self) -> ComputeCost:
        if self.full is None:
            return self.cheap.cost
        return self.cheap.cost + self.full.cost


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    pass

    max_steps: int
    max_full_calls: int
    max_cheap_calls: int | None = None
    max_wall_seconds: float | None = None
    invalid_action_limit: int = 3

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ContractError("max_steps must be positive")
        if self.max_full_calls < 0:
            raise ContractError("max_full_calls must be nonnegative")
        if self.max_cheap_calls is not None and self.max_cheap_calls <= 0:
            raise ContractError("max_cheap_calls must be positive when provided")
        if self.max_wall_seconds is not None and self.max_wall_seconds <= 0:
            raise ContractError("max_wall_seconds must be positive when provided")
        if self.invalid_action_limit <= 0:
            raise ContractError("invalid_action_limit must be positive")


@dataclass(frozen=True, slots=True)
class BudgetState:
    pass

    steps_used: int = 0
    cheap_calls_used: int = 0
    full_calls_attempted: int = 0
    full_calls_solved: int = 0
    full_calls_failed: int = 0
    invalid_actions: int = 0
    wall_seconds: float = 0.0
    solver_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    pass

    step_index: int
    action: CandidateAction
    accepted: bool
    validation_error: str | None
    duplicate_of_step: int | None
    outcome: StepOutcome | None
    terminated: bool
    truncated: bool
    end_reason: EndReason | None


@dataclass(frozen=True, slots=True)
class Incumbents:
    pass

    best_cheap_step: int | None = None
    best_full_step: int | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    pass

    context: PlanningContext
    public_context: PublicContextFeatures
    task_config: TaskConfig
    budget_config: BudgetConfig
    budget_state: BudgetState
    history: tuple[HistoryEntry, ...]
    incumbents: Incumbents


@dataclass(frozen=True, slots=True)
class StepResult:
    pass

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
