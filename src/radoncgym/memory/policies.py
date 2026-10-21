

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Protocol

from radoncgym.core.errors import ContractError
from radoncgym.core.metric_schema import clinical_improvement
from radoncgym.core.models import ObjectiveWeights, Observation, SolverStatus
from radoncgym.memory.semantic import TrainingSemanticResponseAtlas
from radoncgym.memory.physics_response import TrainingFullPhysicsResponseAtlas
from radoncgym.trajectories.serialization import history_entry_to_dict


_FORBIDDEN_REFLECTION_TOKENS = (
    "ground_truth",
    "hidden_m1",
    "hidden_reference",
    "oracle_reference",
    "reference_dose_error",
)


_REFLECTION_EVIDENCE_KINDS = {
    "accepted",
    "requested_full",
    "cheap_metric",
    "cheap_constraint",
    "cheap_status",
    "full_metric",
    "full_constraint",
    "full_status",
}


@dataclass(frozen=True, slots=True)
class ReflectionEvidence:
    pass

    source_step_index: int
    kind: str
    key: str | None
    observed_value: str | float | bool

    def __post_init__(self) -> None:
        if self.source_step_index < 0:
            raise ContractError("reflection evidence step must be nonnegative")
        if self.kind not in _REFLECTION_EVIDENCE_KINDS:
            raise ContractError(f"unsupported reflection evidence kind: {self.kind}")
        keyed = self.kind in {
            "cheap_metric",
            "cheap_constraint",
            "full_metric",
            "full_constraint",
        }
        if keyed != bool(self.key):
            raise ContractError(
                f"reflection evidence {self.kind} has invalid key {self.key!r}"
            )
        if isinstance(self.observed_value, float) and not math.isfinite(
            self.observed_value
        ):
            raise ContractError("reflection evidence value must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_step_index": self.source_step_index,
            "kind": self.kind,
            "key": self.key,
            "observed_value": self.observed_value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReflectionEvidence:
        raw_step = value.get("source_step_index")
        if isinstance(raw_step, bool) or not isinstance(raw_step, int):
            raise ContractError("reflection evidence step must be an integer")
        raw_kind = value.get("kind")
        if not isinstance(raw_kind, str):
            raise ContractError("reflection evidence kind must be a string")
        raw_key = value.get("key")
        if raw_key is not None and not isinstance(raw_key, str):
            raise ContractError("reflection evidence key must be a string or null")
        raw_observed = value.get("observed_value")
        if not isinstance(raw_observed, (str, bool, int, float)):
            raise ContractError(
                "reflection evidence observed_value must be scalar"
            )
        observed: str | float | bool = raw_observed
        if isinstance(raw_observed, int) and not isinstance(raw_observed, bool):
            observed = float(raw_observed)
        return cls(
            source_step_index=raw_step,
            kind=raw_kind,
            key=raw_key,
            observed_value=observed,
        )


@dataclass(frozen=True, slots=True)
class ReflectionRecord:
    pass

    reflection_id: str
    text: str
    source_step_indices: tuple[int, ...]
    generator_id: str
    generator_revision: str
    prompt_sha256: str
    source_episode_id: str
    source_case_id: str
    source_patient_id: str
    source_prediction_set_id: str
    source_split_id: str
    source_history_sha256: str
    evidence: tuple[ReflectionEvidence, ...]
    created_after_step_index: int

    def __post_init__(self) -> None:
        required_identity = (
            self.reflection_id,
            self.generator_id,
            self.generator_revision,
            self.source_episode_id,
            self.source_case_id,
            self.source_patient_id,
            self.source_prediction_set_id,
            self.source_split_id,
        )
        if any(not value for value in required_identity):
            raise ContractError("reflection identity and generator are required")
        for label, digest in (
            ("prompt_sha256", self.prompt_sha256),
            ("source_history_sha256", self.source_history_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ContractError(f"reflection {label} must be lowercase SHA-256")
        normalized = self.text.strip()
        if not normalized:
            raise ContractError("reflection text cannot be empty")
        if len(normalized) > 1200:
            raise ContractError("reflection text exceeds the 1200-character bound")
        lowered = normalized.lower()
        forbidden = [
            token for token in _FORBIDDEN_REFLECTION_TOKENS if token in lowered
        ]
        if forbidden:
            raise ContractError(
                f"reflection contains forbidden hidden-feedback tokens: {forbidden}"
            )
        if not self.source_step_indices:
            raise ContractError("reflection requires at least one public source step")
        if self.source_step_indices[0] < 0 or self.created_after_step_index < 0:
            raise ContractError("reflection steps must be nonnegative")
        if tuple(sorted(set(self.source_step_indices))) != self.source_step_indices:
            raise ContractError("reflection source steps must be sorted and unique")
        if self.source_step_indices[-1] > self.created_after_step_index:
            raise ContractError("reflection cannot cite a future source step")
        if not self.evidence:
            raise ContractError("reflection requires machine-checkable evidence")
        evidence_steps = tuple(
            sorted({item.source_step_index for item in self.evidence})
        )
        if any(step not in self.source_step_indices for step in evidence_steps):
            raise ContractError("reflection evidence must use declared source steps")

    def to_dict(self) -> dict[str, object]:
        return {
            "reflection_id": self.reflection_id,
            "text": self.text,
            "source_step_indices": list(self.source_step_indices),
            "generator_id": self.generator_id,
            "generator_revision": self.generator_revision,
            "prompt_sha256": self.prompt_sha256,
            "source_episode_id": self.source_episode_id,
            "source_case_id": self.source_case_id,
            "source_patient_id": self.source_patient_id,
            "source_prediction_set_id": self.source_prediction_set_id,
            "source_split_id": self.source_split_id,
            "source_history_sha256": self.source_history_sha256,
            "evidence": [item.to_dict() for item in self.evidence],
            "created_after_step_index": self.created_after_step_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReflectionRecord:
        def required_string(name: str) -> str:
            raw = value.get(name)
            if not isinstance(raw, str):
                raise ContractError(f"reflection {name} must be a string")
            return raw

        raw_steps = value.get("source_step_indices")
        if not isinstance(raw_steps, list) or any(
            isinstance(step, bool) or not isinstance(step, int)
            for step in raw_steps
        ):
            raise ContractError(
                "reflection source_step_indices must be an integer list"
            )
        raw_evidence = value.get("evidence")
        if not isinstance(raw_evidence, list) or any(
            not isinstance(item, Mapping) for item in raw_evidence
        ):
            raise ContractError("reflection evidence must be an object list")
        raw_created = value.get("created_after_step_index")
        if isinstance(raw_created, bool) or not isinstance(raw_created, int):
            raise ContractError(
                "reflection created_after_step_index must be an integer"
            )
        return cls(
            reflection_id=required_string("reflection_id"),
            text=required_string("text"),
            source_step_indices=tuple(raw_steps),
            generator_id=required_string("generator_id"),
            generator_revision=required_string("generator_revision"),
            prompt_sha256=required_string("prompt_sha256"),
            source_episode_id=required_string("source_episode_id"),
            source_case_id=required_string("source_case_id"),
            source_patient_id=required_string("source_patient_id"),
            source_prediction_set_id=required_string(
                "source_prediction_set_id"
            ),
            source_split_id=required_string("source_split_id"),
            source_history_sha256=required_string("source_history_sha256"),
            evidence=tuple(
                ReflectionEvidence.from_dict(item) for item in raw_evidence
            ),
            created_after_step_index=raw_created,
        )


def public_history_prefix_sha256(
    observation: Observation,
    created_after_step_index: int,
) -> str:
    pass

    if created_after_step_index < 0 or created_after_step_index >= len(
        observation.history
    ):
        raise ContractError("reflection history prefix is outside the episode")
    payload = [
        history_entry_to_dict(entry)
        for entry in observation.history[: created_after_step_index + 1]
    ]
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def reflection_records_sha256(
    reflections: tuple[ReflectionRecord, ...],
) -> str:
    pass

    payload = sorted(
        (reflection.to_dict() for reflection in reflections),
        key=lambda item: str(item["reflection_id"]),
    )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MemoryPageOperation:
    pass

    operation: str
    step_indices: tuple[int, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.operation not in {"archive_resident", "archive_read"}:
            raise ContractError(f"unsupported memory page operation: {self.operation}")
        if not self.step_indices:
            raise ContractError("page operation requires at least one step")
        if tuple(sorted(set(self.step_indices))) != self.step_indices:
            raise ContractError("page-operation steps must be sorted and unique")
        if not self.reason:
            raise ContractError("page operation requires a reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "step_indices": list(self.step_indices),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ManagedMemoryRecord:
    pass

    record_id: str
    version: int
    kind: str
    text: str
    source_case_id: str
    source_patient_id: str
    source_prediction_set_id: str
    source_split_id: str
    source_history_sha256: str
    evidence: tuple[ReflectionEvidence, ...]
    linked_record_ids: tuple[str, ...]
    created_after_step_index: int
    supersedes_version: int | None = None

    def __post_init__(self) -> None:
        if not self.record_id or self.version < 1:
            raise ContractError("managed memory record identity is invalid")
        if any(
            not value
            for value in (
                self.source_case_id,
                self.source_patient_id,
                self.source_prediction_set_id,
                self.source_split_id,
            )
        ):
            raise ContractError("managed memory source context is required")
        if self.kind not in {"note", "summary"}:
            raise ContractError(f"unsupported managed memory kind: {self.kind}")
        normalized = self.text.strip()
        if not normalized or len(normalized) > 1200:
            raise ContractError("managed memory text must contain 1-1200 characters")
        lowered = normalized.lower()
        forbidden = [
            token for token in _FORBIDDEN_REFLECTION_TOKENS if token in lowered
        ]
        if forbidden:
            raise ContractError(
                f"managed memory contains forbidden hidden-feedback tokens: {forbidden}"
            )
        if len(self.source_history_sha256) != 64:
            raise ContractError("managed memory history digest must be SHA-256")
        if not self.evidence:
            raise ContractError("managed memory requires public evidence")
        if self.created_after_step_index < 0:
            raise ContractError("managed memory creation step must be nonnegative")
        if any(
            item.source_step_index > self.created_after_step_index
            for item in self.evidence
        ):
            raise ContractError("managed memory cannot cite future evidence")
        if tuple(sorted(set(self.linked_record_ids))) != self.linked_record_ids:
            raise ContractError("managed memory links must be sorted and unique")
        if self.record_id in self.linked_record_ids:
            raise ContractError("managed memory cannot link to itself")
        if self.version == 1 and self.supersedes_version is not None:
            raise ContractError("initial managed memory cannot supersede a version")
        if self.version > 1 and self.supersedes_version != self.version - 1:
            raise ContractError("managed memory updates must supersede prior version")

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "version": self.version,
            "kind": self.kind,
            "text": self.text,
            "source_case_id": self.source_case_id,
            "source_patient_id": self.source_patient_id,
            "source_prediction_set_id": self.source_prediction_set_id,
            "source_split_id": self.source_split_id,
            "source_history_sha256": self.source_history_sha256,
            "evidence": [item.to_dict() for item in self.evidence],
            "linked_record_ids": list(self.linked_record_ids),
            "created_after_step_index": self.created_after_step_index,
            "supersedes_version": self.supersedes_version,
        }


@dataclass(frozen=True, slots=True)
class ManagedMemoryOperation:
    pass

    operation_index: int
    operation: str
    record_id: str | None
    record_version: int | None
    target_record_ids: tuple[str, ...]
    created_after_step_index: int
    disposition: str
    reason: str

    def __post_init__(self) -> None:
        if self.operation_index < 0 or self.created_after_step_index < 0:
            raise ContractError("managed memory operation indices must be nonnegative")
        if self.operation not in {
            "add",
            "update",
            "link",
            "discard",
            "retrieve",
            "summarize",
        }:
            raise ContractError(f"unsupported managed memory operation: {self.operation}")
        if self.disposition not in {"accepted", "rejected"}:
            raise ContractError("managed memory operation disposition is invalid")
        if not self.reason:
            raise ContractError("managed memory operation requires a reason")
        if tuple(sorted(set(self.target_record_ids))) != self.target_record_ids:
            raise ContractError("managed memory operation targets must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_index": self.operation_index,
            "operation": self.operation,
            "record_id": self.record_id,
            "record_version": self.record_version,
            "target_record_ids": list(self.target_record_ids),
            "created_after_step_index": self.created_after_step_index,
            "disposition": self.disposition,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RetrievedEpisode:
    pass

    source_episode_id: str
    source_patient_id: str
    source_prediction_set_id: str
    source_split_id: str
    similarity_score: float
    matched_feature_count: int
    selected_step_index: int
    selected_by: str
    weights: tuple[tuple[str, float], ...]
    outcome_summary: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.source_episode_id or not self.source_patient_id:
            raise ContractError("retrieved episode provenance is required")
        if self.source_split_id != "train":
            raise ContractError("external memory sources must use the train split")
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ContractError("retrieval similarity must lie in [0, 1]")
        if self.matched_feature_count <= 0 or self.selected_step_index < 0:
            raise ContractError("retrieval evidence must identify features and a step")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_episode_id": self.source_episode_id,
            "source_patient_id": self.source_patient_id,
            "source_prediction_set_id": self.source_prediction_set_id,
            "source_split_id": self.source_split_id,
            "similarity_score": self.similarity_score,
            "matched_feature_count": self.matched_feature_count,
            "selected_step_index": self.selected_step_index,
            "selected_by": self.selected_by,
            "weights": dict(self.weights),
            "outcome_summary": dict(self.outcome_summary),
        }


@dataclass(frozen=True, slots=True)
class VerifiedResponseRecord:
    pass

    step_index: int
    previous_candidate_step_index: int | None
    previous_full_step_index: int | None
    weights: tuple[tuple[str, float], ...]
    weight_delta: tuple[tuple[str, float], ...]
    cheap_metrics: tuple[tuple[str, float], ...]
    full_metrics: tuple[tuple[str, float], ...]
    full_constraints: tuple[tuple[str, float], ...]
    full_metric_delta: tuple[tuple[str, float], ...]
    full_constraint_delta: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "previous_candidate_step_index": self.previous_candidate_step_index,
            "previous_full_step_index": self.previous_full_step_index,
            "weights": dict(self.weights),
            "weight_delta": dict(self.weight_delta),
            "cheap_metrics": dict(self.cheap_metrics),
            "full_metrics": dict(self.full_metrics),
            "full_constraints": dict(self.full_constraints),
            "full_metric_delta": dict(self.full_metric_delta),
            "full_constraint_delta": dict(self.full_constraint_delta),
        }


@dataclass(frozen=True, slots=True)
class ParetoCandidateRecord:
    pass

    step_index: int
    weights: tuple[tuple[str, float], ...]
    objective_vector: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "weights": dict(self.weights),
            "objective_vector": dict(self.objective_vector),
        }


@dataclass(frozen=True, slots=True)
class ClinicalSubgoalRecord:
    pass

    subgoal_index: int
    criterion: str
    priority_tier: str
    start_step_index: int
    end_step_index: int
    full_step_indices: tuple[int, ...]
    initial_violation: float
    latest_violation: float
    best_violation: float
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "subgoal_index": self.subgoal_index,
            "criterion": self.criterion,
            "priority_tier": self.priority_tier,
            "start_step_index": self.start_step_index,
            "end_step_index": self.end_step_index,
            "full_step_indices": list(self.full_step_indices),
            "initial_violation": self.initial_violation,
            "latest_violation": self.latest_violation,
            "best_violation": self.best_violation,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class MemoryView:
    module_id: str
    authoritative_history_length: int
    selected_step_indices: tuple[int, ...]
    omitted_step_indices: tuple[int, ...]
    selection_reasons: tuple[tuple[int, str], ...]
    retrieved_episodes: tuple[RetrievedEpisode, ...] = ()
    reflections: tuple[ReflectionRecord, ...] = ()
    semantic_atlases: tuple[
        TrainingSemanticResponseAtlas | TrainingFullPhysicsResponseAtlas, ...
    ] = ()
    archived_step_indices: tuple[int, ...] = ()
    page_operations: tuple[MemoryPageOperation, ...] = ()
    managed_records: tuple[ManagedMemoryRecord, ...] = ()
    managed_operations: tuple[ManagedMemoryOperation, ...] = ()
    managed_operation_budget_remaining: int | None = None
    verified_responses: tuple[VerifiedResponseRecord, ...] = ()
    pareto_candidates: tuple[ParetoCandidateRecord, ...] = ()
    clinical_subgoals: tuple[ClinicalSubgoalRecord, ...] = ()
    include_incumbent_pointers: bool = True

    def __post_init__(self) -> None:
        selected = set(self.selected_step_indices)
        omitted = set(self.omitted_step_indices)
        expected = set(range(self.authoritative_history_length))
        if selected & omitted or selected | omitted != expected:
            raise ContractError("memory selection must partition authoritative history")
        if tuple(sorted(selected)) != self.selected_step_indices:
            raise ContractError("selected memory steps must be sorted and unique")
        if tuple(sorted(omitted)) != self.omitted_step_indices:
            raise ContractError("omitted memory steps must be sorted and unique")
        reason_steps = [step for step, _ in self.selection_reasons]
        if set(reason_steps) != selected or len(reason_steps) != len(selected):
            raise ContractError("every selected memory step requires one reason")
        archived = set(self.archived_step_indices)
        if not archived <= omitted:
            raise ContractError("archived steps must be omitted from working context")
        if tuple(sorted(archived)) != self.archived_step_indices:
            raise ContractError("archived memory steps must be sorted and unique")
        for operation in self.page_operations:
            if any(
                step < 0 or step >= self.authoritative_history_length
                for step in operation.step_indices
            ):
                raise ContractError("page operation references an unknown history step")
        for subgoal in self.clinical_subgoals:
            if (
                subgoal.start_step_index < 0
                or subgoal.end_step_index >= self.authoritative_history_length
                or subgoal.start_step_index > subgoal.end_step_index
            ):
                raise ContractError("clinical subgoal references an unknown history step")
        for reflection in self.reflections:
            if reflection.created_after_step_index >= self.authoritative_history_length:
                raise ContractError(
                    "reflection was created outside authoritative history"
                )
        managed_ids = [record.record_id for record in self.managed_records]
        if len(managed_ids) != len(set(managed_ids)):
            raise ContractError("agent-visible managed memory IDs must be unique")
        if self.managed_operation_budget_remaining is not None and (
            self.managed_operation_budget_remaining < 0
        ):
            raise ContractError("managed memory operation budget cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "module_id": self.module_id,
            "authoritative_history_length": self.authoritative_history_length,
            "selected_step_indices": list(self.selected_step_indices),
            "omitted_step_indices": list(self.omitted_step_indices),
            "omitted_count": len(self.omitted_step_indices),
            "selection_reasons": {
                str(step): reason for step, reason in self.selection_reasons
            },
            "retrieved_episodes": [
                episode.to_dict() for episode in self.retrieved_episodes
            ],
            "reflections": [reflection.to_dict() for reflection in self.reflections],
            "semantic_atlases": [
                atlas.to_dict() for atlas in self.semantic_atlases
            ],
            "archived_step_indices": list(self.archived_step_indices),
            "page_operations": [
                operation.to_dict() for operation in self.page_operations
            ],
            "managed_records": [
                record.to_dict() for record in self.managed_records
            ],
            "managed_operations": [
                operation.to_dict() for operation in self.managed_operations
            ],
            "managed_operation_budget_remaining": (
                self.managed_operation_budget_remaining
            ),
            "verified_responses": [
                record.to_dict() for record in self.verified_responses
            ],
            "pareto_candidates": [
                record.to_dict() for record in self.pareto_candidates
            ],
            "clinical_subgoals": [
                record.to_dict() for record in self.clinical_subgoals
            ],
            "include_incumbent_pointers": self.include_incumbent_pointers,
        }


class MemoryPolicy(Protocol):
    @property
    def module_id(self) -> str:
        pass

    def select(self, observation: Observation) -> MemoryView:
        pass


class FullHistoryMemory:
    module_id = "full_history_lossless_v1"

    def select(self, observation: Observation) -> MemoryView:
        steps = tuple(range(len(observation.history)))
        return MemoryView(
            module_id=self.module_id,
            authoritative_history_length=len(observation.history),
            selected_step_indices=steps,
            omitted_step_indices=(),
            selection_reasons=tuple((step, "full_history") for step in steps),
        )


class NoHistoryMemory:
    pass

    module_id = "no_history_control_v1"

    def select(self, observation: Observation) -> MemoryView:
        length = len(observation.history)
        return MemoryView(
            module_id=self.module_id,
            authoritative_history_length=length,
            selected_step_indices=(),
            omitted_step_indices=tuple(range(length)),
            selection_reasons=(),
            include_incumbent_pointers=False,
        )


class RecencyWindowMemory:
    module_id = "recency_window_v1"

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ContractError("recency memory window_size must be positive")
        self.window_size = window_size

    def select(self, observation: Observation) -> MemoryView:
        length = len(observation.history)
        start = max(0, length - self.window_size)
        selected = tuple(range(start, length))
        omitted = tuple(range(start))
        return MemoryView(
            module_id=f"{self.module_id}:L={self.window_size}",
            authoritative_history_length=length,
            selected_step_indices=selected,
            omitted_step_indices=omitted,
            selection_reasons=tuple((step, "recent") for step in selected),
        )


class CriterionResponseContrastMemory:
    pass

    module_id = "criterion_response_contrast_v1"

    def __init__(
        self,
        *,
        recent_steps: int = 2,
        scalar_clinical_only: bool = False,
    ) -> None:
        if recent_steps < 0:
            raise ContractError("criterion contrast recent_steps cannot be negative")
        self.recent_steps = recent_steps
        self.scalar_clinical_only = bool(scalar_clinical_only)

    @property
    def configured_module_id(self) -> str:
        mode = "scalar_clinical" if self.scalar_clinical_only else "criterion"
        return f"{self.module_id}:recent={self.recent_steps}:mode={mode}"

    def select(self, observation: Observation) -> MemoryView:
        reasons: dict[int, set[str]] = {}

        def retain(step: int | None, reason: str) -> None:
            if step is not None:
                reasons.setdefault(step, set()).add(reason)

        retain(observation.incumbents.best_cheap_step, "best_cheap")
        retain(observation.incumbents.best_full_step, "best_full")
        if self.recent_steps:
            for entry in observation.history[-self.recent_steps :]:
                retain(entry.step_index, "recent")

        solved = [
            entry
            for entry in observation.history
            if entry.accepted
            and entry.outcome is not None
            and entry.outcome.full is not None
            and entry.outcome.full.status is SolverStatus.SOLVED
        ]
        active_criterion = None
        if solved:
            latest = solved[-1]
            retain(latest.step_index, "latest_full")
            if self.scalar_clinical_only:
                comparable = [
                    entry
                    for entry in solved
                    if clinical_improvement(entry.outcome.full) is not None
                ]
                if comparable:
                    best = max(
                        comparable,
                        key=lambda entry: (
                            clinical_improvement(entry.outcome.full),
                            -entry.step_index,
                        ),
                    )
                    worst = min(
                        comparable,
                        key=lambda entry: (
                            clinical_improvement(entry.outcome.full),
                            entry.step_index,
                        ),
                    )
                    retain(best.step_index, "scalar_clinical_best")
                    retain(worst.step_index, "scalar_clinical_worst")
            else:
                positive = [
                    (name, value)
                    for name, value in latest.outcome.full.constraints.items
                    if value > 0.0
                ]
                if positive:
                    active_criterion = max(
                        positive,
                        key=lambda item: (item[1], item[0]),
                    )[0]
                    comparable = [
                        entry
                        for entry in solved
                        if active_criterion
                        in entry.outcome.full.constraints.to_dict()
                    ]
                    best = min(
                        comparable,
                        key=lambda entry: (
                            entry.outcome.full.constraints.to_dict()[active_criterion],
                            entry.step_index,
                        ),
                    )
                    worst = max(
                        comparable,
                        key=lambda entry: (
                            entry.outcome.full.constraints.to_dict()[active_criterion],
                            -entry.step_index,
                        ),
                    )
                    retain(best.step_index, f"active_criterion_best:{active_criterion}")
                    retain(worst.step_index, f"active_criterion_worst:{active_criterion}")

        selected = tuple(sorted(reasons))
        omitted = tuple(sorted(set(range(len(observation.history))) - set(selected)))
        criterion = "none" if active_criterion is None else active_criterion
        mode = "scalar_clinical" if self.scalar_clinical_only else "criterion"
        return MemoryView(
            module_id=(
                f"{self.module_id}:recent={self.recent_steps}:"
                f"mode={mode}:active={criterion}"
            ),
            authoritative_history_length=len(observation.history),
            selected_step_indices=selected,
            omitted_step_indices=omitted,
            selection_reasons=tuple(
                (step, "+".join(sorted(reasons[step]))) for step in selected
            ),
        )


class IncumbentFailureArchiveMemory:
    module_id = "incumbent_failure_archive_v1"

    def __init__(self, *, recent_steps: int = 2, failure_steps: int = 2) -> None:
        if recent_steps < 0 or failure_steps < 0:
            raise ContractError("archive memory counts cannot be negative")
        if recent_steps + failure_steps <= 0:
            raise ContractError("archive memory must retain at least one record class")
        self.recent_steps = recent_steps
        self.failure_steps = failure_steps

    def select(self, observation: Observation) -> MemoryView:
        reasons: dict[int, set[str]] = {}

        def retain(step: int | None, reason: str) -> None:
            if step is not None:
                reasons.setdefault(step, set()).add(reason)

        retain(observation.incumbents.best_cheap_step, "best_cheap")
        retain(observation.incumbents.best_full_step, "best_full")
        recent_entries = (
            observation.history[-self.recent_steps :]
            if self.recent_steps
            else ()
        )
        for entry in recent_entries:
            retain(entry.step_index, "recent")

        failures = [
            entry
            for entry in observation.history
            if not entry.accepted
            or (
                entry.outcome is not None
                and clinical_improvement(entry.outcome.cheap) is not None
                and observation.incumbents.best_cheap_step is not None
                and entry.step_index != observation.incumbents.best_cheap_step
            )
        ]
        selected_failures = failures[-self.failure_steps :] if self.failure_steps else ()
        for entry in selected_failures:
            retain(entry.step_index, "failure_or_nonincumbent")

        selected = tuple(sorted(reasons))
        all_steps = set(range(len(observation.history)))
        omitted = tuple(sorted(all_steps - set(selected)))
        return MemoryView(
            module_id=(
                f"{self.module_id}:recent={self.recent_steps}:"
                f"failures={self.failure_steps}"
            ),
            authoritative_history_length=len(observation.history),
            selected_step_indices=selected,
            omitted_step_indices=omitted,
            selection_reasons=tuple(
                (step, "+".join(sorted(reasons[step]))) for step in selected
            ),
        )


class VerifiedResponseMemory:
    pass

    module_id = "verified_response_memory_v1"

    def __init__(self, *, base_policy: MemoryPolicy | None = None) -> None:
        self.base_policy = base_policy or FullHistoryMemory()

    def select(self, observation: Observation) -> MemoryView:
        base = self.base_policy.select(observation)
        records: list[VerifiedResponseRecord] = []
        previous_candidate = None
        previous_full = None
        for entry in observation.history:
            if not entry.accepted or entry.outcome is None:
                continue
            current_weights = entry.outcome.executed_weights
            prior_weights = (
                ObjectiveWeights.all_ones()
                if previous_candidate is None
                else previous_candidate.outcome.executed_weights
            )
            if entry.outcome.full is not None:
                full = entry.outcome.full
                previous_full_result = (
                    None if previous_full is None else previous_full.outcome.full
                )
                records.append(
                    VerifiedResponseRecord(
                        step_index=entry.step_index,
                        previous_candidate_step_index=(
                            None
                            if previous_candidate is None
                            else previous_candidate.step_index
                        ),
                        previous_full_step_index=(
                            None if previous_full is None else previous_full.step_index
                        ),
                        weights=tuple(current_weights.to_dict().items()),
                        weight_delta=tuple(
                            (name, current - prior)
                            for name, current, prior in zip(
                                current_weights.to_dict(),
                                current_weights.values,
                                prior_weights.values,
                                strict=True,
                            )
                        ),
                        cheap_metrics=entry.outcome.cheap.metrics.items,
                        full_metrics=full.metrics.items,
                        full_constraints=full.constraints.items,
                        full_metric_delta=_metric_delta(
                            full.metrics.to_dict(),
                            None
                            if previous_full_result is None
                            else previous_full_result.metrics.to_dict(),
                        ),
                        full_constraint_delta=_metric_delta(
                            full.constraints.to_dict(),
                            None
                            if previous_full_result is None
                            else previous_full_result.constraints.to_dict(),
                        ),
                    )
                )
                previous_full = entry
            previous_candidate = entry
        return _with_performance_records(
            base,
            module_id=f"{self.module_id}:base={base.module_id}",
            verified_responses=tuple(records),
        )


class ParetoTradeoffMemory:
    pass

    module_id = "pareto_tradeoff_memory_v1"

    def __init__(self, *, base_policy: MemoryPolicy | None = None) -> None:
        self.base_policy = base_policy or VerifiedResponseMemory()

    def select(self, observation: Observation) -> MemoryView:
        base = self.base_policy.select(observation)
        candidates: list[ParetoCandidateRecord] = []
        for entry in observation.history:
            if (
                not entry.accepted
                or entry.outcome is None
                or entry.outcome.full is None
                or entry.outcome.full.status is not SolverStatus.SOLVED
            ):
                continue
            full = entry.outcome.full
            metrics = full.metrics.to_dict()
            aggregate_name = (
                "mean_protocol_violation_gy"
                if "mean_protocol_violation_gy" in metrics
                else "objective"
            )
            vector = tuple(
                sorted(
                    [(f"metric:{aggregate_name}", metrics[aggregate_name])]
                    + [
                        (f"constraint:{name}", max(0.0, value))
                        for name, value in full.constraints.items
                    ]
                )
            )
            candidates.append(
                ParetoCandidateRecord(
                    step_index=entry.step_index,
                    weights=tuple(entry.outcome.executed_weights.to_dict().items()),
                    objective_vector=vector,
                )
            )
        common_keys = (
            set.intersection(
                *(set(dict(candidate.objective_vector)) for candidate in candidates)
            )
            if candidates
            else set()
        )
        frontier = tuple(
            candidate
            for candidate in candidates
            if not any(
                _dominates(other, candidate, common_keys)
                for other in candidates
                if other.step_index != candidate.step_index
            )
        )
        return _with_performance_records(
            base,
            module_id=f"{self.module_id}:base={base.module_id}",
            pareto_candidates=frontier,
        )


def _metric_delta(
    current: Mapping[str, float],
    previous: Mapping[str, float] | None,
) -> tuple[tuple[str, float], ...]:
    if previous is None:
        return ()
    return tuple(
        sorted(
            (name, value - previous[name])
            for name, value in current.items()
            if name in previous
        )
    )


def _dominates(
    left: ParetoCandidateRecord,
    right: ParetoCandidateRecord,
    keys: set[str],
) -> bool:
    if not keys:
        return False
    left_values = dict(left.objective_vector)
    right_values = dict(right.objective_vector)
    return all(left_values[key] <= right_values[key] for key in keys) and any(
        left_values[key] < right_values[key] for key in keys
    )


def _with_performance_records(
    base: MemoryView,
    *,
    module_id: str | None = None,
    verified_responses: tuple[VerifiedResponseRecord, ...] | None = None,
    pareto_candidates: tuple[ParetoCandidateRecord, ...] | None = None,
) -> MemoryView:
    return MemoryView(
        module_id=base.module_id if module_id is None else module_id,
        authoritative_history_length=base.authoritative_history_length,
        selected_step_indices=base.selected_step_indices,
        omitted_step_indices=base.omitted_step_indices,
        selection_reasons=base.selection_reasons,
        retrieved_episodes=base.retrieved_episodes,
        reflections=base.reflections,
        semantic_atlases=base.semantic_atlases,
        archived_step_indices=base.archived_step_indices,
        page_operations=base.page_operations,
        managed_records=base.managed_records,
        managed_operations=base.managed_operations,
        managed_operation_budget_remaining=base.managed_operation_budget_remaining,
        verified_responses=(
            base.verified_responses
            if verified_responses is None
            else verified_responses
        ),
        pareto_candidates=(
            base.pareto_candidates
            if pareto_candidates is None
            else pareto_candidates
        ),
        include_incumbent_pointers=base.include_incumbent_pointers,
    )


class ReflectionBufferMemory:
    pass

    module_id = "reflection_buffer_v1"

    def __init__(
        self,
        reflections: tuple[ReflectionRecord, ...],
        *,
        max_reflections: int = 3,
        base_policy: MemoryPolicy | None = None,
    ) -> None:
        if max_reflections <= 0:
            raise ContractError("reflection buffer size must be positive")
        ids = [record.reflection_id for record in reflections]
        if len(ids) != len(set(ids)):
            raise ContractError("reflection IDs must be unique")
        self.reflections = reflections
        self.max_reflections = max_reflections
        self.base_policy = base_policy or IncumbentFailureArchiveMemory()
        self.records_sha256 = reflection_records_sha256(reflections)

    def select(self, observation: Observation) -> MemoryView:
        base = self.base_policy.select(observation)
        active = tuple(
            reflection
            for reflection in self.reflections
            if reflection.created_after_step_index < len(observation.history)
        )[-self.max_reflections :]
        for reflection in active:
            _validate_reflection_against_observation(reflection, observation)
        return MemoryView(
            module_id=(
                f"{self.module_id}:max={self.max_reflections}:"
                f"records={self.records_sha256[:12]}:base={base.module_id}"
            ),
            authoritative_history_length=base.authoritative_history_length,
            selected_step_indices=base.selected_step_indices,
            omitted_step_indices=base.omitted_step_indices,
            selection_reasons=base.selection_reasons,
            retrieved_episodes=base.retrieved_episodes,
            reflections=active,
            semantic_atlases=base.semantic_atlases,
            archived_step_indices=base.archived_step_indices,
            page_operations=base.page_operations,
            managed_records=base.managed_records,
            managed_operations=base.managed_operations,
            managed_operation_budget_remaining=(
                base.managed_operation_budget_remaining
            ),
            include_incumbent_pointers=base.include_incumbent_pointers,
        )


class SemanticResponseAtlasMemory:
    pass

    module_id = "training_semantic_response_atlas_v1"

    def __init__(
        self,
        atlas: TrainingSemanticResponseAtlas,
        *,
        base_policy: MemoryPolicy | None = None,
    ) -> None:
        if not atlas.source_files:
            raise ContractError("agent-visible semantic atlas requires source hashes")
        self.atlas = atlas
        self.base_policy = base_policy or IncumbentFailureArchiveMemory()

    def select(self, observation: Observation) -> MemoryView:
        base = self.base_policy.select(observation)
        return MemoryView(
            module_id=f"{self.module_id}:base={base.module_id}",
            authoritative_history_length=base.authoritative_history_length,
            selected_step_indices=base.selected_step_indices,
            omitted_step_indices=base.omitted_step_indices,
            selection_reasons=base.selection_reasons,
            retrieved_episodes=base.retrieved_episodes,
            reflections=base.reflections,
            semantic_atlases=base.semantic_atlases + (self.atlas,),
            archived_step_indices=base.archived_step_indices,
            page_operations=base.page_operations,
            managed_records=base.managed_records,
            managed_operations=base.managed_operations,
            managed_operation_budget_remaining=(
                base.managed_operation_budget_remaining
            ),
            include_incumbent_pointers=base.include_incumbent_pointers,
        )


class FullPhysicsResponseAtlasMemory:
    pass

    module_id = "training_full_physics_response_atlas_v1"

    def __init__(
        self,
        atlas: TrainingFullPhysicsResponseAtlas,
        *,
        base_policy: MemoryPolicy | None = None,
    ) -> None:
        if not atlas.source_episode_sha256:
            raise ContractError("full-physics response memory requires source hashes")
        self.atlas = atlas
        self.base_policy = base_policy or FullHistoryMemory()

    def select(self, observation: Observation) -> MemoryView:
        base = self.base_policy.select(observation)
        return MemoryView(
            module_id=f"{self.module_id}:base={base.module_id}",
            authoritative_history_length=base.authoritative_history_length,
            selected_step_indices=base.selected_step_indices,
            omitted_step_indices=base.omitted_step_indices,
            selection_reasons=base.selection_reasons,
            retrieved_episodes=base.retrieved_episodes,
            reflections=base.reflections,
            semantic_atlases=base.semantic_atlases + (self.atlas,),
            archived_step_indices=base.archived_step_indices,
            page_operations=base.page_operations,
            managed_records=base.managed_records,
            managed_operations=base.managed_operations,
            managed_operation_budget_remaining=(
                base.managed_operation_budget_remaining
            ),
            verified_responses=base.verified_responses,
            pareto_candidates=base.pareto_candidates,
            include_incumbent_pointers=base.include_incumbent_pointers,
        )


def _validate_reflection_against_observation(
    reflection: ReflectionRecord,
    observation: Observation,
) -> None:
    context = observation.context
    expected_context = (
        context.case_id,
        context.patient_id,
        context.prediction_set_id,
        context.split_id,
    )
    source_context = (
        reflection.source_case_id,
        reflection.source_patient_id,
        reflection.source_prediction_set_id,
        reflection.source_split_id,
    )
    if source_context != expected_context:
        raise ContractError(
            "in-episode reflection source context does not match the observation"
        )
    if any(
        step >= len(observation.history) for step in reflection.source_step_indices
    ):
        raise ContractError("reflection cites a step absent from this episode")
    expected_hash = public_history_prefix_sha256(
        observation, reflection.created_after_step_index
    )
    if reflection.source_history_sha256 != expected_hash:
        raise ContractError("reflection public-history digest does not match")
    for evidence in reflection.evidence:
        actual = resolve_public_reflection_evidence(observation, evidence)
        if not _same_public_value(actual, evidence.observed_value):
            raise ContractError(
                "reflection evidence mismatch at "
                f"step={evidence.source_step_index} kind={evidence.kind} "
                f"key={evidence.key!r}: expected {evidence.observed_value!r}, "
                f"observed {actual!r}"
            )


def resolve_public_reflection_evidence(
    observation: Observation,
    evidence: ReflectionEvidence,
) -> str | float | bool:
    if evidence.source_step_index >= len(observation.history):
        raise ContractError("reflection evidence cites a missing step")
    entry = observation.history[evidence.source_step_index]
    if evidence.kind == "accepted":
        return entry.accepted
    if evidence.kind == "requested_full":
        return entry.action.request_full_verifier
    if entry.outcome is None:
        raise ContractError("reflection evidence requires an evaluated candidate")
    if evidence.kind.startswith("cheap_"):
        result = entry.outcome.cheap
        suffix = evidence.kind.removeprefix("cheap_")
    elif evidence.kind.startswith("full_"):
        result = entry.outcome.full
        suffix = evidence.kind.removeprefix("full_")
        if result is None:
            raise ContractError("reflection evidence cites absent full feedback")
    else:
        raise AssertionError(f"unhandled reflection evidence: {evidence.kind}")
    if suffix == "status":
        return result.status.value
    vector = result.metrics if suffix == "metric" else result.constraints
    value = vector.get(evidence.key or "")
    if value is None:
        raise ContractError(
            f"reflection evidence key is absent: {evidence.kind}:{evidence.key}"
        )
    return value


def _same_public_value(
    actual: str | float | bool,
    expected: str | float | bool,
) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    return type(actual) is type(expected) and actual == expected


class TieredWorkingArchiveMemory:
    pass

    module_id = "tiered_working_archive_v1"

    def __init__(
        self,
        working_steps: int = 4,
        *,
        page_in_incumbents: bool = True,
    ) -> None:
        if working_steps <= 0:
            raise ContractError("tiered working memory size must be positive")
        self.working_steps = working_steps
        self.page_in_incumbents = page_in_incumbents

    def select(self, observation: Observation) -> MemoryView:
        length = len(observation.history)
        working_start = max(0, length - self.working_steps)
        reasons: dict[int, set[str]] = {
            step: {"working_recent"} for step in range(working_start, length)
        }
        archive_reads: set[int] = set()
        if self.page_in_incumbents:
            for step, label in (
                (observation.incumbents.best_cheap_step, "best_cheap"),
                (observation.incumbents.best_full_step, "best_full"),
            ):
                if step is None:
                    continue
                reasons.setdefault(step, set()).add(label)
                if step < working_start:
                    archive_reads.add(step)
        selected = tuple(sorted(reasons))
        omitted = tuple(step for step in range(length) if step not in reasons)
        operations: list[MemoryPageOperation] = []
        if omitted:
            operations.append(
                MemoryPageOperation(
                    operation="archive_resident",
                    step_indices=omitted,
                    reason="outside_bounded_working_set",
                )
            )
        if archive_reads:
            operations.append(
                MemoryPageOperation(
                    operation="archive_read",
                    step_indices=tuple(sorted(archive_reads)),
                    reason="restore_incumbent_to_working_context",
                )
            )
        return MemoryView(
            module_id=(
                f"{self.module_id}:working={self.working_steps}:"
                f"page_incumbents={str(self.page_in_incumbents).lower()}"
            ),
            authoritative_history_length=length,
            selected_step_indices=selected,
            omitted_step_indices=omitted,
            selection_reasons=tuple(
                (step, "+".join(sorted(reasons[step]))) for step in selected
            ),
            archived_step_indices=omitted,
            page_operations=tuple(operations),
        )


class ClinicalSubgoalMemory:
    pass

    module_id = "clinical_subgoal_memory_v1"

    def __init__(self, *, pre_full_working_steps: int = 4) -> None:
        if pre_full_working_steps <= 0:
            raise ContractError("pre_full_working_steps must be positive")
        self.pre_full_working_steps = int(pre_full_working_steps)

    def select(self, observation: Observation) -> MemoryView:
        length = len(observation.history)
        subgoals = _clinical_subgoal_records(observation)
        reasons: dict[int, set[str]] = {}

        def retain(step: int | None, reason: str) -> None:
            if step is not None:
                reasons.setdefault(step, set()).add(reason)

        if subgoals:
            active = subgoals[-1]
            for step in range(active.start_step_index, length):
                retain(step, "active_clinical_subgoal")
        else:
            for step in range(max(0, length - self.pre_full_working_steps), length):
                retain(step, "pre_full_working_context")
        retain(observation.incumbents.best_cheap_step, "best_cheap")
        retain(observation.incumbents.best_full_step, "best_full")

        selected = tuple(sorted(reasons))
        omitted = tuple(step for step in range(length) if step not in reasons)
        operations = ()
        if omitted:
            operations = (
                MemoryPageOperation(
                    operation="archive_resident",
                    step_indices=omitted,
                    reason="prior_subgoal_replaced_by_grounded_gist",
                ),
            )
        return MemoryView(
            module_id=self.module_id,
            authoritative_history_length=length,
            selected_step_indices=selected,
            omitted_step_indices=omitted,
            selection_reasons=tuple(
                (step, "+".join(sorted(reasons[step]))) for step in selected
            ),
            archived_step_indices=omitted,
            page_operations=operations,
            clinical_subgoals=subgoals,
        )


def _clinical_subgoal_records(
    observation: Observation,
) -> tuple[ClinicalSubgoalRecord, ...]:
    from radoncgym.core.incumbents import constraint_priority_group

    observations: list[tuple[int, str, str, float]] = []
    tier_order = {
        "target": 0,
        "serial_or_max_dose_oar": 1,
        "parallel_or_mean_dose_oar": 2,
    }
    for entry in observation.history:
        if (
            not entry.accepted
            or entry.outcome is None
            or entry.outcome.full is None
            or entry.outcome.full.status is not SolverStatus.SOLVED
        ):
            continue
        positive = [
            (
                tier_order[constraint_priority_group(name)],
                -float(value),
                name,
                float(value),
            )
            for name, value in entry.outcome.full.constraints.items
            if value > 0.0
        ]
        if positive:
            _, _, criterion, violation = min(positive)
            tier = constraint_priority_group(criterion)
        else:
            criterion = "protocol_satisfied"
            tier = "none"
            violation = 0.0
        observations.append((entry.step_index, criterion, tier, violation))

    if not observations:
        return ()
    grouped: list[list[tuple[int, str, str, float]]] = []
    for item in observations:
        if not grouped or grouped[-1][-1][1] != item[1]:
            grouped.append([item])
        else:
            grouped[-1].append(item)
    records = []
    for index, group in enumerate(grouped):
        steps = tuple(item[0] for item in group)
        violations = tuple(item[3] for item in group)
        records.append(
            ClinicalSubgoalRecord(
                subgoal_index=index,
                criterion=group[0][1],
                priority_tier=group[0][2],
                start_step_index=steps[0],
                end_step_index=steps[-1],
                full_step_indices=steps,
                initial_violation=violations[0],
                latest_violation=violations[-1],
                best_violation=min(violations),
                status=("active" if index == len(grouped) - 1 else "archived"),
            )
        )
    return tuple(records)
