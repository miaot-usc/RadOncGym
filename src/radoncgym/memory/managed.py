

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from radoncgym.core.errors import ContractError
from radoncgym.core.models import Observation
from radoncgym.memory.policies import (
    IncumbentFailureArchiveMemory,
    ManagedMemoryOperation,
    ManagedMemoryRecord,
    MemoryPolicy,
    MemoryView,
    ReflectionEvidence,
    public_history_prefix_sha256,
    resolve_public_reflection_evidence,
)


@dataclass(frozen=True, slots=True)
class ManagedMemoryRequest:
    pass

    operation: str
    record_id: str | None = None
    expected_version: int | None = None
    text: str | None = None
    kind: str = "note"
    evidence: tuple[ReflectionEvidence, ...] = ()
    target_record_ids: tuple[str, ...] = ()
    reason: str = "agent_request"

    def __post_init__(self) -> None:
        if self.operation not in {
            "add",
            "update",
            "link",
            "discard",
            "retrieve",
            "summarize",
        }:
            raise ContractError(
                f"unsupported managed memory operation: {self.operation}"
            )
        if not self.reason:
            raise ContractError("managed memory request requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "record_id": self.record_id,
            "expected_version": self.expected_version,
            "text": self.text,
            "kind": self.kind,
            "evidence": [item.to_dict() for item in self.evidence],
            "target_record_ids": list(self.target_record_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ManagedMemoryRequest:
        evidence = value.get("evidence", [])
        targets = value.get("target_record_ids", [])
        if not isinstance(evidence, list) or any(
            not isinstance(item, Mapping) for item in evidence
        ):
            raise ContractError("managed memory evidence must be an object list")
        if not isinstance(targets, list) or any(
            not isinstance(item, str) for item in targets
        ):
            raise ContractError("managed memory targets must be a string list")
        expected = value.get("expected_version")
        if expected is not None and (
            isinstance(expected, bool) or not isinstance(expected, int)
        ):
            raise ContractError("managed memory expected_version must be an integer")
        return cls(
            operation=str(value.get("operation", "")),
            record_id=(
                None if value.get("record_id") is None else str(value["record_id"])
            ),
            expected_version=expected,
            text=None if value.get("text") is None else str(value["text"]),
            kind=str(value.get("kind", "note")),
            evidence=tuple(ReflectionEvidence.from_dict(item) for item in evidence),
            target_record_ids=tuple(sorted(set(targets))),
            reason=str(value.get("reason", "agent_request")),
        )


class AgentManagedMemory:
    pass

    module_id = "agent_managed_memory_operations_v1"

    def __init__(
        self,
        *,
        max_operations: int = 16,
        max_active_records: int = 8,
        base_policy: MemoryPolicy | None = None,
    ) -> None:
        if max_operations <= 0 or max_active_records <= 0:
            raise ContractError("managed memory limits must be positive")
        self.max_operations = max_operations
        self.max_active_records = max_active_records
        self.base_policy = base_policy or IncumbentFailureArchiveMemory()
        self.reset()

    @property
    def configured_module_id(self) -> str:
        return (
            f"{self.module_id}:ops={self.max_operations}:"
            f"active={self.max_active_records}:base={self.base_policy.module_id}"
        )

    def reset(self) -> None:
        self._versions: dict[str, list[ManagedMemoryRecord]] = {}
        self._active_ids: set[str] = set()
        self._retrieved_ids: set[str] = set()
        self._operations: list[ManagedMemoryOperation] = []

    @property
    def operation_log(self) -> tuple[ManagedMemoryOperation, ...]:
        return tuple(self._operations)

    @property
    def record_versions(self) -> tuple[ManagedMemoryRecord, ...]:
        return tuple(
            record
            for record_id in sorted(self._versions)
            for record in self._versions[record_id]
        )

    def apply_requests(
        self,
        observation: Observation,
        requests: Iterable[ManagedMemoryRequest],
    ) -> tuple[ManagedMemoryOperation, ...]:
        results = []
        for request in requests:
            results.append(self._apply(observation, request))
        return tuple(results)

    def select(self, observation: Observation) -> MemoryView:
        base = self.base_policy.select(observation)
        visible = tuple(
            self._versions[record_id][-1]
            for record_id in sorted(self._retrieved_ids & self._active_ids)
        )
        for record in visible:
            self._validate_record(observation, record)
        return MemoryView(
            module_id=self.configured_module_id,
            authoritative_history_length=base.authoritative_history_length,
            selected_step_indices=base.selected_step_indices,
            omitted_step_indices=base.omitted_step_indices,
            selection_reasons=base.selection_reasons,
            retrieved_episodes=base.retrieved_episodes,
            reflections=base.reflections,
            semantic_atlases=base.semantic_atlases,
            archived_step_indices=base.archived_step_indices,
            page_operations=base.page_operations,
            managed_records=visible,
            managed_operations=tuple(self._operations),
            managed_operation_budget_remaining=self.max_operations
            - len(self._operations),
            include_incumbent_pointers=base.include_incumbent_pointers,
        )

    def _apply(
        self, observation: Observation, request: ManagedMemoryRequest
    ) -> ManagedMemoryOperation:
        step = max(0, len(observation.history) - 1)
        if len(self._operations) >= self.max_operations:
            raise ContractError("managed memory operation budget exhausted")
        try:
            operation = self._apply_validated(observation, request, step)
        except ContractError as error:
            operation = ManagedMemoryOperation(
                operation_index=len(self._operations),
                operation=request.operation,
                record_id=request.record_id,
                record_version=request.expected_version,
                target_record_ids=request.target_record_ids,
                created_after_step_index=step,
                disposition="rejected",
                reason=str(error),
            )
        self._operations.append(operation)
        return operation

    def _apply_validated(
        self,
        observation: Observation,
        request: ManagedMemoryRequest,
        step: int,
    ) -> ManagedMemoryOperation:
        if not observation.history:
            raise ContractError("managed memory operations require observed history")
        if request.operation in {"add", "summarize"}:
            return self._add(observation, request, step)
        if request.operation == "update":
            return self._update(observation, request, step)
        if request.operation == "link":
            return self._link(observation, request, step)
        if request.operation == "discard":
            return self._discard(request, step)
        if request.operation == "retrieve":
            return self._retrieve(request, step)
        raise ContractError(f"unsupported managed memory operation: {request.operation}")

    def _add(
        self,
        observation: Observation,
        request: ManagedMemoryRequest,
        step: int,
    ) -> ManagedMemoryOperation:
        record_id = request.record_id or ""
        if not record_id or record_id in self._versions:
            raise ContractError("managed memory add requires a new record_id")
        if len(self._active_ids) >= self.max_active_records:
            raise ContractError("managed memory active-record limit reached")
        if request.operation == "summarize" and len(
            {item.source_step_index for item in request.evidence}
        ) < 2:
            raise ContractError("managed summary requires evidence from two steps")
        self._validate_evidence(observation, request.evidence)
        self._require_active_targets(request.target_record_ids)
        record = ManagedMemoryRecord(
            record_id=record_id,
            version=1,
            kind="summary" if request.operation == "summarize" else request.kind,
            text=request.text or "",
            source_case_id=observation.context.case_id,
            source_patient_id=observation.context.patient_id,
            source_prediction_set_id=observation.context.prediction_set_id,
            source_split_id=observation.context.split_id,
            source_history_sha256=public_history_prefix_sha256(observation, step),
            evidence=request.evidence,
            linked_record_ids=request.target_record_ids,
            created_after_step_index=step,
        )
        self._versions[record_id] = [record]
        self._active_ids.add(record_id)
        self._retrieved_ids.add(record_id)
        return self._accepted(request, step, record.version)

    def _update(
        self,
        observation: Observation,
        request: ManagedMemoryRequest,
        step: int,
    ) -> ManagedMemoryOperation:
        current = self._current(request)
        self._validate_evidence(observation, request.evidence)
        self._require_active_targets(request.target_record_ids)
        record = ManagedMemoryRecord(
            record_id=current.record_id,
            version=current.version + 1,
            kind=request.kind,
            text=request.text or "",
            source_case_id=observation.context.case_id,
            source_patient_id=observation.context.patient_id,
            source_prediction_set_id=observation.context.prediction_set_id,
            source_split_id=observation.context.split_id,
            source_history_sha256=public_history_prefix_sha256(observation, step),
            evidence=request.evidence,
            linked_record_ids=request.target_record_ids,
            created_after_step_index=step,
            supersedes_version=current.version,
        )
        self._versions[current.record_id].append(record)
        self._retrieved_ids.add(current.record_id)
        return self._accepted(request, step, record.version)

    def _link(
        self,
        observation: Observation,
        request: ManagedMemoryRequest,
        step: int,
    ) -> ManagedMemoryOperation:
        current = self._current(request)
        self._validate_evidence(observation, request.evidence)
        self._require_active_targets(request.target_record_ids)
        links = tuple(sorted(set(current.linked_record_ids) | set(request.target_record_ids)))
        record = ManagedMemoryRecord(
            record_id=current.record_id,
            version=current.version + 1,
            kind=current.kind,
            text=current.text,
            source_case_id=observation.context.case_id,
            source_patient_id=observation.context.patient_id,
            source_prediction_set_id=observation.context.prediction_set_id,
            source_split_id=observation.context.split_id,
            source_history_sha256=public_history_prefix_sha256(observation, step),
            evidence=request.evidence,
            linked_record_ids=links,
            created_after_step_index=step,
            supersedes_version=current.version,
        )
        self._versions[current.record_id].append(record)
        return self._accepted(request, step, record.version)

    def _discard(
        self, request: ManagedMemoryRequest, step: int
    ) -> ManagedMemoryOperation:
        current = self._current(request)
        self._active_ids.remove(current.record_id)
        self._retrieved_ids.discard(current.record_id)
        return self._accepted(request, step, current.version)

    def _retrieve(
        self, request: ManagedMemoryRequest, step: int
    ) -> ManagedMemoryOperation:
        self._require_active_targets(request.target_record_ids)
        self._retrieved_ids = set(request.target_record_ids)
        return self._accepted(request, step, None)

    def _current(self, request: ManagedMemoryRequest) -> ManagedMemoryRecord:
        if request.record_id is None or request.record_id not in self._active_ids:
            raise ContractError("managed memory record is absent or discarded")
        current = self._versions[request.record_id][-1]
        if request.expected_version != current.version:
            raise ContractError(
                f"stale managed memory version: expected {request.expected_version}, "
                f"current {current.version}"
            )
        return current

    def _validate_evidence(
        self, observation: Observation, evidence: tuple[ReflectionEvidence, ...]
    ) -> None:
        if not evidence:
            raise ContractError("managed memory write requires public evidence")
        for item in evidence:
            actual = resolve_public_reflection_evidence(observation, item)
            if not _same_value(actual, item.observed_value):
                raise ContractError("managed memory evidence value mismatch")

    def _validate_record(
        self, observation: Observation, record: ManagedMemoryRecord
    ) -> None:
        context = observation.context
        if (
            record.source_case_id != context.case_id
            or record.source_patient_id != context.patient_id
            or record.source_prediction_set_id != context.prediction_set_id
            or record.source_split_id != context.split_id
        ):
            raise ContractError("managed memory source context mismatch")
        expected = public_history_prefix_sha256(
            observation, record.created_after_step_index
        )
        if record.source_history_sha256 != expected:
            raise ContractError("managed memory history digest mismatch")
        self._validate_evidence(observation, record.evidence)

    def _require_active_targets(self, targets: tuple[str, ...]) -> None:
        missing = sorted(set(targets) - self._active_ids)
        if missing:
            raise ContractError(f"managed memory link/retrieval targets absent: {missing}")

    def _accepted(
        self,
        request: ManagedMemoryRequest,
        step: int,
        version: int | None,
    ) -> ManagedMemoryOperation:
        return ManagedMemoryOperation(
            operation_index=len(self._operations),
            operation=request.operation,
            record_id=request.record_id,
            record_version=version,
            target_record_ids=request.target_record_ids,
            created_after_step_index=step,
            disposition="accepted",
            reason=request.reason,
        )


def _same_value(
    actual: str | float | bool, expected: str | float | bool
) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), abs_tol=1e-10)
    return actual == expected
