

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from radoncgym.core.errors import ContractError
from radoncgym.core.models import Observation
from radoncgym.memory.policies import (
    MemoryView,
    public_history_prefix_sha256,
    resolve_public_reflection_evidence,
)


_FORBIDDEN_INFORMATION_TOKENS = (
    "ground_truth",
    "hidden_m1",
    "hidden_reference",
    "oracle_reference",
    "reference_dose",
    "reference_error",
)


@dataclass(frozen=True, slots=True)
class MemoryIntegrityFinding:
    pass

    code: str
    severity: str
    component: str
    item_id: str
    detail: str

    def __post_init__(self) -> None:
        if self.severity not in {"error", "warning"}:
            raise ContractError("memory-integrity severity must be error or warning")
        if not all((self.code, self.component, self.item_id, self.detail)):
            raise ContractError("memory-integrity finding fields cannot be empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "component": self.component,
            "item_id": self.item_id,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class MemoryIntegrityReport:
    pass

    module_id: str
    patient_id: str
    history_length: int
    retrieved_episode_count: int
    reflection_count: int
    semantic_atlas_count: int
    managed_record_count: int
    managed_operation_count: int
    findings: tuple[MemoryIntegrityFinding, ...]

    @property
    def error_count(self) -> int:
        return sum(finding.severity == "error" for finding in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(finding.severity == "warning" for finding in self.findings)

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "radoncgym-memory-integrity-report-0.1.0",
            "module_id": self.module_id,
            "patient_id": self.patient_id,
            "history_length": self.history_length,
            "retrieved_episode_count": self.retrieved_episode_count,
            "reflection_count": self.reflection_count,
            "semantic_atlas_count": self.semantic_atlas_count,
            "managed_record_count": self.managed_record_count,
            "managed_operation_count": self.managed_operation_count,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def audit_memory_view(
    observation: Observation,
    view: MemoryView,
    *,
    require_semantic_source_hashes: bool = True,
) -> MemoryIntegrityReport:
    pass

    findings: list[MemoryIntegrityFinding] = []

    def add(
        code: str,
        component: str,
        item_id: str,
        detail: str,
        *,
        severity: str = "error",
    ) -> None:
        findings.append(
            MemoryIntegrityFinding(
                code=code,
                severity=severity,
                component=component,
                item_id=item_id,
                detail=detail,
            )
        )

    if view.authoritative_history_length != len(observation.history):
        add(
            "authoritative_history_length_mismatch",
            "working_memory",
            view.module_id,
            "Memory view was built against a different public-history length.",
        )

    retrieved_ids: set[str] = set()
    for episode in view.retrieved_episodes:
        item_id = episode.source_episode_id
        if item_id in retrieved_ids:
            add(
                "duplicate_retrieved_episode",
                "episodic_retrieval",
                item_id,
                "The same external episode identity appears more than once.",
            )
        retrieved_ids.add(item_id)
        if episode.source_split_id != "train":
            add(
                "nontraining_retrieval_source",
                "episodic_retrieval",
                item_id,
                "External agent-visible memory must originate from train only.",
            )
        if episode.source_patient_id == observation.context.patient_id:
            add(
                "same_patient_retrieval_source",
                "episodic_retrieval",
                item_id,
                "External retrieval reused the query patient's trajectory.",
            )
        for key, _ in episode.outcome_summary:
            if _contains_forbidden_information_token(key):
                add(
                    "forbidden_retrieval_information",
                    "episodic_retrieval",
                    item_id,
                    f"Outcome-summary key {key!r} names hidden/reference feedback.",
                )

    reflection_ids: set[str] = set()
    context = observation.context
    for reflection in view.reflections:
        item_id = reflection.reflection_id
        if item_id in reflection_ids:
            add(
                "duplicate_reflection",
                "reflection",
                item_id,
                "The same reflection identity appears more than once.",
            )
        reflection_ids.add(item_id)
        source_context = (
            reflection.source_case_id,
            reflection.source_patient_id,
            reflection.source_prediction_set_id,
            reflection.source_split_id,
        )
        live_context = (
            context.case_id,
            context.patient_id,
            context.prediction_set_id,
            context.split_id,
        )
        if source_context != live_context:
            add(
                "reflection_context_mismatch",
                "reflection",
                item_id,
                "Reflection source context differs from the live episode.",
            )
            continue
        try:
            expected_hash = public_history_prefix_sha256(
                observation, reflection.created_after_step_index
            )
        except ContractError as error:
            add(
                "reflection_history_range_error",
                "reflection",
                item_id,
                str(error),
            )
            continue
        if expected_hash != reflection.source_history_sha256:
            add(
                "reflection_history_digest_mismatch",
                "reflection",
                item_id,
                "Reflection is not bound to the current immutable public prefix.",
            )
        for evidence_index, evidence in enumerate(reflection.evidence):
            try:
                actual = resolve_public_reflection_evidence(observation, evidence)
            except ContractError as error:
                add(
                    "reflection_evidence_unresolvable",
                    "reflection",
                    f"{item_id}:evidence:{evidence_index}",
                    str(error),
                )
                continue
            if actual != evidence.observed_value:
                add(
                    "reflection_evidence_value_mismatch",
                    "reflection",
                    f"{item_id}:evidence:{evidence_index}",
                    "Stored evidence value differs from immutable public history.",
                )

    atlas_ids: set[str] = set()
    for atlas in view.semantic_atlases:
        item_id = atlas.atlas_id
        if item_id in atlas_ids:
            add(
                "duplicate_semantic_atlas",
                "semantic_memory",
                item_id,
                "The same semantic-atlas identity appears more than once.",
            )
        atlas_ids.add(item_id)
        if atlas.source_split != "train":
            add(
                "nontraining_semantic_source",
                "semantic_memory",
                item_id,
                "Agent-visible semantic memory must be fitted on train only.",
            )
        if require_semantic_source_hashes and not atlas.source_files:
            add(
                "untraceable_semantic_source",
                "semantic_memory",
                item_id,
                "Semantic memory has no content-addressed source manifest.",
            )
        if _contains_forbidden_information_token(atlas.ranking_metric):
            add(
                "forbidden_semantic_ranking_metric",
                "semantic_memory",
                item_id,
                "Semantic ranking metric names hidden/reference feedback.",
            )

    managed_ids: set[str] = set()
    for record in view.managed_records:
        item_id = record.record_id
        if item_id in managed_ids:
            add(
                "duplicate_managed_record",
                "agent_managed_memory",
                item_id,
                "The same managed-memory record appears more than once.",
            )
        managed_ids.add(item_id)
        context = observation.context
        if (
            record.source_case_id != context.case_id
            or record.source_patient_id != context.patient_id
            or record.source_prediction_set_id != context.prediction_set_id
            or record.source_split_id != context.split_id
        ):
            add(
                "managed_record_context_mismatch",
                "agent_managed_memory",
                item_id,
                "Managed record source context differs from the live episode.",
            )
        try:
            expected_hash = public_history_prefix_sha256(
                observation, record.created_after_step_index
            )
        except ContractError as error:
            add(
                "managed_record_history_range_error",
                "agent_managed_memory",
                item_id,
                str(error),
            )
            continue
        if expected_hash != record.source_history_sha256:
            add(
                "managed_record_history_digest_mismatch",
                "agent_managed_memory",
                item_id,
                "Managed record is not bound to immutable public history.",
            )
        for evidence_index, evidence in enumerate(record.evidence):
            try:
                actual = resolve_public_reflection_evidence(observation, evidence)
            except ContractError as error:
                add(
                    "managed_record_evidence_unresolvable",
                    "agent_managed_memory",
                    f"{item_id}:evidence:{evidence_index}",
                    str(error),
                )
                continue
            if actual != evidence.observed_value:
                add(
                    "managed_record_evidence_value_mismatch",
                    "agent_managed_memory",
                    f"{item_id}:evidence:{evidence_index}",
                    "Managed evidence differs from immutable public history.",
                )

    for expected_index, operation in enumerate(view.managed_operations):
        if operation.operation_index != expected_index:
            add(
                "managed_operation_sequence_gap",
                "agent_managed_memory",
                str(operation.operation_index),
                "Managed-memory operation log is not contiguous.",
            )

    findings.sort(
        key=lambda finding: (
            finding.severity,
            finding.component,
            finding.code,
            finding.item_id,
        )
    )
    return MemoryIntegrityReport(
        module_id=view.module_id,
        patient_id=context.patient_id,
        history_length=len(observation.history),
        retrieved_episode_count=len(view.retrieved_episodes),
        reflection_count=len(view.reflections),
        semantic_atlas_count=len(view.semantic_atlases),
        managed_record_count=len(view.managed_records),
        managed_operation_count=len(view.managed_operations),
        findings=tuple(findings),
    )


def require_memory_integrity(
    observation: Observation,
    view: MemoryView,
    *,
    require_semantic_source_hashes: bool = True,
) -> MemoryIntegrityReport:
    pass

    report = audit_memory_view(
        observation,
        view,
        require_semantic_source_hashes=require_semantic_source_hashes,
    )
    if not report.passed:
        codes = sorted({finding.code for finding in report.findings})
        raise ContractError(f"memory integrity audit failed: {codes}")
    return report


def _contains_forbidden_information_token(value: str) -> bool:
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    return any(token in normalized for token in _FORBIDDEN_INFORMATION_TOKENS)
