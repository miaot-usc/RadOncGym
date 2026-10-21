

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Mapping, Protocol

from radoncgym.core.errors import ContractError
from radoncgym.core.constants import (
    FULL_CLINICAL_IMPROVEMENT_METRIC,
    PREDICTED_CLINICAL_IMPROVEMENT_METRIC,
)
from radoncgym.core.metric_schema import clinical_improvement
from radoncgym.core.models import Observation
from radoncgym.memory.policies import (
    IncumbentFailureArchiveMemory,
    MemoryPolicy,
    MemoryView,
    ReflectionBufferMemory,
    ReflectionEvidence,
    ReflectionRecord,
    public_history_prefix_sha256,
    resolve_public_reflection_evidence,
)
from radoncgym.trajectories.serialization import history_entry_to_dict


_DETERMINISTIC_TEMPLATE = (
    "Summarize only machine-checkable public evidence from one completed "
    "RadOncGym transition. Do not infer hidden or clinical facts."
)
_DETERMINISTIC_TEMPLATE_SHA256 = hashlib.sha256(
    _DETERMINISTIC_TEMPLATE.encode("utf-8")
).hexdigest()
_REFLECTION_TRIGGERS = {"every_evaluated_step", "full_feedback_only"}
_REFLECTION_DISPOSITIONS = {
    "not_triggered",
    "accepted",
    "abstained",
    "rejected_invalid",
    "discarded_control",
}
_LLM_REFLECTION_SYSTEM_PROMPT = """You are a RadOncGym reflection module.
Use only the supplied public transition. Never infer hidden reference quality,
clinical acceptability, causality, or facts not present in that transition.
Return one JSON object matching the required schema. A reflection is an
advisory planning hypothesis, not a verified clinical statement. Cite at least
one supplied evidence item unless abstain=true. Evidence entries identify
fields; the environment resolves their values from the immutable history."""


@dataclass(frozen=True, slots=True)
class ReflectionModelResult:
    pass

    raw_text: str
    requested_model: str
    response_model: str
    response_id: str
    latency_seconds: float
    usage: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.requested_model or not self.response_model:
            raise ContractError("reflection result requires model identity")
        if self.latency_seconds < 0 or any(value < 0 for _, value in self.usage):
            raise ContractError("reflection model cost cannot be negative")


class ReflectionModelProvider(Protocol):
    pass

    @property
    def provider_name(self) -> str:
        pass

    @property
    def requested_model(self) -> str:
        pass

    @property
    def configuration(self) -> Mapping[str, str]:
        pass

    def reset(self, *, seed: int) -> None:
        pass

    def complete_reflection(self, *, system: str, user: str) -> ReflectionModelResult:
        pass


class OpenAIReflectionProvider:
    pass

    def __init__(
        self,
        *,
        model: str = "gpt-5.6",
        reasoning_effort: str = "medium",
    ) -> None:
        if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ContractError("unsupported reflection reasoning effort")
        self._model = model
        self._reasoning_effort = reasoning_effort

    @property
    def provider_name(self) -> str:
        return "openai_responses_reflection"

    @property
    def requested_model(self) -> str:
        return self._model

    @property
    def configuration(self) -> Mapping[str, str]:
        return {
            "reasoning_effort": self._reasoning_effort,
            "structured_output": "radoncgym_evidence_reflection_v1",
            "store": "false",
        }

    def reset(self, *, seed: int) -> None:
        del seed

    def complete_reflection(self, *, system: str, user: str) -> ReflectionModelResult:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ContractError(
                "OpenAI reflection requires the optional 'frontier' dependency"
            ) from error

        started = time.monotonic()
        response = OpenAI(max_retries=1).responses.create(
            model=self._model,
            reasoning={"effort": self._reasoning_effort},
            store=False,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text={"format": _reflection_json_schema()},
        )
        latency = time.monotonic() - started
        return ReflectionModelResult(
            raw_text=str(response.output_text),
            requested_model=self._model,
            response_model=str(getattr(response, "model", self._model)),
            response_id=str(getattr(response, "id", "unavailable")),
            latency_seconds=latency,
            usage=tuple(sorted(_normalize_usage(getattr(response, "usage", None)).items())),
        )


class CompletionReflectionProviderAdapter:
    pass

    def __init__(self, provider: Any) -> None:
        required = (
            "provider_name",
            "requested_model",
            "configuration",
            "reset",
            "complete",
        )
        if any(not hasattr(provider, name) for name in required):
            raise ContractError("reflection adapter received an incompatible provider")
        self.provider = provider

    @property
    def provider_name(self) -> str:
        return f"{self.provider.provider_name}_reflection_adapter"

    @property
    def requested_model(self) -> str:
        return str(self.provider.requested_model)

    @property
    def configuration(self) -> Mapping[str, str]:
        return {
            **dict(self.provider.configuration),
            "structured_output": "prompt_enforced_radoncgym_evidence_reflection_v1",
        }

    def reset(self, *, seed: int) -> None:
        self.provider.reset(seed=seed)

    def complete_reflection(self, *, system: str, user: str) -> ReflectionModelResult:
        result = self.provider.complete(system=system, user=user)
        return ReflectionModelResult(
            raw_text=result.raw_text,
            requested_model=result.requested_model,
            response_model=result.response_model,
            response_id=result.response_id,
            latency_seconds=result.latency_seconds,
            usage=result.usage,
        )


@dataclass(frozen=True, slots=True)
class ReflectionGeneration:
    pass

    source_step_index: int
    trigger: str
    trigger_fired: bool
    record: ReflectionRecord | None
    generator_call_count: int
    latency_seconds: float
    disposition: str
    usage: tuple[tuple[str, int], ...] = ()
    raw_response: str | None = None
    error: str | None = None
    requested_model: str | None = None
    response_model: str | None = None
    response_id: str | None = None

    def __post_init__(self) -> None:
        if self.source_step_index < 0:
            raise ContractError("reflection generation step must be nonnegative")
        if not self.trigger:
            raise ContractError("reflection generation trigger is required")
        if self.disposition not in _REFLECTION_DISPOSITIONS:
            raise ContractError(f"unsupported reflection disposition: {self.disposition}")
        if self.trigger_fired != (self.disposition != "not_triggered"):
            raise ContractError("reflection trigger and disposition disagree")
        if (self.record is not None) != (self.disposition == "accepted"):
            raise ContractError("only accepted reflection generations carry a record")
        if self.generator_call_count < 0 or self.latency_seconds < 0:
            raise ContractError("reflection generation cost cannot be negative")
        if any(value < 0 for _, value in self.usage):
            raise ContractError("reflection generation usage cannot be negative")
        if self.disposition == "not_triggered" and self.generator_call_count != 0:
            raise ContractError("untriggered reflection cannot report a model call")
        if self.disposition == "rejected_invalid" and not self.error:
            raise ContractError("rejected reflection requires an error")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_step_index": self.source_step_index,
            "trigger": self.trigger,
            "trigger_fired": self.trigger_fired,
            "record": None if self.record is None else self.record.to_dict(),
            "generator_call_count": self.generator_call_count,
            "latency_seconds": self.latency_seconds,
            "disposition": self.disposition,
            "usage": dict(self.usage),
            "raw_response": self.raw_response,
            "error": self.error,
            "requested_model": self.requested_model,
            "response_model": self.response_model,
            "response_id": self.response_id,
        }


class ReflectionGenerator(Protocol):
    @property
    def module_id(self) -> str:
        pass

    def reset(self, *, seed: int) -> None:
        pass

    def generate(
        self,
        observation: Observation,
        *,
        source_step_index: int,
        source_episode_id: str,
    ) -> ReflectionGeneration:
        pass


class DeterministicEvidenceReflectionGenerator:
    pass

    module_id = "deterministic_public_evidence_reflector_v1"

    def __init__(self, *, trigger: str = "every_evaluated_step") -> None:
        if trigger not in _REFLECTION_TRIGGERS:
            raise ContractError(f"unsupported reflection trigger: {trigger}")
        self.trigger = trigger

    @property
    def configured_module_id(self) -> str:
        return f"{self.module_id}:trigger={self.trigger}"

    def reset(self, *, seed: int) -> None:
        del seed

    def generate(
        self,
        observation: Observation,
        *,
        source_step_index: int,
        source_episode_id: str,
    ) -> ReflectionGeneration:
        if source_step_index < 0 or source_step_index >= len(observation.history):
            raise ContractError("reflector source step is outside public history")
        entry = observation.history[source_step_index]
        has_full = entry.outcome is not None and entry.outcome.full is not None
        should_generate = entry.outcome is not None and (
            self.trigger == "every_evaluated_step" or has_full
        )
        if not should_generate:
            return ReflectionGeneration(
                source_step_index=source_step_index,
                trigger=self.trigger,
                trigger_fired=False,
                record=None,
                generator_call_count=0,
                latency_seconds=0.0,
                disposition="not_triggered",
            )

        if entry.outcome is None:
            raise AssertionError("triggered reflection lost its public outcome")
        evidence: list[ReflectionEvidence] = [
            ReflectionEvidence(
                source_step_index=source_step_index,
                kind="accepted",
                key=None,
                observed_value=entry.accepted,
            ),
            ReflectionEvidence(
                source_step_index=source_step_index,
                kind="requested_full",
                key=None,
                observed_value=entry.action.request_full_verifier,
            ),
        ]
        statements = [f"Step {source_step_index} was accepted={entry.accepted}."]
        cheap_improvement = clinical_improvement(entry.outcome.cheap)
        if cheap_improvement is not None:
            evidence.append(
                ReflectionEvidence(
                    source_step_index=source_step_index,
                    kind="cheap_metric",
                    key=PREDICTED_CLINICAL_IMPROVEMENT_METRIC,
                    observed_value=cheap_improvement,
                )
            )
            statements.append(
                f"Its predicted clinical improvement was {cheap_improvement:.8g}."
            )
        full = entry.outcome.full
        if full is not None:
            full_improvement = clinical_improvement(full)
            if full_improvement is not None:
                evidence.append(
                    ReflectionEvidence(
                        source_step_index=source_step_index,
                        kind="full_metric",
                        key=FULL_CLINICAL_IMPROVEMENT_METRIC,
                        observed_value=full_improvement,
                    )
                )
                statements.append(
                    f"Its full clinical improvement was {full_improvement:.8g}."
                )
            positive = [
                (name, value)
                for name, value in full.constraints.items
                if value > 0
            ]
            if positive:
                name, value = max(positive, key=lambda item: (item[1], item[0]))
                evidence.append(
                    ReflectionEvidence(
                        source_step_index=source_step_index,
                        kind="full_constraint",
                        key=name,
                        observed_value=value,
                    )
                )
                statements.append(
                    f"The largest positive reported constraint was {name}="
                    f"{value:.8g}."
                )
            else:
                statements.append("No positive full constraint was reported.")

        text = " ".join(statements)
        identity_payload = {
            "source_episode_id": source_episode_id,
            "source_step_index": source_step_index,
            "source_history_sha256": public_history_prefix_sha256(
                observation, source_step_index
            ),
            "text": text,
            "evidence": [item.to_dict() for item in evidence],
        }
        canonical = json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        reflection_id = "reflection-" + hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()[:20]
        context = observation.context
        record = ReflectionRecord(
            reflection_id=reflection_id,
            text=text,
            source_step_indices=(source_step_index,),
            generator_id=self.configured_module_id,
            generator_revision="1.0.0",
            prompt_sha256=_DETERMINISTIC_TEMPLATE_SHA256,
            source_episode_id=source_episode_id,
            source_case_id=context.case_id,
            source_patient_id=context.patient_id,
            source_prediction_set_id=context.prediction_set_id,
            source_split_id=context.split_id,
            source_history_sha256=identity_payload["source_history_sha256"],
            evidence=tuple(evidence),
            created_after_step_index=source_step_index,
        )
        return ReflectionGeneration(
            source_step_index=source_step_index,
            trigger=self.trigger,
            trigger_fired=True,
            record=record,
            generator_call_count=0,
            latency_seconds=0.0,
            disposition="accepted",
        )


class LLMEvidenceReflectionGenerator:
    pass

    module_id = "llm_evidence_reflector_v1"

    def __init__(
        self,
        *,
        provider: ReflectionModelProvider,
        trigger: str = "full_feedback_only",
        max_evidence_items: int = 6,
        retain_record: bool = True,
    ) -> None:
        if trigger not in _REFLECTION_TRIGGERS:
            raise ContractError(f"unsupported reflection trigger: {trigger}")
        if not 0 < max_evidence_items <= 6:
            raise ContractError("reflection evidence limit must be in [1, 6]")
        self.provider = provider
        self.trigger = trigger
        self.max_evidence_items = max_evidence_items
        self.retain_record = retain_record
        configuration = json.dumps(
            dict(provider.configuration),
            sort_keys=True,
            separators=(",", ":"),
        )
        self._configuration_sha256 = hashlib.sha256(
            configuration.encode("utf-8")
        ).hexdigest()

    @property
    def configured_module_id(self) -> str:
        retention = "retain" if self.retain_record else "discard_control"
        return (
            f"{self.module_id}:provider={self.provider.provider_name}:"
            f"model={self.provider.requested_model}:trigger={self.trigger}:"
            f"evidence={self.max_evidence_items}:retention={retention}:"
            f"config={self._configuration_sha256[:12]}"
        )

    def reset(self, *, seed: int) -> None:
        self.provider.reset(seed=seed)

    def generate(
        self,
        observation: Observation,
        *,
        source_step_index: int,
        source_episode_id: str,
    ) -> ReflectionGeneration:
        if source_step_index < 0 or source_step_index >= len(observation.history):
            raise ContractError("reflector source step is outside public history")
        entry = observation.history[source_step_index]
        has_full = entry.outcome is not None and entry.outcome.full is not None
        should_generate = entry.outcome is not None and (
            self.trigger == "every_evaluated_step" or has_full
        )
        if not should_generate:
            return ReflectionGeneration(
                source_step_index=source_step_index,
                trigger=self.trigger,
                trigger_fired=False,
                record=None,
                generator_call_count=0,
                latency_seconds=0.0,
                disposition="not_triggered",
            )

        system = _LLM_REFLECTION_SYSTEM_PROMPT
        user = json.dumps(
            {
                "source_step_index": source_step_index,
                "public_transition": history_entry_to_dict(entry),
                "allowed_evidence_kinds": sorted(
                    kind
                    for kind in {
                        "accepted",
                        "requested_full",
                        "cheap_metric",
                        "cheap_constraint",
                        "cheap_status",
                        "full_metric",
                        "full_constraint",
                        "full_status",
                    }
                ),
                "max_evidence_items": self.max_evidence_items,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        prompt_sha256 = hashlib.sha256(
            (system + "\n" + user).encode("utf-8")
        ).hexdigest()
        result = self.provider.complete_reflection(system=system, user=user)
        common = {
            "source_step_index": source_step_index,
            "trigger": self.trigger,
            "trigger_fired": True,
            "generator_call_count": 1,
            "latency_seconds": result.latency_seconds,
            "usage": result.usage,
            "raw_response": result.raw_text,
            "requested_model": result.requested_model,
            "response_model": result.response_model,
            "response_id": result.response_id,
        }
        try:
            value = _parse_reflection_response(result.raw_text)
            if value["abstain"]:
                return ReflectionGeneration(
                    record=None,
                    disposition="abstained",
                    **common,
                )
            evidence_specs = value["evidence"]
            if len(evidence_specs) > self.max_evidence_items:
                raise ContractError("reflection exceeds configured evidence limit")
            evidence = tuple(
                _resolve_evidence_spec(
                    observation,
                    source_step_index=source_step_index,
                    value=item,
                )
                for item in evidence_specs
            )
            context = observation.context
            source_hash = public_history_prefix_sha256(
                observation, source_step_index
            )
            identity_payload = {
                "source_episode_id": source_episode_id,
                "source_step_index": source_step_index,
                "source_history_sha256": source_hash,
                "text": value["reflection"],
                "evidence": [item.to_dict() for item in evidence],
                "generator_id": self.configured_module_id,
            }
            canonical = json.dumps(
                identity_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            record = ReflectionRecord(
                reflection_id="reflection-"
                + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20],
                text=value["reflection"],
                source_step_indices=(source_step_index,),
                generator_id=self.configured_module_id,
                generator_revision="1.0.0",
                prompt_sha256=prompt_sha256,
                source_episode_id=source_episode_id,
                source_case_id=context.case_id,
                source_patient_id=context.patient_id,
                source_prediction_set_id=context.prediction_set_id,
                source_split_id=context.split_id,
                source_history_sha256=source_hash,
                evidence=evidence,
                created_after_step_index=source_step_index,
            )
            if not self.retain_record:
                return ReflectionGeneration(
                    record=None,
                    disposition="discarded_control",
                    **common,
                )
            return ReflectionGeneration(
                record=record,
                disposition="accepted",
                **common,
            )
        except ContractError as error:
            return ReflectionGeneration(
                record=None,
                disposition="rejected_invalid",
                error=str(error),
                **common,
            )


class OnlineReflectionBufferMemory:
    pass

    module_id = "online_reflection_buffer_v1"

    def __init__(
        self,
        *,
        max_reflections: int = 3,
        base_policy: MemoryPolicy | None = None,
    ) -> None:
        if max_reflections <= 0:
            raise ContractError("online reflection buffer size must be positive")
        self.max_reflections = max_reflections
        self.base_policy = base_policy or IncumbentFailureArchiveMemory()
        self._records: list[ReflectionRecord] = []
        self._record_ids: set[str] = set()

    @property
    def configured_module_id(self) -> str:
        return (
            f"{self.module_id}:max={self.max_reflections}:"
            f"base={self.base_policy.module_id}"
        )

    @property
    def records(self) -> tuple[ReflectionRecord, ...]:
        return tuple(self._records)

    def reset(self) -> None:
        self._records.clear()
        self._record_ids.clear()

    def append(self, record: ReflectionRecord) -> None:
        if record.reflection_id in self._record_ids:
            raise ContractError(
                f"duplicate online reflection ID: {record.reflection_id}"
            )
        self._records.append(record)
        self._record_ids.add(record.reflection_id)

    def select(self, observation: Observation) -> MemoryView:
        view = ReflectionBufferMemory(
            self.records,
            max_reflections=self.max_reflections,
            base_policy=self.base_policy,
        ).select(observation)
        return MemoryView(
            module_id=(
                f"{self.configured_module_id}:"
                f"records={_short_records_digest(self.records)}"
            ),
            authoritative_history_length=view.authoritative_history_length,
            selected_step_indices=view.selected_step_indices,
            omitted_step_indices=view.omitted_step_indices,
            selection_reasons=view.selection_reasons,
            retrieved_episodes=view.retrieved_episodes,
            reflections=view.reflections,
            semantic_atlases=view.semantic_atlases,
            archived_step_indices=view.archived_step_indices,
            page_operations=view.page_operations,
            managed_records=view.managed_records,
            managed_operations=view.managed_operations,
            managed_operation_budget_remaining=(
                view.managed_operation_budget_remaining
            ),
            include_incumbent_pointers=view.include_incumbent_pointers,
        )


def _short_records_digest(records: tuple[ReflectionRecord, ...]) -> str:
    payload = [record.to_dict() for record in records]
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _parse_reflection_response(raw_text: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ContractError(f"reflection response is not valid JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise ContractError("reflection response must be one JSON object")
    if set(value) != {"abstain", "reflection", "evidence"}:
        raise ContractError("reflection response fields must be abstain/reflection/evidence")
    if not isinstance(value["abstain"], bool):
        raise ContractError("reflection abstain must be boolean")
    if not isinstance(value["reflection"], str):
        raise ContractError("reflection text must be a string")
    evidence = value["evidence"]
    if not isinstance(evidence, list) or any(
        not isinstance(item, Mapping) for item in evidence
    ):
        raise ContractError("reflection evidence must be an object list")
    if value["abstain"]:
        if value["reflection"].strip() or evidence:
            raise ContractError("abstention must use empty reflection and evidence")
    elif not value["reflection"].strip() or not evidence:
        raise ContractError("non-abstaining reflection requires text and evidence")
    return {
        "abstain": value["abstain"],
        "reflection": value["reflection"].strip(),
        "evidence": evidence,
    }


def _resolve_evidence_spec(
    observation: Observation,
    *,
    source_step_index: int,
    value: Mapping[str, Any],
) -> ReflectionEvidence:
    if set(value) != {"kind", "key"}:
        raise ContractError("reflection evidence fields must be kind/key")
    kind = value.get("kind")
    key = value.get("key")
    if not isinstance(kind, str):
        raise ContractError("reflection evidence kind must be a string")
    if key is not None and not isinstance(key, str):
        raise ContractError("reflection evidence key must be string or null")
    unresolved = ReflectionEvidence(
        source_step_index=source_step_index,
        kind=kind,
        key=key,
        observed_value=False,
    )
    observed = resolve_public_reflection_evidence(observation, unresolved)
    return ReflectionEvidence(
        source_step_index=source_step_index,
        kind=kind,
        key=key,
        observed_value=observed,
    )


def _reflection_json_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "radoncgym_evidence_reflection_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["abstain", "reflection", "evidence"],
            "properties": {
                "abstain": {"type": "boolean"},
                "reflection": {"type": "string", "maxLength": 1200},
                "evidence": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["kind", "key"],
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": sorted(
                                    {
                                        "accepted",
                                        "requested_full",
                                        "cheap_metric",
                                        "cheap_constraint",
                                        "cheap_status",
                                        "full_metric",
                                        "full_constraint",
                                        "full_status",
                                    }
                                ),
                            },
                            "key": {"type": ["string", "null"]},
                        },
                    },
                },
            },
        },
    }


def _normalize_usage(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    result: dict[str, int] = {}
    for target, candidates in {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }.items():
        for candidate in candidates:
            item = getattr(value, candidate, None)
            if item is not None:
                result[target] = int(item)
                break
    return result
