

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Mapping, Protocol

from radoncgym.agents.codex_runtime import (
    codex_concurrency_slot,
    codex_transient_backoff_seconds,
    is_transient_codex_failure,
)
from radoncgym.core.constants import OBJECTIVE_NAMES
from radoncgym.core.errors import ContractError


CRITIC_MODULE_ID = "evaluator_planner_decomposition_v1"
CANDIDATE_CRITIC_MODULE_ID = "anticipatory_criterion_critic_v1"


class ProviderRoleOutputError(ContractError):
    pass

    def __init__(self, *, role: str, parse_error: str, result: Any) -> None:
        super().__init__(f"{role} structured output invalid: {parse_error}")
        self.role = role
        self.parse_error = parse_error
        self.result = result


class StructuredCriticProvider(Protocol):
    pass

    @property
    def provider_name(self) -> str:
        ...

    @property
    def requested_model(self) -> str:
        ...

    @property
    def configuration(self) -> Mapping[str, str]:
        ...

    def reset(self, *, seed: int) -> None:
        ...

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        json_schema: Mapping[str, Any],
    ) -> Any:
        ...

    def complete_structured_with_images(
        self,
        *,
        system: str,
        user: str,
        image_paths: tuple[str, ...],
        json_schema: Mapping[str, Any],
    ) -> Any:
        ...


def _complete_structured_role(
    *,
    provider: StructuredCriticProvider,
    system: str,
    user: str,
    image_paths: tuple[str, ...],
    json_schema: Mapping[str, Any],
) -> Any:
    pass

    if not image_paths:
        return provider.complete_structured(
            system=system,
            user=user,
            json_schema=json_schema,
        )
    complete = getattr(provider, "complete_structured_with_images", None)
    if complete is None:
        complete = getattr(provider, "complete_with_images_structured", None)
    if complete is None:
        raise ContractError(
            "structured critic provider cannot accept image-bearing role calls"
        )
    return complete(
        system=system,
        user=user,
        image_paths=image_paths,
        json_schema=json_schema,
    )


def _validated_image_paths(image_paths: tuple[str, ...]) -> tuple[str, ...]:
    resolved: list[str] = []
    for raw_path in image_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ContractError(f"critic image does not exist: {path}")
        resolved.append(str(path))
    return tuple(resolved)


CRITIC_SYSTEM_PROMPT = """You are the evidence-grounded evaluator in a
radiotherapy evaluator/planner pair. Diagnose the supplied public RadOncGym
state before the planner chooses its next objective-weight action. Use only
values in the observation. Never claim access to an expert reference dose.
Prioritize target coverage, then critical serial or max-dose organs, then
discretionary mean-dose sparing. Identify uncertainty and cheap/full
disagreement explicitly. Your output is advisory: do not emit a complete weight
vector and do not decide the planner's action. Return exactly the required JSON.
"""


CANDIDATE_CRITIC_SYSTEM_PROMPT = """You are a pre-execution radiotherapy
candidate critic. Review one proposed RadOncGym action against only the public
observation and history supplied below. Check for avoidable target compromise,
unsupported weight-response assumptions, duplicate or negligible changes, and
whether a requested full physics call is likely to provide useful information.
Never claim access to an expert reference dose. Do not emit replacement weights
and do not decide the final action: the planner owns both the weight vector and
the cheap/full decision. Return exactly the required JSON.
"""


def _run_codex_with_retries(
    *,
    command: list[str],
    prompt: str,
    output_path: Path,
    timeout_seconds: float,
    max_transient_attempts: int,
    component: str,
) -> tuple[subprocess.CompletedProcess[str], int]:
    pass

    attempts_used = 0
    while attempts_used < max_transient_attempts:
        attempts_used += 1
        output_path.unlink(missing_ok=True)
        try:
            with codex_concurrency_slot():
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    check=False,
                )
        except OSError as error:
            raise ContractError(
                f"Codex {component} execution failed: {error}"
            ) from error
        except subprocess.TimeoutExpired as error:
            if attempts_used >= max_transient_attempts:
                raise ContractError(
                    f"Codex {component} exhausted transient retries after "
                    f"{attempts_used} attempts: {error}"
                ) from error
            time.sleep(codex_transient_backoff_seconds(attempts_used))
            continue
        if completed.returncode == 0:
            return completed, attempts_used
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or "no CLI diagnostics"
        )[-4000:]
        if (
            attempts_used < max_transient_attempts
            and is_transient_codex_failure(detail)
        ):
            time.sleep(codex_transient_backoff_seconds(attempts_used))
            continue
        raise ContractError(
            f"Codex {component} exited with status {completed.returncode} after "
            f"{attempts_used} attempt(s): {detail}"
        )
    raise AssertionError("unreachable Codex retry state")


@dataclass(frozen=True, slots=True)
class CandidateCriticIssue:
    issue_type: str
    evidence: str
    severity: str

    def __post_init__(self) -> None:
        if self.issue_type not in {
            "target_compromise",
            "unsupported_response",
            "duplicate_or_negligible",
            "low_value_full_request",
            "clinical_tradeoff",
        }:
            raise ContractError(
                f"candidate critic issue type is invalid: {self.issue_type}"
            )
        if self.severity not in {"low", "medium", "high"}:
            raise ContractError(
                f"candidate critic issue severity is invalid: {self.severity}"
            )
        if not self.evidence.strip():
            raise ContractError("candidate critic issue requires evidence")

    def to_dict(self) -> dict[str, str]:
        return {
            "issue_type": self.issue_type,
            "evidence": self.evidence,
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class CandidateCritique:
    verdict: str
    issues: tuple[CandidateCriticIssue, ...]
    revision_guidance: str
    full_request_assessment: str
    confidence: str

    def __post_init__(self) -> None:
        if self.verdict not in {"accept", "revise"}:
            raise ContractError("candidate critic verdict is invalid")
        if self.full_request_assessment not in {
            "worthwhile",
            "uncertain",
            "low_value",
        }:
            raise ContractError("candidate critic full-request assessment is invalid")
        if self.confidence not in {"low", "medium", "high"}:
            raise ContractError("candidate critic confidence is invalid")
        if not self.revision_guidance.strip():
            raise ContractError("candidate critic requires revision guidance")
        if len(self.issues) > 5:
            raise ContractError("candidate critic permits at most five issues")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": CANDIDATE_CRITIC_MODULE_ID,
            "verdict": self.verdict,
            "issues": [item.to_dict() for item in self.issues],
            "revision_guidance": self.revision_guidance,
            "full_request_assessment": self.full_request_assessment,
            "confidence": self.confidence,
            "advisory_only": True,
            "planner_owns_final_action": True,
        }


@dataclass(frozen=True, slots=True)
class CandidateCriticGeneration:
    critique: CandidateCritique
    raw_text: str
    latency_seconds: float
    response_id: str
    model: str
    usage: tuple[tuple[str, int], ...] = ()
    provider_metadata: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = {
            "critique": self.critique.to_dict(),
            "raw_text": self.raw_text,
            "latency_seconds": self.latency_seconds,
            "response_id": self.response_id,
            "model": self.model,
        }
        if self.usage:
            value["usage"] = dict(self.usage)
        if self.provider_metadata:
            value["provider_metadata"] = dict(self.provider_metadata)
        return value


class CodexCandidateCritic:
    pass

    module_id = CANDIDATE_CRITIC_MODULE_ID

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        reasoning_effort: str = "medium",
        timeout_seconds: float = 1200.0,
        executable: str = "codex",
        max_transient_attempts: int = 7,
    ) -> None:
        if not model or not executable or timeout_seconds <= 0:
            raise ContractError("Codex candidate critic configuration is invalid")
        if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ContractError("unsupported candidate critic reasoning effort")
        if max_transient_attempts < 1:
            raise ContractError("candidate critic attempts must be positive")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = float(timeout_seconds)
        self.executable = executable
        self.max_transient_attempts = int(max_transient_attempts)
        self._turn_index = 0

    @property
    def configured_module_id(self) -> str:
        return (
            f"{self.module_id}:scope=full_requests:model={self.model}:"
            f"reasoning={self.reasoning_effort}"
        )

    def reset(self, *, seed: int) -> None:
        del seed
        self._turn_index = 0

    def generate(
        self,
        *,
        observation_json: str,
        candidate_action: Mapping[str, Any],
        image_paths: tuple[str, ...] = (),
    ) -> CandidateCriticGeneration:
        prompt = (
            CANDIDATE_CRITIC_SYSTEM_PROMPT
            + "\n\n<PUBLIC_OBSERVATION>\n"
            + observation_json
            + "\n</PUBLIC_OBSERVATION>\n\n<PROPOSED_ACTION>\n"
            + json.dumps(candidate_action, sort_keys=True, separators=(",", ":"))
            + "\n</PROPOSED_ACTION>"
        )
        with tempfile.TemporaryDirectory(
            prefix="radoncgym-candidate-critic-"
        ) as directory:
            root = Path(directory)
            schema_path = root / "candidate-critic-schema.json"
            output_path = root / "candidate-critique.json"
            schema_path.write_text(
                json.dumps(
                    _candidate_critic_schema(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--model",
                self.model,
                "-c",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "-C",
                str(root),
            ]
            for image_path in _validated_image_paths(image_paths):
                command.extend(("--image", image_path))
            command.append("-")
            started = time.monotonic()
            completed, attempts_used = _run_codex_with_retries(
                command=command,
                prompt=prompt,
                output_path=output_path,
                timeout_seconds=self.timeout_seconds,
                max_transient_attempts=self.max_transient_attempts,
                component="candidate critic",
            )
            latency = time.monotonic() - started
            if not output_path.is_file():
                raise ContractError(
                    "Codex candidate critic did not write structured output"
                )
            raw = output_path.read_text(encoding="utf-8").strip()
        critique = _parse_candidate_critique(raw)
        response_id = f"codex-candidate-critic-turn-{self._turn_index:04d}"
        self._turn_index += 1
        return CandidateCriticGeneration(
            critique=critique,
            raw_text=raw,
            latency_seconds=latency,
            response_id=response_id,
            model=self.model,
            provider_metadata=(
                ("transient_attempts_used", str(attempts_used)),
                ("transient_retries_used", str(attempts_used - 1)),
            ),
        )


class ProviderCandidateCritic:
    pass

    module_id = CANDIDATE_CRITIC_MODULE_ID

    def __init__(self, provider: StructuredCriticProvider) -> None:
        self.provider = provider
        self.model = provider.requested_model

    @property
    def configured_module_id(self) -> str:
        effort = self.provider.configuration.get(
            "effort",
            self.provider.configuration.get("reasoning_effort", "unspecified"),
        )
        return (
            f"{self.module_id}:scope=full_requests:model={self.model}:"
            f"reasoning={effort}:provider={self.provider.provider_name}"
        )

    def reset(self, *, seed: int) -> None:
        self.provider.reset(seed=seed)

    def generate(
        self,
        *,
        observation_json: str,
        candidate_action: Mapping[str, Any],
        image_paths: tuple[str, ...] = (),
    ) -> CandidateCriticGeneration:
        user = (
            "<PUBLIC_OBSERVATION>\n"
            + observation_json
            + "\n</PUBLIC_OBSERVATION>\n\n<PROPOSED_ACTION>\n"
            + json.dumps(candidate_action, sort_keys=True, separators=(",", ":"))
            + "\n</PROPOSED_ACTION>"
        )
        validated_images = _validated_image_paths(image_paths)
        result = _complete_structured_role(
            provider=self.provider,
            system=CANDIDATE_CRITIC_SYSTEM_PROMPT,
            user=user,
            image_paths=validated_images,
            json_schema=_candidate_critic_schema(),
        )
        raw = result.raw_text
        try:
            critique = _parse_candidate_critique(raw)
        except ContractError as error:
            raise ProviderRoleOutputError(
                role="candidate_critic",
                parse_error=str(error),
                result=result,
            ) from error
        return CandidateCriticGeneration(
            critique=critique,
            raw_text=raw,
            latency_seconds=result.latency_seconds,
            response_id=result.response_id,
            model=result.response_model,
            usage=result.usage,
            provider_metadata=result.provider_metadata,
        )


@dataclass(frozen=True, slots=True)
class CriticIssue:
    criterion: str
    status: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        return {
            "criterion": self.criterion,
            "status": self.status,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class CriticAdjustment:
    objective_name: str
    direction: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "objective_name": self.objective_name,
            "direction": self.direction,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PlanningCritique:
    summary: str
    priority_issues: tuple[CriticIssue, ...]
    recommended_adjustments: tuple[CriticAdjustment, ...]
    verifier_request_worth_considering: bool
    verifier_reason: str
    confidence: str

    def __post_init__(self) -> None:
        if not self.summary or not self.verifier_reason:
            raise ContractError("planning critique requires summary and verifier reason")
        if self.confidence not in {"low", "medium", "high"}:
            raise ContractError("planning critique confidence is invalid")
        for adjustment in self.recommended_adjustments:
            if adjustment.objective_name not in OBJECTIVE_NAMES:
                raise ContractError(
                    f"critic used unknown objective: {adjustment.objective_name}"
                )
            if adjustment.direction not in {"increase", "decrease", "hold"}:
                raise ContractError(
                    f"critic used invalid direction: {adjustment.direction}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": CRITIC_MODULE_ID,
            "summary": self.summary,
            "priority_issues": [item.to_dict() for item in self.priority_issues],
            "recommended_adjustments": [
                item.to_dict() for item in self.recommended_adjustments
            ],
            "verifier_request_worth_considering": (
                self.verifier_request_worth_considering
            ),
            "verifier_reason": self.verifier_reason,
            "confidence": self.confidence,
            "advisory_only": True,
        }


@dataclass(frozen=True, slots=True)
class CriticGeneration:
    critique: PlanningCritique
    raw_text: str
    latency_seconds: float
    response_id: str
    model: str
    usage: tuple[tuple[str, int], ...] = ()
    provider_metadata: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = {
            "critique": self.critique.to_dict(),
            "raw_text": self.raw_text,
            "latency_seconds": self.latency_seconds,
            "response_id": self.response_id,
            "model": self.model,
        }
        if self.usage:
            value["usage"] = dict(self.usage)
        if self.provider_metadata:
            value["provider_metadata"] = dict(self.provider_metadata)
        return value


class CodexPlanningCritic:
    pass

    module_id = CRITIC_MODULE_ID

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        reasoning_effort: str = "medium",
        timeout_seconds: float = 1200.0,
        executable: str = "codex",
        max_transient_attempts: int = 7,
    ) -> None:
        if not model or not executable or timeout_seconds <= 0:
            raise ContractError("Codex planning critic configuration is invalid")
        if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ContractError("unsupported critic reasoning effort")
        if max_transient_attempts < 1:
            raise ContractError("planning critic attempts must be positive")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = float(timeout_seconds)
        self.executable = executable
        self.max_transient_attempts = int(max_transient_attempts)
        self._turn_index = 0

    @property
    def configured_module_id(self) -> str:
        return (
            f"{self.module_id}:model={self.model}:"
            f"reasoning={self.reasoning_effort}"
        )

    def reset(self, *, seed: int) -> None:
        del seed
        self._turn_index = 0

    def generate(
        self,
        *,
        observation_json: str,
        image_paths: tuple[str, ...] = (),
    ) -> CriticGeneration:
        prompt = (
            CRITIC_SYSTEM_PROMPT
            + "\n\n<PUBLIC_OBSERVATION>\n"
            + observation_json
            + "\n</PUBLIC_OBSERVATION>"
        )
        with tempfile.TemporaryDirectory(prefix="radoncgym-critic-") as directory:
            root = Path(directory)
            schema_path = root / "critic-schema.json"
            output_path = root / "critique.json"
            schema_path.write_text(
                json.dumps(
                    _critic_schema(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--model",
                self.model,
                "-c",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "-C",
                str(root),
            ]
            for image_path in _validated_image_paths(image_paths):
                command.extend(("--image", image_path))
            command.append("-")
            started = time.monotonic()
            completed, attempts_used = _run_codex_with_retries(
                command=command,
                prompt=prompt,
                output_path=output_path,
                timeout_seconds=self.timeout_seconds,
                max_transient_attempts=self.max_transient_attempts,
                component="planning critic",
            )
            latency = time.monotonic() - started
            if not output_path.is_file():
                raise ContractError("Codex critic did not write structured output")
            raw = output_path.read_text(encoding="utf-8").strip()
        critique = _parse_critique(raw)
        response_id = f"codex-critic-turn-{self._turn_index:04d}"
        self._turn_index += 1
        return CriticGeneration(
            critique=critique,
            raw_text=raw,
            latency_seconds=latency,
            response_id=response_id,
            model=self.model,
            provider_metadata=(
                ("transient_attempts_used", str(attempts_used)),
                ("transient_retries_used", str(attempts_used - 1)),
            ),
        )


class ProviderPlanningCritic:
    pass

    module_id = CRITIC_MODULE_ID

    def __init__(self, provider: StructuredCriticProvider) -> None:
        self.provider = provider
        self.model = provider.requested_model

    @property
    def configured_module_id(self) -> str:
        effort = self.provider.configuration.get(
            "effort",
            self.provider.configuration.get("reasoning_effort", "unspecified"),
        )
        return (
            f"{self.module_id}:model={self.model}:reasoning={effort}:"
            f"provider={self.provider.provider_name}"
        )

    def reset(self, *, seed: int) -> None:
        self.provider.reset(seed=seed)

    def generate(
        self,
        *,
        observation_json: str,
        image_paths: tuple[str, ...] = (),
    ) -> CriticGeneration:
        user = (
            "<PUBLIC_OBSERVATION>\n"
            + observation_json
            + "\n</PUBLIC_OBSERVATION>"
        )
        validated_images = _validated_image_paths(image_paths)
        result = _complete_structured_role(
            provider=self.provider,
            system=CRITIC_SYSTEM_PROMPT,
            user=user,
            image_paths=validated_images,
            json_schema=_critic_schema(),
        )
        raw = result.raw_text
        try:
            critique = _parse_critique(raw)
        except ContractError as error:
            raise ProviderRoleOutputError(
                role="planning_critic",
                parse_error=str(error),
                result=result,
            ) from error
        return CriticGeneration(
            critique=critique,
            raw_text=raw,
            latency_seconds=result.latency_seconds,
            response_id=result.response_id,
            model=result.response_model,
            usage=result.usage,
            provider_metadata=result.provider_metadata,
        )


def _critic_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "minLength": 1},
            "priority_issues": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["violated", "satisfied", "tradeoff", "uncertain"],
                        },
                        "evidence": {"type": "string", "minLength": 1},
                    },
                    "required": ["criterion", "status", "evidence"],
                    "additionalProperties": False,
                },
            },
            "recommended_adjustments": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "objective_name": {
                            "type": "string",
                            "enum": list(OBJECTIVE_NAMES),
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["increase", "decrease", "hold"],
                        },
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": ["objective_name", "direction", "reason"],
                    "additionalProperties": False,
                },
            },
            "verifier_request_worth_considering": {"type": "boolean"},
            "verifier_reason": {"type": "string", "minLength": 1},
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
        },
        "required": [
            "summary",
            "priority_issues",
            "recommended_adjustments",
            "verifier_request_worth_considering",
            "verifier_reason",
            "confidence",
        ],
        "additionalProperties": False,
    }


def _parse_critique(raw: str) -> PlanningCritique:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ContractError(f"invalid critic JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ContractError("critic output must be a JSON object")
    required = {
        "summary",
        "priority_issues",
        "recommended_adjustments",
        "verifier_request_worth_considering",
        "verifier_reason",
        "confidence",
    }
    if set(value) != required:
        raise ContractError(
            f"critic fields differ from contract: {sorted(set(value) ^ required)}"
        )
    issues = value["priority_issues"]
    adjustments = value["recommended_adjustments"]
    if not isinstance(issues, list) or not isinstance(adjustments, list):
        raise ContractError("critic issues and adjustments must be lists")
    if len(issues) > 5 or len(adjustments) > 5:
        raise ContractError("critic permits at most five issues or adjustments")
    if any(not isinstance(item, dict) for item in issues + adjustments):
        raise ContractError("critic issue and adjustment entries must be objects")
    issue_fields = {"criterion", "status", "evidence"}
    adjustment_fields = {"objective_name", "direction", "reason"}
    if any(set(item) != issue_fields for item in issues):
        raise ContractError("critic issue fields differ from contract")
    if any(set(item) != adjustment_fields for item in adjustments):
        raise ContractError("critic adjustment fields differ from contract")
    request_worth = value["verifier_request_worth_considering"]
    if not isinstance(request_worth, bool):
        raise ContractError("critic verifier recommendation must be boolean")
    return PlanningCritique(
        summary=str(value["summary"]),
        priority_issues=tuple(
            CriticIssue(
                criterion=str(item["criterion"]),
                status=str(item["status"]),
                evidence=str(item["evidence"]),
            )
            for item in issues
        ),
        recommended_adjustments=tuple(
            CriticAdjustment(
                objective_name=str(item["objective_name"]),
                direction=str(item["direction"]),
                reason=str(item["reason"]),
            )
            for item in adjustments
        ),
        verifier_request_worth_considering=request_worth,
        verifier_reason=str(value["verifier_reason"]),
        confidence=str(value["confidence"]),
    )


def _candidate_critic_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["accept", "revise"]},
            "issues": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "issue_type": {
                            "type": "string",
                            "enum": [
                                "target_compromise",
                                "unsupported_response",
                                "duplicate_or_negligible",
                                "low_value_full_request",
                                "clinical_tradeoff",
                            ],
                        },
                        "evidence": {"type": "string", "minLength": 1},
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                    "required": ["issue_type", "evidence", "severity"],
                    "additionalProperties": False,
                },
            },
            "revision_guidance": {"type": "string", "minLength": 1},
            "full_request_assessment": {
                "type": "string",
                "enum": ["worthwhile", "uncertain", "low_value"],
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
        },
        "required": [
            "verdict",
            "issues",
            "revision_guidance",
            "full_request_assessment",
            "confidence",
        ],
        "additionalProperties": False,
    }


def _parse_candidate_critique(raw: str) -> CandidateCritique:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ContractError(
            f"invalid candidate critic JSON: {error.msg}"
        ) from error
    if not isinstance(value, dict):
        raise ContractError("candidate critic output must be a JSON object")
    required = {
        "verdict",
        "issues",
        "revision_guidance",
        "full_request_assessment",
        "confidence",
    }
    if set(value) != required:
        raise ContractError(
            "candidate critic fields differ from contract: "
            f"{sorted(set(value) ^ required)}"
        )
    issues = value["issues"]
    if not isinstance(issues, list) or any(
        not isinstance(item, dict) for item in issues
    ):
        raise ContractError("candidate critic issues must be an object list")
    if len(issues) > 5:
        raise ContractError("candidate critic permits at most five issues")
    issue_fields = {"issue_type", "evidence", "severity"}
    if any(set(item) != issue_fields for item in issues):
        raise ContractError("candidate critic issue fields differ from contract")
    return CandidateCritique(
        verdict=str(value["verdict"]),
        issues=tuple(
            CandidateCriticIssue(
                issue_type=str(item["issue_type"]),
                evidence=str(item["evidence"]),
                severity=str(item["severity"]),
            )
            for item in issues
        ),
        revision_guidance=str(value["revision_guidance"]),
        full_request_assessment=str(value["full_request_assessment"]),
        confidence=str(value["confidence"]),
    )
