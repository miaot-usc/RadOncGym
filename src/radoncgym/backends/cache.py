

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np

from radoncgym.core.errors import BackendContractError, ContractError
from radoncgym.core.models import (
    ComputeCost,
    EvaluationResult,
    FeedbackSource,
    MetricVector,
    ObjectiveWeights,
    PlanningContext,
    SolverStatus,
)


EXACT_CACHE_SCHEMA_VERSION = "exact-full-cache-0.1.0"
PORTFOLIO_EXACT_CACHE_SCHEMA_VERSION = (
    "prediction-portfolio-exact-full-cache-0.1.0"
)
PORTFOLIO_EXACT_CACHE_KEY_VERSION = (
    "prediction-portfolio-exact-cache-key-0.1.0"
)


class ExactCacheMiss(KeyError):
    pass


@dataclass(frozen=True, slots=True)
class PortfolioExactCacheIdentity:
    pass

    objective_mapping_version: str
    solver_profile_id: str
    solver_parameters_sha256: str
    dataset_manifest_id: str
    dataset_manifest_version: str
    dataset_manifest_sha256: str
    public_metric_schema_version: str
    public_constraint_schema_version: str
    patient_public_feature_schema_version: str
    branch_public_feature_schema_version: str
    cache_key_version: str = PORTFOLIO_EXACT_CACHE_KEY_VERSION

    def __post_init__(self) -> None:
        required = {
            "cache_key_version": self.cache_key_version,
            "objective_mapping_version": self.objective_mapping_version,
            "solver_profile_id": self.solver_profile_id,
            "dataset_manifest_id": self.dataset_manifest_id,
            "dataset_manifest_version": self.dataset_manifest_version,
            "public_metric_schema_version": self.public_metric_schema_version,
            "public_constraint_schema_version": (
                self.public_constraint_schema_version
            ),
            "patient_public_feature_schema_version": (
                self.patient_public_feature_schema_version
            ),
            "branch_public_feature_schema_version": (
                self.branch_public_feature_schema_version
            ),
        }
        for name, value in required.items():
            if not value:
                raise ContractError(f"{name} is required for portfolio cache identity")
        for name, value in (
            ("solver_parameters_sha256", self.solver_parameters_sha256),
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ContractError(
                    f"{name} must be 64 lowercase hexadecimal characters"
                )
        if self.cache_key_version != PORTFOLIO_EXACT_CACHE_KEY_VERSION:
            raise ContractError("unsupported portfolio exact cache key version")

    def to_dict(self) -> dict[str, str]:
        return {
            "cache_key_version": self.cache_key_version,
            "objective_mapping_version": self.objective_mapping_version,
            "solver_profile_id": self.solver_profile_id,
            "solver_parameters_sha256": self.solver_parameters_sha256,
            "dataset_manifest_id": self.dataset_manifest_id,
            "dataset_manifest_version": self.dataset_manifest_version,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "public_metric_schema_version": self.public_metric_schema_version,
            "public_constraint_schema_version": (
                self.public_constraint_schema_version
            ),
            "patient_public_feature_schema_version": (
                self.patient_public_feature_schema_version
            ),
            "branch_public_feature_schema_version": (
                self.branch_public_feature_schema_version
            ),
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "PortfolioExactCacheIdentity":
        expected = {
            "cache_key_version",
            "objective_mapping_version",
            "solver_profile_id",
            "solver_parameters_sha256",
            "dataset_manifest_id",
            "dataset_manifest_version",
            "dataset_manifest_sha256",
            "public_metric_schema_version",
            "public_constraint_schema_version",
            "patient_public_feature_schema_version",
            "branch_public_feature_schema_version",
        }
        actual = set(value)
        if actual != expected:
            raise ContractError(
                "portfolio cache identity fields mismatch: "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        return cls(**{name: str(value[name]) for name in expected})


def canonical_solver_parameters_sha256(
    parameters: Mapping[str, Any],
) -> str:
    pass

    if not parameters:
        raise ContractError("canonical solver parameters cannot be empty")
    canonical = json.dumps(
        dict(parameters),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_exact_cache_key(
    context: PlanningContext,
    weights: ObjectiveWeights,
    *,
    backend_profile_id: str,
) -> str:
    if not backend_profile_id:
        raise ContractError("backend_profile_id is required for exact cache keys")
    payload = {
        "patient_id": context.patient_id,
        "prediction_set_id": context.prediction_set_id,
        "planning_formulation": context.planning_formulation,
        "clinical_profile_id": context.clinical_profile_id,
        "environment_version": context.environment_version,
        "backend_profile_id": backend_profile_id,
        "weights": weights.to_dict(),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_portfolio_exact_cache_key(
    context: PlanningContext,
    weights: ObjectiveWeights,
    *,
    identity: PortfolioExactCacheIdentity,
) -> str:
    pass

    if identity.solver_profile_id == "":
        raise ContractError("solver profile is required for portfolio cache keys")
    payload = {
        "cache_key_version": identity.cache_key_version,
        "case_id": context.case_id,
        "patient_id": context.patient_id,
        "prediction_set_id": context.prediction_set_id,
        "split_id": context.split_id,
        "planning_formulation": context.planning_formulation,
        "clinical_profile_id": context.clinical_profile_id,
        "environment_version": context.environment_version,
        "weights": weights.to_dict(),
        "portfolio_identity": identity.to_dict(),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExactFullCacheRecord:
    pass

    cache_key: str
    context: PlanningContext
    weights: ObjectiveWeights
    backend_profile_id: str
    status: SolverStatus
    metrics: MetricVector
    constraints: MetricVector
    original_wall_seconds: float
    original_solver_seconds: float
    solver_id: str
    source_record_id: str
    dose_artifact: str | None = None
    portfolio_identity: PortfolioExactCacheIdentity | None = None
    schema_version: str = EXACT_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.portfolio_identity is None:
            if self.schema_version != EXACT_CACHE_SCHEMA_VERSION:
                raise ContractError(
                    "portfolio exact cache records require portfolio identity"
                )
            expected_key = make_exact_cache_key(
                self.context,
                self.weights,
                backend_profile_id=self.backend_profile_id,
            )
        else:
            if self.schema_version != PORTFOLIO_EXACT_CACHE_SCHEMA_VERSION:
                raise ContractError(
                    "portfolio identity requires the portfolio exact-cache schema"
                )
            if (
                self.portfolio_identity.solver_profile_id
                != self.backend_profile_id
            ):
                raise ContractError(
                    "portfolio cache solver profile does not match backend profile"
                )
            expected_key = make_portfolio_exact_cache_key(
                self.context,
                self.weights,
                identity=self.portfolio_identity,
            )
        if self.cache_key != expected_key:
            raise ContractError("exact cache record key does not match its query")
        if self.original_wall_seconds < 0 or self.original_solver_seconds < 0:
            raise ContractError("original cache costs must be nonnegative")
        if not self.solver_id or not self.source_record_id:
            raise ContractError("solver_id and source_record_id are required")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "cache_key": self.cache_key,
            "context": {
                "case_id": self.context.case_id,
                "patient_id": self.context.patient_id,
                "prediction_set_id": self.context.prediction_set_id,
                "split_id": self.context.split_id,
                "clinical_profile_id": self.context.clinical_profile_id,
                "planning_formulation": self.context.planning_formulation,
                "environment_version": self.context.environment_version,
            },
            "weights": self.weights.to_dict(),
            "backend_profile_id": self.backend_profile_id,
            "status": self.status.value,
            "metrics": [
                {"name": name, "value": value} for name, value in self.metrics.items
            ],
            "constraints": [
                {"name": name, "value": value}
                for name, value in self.constraints.items
            ],
            "original_wall_seconds": self.original_wall_seconds,
            "original_solver_seconds": self.original_solver_seconds,
            "solver_id": self.solver_id,
            "source_record_id": self.source_record_id,
            "dose_artifact": self.dose_artifact,
        }
        if self.portfolio_identity is not None:
            result["portfolio_identity"] = self.portfolio_identity.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExactFullCacheRecord":
        schema_version = str(value.get("schema_version"))
        if schema_version not in {
            EXACT_CACHE_SCHEMA_VERSION,
            PORTFOLIO_EXACT_CACHE_SCHEMA_VERSION,
        }:
            raise ContractError("unsupported exact cache schema")
        identity_value = value.get("portfolio_identity")
        if schema_version == EXACT_CACHE_SCHEMA_VERSION and identity_value is not None:
            raise ContractError(
                "legacy exact cache record cannot claim portfolio identity"
            )
        if (
            schema_version == PORTFOLIO_EXACT_CACHE_SCHEMA_VERSION
            and identity_value is None
        ):
            raise ContractError("portfolio exact cache record lacks portfolio identity")
        if schema_version == PORTFOLIO_EXACT_CACHE_SCHEMA_VERSION:
            expected_fields = {
                "schema_version",
                "cache_key",
                "context",
                "weights",
                "backend_profile_id",
                "status",
                "metrics",
                "constraints",
                "original_wall_seconds",
                "original_solver_seconds",
                "solver_id",
                "source_record_id",
                "dose_artifact",
                "portfolio_identity",
            }
            actual_fields = set(value)
            if actual_fields != expected_fields:
                raise ContractError(
                    "portfolio exact cache record fields mismatch: "
                    f"missing={sorted(expected_fields - actual_fields)}, "
                    f"extra={sorted(actual_fields - expected_fields)}"
                )
        context_value = _mapping(value.get("context"), "context")
        if schema_version == PORTFOLIO_EXACT_CACHE_SCHEMA_VERSION:
            expected_context_fields = {
                "case_id",
                "patient_id",
                "prediction_set_id",
                "split_id",
                "clinical_profile_id",
                "planning_formulation",
                "environment_version",
            }
            if set(context_value) != expected_context_fields:
                raise ContractError(
                    "portfolio exact cache context fields mismatch: "
                    f"missing={sorted(expected_context_fields - set(context_value))}, "
                    f"extra={sorted(set(context_value) - expected_context_fields)}"
                )
        return cls(
            cache_key=str(value["cache_key"]),
            context=PlanningContext(
                case_id=str(context_value["case_id"]),
                patient_id=str(context_value["patient_id"]),
                prediction_set_id=str(context_value["prediction_set_id"]),
                split_id=str(context_value["split_id"]),
                clinical_profile_id=str(context_value["clinical_profile_id"]),
                planning_formulation=str(context_value["planning_formulation"]),
                environment_version=str(context_value["environment_version"]),
            ),
            weights=ObjectiveWeights.from_mapping(
                {
                    str(name): float(weight)
                    for name, weight in _mapping(
                        value.get("weights"), "weights"
                    ).items()
                }
            ),
            backend_profile_id=str(value["backend_profile_id"]),
            status=SolverStatus(str(value["status"])),
            metrics=_metric_vector(value.get("metrics"), "metrics"),
            constraints=_metric_vector(value.get("constraints"), "constraints"),
            original_wall_seconds=float(value["original_wall_seconds"]),
            original_solver_seconds=float(value["original_solver_seconds"]),
            solver_id=str(value["solver_id"]),
            source_record_id=str(value["source_record_id"]),
            dose_artifact=(
                None
                if value.get("dose_artifact") is None
                else str(value["dose_artifact"])
            ),
            portfolio_identity=(
                None
                if identity_value is None
                else PortfolioExactCacheIdentity.from_dict(
                    _mapping(identity_value, "portfolio_identity")
                )
            ),
            schema_version=schema_version,
        )


class ExactFullCacheBackend:
    pass

    def __init__(
        self,
        records: Iterable[ExactFullCacheRecord],
        *,
        backend_profile_id: str,
        portfolio_identity: PortfolioExactCacheIdentity | None = None,
    ) -> None:
        self.backend_profile_id = backend_profile_id
        self.portfolio_identity = portfolio_identity
        self.last_record: ExactFullCacheRecord | None = None
        self._records: dict[str, ExactFullCacheRecord] = {}
        for record in records:
            if (
                record.backend_profile_id != backend_profile_id
                or record.portfolio_identity != portfolio_identity
            ):
                continue
            existing = self._records.get(record.cache_key)
            if existing is not None and not _equivalent_evaluation(existing, record):
                raise BackendContractError(
                    f"divergent exact cache records for key {record.cache_key}"
                )
            if existing is None:
                self._records[record.cache_key] = record
        if not self._records:
            raise ContractError(
                f"no exact records for backend profile {backend_profile_id!r}"
            )

    @classmethod
    def from_jsonl(
        cls,
        paths: Iterable[str | Path],
        *,
        backend_profile_id: str,
        portfolio_identity: PortfolioExactCacheIdentity | None = None,
    ) -> "ExactFullCacheBackend":
        records: list[ExactFullCacheRecord] = []
        for path in paths:
            with Path(path).open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        value = json.loads(line)
                        records.append(ExactFullCacheRecord.from_dict(value))
        return cls(
            records,
            backend_profile_id=backend_profile_id,
            portfolio_identity=portfolio_identity,
        )

    def evaluate(
        self, context: PlanningContext, weights: ObjectiveWeights
    ) -> EvaluationResult:
        started = time.perf_counter()
        key = (
            make_exact_cache_key(
                context, weights, backend_profile_id=self.backend_profile_id
            )
            if self.portfolio_identity is None
            else make_portfolio_exact_cache_key(
                context, weights, identity=self.portfolio_identity
            )
        )
        try:
            record = self._records[key]
        except KeyError as error:
            raise ExactCacheMiss(key) from error
        self.last_record = record
        replay_seconds = time.perf_counter() - started
        return EvaluationResult(
            source=FeedbackSource.FULL_EXACT_CACHE,
            status=record.status,
            metrics=record.metrics,
            constraints=record.constraints,
            cost=ComputeCost(
                wall_seconds=replay_seconds,
                solver_seconds=record.original_solver_seconds,
                full_calls=1,
            ),
            uncertainty=0.0,
            message=f"exact_cache:{record.cache_key}",
        )


class ApproximateNeighborCacheBackend:
    pass

    def __init__(self, records: Iterable[ExactFullCacheRecord]) -> None:
        self._by_context: dict[tuple[str, str, str, str], list[ExactFullCacheRecord]] = {}
        for record in records:
            key = (
                record.context.patient_id,
                record.context.prediction_set_id,
                record.context.planning_formulation,
                record.context.clinical_profile_id,
            )
            self._by_context.setdefault(key, []).append(record)
        if not self._by_context:
            raise ContractError("approximate neighbor cache requires records")

    def evaluate(
        self, context: PlanningContext, weights: ObjectiveWeights
    ) -> EvaluationResult:
        started = time.perf_counter()
        key = (
            context.patient_id,
            context.prediction_set_id,
            context.planning_formulation,
            context.clinical_profile_id,
        )
        candidates = self._by_context.get(key)
        if not candidates:
            raise ExactCacheMiss("no neighbor candidates for context")
        target = np.asarray(weights.values, dtype=np.float64)
        record = min(
            candidates,
            key=lambda candidate: float(
                np.linalg.norm(np.asarray(candidate.weights.values) - target)
            ),
        )
        distance = float(
            np.linalg.norm(np.asarray(record.weights.values) - target)
        )
        return EvaluationResult(
            source=FeedbackSource.APPROXIMATE_CACHE_SURROGATE,
            status=record.status,
            metrics=record.metrics,
            constraints=record.constraints,
            cost=ComputeCost(
                wall_seconds=time.perf_counter() - started,
                cheap_calls=1,
            ),
            uncertainty=distance,
            message=f"nearest_cache:{record.cache_key}",
        )


def import_legacy_full_row(
    value: Mapping[str, Any],
    *,
    split_id: str,
    backend_profile_id: str,
    environment_version: str = "1.0.0-draft",
) -> ExactFullCacheRecord:
    pass

    patient_id = str(value["patient_id"])
    prediction_set_id = str(value["prediction_set"])
    formulation = str(value["plan_family"])
    context = PlanningContext(
        case_id=f"{patient_id}:{prediction_set_id}:{formulation}",
        patient_id=patient_id,
        prediction_set_id=prediction_set_id,
        split_id=split_id,
        planning_formulation=formulation,
        environment_version=environment_version,
    )
    weights = ObjectiveWeights.from_mapping(
        {
            str(name): float(weight)
            for name, weight in _mapping(
                value.get("objective_weights"), "objective_weights"
            ).items()
        }
    )
    safe_protocol = {
        str(name): float(metric)
        for name, metric in _mapping(value.get("new_metrics"), "new_metrics").items()
    }
    violations = {
        str(name): float(metric)
        for name, metric in _mapping(
            value.get("criteria_violations"), "criteria_violations"
        ).items()
    }
    raw_mean_violation = value.get("clinical_score")
    mean_violation = (
        float(raw_mean_violation)
        if raw_mean_violation is not None
        else float(np.mean(tuple(violations.values())))
        if violations
        else 0.0
    )
    metrics = {
        "objective": mean_violation,
        "mean_protocol_violation_gy": mean_violation,
        **safe_protocol,
    }
    metadata = _mapping(value.get("backend_metadata", {}), "backend_metadata")
    solve_status = value.get("solve_status", metadata.get("solve_status"))
    status = SolverStatus.SOLVED if solve_status == 0 else SolverStatus.ERROR
    solver_seconds = float(
        value.get("solve_time_seconds", metadata.get("solve_time_seconds", 0.0))
    )
    source_record_id = str(
        value.get("cache_key", value.get("source_audit_id", "legacy_unknown"))
    )
    cache_key = make_exact_cache_key(
        context, weights, backend_profile_id=backend_profile_id
    )
    return ExactFullCacheRecord(
        cache_key=cache_key,
        context=context,
        weights=weights,
        backend_profile_id=backend_profile_id,
        status=status,
        metrics=MetricVector.from_mapping(metrics),
        constraints=MetricVector.from_mapping(violations),
        original_wall_seconds=solver_seconds,
        original_solver_seconds=solver_seconds,
        solver_id=str(value.get("solver", metadata.get("solver", "unknown"))),
        source_record_id=source_record_id,
        dose_artifact=(
            None if value.get("dose_path") is None else str(value["dose_path"])
        ),
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping")
    return value


def _metric_vector(value: Any, name: str) -> MetricVector:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be an ordered list")
    return MetricVector(
        tuple(
            (str(_mapping(item, name)["name"]), float(_mapping(item, name)["value"]))
            for item in value
        )
    )


def _equivalent_evaluation(
    first: ExactFullCacheRecord, second: ExactFullCacheRecord
) -> bool:
    pass

    return (
        first.context == second.context
        and first.weights == second.weights
        and first.backend_profile_id == second.backend_profile_id
        and first.status == second.status
        and first.metrics == second.metrics
        and first.constraints == second.constraints
        and first.solver_id == second.solver_id
    )
