

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

import numpy as np

from radoncgym.core.constants import OBJECTIVE_NAMES
from radoncgym.core.errors import ContractError
from radoncgym.trajectories.generated_plan import GeneratedPlanTransition


FULL_PHYSICS_RESPONSE_ATLAS_SCHEMA_VERSION = (
    "training-full-physics-response-atlas-0.1.0"
)


@dataclass(frozen=True, slots=True)
class FullPhysicsResponseEntry:
    pass

    objective_name: str
    direction: str
    comparison_count: int
    episode_count: int
    patient_count: int
    median_absolute_log_weight_change: float
    median_objective_improvement: float
    objective_improvement_interquartile_low: float
    objective_improvement_interquartile_high: float
    objective_improvement_success_rate: float
    median_constraint_improvement: float
    patient_bootstrap_objective_low: float
    patient_bootstrap_objective_high: float

    def __post_init__(self) -> None:
        if self.objective_name not in OBJECTIVE_NAMES:
            raise ContractError("physics response entry has unknown objective")
        if self.direction not in {"increase", "decrease"}:
            raise ContractError("physics response direction is invalid")
        if min(self.comparison_count, self.episode_count, self.patient_count) <= 0:
            raise ContractError("physics response counts must be positive")
        if not 0.0 <= self.objective_improvement_success_rate <= 1.0:
            raise ContractError("physics response success rate must lie in [0, 1]")

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class TrainingFullPhysicsResponseAtlas:
    pass

    atlas_id: str
    source_split: str
    transition_count: int
    comparison_count: int
    episode_count: int
    patient_count: int
    bootstrap_samples: int
    bootstrap_seed: int
    minimum_absolute_log_weight_change: float
    source_episode_sha256: tuple[str, ...]
    entries: tuple[FullPhysicsResponseEntry, ...]

    def __post_init__(self) -> None:
        if self.source_split != "train":
            raise ContractError("full-physics response atlas must use train split")
        if not self.atlas_id:
            raise ContractError("full-physics response atlas requires an ID")
        if min(
            self.transition_count,
            self.comparison_count,
            self.episode_count,
            self.patient_count,
            self.bootstrap_samples,
        ) <= 0:
            raise ContractError("full-physics response atlas counts must be positive")
        if self.minimum_absolute_log_weight_change < 0:
            raise ContractError("minimum log-weight change cannot be negative")
        if not self.entries:
            raise ContractError("full-physics response atlas cannot be empty")
        if tuple(sorted(set(self.source_episode_sha256))) != self.source_episode_sha256:
            raise ContractError("source episode hashes must be sorted and unique")
        if any(len(value) != 64 for value in self.source_episode_sha256):
            raise ContractError("source episode hashes must be SHA-256 values")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": FULL_PHYSICS_RESPONSE_ATLAS_SCHEMA_VERSION,
            "atlas_id": self.atlas_id,
            "source_split": self.source_split,
            "transition_count": self.transition_count,
            "comparison_count": self.comparison_count,
            "episode_count": self.episode_count,
            "patient_count": self.patient_count,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "minimum_absolute_log_weight_change": (
                self.minimum_absolute_log_weight_change
            ),
            "source_episode_sha256": list(self.source_episode_sha256),
            "entries": [entry.to_dict() for entry in self.entries],
            "interpretation": (
                "Training-only associations over consecutive solved generated "
                "plans. Simultaneous weight changes share the observed response; "
                "entries are not independent causal effects or monotonic rules."
            ),
            "hidden_reference_used": False,
        }

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingFullPhysicsResponseAtlas:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ContractError("full-physics response atlas must be an object")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> TrainingFullPhysicsResponseAtlas:
        if payload.get("schema_version") != FULL_PHYSICS_RESPONSE_ATLAS_SCHEMA_VERSION:
            raise ContractError("unsupported full-physics response atlas schema")
        raw_entries = payload.get("entries")
        raw_hashes = payload.get("source_episode_sha256")
        if not isinstance(raw_entries, list) or not isinstance(raw_hashes, list):
            raise ContractError("full-physics response atlas entries/hashes are invalid")
        return cls(
            atlas_id=str(payload["atlas_id"]),
            source_split=str(payload["source_split"]),
            transition_count=int(payload["transition_count"]),
            comparison_count=int(payload["comparison_count"]),
            episode_count=int(payload["episode_count"]),
            patient_count=int(payload["patient_count"]),
            bootstrap_samples=int(payload["bootstrap_samples"]),
            bootstrap_seed=int(payload["bootstrap_seed"]),
            minimum_absolute_log_weight_change=float(
                payload["minimum_absolute_log_weight_change"]
            ),
            source_episode_sha256=tuple(str(value) for value in raw_hashes),
            entries=tuple(
                FullPhysicsResponseEntry(
                    objective_name=str(item["objective_name"]),
                    direction=str(item["direction"]),
                    comparison_count=int(item["comparison_count"]),
                    episode_count=int(item["episode_count"]),
                    patient_count=int(item["patient_count"]),
                    median_absolute_log_weight_change=float(
                        item["median_absolute_log_weight_change"]
                    ),
                    median_objective_improvement=float(
                        item["median_objective_improvement"]
                    ),
                    objective_improvement_interquartile_low=float(
                        item["objective_improvement_interquartile_low"]
                    ),
                    objective_improvement_interquartile_high=float(
                        item["objective_improvement_interquartile_high"]
                    ),
                    objective_improvement_success_rate=float(
                        item["objective_improvement_success_rate"]
                    ),
                    median_constraint_improvement=float(
                        item["median_constraint_improvement"]
                    ),
                    patient_bootstrap_objective_low=float(
                        item["patient_bootstrap_objective_low"]
                    ),
                    patient_bootstrap_objective_high=float(
                        item["patient_bootstrap_objective_high"]
                    ),
                )
                for item in raw_entries
                if isinstance(item, Mapping)
            ),
        )


@dataclass(frozen=True, slots=True)
class _Response:
    episode_id: str
    patient_id: str
    objective_name: str
    direction: str
    absolute_log_weight_change: float
    objective_improvement: float
    constraint_improvement: float


def build_training_full_physics_response_atlas(
    transitions: Iterable[GeneratedPlanTransition],
    *,
    minimum_absolute_log_weight_change: float = 1e-6,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260715,
) -> TrainingFullPhysicsResponseAtlas:
    pass

    if (
        minimum_absolute_log_weight_change < 0
        or not math.isfinite(minimum_absolute_log_weight_change)
    ):
        raise ContractError("minimum log-weight change must be finite and nonnegative")
    if bootstrap_samples <= 0:
        raise ContractError("bootstrap_samples must be positive")
    records = tuple(transitions)
    if not records:
        raise ContractError("full-physics response atlas input cannot be empty")
    if any(record.split_id != "train" for record in records):
        raise ContractError("full-physics response atlas accepts train split only")

    by_episode: dict[str, list[GeneratedPlanTransition]] = {}
    source_hashes: set[str] = set()
    patients: set[str] = set()
    for record in records:
        if record.objective_names != OBJECTIVE_NAMES:
            raise ContractError("generated-plan objective order is invalid")
        if len(record.weights) != len(OBJECTIVE_NAMES):
            raise ContractError("generated-plan weight vector has invalid length")
        by_episode.setdefault(record.episode_id, []).append(record)
        source_hashes.add(record.source_episode_sha256)
        patients.add(record.patient_id)

    responses: list[_Response] = []
    comparison_count = 0
    for episode_id, episode_records in sorted(by_episode.items()):
        ordered = sorted(episode_records, key=lambda item: item.full_call_index)
        indices = [item.full_call_index for item in ordered]
        if indices != list(range(1, len(ordered) + 1)):
            raise ContractError(
                f"full-call sequence is not contiguous for episode {episode_id}"
            )
        if len({item.patient_id for item in ordered}) != 1:
            raise ContractError("one episode cannot contain multiple patients")
        for previous, current in zip(ordered, ordered[1:], strict=False):
            comparison_count += 1
            objective_improvement = (
                previous.generated_objective - current.generated_objective
            )
            constraint_improvement = (
                previous.positive_constraint_sum - current.positive_constraint_sum
            )
            for name, old, new in zip(
                OBJECTIVE_NAMES,
                previous.weights,
                current.weights,
                strict=True,
            ):
                if min(old, new) <= 0 or not all(map(math.isfinite, (old, new))):
                    raise ContractError("physics response weights must be finite positive")
                log_change = math.log(new / old)
                if abs(log_change) <= minimum_absolute_log_weight_change:
                    continue
                responses.append(
                    _Response(
                        episode_id=episode_id,
                        patient_id=current.patient_id,
                        objective_name=name,
                        direction="increase" if log_change > 0 else "decrease",
                        absolute_log_weight_change=abs(log_change),
                        objective_improvement=objective_improvement,
                        constraint_improvement=constraint_improvement,
                    )
                )
    if comparison_count == 0:
        raise ContractError("atlas requires at least two full plans in one episode")
    if not responses:
        raise ContractError("atlas found no material weight changes")

    grouped: dict[tuple[str, str], list[_Response]] = {}
    for response in responses:
        grouped.setdefault(
            (response.objective_name, response.direction), []
        ).append(response)
    rng = np.random.default_rng(bootstrap_seed)
    entries = tuple(
        _summarize_group(
            objective_name,
            direction,
            grouped[(objective_name, direction)],
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        for objective_name in OBJECTIVE_NAMES
        for direction in ("increase", "decrease")
        if (objective_name, direction) in grouped
    )
    return TrainingFullPhysicsResponseAtlas(
        atlas_id="training-full-physics-response-atlas-v1",
        source_split="train",
        transition_count=len(records),
        comparison_count=comparison_count,
        episode_count=len(by_episode),
        patient_count=len(patients),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        minimum_absolute_log_weight_change=minimum_absolute_log_weight_change,
        source_episode_sha256=tuple(sorted(source_hashes)),
        entries=entries,
    )


def _summarize_group(
    objective_name: str,
    direction: str,
    rows: list[_Response],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> FullPhysicsResponseEntry:
    objective = np.asarray([row.objective_improvement for row in rows], dtype=float)
    constraint = np.asarray([row.constraint_improvement for row in rows], dtype=float)
    log_change = np.asarray(
        [row.absolute_log_weight_change for row in rows], dtype=float
    )
    by_patient: dict[str, list[float]] = {}
    for row in rows:
        by_patient.setdefault(row.patient_id, []).append(row.objective_improvement)
    patient_values = np.asarray(
        [median(by_patient[patient]) for patient in sorted(by_patient)], dtype=float
    )
    bootstrap = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        sampled = rng.choice(patient_values, size=len(patient_values), replace=True)
        bootstrap[index] = float(np.median(sampled))
    return FullPhysicsResponseEntry(
        objective_name=objective_name,
        direction=direction,
        comparison_count=len(rows),
        episode_count=len({row.episode_id for row in rows}),
        patient_count=len(by_patient),
        median_absolute_log_weight_change=float(np.median(log_change)),
        median_objective_improvement=float(np.median(objective)),
        objective_improvement_interquartile_low=float(np.quantile(objective, 0.25)),
        objective_improvement_interquartile_high=float(np.quantile(objective, 0.75)),
        objective_improvement_success_rate=float(np.mean(objective > 0.0)),
        median_constraint_improvement=float(np.median(constraint)),
        patient_bootstrap_objective_low=float(np.quantile(bootstrap, 0.025)),
        patient_bootstrap_objective_high=float(np.quantile(bootstrap, 0.975)),
    )
