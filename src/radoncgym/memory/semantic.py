

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping
from typing import Iterable

import numpy as np

from radoncgym.core.constants import OBJECTIVE_NAMES
from radoncgym.core.errors import ContractError
from radoncgym.core.metric_schema import clinical_improvement
from radoncgym.trajectories.schema import TrajectoryEpisode


@dataclass(frozen=True, slots=True)
class SemanticResponseEntry:
    objective_name: str
    elite_geometric_median: float
    elite_interquartile_low: float
    elite_interquartile_high: float
    all_geometric_median: float
    elite_minus_all_log_median: float
    patient_bootstrap_low: float
    patient_bootstrap_high: float
    elite_candidate_count: int
    patient_count: int

    def to_dict(self) -> dict[str, str | float | int]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class SemanticSourceFile:
    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        if not self.path or self.bytes <= 0:
            raise ContractError("semantic source file requires path and bytes")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ContractError("semantic source file requires lowercase SHA-256")

    def to_dict(self) -> dict[str, str | int]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class TrainingSemanticResponseAtlas:
    atlas_id: str
    source_split: str
    ranking_metric: str
    elite_fraction: float
    episode_count: int
    context_count: int
    patient_count: int
    candidate_count: int
    bootstrap_samples: int
    bootstrap_seed: int
    entries: tuple[SemanticResponseEntry, ...]
    source_files: tuple[SemanticSourceFile, ...] = ()

    def __post_init__(self) -> None:
        if self.source_split != "train":
            raise ContractError("semantic response atlas must use train split")
        if tuple(entry.objective_name for entry in self.entries) != OBJECTIVE_NAMES:
            raise ContractError("semantic response atlas objective order is invalid")
        hashes = [source.sha256 for source in self.source_files]
        if len(hashes) != len(set(hashes)):
            raise ContractError("semantic response atlas source hashes must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "training-semantic-response-atlas-0.1.0",
            "atlas_id": self.atlas_id,
            "source_split": self.source_split,
            "ranking_metric": self.ranking_metric,
            "lower_is_better": False,
            "elite_fraction": self.elite_fraction,
            "episode_count": self.episode_count,
            "context_count": self.context_count,
            "patient_count": self.patient_count,
            "candidate_count": self.candidate_count,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "interpretation": (
                "Associative elite cheap-surrogate weight summary; not a causal "
                "effect estimate and not physics-verified plan evidence."
            ),
            "entries": [entry.to_dict() for entry in self.entries],
            "source_files": [source.to_dict() for source in self.source_files],
        }

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        require_sources: bool = True,
    ) -> TrainingSemanticResponseAtlas:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ContractError("semantic response atlas must be a JSON object")
        atlas = cls.from_dict(payload)
        if require_sources and not atlas.source_files:
            raise ContractError("agent-visible semantic atlas requires source hashes")
        return atlas

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> TrainingSemanticResponseAtlas:
        raw_entries = payload.get("entries")
        raw_sources = payload.get("source_files", [])
        if not isinstance(raw_entries, list) or not isinstance(raw_sources, list):
            raise ContractError("semantic response atlas entries/sources are invalid")
        return cls(
            atlas_id=str(payload["atlas_id"]),
            source_split=str(payload["source_split"]),
            ranking_metric=str(payload["ranking_metric"]),
            elite_fraction=float(payload["elite_fraction"]),
            episode_count=int(payload["episode_count"]),
            context_count=int(payload["context_count"]),
            patient_count=int(payload["patient_count"]),
            candidate_count=int(payload["candidate_count"]),
            bootstrap_samples=int(payload["bootstrap_samples"]),
            bootstrap_seed=int(payload["bootstrap_seed"]),
            entries=tuple(
                SemanticResponseEntry(
                    objective_name=str(item["objective_name"]),
                    elite_geometric_median=float(item["elite_geometric_median"]),
                    elite_interquartile_low=float(item["elite_interquartile_low"]),
                    elite_interquartile_high=float(item["elite_interquartile_high"]),
                    all_geometric_median=float(item["all_geometric_median"]),
                    elite_minus_all_log_median=float(
                        item["elite_minus_all_log_median"]
                    ),
                    patient_bootstrap_low=float(item["patient_bootstrap_low"]),
                    patient_bootstrap_high=float(item["patient_bootstrap_high"]),
                    elite_candidate_count=int(item["elite_candidate_count"]),
                    patient_count=int(item["patient_count"]),
                )
                for item in raw_entries
            ),
            source_files=tuple(
                SemanticSourceFile(
                    path=str(item["path"]),
                    sha256=str(item["sha256"]),
                    bytes=int(item["bytes"]),
                )
                for item in raw_sources
            ),
        )


def build_training_semantic_response_atlas(
    episodes: Iterable[TrajectoryEpisode],
    *,
    elite_fraction: float = 0.25,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260715,
) -> TrainingSemanticResponseAtlas:
    pass

    if not 0.0 < elite_fraction <= 1.0:
        raise ContractError("semantic atlas elite_fraction must lie in (0, 1]")
    if bootstrap_samples <= 0:
        raise ContractError("semantic atlas bootstrap_samples must be positive")
    by_context: dict[
        tuple[str, str], list[tuple[float, tuple[float, ...]]]
    ] = {}
    episode_ids: set[str] = set()
    for episode in episodes:
        if episode.episode_id in episode_ids:
            raise ContractError(f"duplicate semantic-atlas episode: {episode.episode_id}")
        episode_ids.add(episode.episode_id)
        observation = episode.final_observation
        if observation.context.split_id != "train":
            raise ContractError("semantic atlas accepts training trajectories only")
        context_key = (
            observation.context.patient_id,
            observation.context.prediction_set_id,
        )
        rows = by_context.setdefault(context_key, [])
        for entry in observation.history:
            if not entry.accepted or entry.outcome is None:
                continue
            objective = clinical_improvement(entry.outcome.cheap)
            if objective is None or not math.isfinite(objective):
                raise ContractError("semantic atlas requires finite cheap improvement")
            values = entry.outcome.executed_weights.values
            if any(value <= 0 or not math.isfinite(value) for value in values):
                raise ContractError("semantic atlas weights must be finite and positive")
            rows.append((objective, values))
    if not episode_ids or not by_context:
        raise ContractError("semantic atlas input cannot be empty")

    elite_by_patient: dict[str, list[np.ndarray]] = {}
    all_logs: list[np.ndarray] = []
    elite_count = 0
    candidate_count = 0
    for (patient_id, _), rows in sorted(by_context.items()):
        if not rows:
            raise ContractError("semantic atlas context has no evaluated candidate")
        ordered = sorted(rows, key=lambda row: row[0], reverse=True)
        keep = max(1, math.ceil(len(ordered) * elite_fraction))
        elite = ordered[:keep]
        elite_logs = [np.log(np.asarray(weights, dtype=np.float64)) for _, weights in elite]
        elite_by_patient.setdefault(patient_id, []).extend(elite_logs)
        all_logs.extend(
            np.log(np.asarray(weights, dtype=np.float64)) for _, weights in rows
        )
        elite_count += len(elite)
        candidate_count += len(rows)

    patient_ids = sorted(elite_by_patient)
    patient_arrays = {
        patient_id: np.stack(values) for patient_id, values in elite_by_patient.items()
    }
    elite_matrix = np.concatenate([patient_arrays[p] for p in patient_ids], axis=0)
    all_matrix = np.stack(all_logs)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty((bootstrap_samples, len(OBJECTIVE_NAMES)), dtype=np.float64)
    for index in range(bootstrap_samples):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        sample_matrix = np.concatenate([patient_arrays[str(p)] for p in sampled], axis=0)
        bootstrap[index] = np.median(sample_matrix, axis=0)

    entries = []
    for objective_index, objective_name in enumerate(OBJECTIVE_NAMES):
        elite_values = elite_matrix[:, objective_index]
        all_values = all_matrix[:, objective_index]
        entries.append(
            SemanticResponseEntry(
                objective_name=objective_name,
                elite_geometric_median=float(np.exp(np.median(elite_values))),
                elite_interquartile_low=float(np.exp(np.quantile(elite_values, 0.25))),
                elite_interquartile_high=float(np.exp(np.quantile(elite_values, 0.75))),
                all_geometric_median=float(np.exp(np.median(all_values))),
                elite_minus_all_log_median=float(
                    np.median(elite_values) - np.median(all_values)
                ),
                patient_bootstrap_low=float(
                    np.exp(np.quantile(bootstrap[:, objective_index], 0.025))
                ),
                patient_bootstrap_high=float(
                    np.exp(np.quantile(bootstrap[:, objective_index], 0.975))
                ),
                elite_candidate_count=elite_count,
                patient_count=len(patient_ids),
            )
        )
    return TrainingSemanticResponseAtlas(
        atlas_id="training-semantic-response-atlas-v1",
        source_split="train",
        ranking_metric="predicted_clinical_improvement_from_all_ones",
        elite_fraction=elite_fraction,
        episode_count=len(episode_ids),
        context_count=len(by_context),
        patient_count=len(patient_ids),
        candidate_count=candidate_count,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        entries=tuple(entries),
    )
