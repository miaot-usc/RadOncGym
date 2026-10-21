

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from radoncgym.core.errors import ContractError
from radoncgym.core.models import Observation
from radoncgym.trajectories.serialization import (
    observation_from_dict,
    observation_to_dict,
)


TRAJECTORY_SCHEMA_VERSION = "radonctraj-0.4.0"


@dataclass(frozen=True, slots=True)
class PolicyProvenance:
    pass

    family: str
    name: str
    version: str
    checkpoint_sha256: str | None = None
    prompt_template_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.family or not self.name or not self.version:
            raise ContractError("Policy family, name, and version are required")


@dataclass(frozen=True, slots=True)
class TrajectoryEpisode:
    pass

    episode_id: str
    task_id: str
    seed: int
    policy: PolicyProvenance
    final_observation: Observation
    started_at_utc: str
    ended_at_utc: str
    software_version: str
    reward_version: str
    rewards: tuple[float, ...]
    schema_version: str = TRAJECTORY_SCHEMA_VERSION
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        required = {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "software_version": self.software_version,
            "reward_version": self.reward_version,
            "schema_version": self.schema_version,
        }
        for name, value in required.items():
            if not value:
                raise ContractError(f"{name} is required")
        if not self.reward_version:
            raise ContractError("reward_version is required")
        if len(self.rewards) != len(self.final_observation.history):
            raise ContractError(
                "reward trace length must equal the number of environment steps"
            )
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in self.rewards
        ):
            raise ContractError("reward trace must contain finite numeric values")
        if self.task_id != self.final_observation.task_config.task_id:
            raise ContractError(
                "trajectory task_id must match final observation task contract"
            )
        metadata_keys = [key for key, _ in self.metadata]
        if len(metadata_keys) != len(set(metadata_keys)):
            raise ContractError("Trajectory metadata keys must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "policy": {
                "family": self.policy.family,
                "name": self.policy.name,
                "version": self.policy.version,
                "checkpoint_sha256": self.policy.checkpoint_sha256,
                "prompt_template_sha256": self.policy.prompt_template_sha256,
            },
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "software_version": self.software_version,
            "reward_version": self.reward_version,
            "rewards": list(self.rewards),
            "metadata": dict(self.metadata),
            "final_observation": observation_to_dict(self.final_observation),
        }

    @classmethod
    def from_dict(cls, value: Any) -> TrajectoryEpisode:
        if not isinstance(value, Mapping):
            raise ContractError("Trajectory episode must be a mapping")
        policy_data = value.get("policy")
        if not isinstance(policy_data, Mapping):
            raise ContractError("Trajectory policy must be a mapping")
        metadata_data = value.get("metadata", {})
        if not isinstance(metadata_data, Mapping):
            raise ContractError("Trajectory metadata must be a mapping")
        schema_version = str(value["schema_version"])
        if schema_version != TRAJECTORY_SCHEMA_VERSION:
            raise ContractError(
                f"Unsupported trajectory schema {schema_version!r}; "
                f"expected {TRAJECTORY_SCHEMA_VERSION!r}"
            )
        return cls(
            episode_id=str(value["episode_id"]),
            task_id=str(value["task_id"]),
            seed=int(value["seed"]),
            policy=PolicyProvenance(
                family=str(policy_data["family"]),
                name=str(policy_data["name"]),
                version=str(policy_data["version"]),
                checkpoint_sha256=(
                    None
                    if policy_data.get("checkpoint_sha256") is None
                    else str(policy_data["checkpoint_sha256"])
                ),
                prompt_template_sha256=(
                    None
                    if policy_data.get("prompt_template_sha256") is None
                    else str(policy_data["prompt_template_sha256"])
                ),
            ),
            final_observation=observation_from_dict(value["final_observation"]),
            started_at_utc=str(value["started_at_utc"]),
            ended_at_utc=str(value["ended_at_utc"]),
            software_version=str(value["software_version"]),
            reward_version=str(value["reward_version"]),
            rewards=tuple(float(item) for item in value["rewards"]),
            schema_version=schema_version,
            metadata=tuple(
                (str(key), str(metadata_value))
                for key, metadata_value in metadata_data.items()
            ),
        )

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
