

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from radoncgym.core.constants import OBJECTIVE_NAMES
from radoncgym.core.metric_schema import (
    PUBLIC_CONSTRAINT_METRIC_NAMES,
    PUBLIC_PLAN_METRIC_NAMES,
)
from radoncgym.core.models import BudgetConfig, Observation
from radoncgym.envs.gymnasium_env import RadOncGymnasiumEnv, make_synthetic_env


_BINARY_CONTROL_THRESHOLD = np.float32(0.5)
NORMALIZED_OBSERVATION_SCHEMA_VERSION = "normalized-structured-vector-0.3.0"
NORMALIZED_ACTION_SCHEMA_VERSION = "normalized-continuous-action-0.1.0"


class NormalizedVectorCodec:
    pass

    def __init__(
        self,
        *,
        public_context_feature_names: tuple[str, ...],
        budget_config: BudgetConfig,
    ) -> None:
        self.public_context_feature_names = public_context_feature_names
        self.budget_config = budget_config
        self.max_steps = budget_config.max_steps
        self.num_objectives = len(OBJECTIVE_NAMES)
        self.observation_feature_names = self._build_feature_names()
        self.action_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(self.num_objectives + 2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(len(self.observation_feature_names),),
            dtype=np.float32,
        )

    @classmethod
    def from_observation(cls, observation: Observation) -> NormalizedVectorCodec:
        pass

        return cls(
            public_context_feature_names=tuple(
                name for name, _ in observation.public_context.items
            ),
            budget_config=observation.budget_config,
        )

    def canonical_action(self, action: np.ndarray) -> dict[str, Any]:
        pass

        value = np.asarray(action, dtype=np.float32)
        if value.shape != self.action_space.shape:
            raise ValueError(
                f"normalized action shape {value.shape} != {self.action_space.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("normalized action must contain only finite values")
        clipped = np.clip(value, -1.0, 1.0)
        stop = bool(clipped[-1] > _BINARY_CONTROL_THRESHOLD)
        request_full = bool(clipped[-2] > _BINARY_CONTROL_THRESHOLD) and not stop
        weights = np.power(np.float32(4.0), clipped[: self.num_objectives])
        return {
            "weights": weights.astype(np.float32),
            "request_full_verifier": int(request_full),
            "stop": int(stop),
        }

    def normalized_action(
        self,
        weights: np.ndarray,
        *,
        request_full_verifier: bool,
        stop: bool,
    ) -> np.ndarray:
        pass

        raw_weights = np.asarray(weights, dtype=np.float32)
        if raw_weights.shape != (self.num_objectives,):
            raise ValueError("canonical weight vector has the wrong shape")
        if np.any(raw_weights < 0.25) or np.any(raw_weights > 4.0):
            raise ValueError("canonical weights must lie in [0.25, 4.0]")
        result = np.empty(self.num_objectives + 2, dtype=np.float32)
        result[: self.num_objectives] = np.log(raw_weights) / np.log(4.0)
        result[-2] = np.float32(1.0 if request_full_verifier and not stop else -1.0)
        result[-1] = np.float32(1.0 if stop else -1.0)
        return result

    def encode_observation(
        self,
        observation: dict[str, np.ndarray],
    ) -> np.ndarray:
        pass

        weights = np.log(observation["weights_history"]) / np.log(4.0)
        budget = observation["budget_usage"].astype(np.float32, copy=True)
        cheap_limit = max(
            self.budget_config.max_cheap_calls or self.max_steps,
            1,
        )
        full_limit = max(self.budget_config.max_full_calls, 1)
        budget[:6] = np.asarray(
            (
                budget[0] / self.max_steps,
                budget[1] / cheap_limit,
                budget[2] / full_limit,
                budget[3] / full_limit,
                budget[4] / full_limit,
                budget[5] / self.budget_config.invalid_action_limit,
            ),
            dtype=np.float32,
        )
        budget[6:] = _nonnegative_log_squash(budget[6:])
        incumbents = observation["incumbent_steps"].astype(np.float32)
        present = incumbents >= 0
        incumbents[present] /= max(self.max_steps - 1, 1)

        parts = (
            _signed_log_squash(observation["public_context"]),
            observation["clinical_preference"].astype(np.float32),
            weights.reshape(-1),
            observation["action_flags"].astype(np.float32).reshape(-1),
            observation["history_mask"].astype(np.float32),
            observation["accepted_mask"].astype(np.float32),
            _signed_log_squash(observation["cheap_metrics"]).reshape(-1),
            observation["cheap_metrics_mask"].astype(np.float32).reshape(-1),
            _signed_log_squash(observation["full_metrics"]).reshape(-1),
            observation["full_metrics_mask"].astype(np.float32).reshape(-1),
            _nonnegative_log_squash(observation["full_constraints"]).reshape(-1),
            observation["full_constraints_mask"].astype(np.float32).reshape(-1),
            budget,
            incumbents,
        )
        encoded = np.concatenate(parts).astype(np.float32, copy=False)
        encoded = np.clip(encoded, -1.0, 1.0)
        if not self.observation_space.contains(encoded):
            raise RuntimeError("normalized observation violates observation_space")
        return encoded

    def _build_feature_names(self) -> tuple[str, ...]:
        names = list(self.public_context_feature_names)
        names.extend(
            f"clinical_preference:{objective}" for objective in OBJECTIVE_NAMES
        )
        for step in range(self.max_steps):
            names.extend(
                f"history:{step}:weight:{objective}" for objective in OBJECTIVE_NAMES
            )
        for step in range(self.max_steps):
            names.extend(
                (
                    f"history:{step}:requested_full",
                    f"history:{step}:stop",
                )
            )
        for field in ("history_mask", "accepted_mask"):
            names.extend(f"history:{step}:{field}" for step in range(self.max_steps))
        for source in ("cheap", "full"):
            for step in range(self.max_steps):
                names.extend(
                    f"history:{step}:{source}_metric:{metric}"
                    for metric in PUBLIC_PLAN_METRIC_NAMES
                )
            for step in range(self.max_steps):
                names.extend(
                    f"history:{step}:{source}_metric_present:{metric}"
                    for metric in PUBLIC_PLAN_METRIC_NAMES
                )
        for step in range(self.max_steps):
            names.extend(
                f"history:{step}:full_constraint:{metric}"
                for metric in PUBLIC_CONSTRAINT_METRIC_NAMES
            )
        for step in range(self.max_steps):
            names.extend(
                f"history:{step}:full_constraint_present:{metric}"
                for metric in PUBLIC_CONSTRAINT_METRIC_NAMES
            )
        names.extend(
            (
                "budget:steps_used_fraction",
                "budget:cheap_calls_used_fraction",
                "budget:full_calls_attempted_fraction",
                "budget:full_calls_solved_fraction",
                "budget:full_calls_failed_fraction",
                "budget:invalid_actions_fraction",
                "budget:wall_seconds_squashed",
                "budget:solver_seconds_squashed",
                "incumbent:best_cheap_step_fraction",
                "incumbent:best_full_step_fraction",
            )
        )
        if len(names) != len(set(names)):
            raise RuntimeError("normalized observation feature names are not unique")
        return tuple(names)


class RadOncNormalizedVectorEnv(gym.Env[np.ndarray, np.ndarray]):
    pass

    metadata = {"render_modes": []}

    def __init__(self, env: RadOncGymnasiumEnv) -> None:
        super().__init__()
        self.env = env
        self.canonical_env = env
        self.max_steps = env.max_steps
        self.num_objectives = len(OBJECTIVE_NAMES)
        self.codec = NormalizedVectorCodec(
            public_context_feature_names=tuple(
                name for name, _ in env.engine.public_context.items
            ),
            budget_config=env.engine.budget_config,
        )
        self.observation_feature_names = self.codec.observation_feature_names
        self.action_space = self.codec.action_space
        self.observation_space = self.codec.observation_space

    def close(self) -> None:
        self.env.close()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        observation, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info.update(
            {
                "normalized_observation_schema_version": (
                    NORMALIZED_OBSERVATION_SCHEMA_VERSION
                ),
                "normalized_action_schema_version": NORMALIZED_ACTION_SCHEMA_VERSION,
                "normalized_observation_feature_names": (
                    self.observation_feature_names
                ),
            }
        )
        return self.codec.encode_observation(observation), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        canonical = self.canonical_action(action)
        observation, reward, terminated, truncated, info = self.env.step(canonical)
        return (
            self.codec.encode_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def canonical_action(self, action: np.ndarray) -> dict[str, Any]:
        pass

        return self.codec.canonical_action(action)

    def normalized_action(
        self,
        weights: np.ndarray,
        *,
        request_full_verifier: bool,
        stop: bool,
    ) -> np.ndarray:
        pass

        return self.codec.normalized_action(
            weights,
            request_full_verifier=request_full_verifier,
            stop=stop,
        )


def _signed_log_squash(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    transformed = np.sign(array) * np.log1p(np.abs(array))
    return transformed / (np.float32(1.0) + np.abs(transformed))


def _nonnegative_log_squash(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), np.float32(0.0))
    transformed = np.log1p(array)
    return transformed / (np.float32(1.0) + transformed)


def make_normalized_synthetic_env(
    *,
    max_steps: int = 8,
    max_full_calls: int = 2,
    max_cheap_calls: int | None = None,
) -> RadOncNormalizedVectorEnv:
    pass

    return RadOncNormalizedVectorEnv(
        make_synthetic_env(
            max_steps=max_steps,
            max_full_calls=max_full_calls,
            max_cheap_calls=max_cheap_calls,
        )
    )
