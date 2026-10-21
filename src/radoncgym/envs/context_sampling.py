

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from radoncgym.core.engine import RadOncEngine
from radoncgym.core.errors import ContractError
from radoncgym.core.models import PlanningContext
from radoncgym.envs.gymnasium_env import RadOncGymnasiumEnv
from radoncgym.envs.normalized import RadOncNormalizedVectorEnv


EngineFactory = Callable[[PlanningContext], RadOncEngine]


class ContextSamplingMode(str, Enum):
    pass

    UNIFORM_CONTEXT = "uniform_context"
    UNIFORM_PATIENT = "uniform_patient"


class RadOncContextSamplingEnv(gym.Env[np.ndarray, np.ndarray]):
    pass

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        contexts: Sequence[PlanningContext],
        engine_factory: EngineFactory,
        sampling_mode: ContextSamplingMode = ContextSamplingMode.UNIFORM_PATIENT,
    ) -> None:
        super().__init__()
        if not contexts:
            raise ContractError("context sampler requires at least one context")
        self.contexts = tuple(contexts)
        if len({context.case_id for context in self.contexts}) != len(self.contexts):
            raise ContractError("context sampler case IDs must be unique")
        split_ids = {context.split_id for context in self.contexts}
        if len(split_ids) != 1:
            raise ContractError(
                "one context sampler cannot mix patient splits; use separate train/eval envs"
            )
        self.split_id = next(iter(split_ids))
        self.engine_factory = engine_factory
        self.sampling_mode = ContextSamplingMode(sampling_mode)
        self._patient_indices: dict[str, tuple[int, ...]] = self._index_patients()
        self._patient_ids = tuple(sorted(self._patient_indices))
        self._active: RadOncNormalizedVectorEnv | None = None
        self._active_context_index: int | None = None

        prototype = self._build(0)
        self.action_space = prototype.action_space
        self.observation_space = prototype.observation_space
        self.observation_feature_names = prototype.observation_feature_names
        prototype.close()

    @property
    def active_context(self) -> PlanningContext:
        if self._active_context_index is None:
            raise RuntimeError("reset() must be called before reading active_context")
        return self.contexts[self._active_context_index]

    @property
    def active_engine(self) -> RadOncEngine:
        pass

        if self._active is None:
            raise RuntimeError("reset() must be called before reading active_engine")
        return self._active.canonical_env.engine

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        unknown = sorted(set(options) - {"context_index"})
        if unknown:
            raise ContractError(f"unknown context-sampler reset options: {unknown}")
        if "context_index" in options:
            index = int(options["context_index"])
            if index < 0 or index >= len(self.contexts):
                raise ContractError("context_index is outside the sampler registry")
        else:
            index = self._sample_index()

        if self._active is not None:
            self._active.close()
        self._active = self._build(index)
        self._active_context_index = index
        episode_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
        observation, info = self._active.reset(seed=episode_seed)
        info = dict(info)
        info.update(
            {
                "context_sampler_mode": self.sampling_mode.value,
                "context_registry_size": len(self.contexts),
                "context_index": index,
                "case_id": self.active_context.case_id,
                "patient_id": self.active_context.patient_id,
                "prediction_set_id": self.active_context.prediction_set_id,
                "split_id": self.active_context.split_id,
            }
        )
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._active is None:
            raise RuntimeError("reset() must be called before step()")
        observation, reward, terminated, truncated, info = self._active.step(action)
        info = dict(info)
        info.update(
            {
                "case_id": self.active_context.case_id,
                "patient_id": self.active_context.patient_id,
                "prediction_set_id": self.active_context.prediction_set_id,
                "split_id": self.active_context.split_id,
            }
        )
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self._active is not None:
            self._active.close()
            self._active = None

    def _index_patients(self) -> dict[str, tuple[int, ...]]:
        mutable: dict[str, list[int]] = defaultdict(list)
        for index, context in enumerate(self.contexts):
            mutable[context.patient_id].append(index)
        return {patient: tuple(indices) for patient, indices in mutable.items()}

    def _sample_index(self) -> int:
        if self.sampling_mode is ContextSamplingMode.UNIFORM_CONTEXT:
            return int(self.np_random.integers(len(self.contexts)))
        patient = self._patient_ids[int(self.np_random.integers(len(self._patient_ids)))]
        indices = self._patient_indices[patient]
        return indices[int(self.np_random.integers(len(indices)))]

    def _build(self, index: int) -> RadOncNormalizedVectorEnv:
        wrapped = RadOncNormalizedVectorEnv(
            RadOncGymnasiumEnv(self.engine_factory(self.contexts[index]))
        )
        if hasattr(self, "observation_space"):
            if wrapped.observation_space != self.observation_space:
                raise ContractError("sampled contexts do not share an observation space")
            if wrapped.action_space != self.action_space:
                raise ContractError("sampled contexts do not share an action space")
            if wrapped.observation_feature_names != self.observation_feature_names:
                raise ContractError("sampled contexts do not share a feature schema")
        return wrapped


class RadOncWeightOnlyEnv(gym.ActionWrapper[np.ndarray, np.ndarray, np.ndarray]):
    pass

    def __init__(self, env: gym.Env[np.ndarray, np.ndarray]) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box) or env.action_space.shape != (14,):
            raise ContractError("weight-only wrapper requires the normalized 14-D action")
        self.action_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(12,),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (12,) or not np.all(np.isfinite(value)):
            raise ContractError("weight-only action must be a finite 12-vector")
        canonical = np.full(14, -1.0, dtype=np.float32)
        canonical[:12] = np.clip(value, -1.0, 1.0)
        return canonical


class RadOncCoordinateActionEnv(gym.Wrapper[np.ndarray, int, np.ndarray, np.ndarray]):
    pass

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: gym.Env[np.ndarray, np.ndarray],
        *,
        normalized_step: float = 0.25,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box) or env.action_space.shape != (12,):
            raise ContractError("coordinate wrapper requires the 12-D weight action")
        if not 0.0 < normalized_step <= 1.0:
            raise ContractError("coordinate normalized_step must lie in (0, 1]")
        self.normalized_step = np.float32(normalized_step)
        self.num_objectives = int(env.action_space.shape[0])
        self.retain_action = 2 * self.num_objectives
        self.action_space = spaces.Discrete(self.retain_action + 1)
        self._weights = np.zeros(self.num_objectives, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._weights.fill(0.0)
        observation, info = self.env.reset(seed=seed, options=options)
        result = dict(info)
        result.update(self._coordinate_info(None))
        return observation, result

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_index = int(action)
        if not self.action_space.contains(action_index):
            raise ContractError("coordinate action is outside the discrete space")
        changed_objective: int | None = None
        direction = 0
        if action_index != self.retain_action:
            changed_objective = action_index // 2
            direction = -1 if action_index % 2 == 0 else 1
            self._weights[changed_objective] = np.clip(
                self._weights[changed_objective]
                + np.float32(direction) * self.normalized_step,
                -1.0,
                1.0,
            )
        observation, reward, terminated, truncated, info = self.env.step(
            self._weights.copy()
        )
        result = dict(info)
        result.update(self._coordinate_info((changed_objective, direction)))
        return observation, reward, terminated, truncated, result

    def _coordinate_info(
        self,
        change: tuple[int | None, int] | None,
    ) -> dict[str, Any]:
        objective, direction = (None, 0) if change is None else change
        return {
            "coordinate_action_schema_version": "coordinate-log-weight-0.1.0",
            "coordinate_normalized_step": float(self.normalized_step),
            "coordinate_objective_index": objective,
            "coordinate_direction": direction,
            "coordinate_retain_action": self.retain_action,
            "coordinate_normalized_weights": self._weights.tolist(),
        }
