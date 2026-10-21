

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from radoncgym.adapters.dvh import DVHObservation
from radoncgym.core.errors import ContractError
from radoncgym.core.models import FeedbackSource
from radoncgym.envs.normalized import RadOncNormalizedVectorEnv


DVH_VECTOR_SCHEMA_VERSION = "normalized-dvh-vector-0.1.0"


class RadOncDVHVectorEnv(gym.Env[np.ndarray, np.ndarray]):
    pass

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: RadOncNormalizedVectorEnv,
        dvh_observation: DVHObservation,
    ) -> None:
        super().__init__()
        context = env.canonical_env.engine.context
        if dvh_observation.patient_id != context.patient_id:
            raise ContractError("DVH and environment patient IDs differ")
        if dvh_observation.prediction_set_id != context.prediction_set_id:
            raise ContractError("DVH and environment prediction IDs differ")
        if dvh_observation.source is not FeedbackSource.CONTEXT:
            raise ContractError("DVH-v0 initial curves must use context provenance")
        self.env = env
        self.dvh_observation = dvh_observation
        self.action_space = env.action_space
        self._dvh_values = (
            dvh_observation.as_array(dtype=np.dtype(np.float32)).reshape(-1)
            * np.float32(2.0)
            - np.float32(1.0)
        )
        self._dvh_mask = np.asarray(
            [1.0 if present else -1.0 for present in dvh_observation.present_mask],
            dtype=np.float32,
        )
        self.observation_feature_names = (
            env.observation_feature_names
            + tuple(
                f"dvh:context:{roi}:volume_at_or_above:{dose:.1f}_gy"
                for roi in dvh_observation.roi_names
                for dose in dvh_observation.dose_bins_gy
            )
            + tuple(
                f"dvh:context:{roi}:present" for roi in dvh_observation.roi_names
            )
        )
        self.observation_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(len(self.observation_feature_names),),
            dtype=np.float32,
        )

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
                "dvh_vector_schema_version": DVH_VECTOR_SCHEMA_VERSION,
                "dvh_observation_schema_version": self.dvh_observation.schema_version,
                "dvh_source": self.dvh_observation.source.value,
                "dvh_dose_bins_gy": self.dvh_observation.dose_bins_gy,
                "dvh_roi_names": self.dvh_observation.roi_names,
                "normalized_observation_feature_names": (
                    self.observation_feature_names
                ),
            }
        )
        return self._append_dvh(observation), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._append_dvh(observation), reward, terminated, truncated, info

    def canonical_action(self, action: np.ndarray) -> dict[str, Any]:
        return self.env.canonical_action(action)

    def normalized_action(
        self,
        weights: np.ndarray,
        *,
        request_full_verifier: bool,
        stop: bool,
    ) -> np.ndarray:
        return self.env.normalized_action(
            weights,
            request_full_verifier=request_full_verifier,
            stop=stop,
        )

    def close(self) -> None:
        self.env.close()

    def _append_dvh(self, observation: np.ndarray) -> np.ndarray:
        result = np.concatenate(
            (observation, self._dvh_values, self._dvh_mask)
        ).astype(np.float32, copy=False)
        if not self.observation_space.contains(result):
            raise RuntimeError("DVH-v0 observation violates observation_space")
        return result
