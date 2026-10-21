

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from radoncgym.backends.synthetic import SyntheticCheapBackend, SyntheticFullBackend
from radoncgym.clinical import (
    CLINICAL_PREFERENCE_SCHEMA_VERSION,
    clinical_preference_profile,
)
from radoncgym.core.constants import (
    MAX_OBJECTIVE_WEIGHT,
    MIN_OBJECTIVE_WEIGHT,
    OBJECTIVE_NAMES,
)
from radoncgym.core.engine import RadOncEngine
from radoncgym.core.metric_schema import (
    PUBLIC_CONSTRAINT_METRIC_NAMES,
    PUBLIC_CONSTRAINT_SCHEMA_VERSION,
    PUBLIC_PLAN_METRIC_NAMES,
    PUBLIC_PLAN_METRIC_SCHEMA_VERSION,
    clinical_improvement,
    vectorize_public_metrics,
)
from radoncgym.core.models import (
    BudgetConfig,
    CandidateAction,
    ObjectiveWeights,
    Observation,
    PlanningContext,
    PublicContextFeatures,
)
from radoncgym.rewards.base import VisibleIncumbentImprovementReward


class RadOncGymnasiumEnv(gym.Env[dict[str, np.ndarray], Mapping[str, Any]]):
    pass

    metadata = {"render_modes": []}

    def __init__(self, engine: RadOncEngine) -> None:
        super().__init__()
        self.engine = engine
        self.max_steps = engine.budget_config.max_steps
        num_objectives = len(OBJECTIVE_NAMES)
        num_plan_metrics = len(PUBLIC_PLAN_METRIC_NAMES)
        num_constraints = len(PUBLIC_CONSTRAINT_METRIC_NAMES)

        self.action_space = spaces.Dict(
            {
                "weights": spaces.Box(
                    low=np.float32(MIN_OBJECTIVE_WEIGHT),
                    high=np.float32(MAX_OBJECTIVE_WEIGHT),
                    shape=(num_objectives,),
                    dtype=np.float32,
                ),
                "request_full_verifier": spaces.Discrete(2),
                "stop": spaces.Discrete(2),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "public_context": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(engine.public_context.items),),
                    dtype=np.float32,
                ),
                "clinical_preference": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(num_objectives,),
                    dtype=np.float32,
                ),
                "weights_history": spaces.Box(
                    low=np.float32(MIN_OBJECTIVE_WEIGHT),
                    high=np.float32(MAX_OBJECTIVE_WEIGHT),
                    shape=(self.max_steps, num_objectives),
                    dtype=np.float32,
                ),
                "action_flags": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_steps, 2),
                    dtype=np.int8,
                ),
                "history_mask": spaces.MultiBinary(self.max_steps),
                "accepted_mask": spaces.MultiBinary(self.max_steps),
                "cheap_clinical_improvement": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_steps,),
                    dtype=np.float32,
                ),
                "cheap_metric_mask": spaces.MultiBinary(self.max_steps),
                "cheap_metrics": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_steps, num_plan_metrics),
                    dtype=np.float32,
                ),
                "cheap_metrics_mask": spaces.MultiBinary(
                    (self.max_steps, num_plan_metrics)
                ),
                "full_clinical_improvement": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_steps,),
                    dtype=np.float32,
                ),
                "full_metric_mask": spaces.MultiBinary(self.max_steps),
                "full_metrics": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_steps, num_plan_metrics),
                    dtype=np.float32,
                ),
                "full_metrics_mask": spaces.MultiBinary(
                    (self.max_steps, num_plan_metrics)
                ),
                "full_constraint_sum": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.max_steps,),
                    dtype=np.float32,
                ),
                "full_constraints": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.max_steps, num_constraints),
                    dtype=np.float32,
                ),
                "full_constraints_mask": spaces.MultiBinary(
                    (self.max_steps, num_constraints)
                ),
                "budget_usage": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(8,),
                    dtype=np.float32,
                ),
                "incumbent_steps": spaces.Box(
                    low=-1,
                    high=self.max_steps - 1,
                    shape=(2,),
                    dtype=np.int32,
                ),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        observation, info = self.engine.reset(seed=seed)
        info.update(
            {
                "public_plan_metric_schema_version": (
                    PUBLIC_PLAN_METRIC_SCHEMA_VERSION
                ),
                "public_plan_metric_names": PUBLIC_PLAN_METRIC_NAMES,
                "public_constraint_schema_version": (
                    PUBLIC_CONSTRAINT_SCHEMA_VERSION
                ),
                "public_constraint_metric_names": PUBLIC_CONSTRAINT_METRIC_NAMES,
                "clinical_preference_schema_version": (
                    CLINICAL_PREFERENCE_SCHEMA_VERSION
                ),
                "clinical_preference_objective_names": OBJECTIVE_NAMES,
            }
        )
        return self._encode_observation(observation), info

    def step(
        self, action: Mapping[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        stop = bool(action["stop"])
        weights = None
        if not stop:
            weights = ObjectiveWeights(
                tuple(float(value) for value in action["weights"])
            )
        canonical_action = CandidateAction(
            weights=weights,
            request_full_verifier=(
                bool(action["request_full_verifier"]) and not stop
            ),
            stop=stop,
        )
        result = self.engine.step(canonical_action)
        return (
            self._encode_observation(result.observation),
            result.reward,
            result.terminated,
            result.truncated,
            dict(result.info),
        )

    def _encode_observation(self, observation: Observation) -> dict[str, np.ndarray]:
        encoded = encode_structured_observation(observation)
        if not self.observation_space.contains(encoded):
            raise RuntimeError("Encoded observation violates observation_space")
        return encoded


def encode_structured_observation(
    observation: Observation,
) -> dict[str, np.ndarray]:
    pass

    max_steps = observation.budget_config.max_steps
    num_objectives = len(OBJECTIVE_NAMES)
    weights_history = np.full(
        (max_steps, num_objectives),
        1.0,
        dtype=np.float32,
    )
    action_flags = np.zeros((max_steps, 2), dtype=np.int8)
    history_mask = np.zeros(max_steps, dtype=np.int8)
    accepted_mask = np.zeros(max_steps, dtype=np.int8)
    cheap_improvement = np.zeros(max_steps, dtype=np.float32)
    cheap_metric_mask = np.zeros(max_steps, dtype=np.int8)
    cheap_metrics = np.zeros(
        (max_steps, len(PUBLIC_PLAN_METRIC_NAMES)), dtype=np.float32
    )
    cheap_metrics_mask = np.zeros_like(cheap_metrics, dtype=np.int8)
    full_improvement = np.zeros(max_steps, dtype=np.float32)
    full_metric_mask = np.zeros(max_steps, dtype=np.int8)
    full_metrics = np.zeros_like(cheap_metrics)
    full_metrics_mask = np.zeros_like(cheap_metrics_mask)
    full_constraint_sum = np.zeros(max_steps, dtype=np.float32)
    full_constraints = np.zeros(
        (max_steps, len(PUBLIC_CONSTRAINT_METRIC_NAMES)), dtype=np.float32
    )
    full_constraints_mask = np.zeros_like(full_constraints, dtype=np.int8)

    for entry in observation.history:
        index = entry.step_index
        if index >= max_steps:
            raise RuntimeError("Core history exceeds declared Gymnasium max_steps")
        history_mask[index] = 1
        accepted_mask[index] = int(entry.accepted)
        action_flags[index, 0] = int(entry.action.request_full_verifier)
        action_flags[index, 1] = int(entry.action.stop)
        if entry.action.weights is not None:
            weights_history[index] = np.asarray(
                entry.action.weights.values, dtype=np.float32
            )
        if entry.outcome is None:
            continue
        cheap_metrics[index], cheap_metrics_mask[index] = vectorize_public_metrics(
            entry.outcome.cheap.metrics,
            names=PUBLIC_PLAN_METRIC_NAMES,
        )
        cheap_value = clinical_improvement(entry.outcome.cheap)
        if cheap_value is not None:
            cheap_improvement[index] = np.float32(cheap_value)
            cheap_metric_mask[index] = 1
        if entry.outcome.full is None:
            continue
        full_metrics[index], full_metrics_mask[index] = vectorize_public_metrics(
            entry.outcome.full.metrics,
            names=PUBLIC_PLAN_METRIC_NAMES,
        )
        full_constraints[index], full_constraints_mask[index] = (
            vectorize_public_metrics(
                entry.outcome.full.constraints,
                names=PUBLIC_CONSTRAINT_METRIC_NAMES,
            )
        )
        full_value = clinical_improvement(entry.outcome.full)
        if full_value is not None:
            full_improvement[index] = np.float32(full_value)
            full_metric_mask[index] = 1
        full_constraint_sum[index] = np.float32(
            entry.outcome.full.constraints.sum()
        )

    state = observation.budget_state
    budget_usage = np.asarray(
        (
            state.steps_used,
            state.cheap_calls_used,
            state.full_calls_attempted,
            state.full_calls_solved,
            state.full_calls_failed,
            state.invalid_actions,
            state.wall_seconds,
            state.solver_seconds,
        ),
        dtype=np.float32,
    )
    incumbent_steps = np.asarray(
        (
            -1
            if observation.incumbents.best_cheap_step is None
            else observation.incumbents.best_cheap_step,
            -1
            if observation.incumbents.best_full_step is None
            else observation.incumbents.best_full_step,
        ),
        dtype=np.int32,
    )
    return {
        "public_context": np.asarray(
            [value for _, value in observation.public_context.items],
            dtype=np.float32,
        ),
        "clinical_preference": np.asarray(
            clinical_preference_profile(
                observation.context.clinical_profile_id
            ).objective_emphasis,
            dtype=np.float32,
        ),
        "weights_history": weights_history,
        "action_flags": action_flags,
        "history_mask": history_mask,
        "accepted_mask": accepted_mask,
        "cheap_clinical_improvement": cheap_improvement,
        "cheap_metric_mask": cheap_metric_mask,
        "cheap_metrics": cheap_metrics,
        "cheap_metrics_mask": cheap_metrics_mask,
        "full_clinical_improvement": full_improvement,
        "full_metric_mask": full_metric_mask,
        "full_metrics": full_metrics,
        "full_metrics_mask": full_metrics_mask,
        "full_constraint_sum": full_constraint_sum,
        "full_constraints": full_constraints,
        "full_constraints_mask": full_constraints_mask,
        "budget_usage": budget_usage,
        "incumbent_steps": incumbent_steps,
    }


def make_synthetic_env(
    *,
    max_steps: int = 8,
    max_full_calls: int = 2,
    max_cheap_calls: int | None = None,
) -> RadOncGymnasiumEnv:
    pass

    context = PlanningContext(
        case_id="synthetic_pt_000:set_1:absolute_mean",
        patient_id="synthetic_pt_000",
        prediction_set_id="set_1",
        split_id="synthetic",
    )
    full_backend = SyntheticFullBackend()
    anchor = SyntheticFullBackend().evaluate(
        context, ObjectiveWeights.all_ones()
    )
    engine = RadOncEngine(
        context=context,
        public_context=PublicContextFeatures(
            schema_version="synthetic-context-0.1.0",
            items=(
                ("anatomy:target_volume_cc", 72.0),
                ("anatomy:oar_overlap_fraction", 0.18),
                ("prediction:mean_dose_gy", 34.5),
                ("prediction:target_d95_gy", 68.2),
            ),
        ),
        budget=BudgetConfig(
            max_steps=max_steps,
            max_full_calls=max_full_calls,
            max_cheap_calls=max_cheap_calls,
        ),
        cheap_backend=SyntheticCheapBackend(),
        full_backend=full_backend,
        reward_function=VisibleIncumbentImprovementReward(),
        all_ones_clinical_objective=float(
            anchor.metrics.get("objective")
        ),
    )
    return RadOncGymnasiumEnv(engine)
