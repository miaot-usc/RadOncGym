

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np

from radoncgym.backends.disabled import DisabledFullBackend
from radoncgym.core.errors import ContractError
from radoncgym.core.models import BudgetConfig, PlanningContext
from radoncgym.data.manifest import PredictionTier, load_dataset_manifest
from radoncgym.data.openkbp import OpenKBPPublicCaseLoader
from radoncgym.envs.context_sampling import (
    ContextSamplingMode,
    RadOncCoordinateActionEnv,
    RadOncContextSamplingEnv,
    RadOncWeightOnlyEnv,
)
from radoncgym.envs.factory import make_openkbp_engine
from radoncgym.rewards import VisibleIncumbentImprovementReward
from radoncgym.surrogates import RankHGBBackend
from radoncgym.tasks import get_task_config


def manifest_contexts(
    *,
    manifest_path: str | Path,
    split_name: str,
    prediction_set_ids: Sequence[str] | None = None,
) -> tuple[PlanningContext, ...]:
    pass

    manifest = load_dataset_manifest(manifest_path)
    canonical = manifest.predictions_for_tier(PredictionTier.CANONICAL)
    selected = canonical if prediction_set_ids is None else tuple(prediction_set_ids)
    unknown = sorted(set(selected) - set(canonical))
    if unknown:
        raise ContractError(f"noncanonical or unknown prediction sets: {unknown}")
    if not selected or len(selected) != len(set(selected)):
        raise ContractError("prediction_set_ids must be non-empty and unique")
    return tuple(
        context
        for context in manifest.iter_contexts(
            split_names=(split_name,),
            prediction_tiers=(PredictionTier.CANONICAL,),
        )
        if context.prediction_set_id in selected
    )


def make_rank_hgb_online_env(
    *,
    manifest_path: str | Path,
    openkbp_root: str | Path,
    artifact_path: str | Path,
    split_name: str,
    prediction_set_ids: Sequence[str] | None = None,
    max_steps: int = 8,
    sampling_mode: ContextSamplingMode = ContextSamplingMode.UNIFORM_PATIENT,
    weight_only: bool = True,
    coordinate_normalized_step: float | None = None,
) -> gym.Env[np.ndarray, np.ndarray]:
    pass

    contexts = manifest_contexts(
        manifest_path=manifest_path,
        split_name=split_name,
        prediction_set_ids=prediction_set_ids,
    )
    loader = OpenKBPPublicCaseLoader(openkbp_root)
    backend = RankHGBBackend(artifact_path, loader)
    budget = BudgetConfig(
        max_steps=max_steps,
        max_full_calls=0,
        max_cheap_calls=max_steps,
        invalid_action_limit=max_steps,
    )
    reward = VisibleIncumbentImprovementReward(
        baseline_value=0.0,
        invalid_action_penalty=0.0,
    )

    def engine_factory(context: PlanningContext):
        return make_openkbp_engine(
            context=context,
            budget=budget,
            task_config=get_task_config("BudgetedMultiFidelity-v0"),
            cheap_backend=backend,
            full_backend=DisabledFullBackend(),
            public_case_loader=loader,
            openkbp_root=openkbp_root,
            reward_function=reward,
            all_ones_clinical_objective=None,
        )

    base = RadOncContextSamplingEnv(
        contexts=contexts,
        engine_factory=engine_factory,
        sampling_mode=sampling_mode,
    )
    if weight_only:
        weights = RadOncWeightOnlyEnv(base)
        if coordinate_normalized_step is not None:
            return RadOncCoordinateActionEnv(
                weights,
                normalized_step=coordinate_normalized_step,
            )
        return weights
    if coordinate_normalized_step is not None:
        raise ContractError("coordinate actions require weight_only=True")
    return base
