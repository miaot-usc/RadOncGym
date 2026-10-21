

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from radoncgym.adapters.structured import vectorize_public_case
from radoncgym.core.constants import (
    OBJECTIVE_NAMES,
    PREDICTED_CLINICAL_IMPROVEMENT_METRIC,
    PRIMARY_FORMULATION,
)
from radoncgym.core.errors import ContractError
from radoncgym.core.models import (
    ComputeCost,
    EvaluationResult,
    FeedbackSource,
    MetricVector,
    ObjectiveWeights,
    PlanningContext,
    SolverStatus,
)
from radoncgym.data.openkbp import OpenKBPPublicCaseLoader, PublicCaseFeatures


RANK_HGB_FEATURE_VERSION = "rank-hgb-public-context-0.1.0"
RANK_HGB_ARTIFACT_VERSION = "rank-hgb-ensemble-0.1.0"
RANK_HGB_CLINICAL_TARGET = "context_delta:mean_protocol_violation_gy"
RANK_HGB_TARGET_OUTPUT_METRICS = {
    RANK_HGB_CLINICAL_TARGET: PREDICTED_CLINICAL_IMPROVEMENT_METRIC,
}
FROZEN_VERIFIER_FILENAME = (
    "rank_hgb_generated_clinical_fullres_top10_optimized_v1.joblib"
)
FROZEN_VERIFIER_FEATURE_SCHEMA_FILENAME = "verifier_feature_indices.json"


def frozen_verifier_artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / FROZEN_VERIFIER_FILENAME


def frozen_verifier_feature_schema_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "assets"
        / FROZEN_VERIFIER_FEATURE_SCHEMA_FILENAME
    )


def rank_hgb_feature_names(case: PublicCaseFeatures) -> tuple[str, ...]:
    public = vectorize_public_case(case, include_prediction_identity=True)
    return tuple(f"weight:{name}" for name in OBJECTIVE_NAMES) + public.names


def rank_hgb_feature_vector(
    weights: ObjectiveWeights, case: PublicCaseFeatures
) -> np.ndarray:
    pass

    public = vectorize_public_case(case, include_prediction_identity=True)
    values = np.concatenate(
        (
            np.asarray(weights.values, dtype=np.float32),
            public.as_array(dtype=np.dtype(np.float32)),
        )
    )
    if values.shape != (len(rank_hgb_feature_names(case)),):
        raise ContractError(f"unexpected rank-HGB feature shape: {values.shape}")
    return values


class RankHGBBackend:
    pass

    def __init__(
        self,
        artifact_path: str | Path,
        public_case_loader: OpenKBPPublicCaseLoader,
    ) -> None:
        self.artifact_path = Path(artifact_path).resolve()
        self.public_case_loader = public_case_loader
        self.artifact = _load_artifact(self.artifact_path)
        _validate_artifact(self.artifact)
        self.models = tuple(self.artifact["models"])
        self.conformal_q90 = float(self.artifact["uncertainty"]["absolute_residual_q90"])

    def evaluate(
        self, context: PlanningContext, weights: ObjectiveWeights
    ) -> EvaluationResult:
        if context.planning_formulation != PRIMARY_FORMULATION:
            raise ContractError(
                f"rank HGB v1 supports only {PRIMARY_FORMULATION!r}"
            )
        started = time.perf_counter()
        case = self.public_case_loader.load(context)
        candidate = rank_hgb_feature_vector(weights, case).reshape(1, -1)
        baseline = rank_hgb_feature_vector(
            ObjectiveWeights.all_ones(), case
        ).reshape(1, -1)
        member_deltas = np.asarray(
            [
                float(model.predict(candidate)[0] - model.predict(baseline)[0])
                for model in self.models
            ],
            dtype=np.float64,
        )
        predicted_delta = float(np.mean(member_deltas))
        ensemble_std = float(np.std(member_deltas))
        uncertainty = math.hypot(ensemble_std, self.conformal_q90)
        wall_seconds = time.perf_counter() - started
        target = str(self.artifact["target"])
        target_metric = rank_hgb_target_output_metric(target)
        predicted_improvement = -predicted_delta
        return EvaluationResult(
            source=FeedbackSource.CHEAP_LEARNED_SURROGATE,
            status=SolverStatus.NOT_APPLICABLE,
            metrics=MetricVector.from_mapping(
                {
                    target_metric: predicted_improvement,
                    "ensemble_std": ensemble_std,
                }
            ),
            constraints=MetricVector(),
            cost=ComputeCost(wall_seconds=wall_seconds, cheap_calls=1),
            uncertainty=uncertainty,
            message=(
                "rank_hgb:context_relative_to_all_ones:"
                f"{self.artifact_path.name}"
            ),
        )


def _load_artifact(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import joblib

        artifact = joblib.load(path)
    except ModuleNotFoundError as error:
        raise ContractError(
            "loading a rank HGB artifact requires the 'surrogate' dependencies"
        ) from error
    if not isinstance(artifact, Mapping):
        raise ContractError("rank HGB artifact must be a mapping")
    return artifact


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != RANK_HGB_ARTIFACT_VERSION:
        raise ContractError("unsupported rank HGB artifact version")
    if artifact.get("feature_schema_version") != RANK_HGB_FEATURE_VERSION:
        raise ContractError("unsupported rank HGB feature version")
    if artifact.get("planning_formulation") != PRIMARY_FORMULATION:
        raise ContractError("rank HGB artifact has the wrong planning formulation")
    if artifact.get("target") not in RANK_HGB_TARGET_OUTPUT_METRICS:
        raise ContractError("rank HGB artifact has an unsupported target")
    models = artifact.get("models")
    if not isinstance(models, (list, tuple)) or not models:
        raise ContractError("rank HGB artifact contains no ensemble members")
    feature_names = artifact.get("feature_names")
    if not isinstance(feature_names, (list, tuple)) or not feature_names:
        raise ContractError("rank HGB artifact contains no feature schema")
    if any("reference" in str(name).lower() for name in feature_names):
        raise ContractError("rank HGB runtime feature schema contains hidden inputs")
    uncertainty = artifact.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise ContractError("rank HGB uncertainty metadata is required")
    q90 = float(uncertainty.get("absolute_residual_q90", -1.0))
    if q90 < 0 or not math.isfinite(q90):
        raise ContractError("rank HGB validation uncertainty must be nonnegative")


def rank_hgb_target_output_metric(target: str) -> str:
    pass

    try:
        return RANK_HGB_TARGET_OUTPUT_METRICS[target]
    except KeyError as error:
        raise ContractError(f"unsupported rank HGB target: {target!r}") from error
