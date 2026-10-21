from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from radoncgym.core.constants import PRIMARY_FORMULATION
from radoncgym.surrogates.rank_hgb import (
    RANK_HGB_ARTIFACT_VERSION,
    RANK_HGB_CLINICAL_TARGET,
    RANK_HGB_FEATURE_VERSION,
)


CONFIGURATIONS = (
    ("squared_error", 0.03, 31, 30, 3.0),
    ("squared_error", 0.05, 15, 30, 3.0),
    ("squared_error", 0.10, 31, 30, 3.0),
    ("absolute_error", 0.03, 31, 30, 3.0),
    ("absolute_error", 0.10, 15, 30, 0.3),
    ("absolute_error", 0.10, 31, 30, 3.0),
)
SEEDS = (17, 29, 43, 71)


def paper_offline_weight_design(*, seed: int, size: int = 16) -> np.ndarray:
    if size < 2:
        raise ValueError("offline design size must be at least two")
    from scipy.stats import qmc

    engine = qmc.Sobol(d=12, scramble=True, seed=seed)
    exponent = int(np.ceil(np.log2(size - 1)))
    unit = engine.random_base2(exponent)[: size - 1]
    candidates = 0.25 * np.power(16.0, unit)
    return np.vstack((np.ones((1, 12), dtype=np.float64), candidates))


def anchored_context_delta_rows(
    x: np.ndarray,
    clinical_scores: np.ndarray,
    context_ids: Sequence[str],
    solved_mask: np.ndarray,
    anchor_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(x)
    scores = np.asarray(clinical_scores, dtype=np.float64)
    solved = np.asarray(solved_mask, dtype=bool)
    anchors = np.asarray(anchor_mask, dtype=bool)
    contexts = np.asarray(tuple(str(item) for item in context_ids), dtype=object)
    row_count = values.shape[0]
    if values.ndim != 2 or any(
        array.shape != (row_count,)
        for array in (scores, solved, anchors, contexts)
    ):
        raise ValueError("anchored rows must have aligned first dimensions")
    retained: list[int] = []
    targets: list[float] = []
    for context_id in dict.fromkeys(contexts.tolist()):
        members = np.flatnonzero(contexts == context_id)
        anchor_rows = members[solved[members] & anchors[members]]
        if anchor_rows.size != 1:
            continue
        anchor_score = float(scores[int(anchor_rows[0])])
        for index in members[solved[members]]:
            retained.append(int(index))
            targets.append(float(scores[int(index)]) - anchor_score)
    indices = np.asarray(retained, dtype=np.int64)
    return values[indices], np.asarray(targets), indices


def fit_verifier(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    sample_weight: np.ndarray,
    feature_names: Sequence[str],
    max_iter: int = 800,
) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor

    values = np.asarray(x, dtype=np.float32)
    targets = np.asarray(y, dtype=np.float64)
    train = np.asarray(train_mask, dtype=bool)
    validation = np.asarray(validation_mask, dtype=bool)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 309:
        raise ValueError("x must have shape (n, 309)")
    if not np.any(train) or not np.any(validation):
        raise ValueError("train and validation partitions must be nonempty")
    models = []
    for loss, learning_rate, leaves, minimum, regularization in CONFIGURATIONS:
        for seed in SEEDS:
            model = HistGradientBoostingRegressor(
                loss=loss,
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_leaf_nodes=leaves,
                min_samples_leaf=minimum,
                l2_regularization=regularization,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=50,
                random_state=seed,
            )
            model.fit(values[train], targets[train], sample_weight=weights[train])
            models.append(model)
    baseline = values[validation].copy()
    baseline[:, :12] = 1.0
    predictions = np.asarray(
        [
            model.predict(values[validation]) - model.predict(baseline)
            for model in models
        ]
    )
    mean_prediction = np.mean(predictions, axis=0)
    q90 = float(
        np.quantile(
            np.abs(mean_prediction - targets[validation]),
            0.9,
            method="higher",
        )
    )
    return {
        "schema_version": RANK_HGB_ARTIFACT_VERSION,
        "feature_schema_version": RANK_HGB_FEATURE_VERSION,
        "feature_names": list(feature_names),
        "planning_formulation": PRIMARY_FORMULATION,
        "target": RANK_HGB_CLINICAL_TARGET,
        "models": models,
        "uncertainty": {
            "method": "ensemble_std_plus_patient_validation_absolute_residual_q90",
            "absolute_residual_q90": q90,
        },
    }
