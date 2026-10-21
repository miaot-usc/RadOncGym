

from __future__ import annotations

import random
from statistics import mean, median
from typing import Any, Iterable

from radoncgym.metrics.comparison import PairedEpisodeDelta


def interquartile_mean(values: Iterable[float]) -> float:
    pass

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("interquartile mean requires at least one value")
    lower = len(ordered) * 0.25
    upper = len(ordered) * 0.75
    total = 0.0
    mass = 0.0
    for index, value in enumerate(ordered):
        overlap = max(0.0, min(index + 1.0, upper) - max(float(index), lower))
        total += overlap * value
        mass += overlap
    return total / mass


def patient_clustered_paired_statistics(
    deltas: Iterable[PairedEpisodeDelta],
    *,
    bootstrap_repetitions: int = 10_000,
    seed: int = 731_991,
) -> dict[str, Any]:
    pass

    values = tuple(deltas)
    if not values:
        raise ValueError("at least one paired delta is required")
    if bootstrap_repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    by_patient: dict[str, list[float]] = {}
    for item in values:
        by_patient.setdefault(item.patient_id, []).append(
            float(item.objective_improvement)
        )
    patient_means = {
        patient_id: mean(improvements)
        for patient_id, improvements in sorted(by_patient.items())
    }
    cluster_values = tuple(patient_means.values())
    rng = random.Random(seed)
    bootstrapped_iqm = []
    for _ in range(bootstrap_repetitions):
        sample = [rng.choice(cluster_values) for _ in cluster_values]
        bootstrapped_iqm.append(interquartile_mean(sample))
    ordered_bootstrap = sorted(bootstrapped_iqm)
    return {
        "paired_contexts": len(values),
        "patient_clusters": len(cluster_values),
        "mean_patient_improvement": mean(cluster_values),
        "median_patient_improvement": median(cluster_values),
        "iqm_patient_improvement": interquartile_mean(cluster_values),
        "patient_win_rate": sum(value > 0.0 for value in cluster_values)
        / len(cluster_values),
        "patient_harm_rate": sum(value < 0.0 for value in cluster_values)
        / len(cluster_values),
        "probability_iqm_improvement": sum(
            value > 0.0 for value in bootstrapped_iqm
        )
        / bootstrap_repetitions,
        "bootstrap_iqm_95_percent_interval": [
            _empirical_quantile(ordered_bootstrap, 0.025),
            _empirical_quantile(ordered_bootstrap, 0.975),
        ],
        "bootstrap_interval_type": "percentile",
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": seed,
        "patient_cluster_means": patient_means,
        "interpretation": (
            "Prediction sets and seeds are averaged within patient before "
            "resampling, so correlated contexts do not become independent patients."
        ),
    }


def patient_clustered_mean_interval(
    values: Iterable[tuple[str, float]],
    *,
    bootstrap_repetitions: int = 10_000,
    seed: int = 20_260_824,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for patient_id, value in values:
        grouped.setdefault(str(patient_id), []).append(float(value))
    if not grouped:
        raise ValueError("at least one patient is required")
    patient_means = tuple(mean(group) for _, group in sorted(grouped.items()))
    rng = random.Random(seed)
    bootstrap = sorted(
        mean(rng.choice(patient_means) for _ in patient_means)
        for _ in range(bootstrap_repetitions)
    )
    return {
        "patient_clusters": len(patient_means),
        "mean": mean(patient_means),
        "ci_low": _empirical_quantile(bootstrap, 0.025),
        "ci_high": _empirical_quantile(bootstrap, 0.975),
        "bootstrap_interval_type": "percentile",
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": seed,
    }


def _empirical_quantile(ordered: list[float], probability: float) -> float:
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
