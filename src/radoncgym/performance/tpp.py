

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping, Sequence

from radoncgym.core.constants import OBJECTIVE_NAMES, PRIMARY_FORMULATION
from radoncgym.core.errors import ContractError


TPP_PROFILE_SCHEMA_VERSION = "training-tpp-distribution-profile-0.1.0"


def load_training_patient_ids(manifest_path: str | Path) -> frozenset[str]:
    pass

    value = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    try:
        patients = value["splits"]["train"]
    except (KeyError, TypeError) as error:
        raise ContractError("manifest is missing splits.train") from error
    if not isinstance(patients, list) or not patients:
        raise ContractError("manifest training split must be a nonempty list")
    result = frozenset(str(patient) for patient in patients)
    if len(result) != len(patients):
        raise ContractError("manifest training split contains duplicate patients")
    return result


def build_tpp_distribution_profile(
    records: Iterable[Mapping[str, Any]],
    *,
    train_patient_ids: frozenset[str],
    source_sha256: str,
    formulation: str = PRIMARY_FORMULATION,
    minimum_candidates_per_context: int = 12,
    minimum_abs_correlation: float = 0.20,
    minimum_sign_consistency: float = 0.75,
    minimum_association_contexts: int = 10,
) -> dict[str, Any]:
    pass

    if formulation != PRIMARY_FORMULATION:
        raise ContractError(
            f"TPP profile currently supports only {PRIMARY_FORMULATION!r}"
        )
    if minimum_candidates_per_context < 2:
        raise ContractError("minimum candidates per context must be at least two")
    if not 0.0 <= minimum_abs_correlation <= 1.0:
        raise ContractError("minimum absolute correlation must be in [0, 1]")
    if not 0.5 <= minimum_sign_consistency <= 1.0:
        raise ContractError("minimum sign consistency must be in [0.5, 1]")
    if minimum_association_contexts < 2:
        raise ContractError("minimum association contexts must be at least two")
    if len(source_sha256) != 64:
        raise ContractError("source_sha256 must be a SHA-256 hex digest")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    eligible_records = 0
    for source in records:
        if source.get("plan_family") != formulation:
            continue
        if int(source.get("solve_status", -1)) != 0:
            continue
        patient_id = str(source.get("patient_id", ""))
        if patient_id not in train_patient_ids:
            raise ContractError(
                "TPP profile source contains a non-training patient: "
                f"{patient_id}"
            )
        prediction_set_id = str(source.get("prediction_set", ""))
        if not patient_id or not prediction_set_id:
            raise ContractError("TPP source record lacks context identity")
        grouped[(patient_id, prediction_set_id)].append(
            _whitelist_generated_plan_record(source)
        )
        eligible_records += 1
    if not grouped:
        raise ContractError("TPP source has no eligible generated plans")

    dense_contexts = {
        context: rows
        for context, rows in grouped.items()
        if len(rows) >= minimum_candidates_per_context
    }
    if not dense_contexts:
        raise ContractError("no TPP source context meets the density threshold")



    selected = [
        min(
            rows,
            key=lambda row: (
                row["clinical_score"],
                row["source_record_id"],
            ),
        )
        for _, rows in sorted(dense_contexts.items())
    ]
    distributions = []
    for name in OBJECTIVE_NAMES:
        values = sorted(float(row["weights"][name]) for row in selected)
        distributions.append(
            {
                "objective": name,
                "minimum": values[0],
                "q10": _quantile(values, 0.10),
                "q25": _quantile(values, 0.25),
                "median": _quantile(values, 0.50),
                "q75": _quantile(values, 0.75),
                "q90": _quantile(values, 0.90),
                "maximum": values[-1],
            }
        )

    associations = _stable_associations(
        dense_contexts,
        minimum_abs_correlation=minimum_abs_correlation,
        minimum_sign_consistency=minimum_sign_consistency,
        minimum_contexts=minimum_association_contexts,
    )
    return {
        "schema_version": TPP_PROFILE_SCHEMA_VERSION,
        "profile_id": "openkbp-train-generated-plan-tpp-distribution-v1",
        "information_boundary": "training_generated_plan_only",
        "planning_formulation": formulation,
        "source": {
            "sha256": source_sha256,
            "solver_scope": "downsampled_GLOP_generated_plans",
            "contains_reference_metrics": False,
            "selection_metric": "generated_plan_clinical_score_lower_is_better",
        },
        "coverage": {
            "eligible_records": eligible_records,
            "eligible_contexts": len(grouped),
            "dense_contexts": len(dense_contexts),
            "distinct_training_patients": len(
                {patient_id for patient_id, _ in dense_contexts}
            ),
            "selected_vectors": len(selected),
            "minimum_candidates_per_context": minimum_candidates_per_context,
        },
        "selection": {
            "rule": "one_lowest_generated_clinical_score_vector_per_context",
            "context_definition": "patient_id_and_prediction_set",
            "patient_vectors_serialized": False,
        },
        "parameter_distributions": distributions,
        "stable_generated_plan_associations": associations,
        "association_interpretation": (
            "Negative median Spearman means increasing the objective weight was "
            "associated with a lower generated-plan protocol violation. These "
            "joint randomized-plan associations are not isolated causal effects."
        ),
        "association_filter": {
            "minimum_absolute_median_spearman": minimum_abs_correlation,
            "minimum_sign_consistency": minimum_sign_consistency,
            "minimum_contexts": minimum_association_contexts,
        },
        "usage": {
            "role": "soft_empirical_prior_not_action_constraint",
            "extrapolation": (
                "Prefer the empirical q10-q90 region initially; extrapolate only "
                "when current-patient cheap or full response evidence supports it."
            ),
        },
    }


def read_jsonl_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    pass

    result = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContractError(
                    f"TPP source line {line_number} must be a JSON object"
                )
            result.append(value)
    if not result:
        raise ContractError("TPP source JSONL is empty")
    return tuple(result)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tpp_distribution_profile(profile: Mapping[str, Any]) -> None:
    pass

    if profile.get("schema_version") != TPP_PROFILE_SCHEMA_VERSION:
        raise ContractError("unsupported training TPP distribution profile")
    if profile.get("information_boundary") != "training_generated_plan_only":
        raise ContractError("TPP profile has an unsafe information boundary")
    if profile.get("planning_formulation") != PRIMARY_FORMULATION:
        raise ContractError("TPP profile planning formulation is incompatible")
    source = profile.get("source")
    if not isinstance(source, Mapping) or source.get("contains_reference_metrics") is not False:
        raise ContractError("TPP profile does not prove reference exclusion")
    distributions = profile.get("parameter_distributions")
    if not isinstance(distributions, list):
        raise ContractError("TPP profile parameter distributions must be a list")
    names = [
        str(value.get("objective"))
        for value in distributions
        if isinstance(value, Mapping)
    ]
    if names != list(OBJECTIVE_NAMES):
        raise ContractError("TPP profile objective order or coverage is invalid")
    forbidden = ("reference", "hidden_m1", "dvh_score", "dose_score")
    for key in _walk_keys(profile):
        if any(token in key.lower() for token in forbidden):
            if key == "contains_reference_metrics":
                continue
            raise ContractError(f"TPP profile contains forbidden field: {key}")


def _whitelist_generated_plan_record(source: Mapping[str, Any]) -> dict[str, Any]:
    weights = source.get("objective_weights")
    violations = source.get("criteria_violations")
    if not isinstance(weights, Mapping) or not isinstance(violations, Mapping):
        raise ContractError("TPP source lacks weights or generated-plan violations")
    if set(weights) != set(OBJECTIVE_NAMES):
        raise ContractError("TPP source objective weights are incomplete")
    return {
        "source_record_id": str(source.get("cache_key", "")),
        "weights": {name: float(weights[name]) for name in OBJECTIVE_NAMES},
        "clinical_score": float(source["clinical_score"]),
        "criteria_violations": {
            str(name): float(value) for name, value in violations.items()
        },
    }


def _stable_associations(
    contexts: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    *,
    minimum_abs_correlation: float,
    minimum_sign_consistency: float,
    minimum_contexts: int,
) -> list[dict[str, Any]]:
    criteria = sorted(
        {
            criterion
            for rows in contexts.values()
            for row in rows
            for criterion in row["criteria_violations"]
        }
    )
    result = []
    for criterion in criteria:
        for objective in OBJECTIVE_NAMES:
            correlations = []
            for rows in contexts.values():
                paired = [
                    row
                    for row in rows
                    if criterion in row["criteria_violations"]
                ]
                if len(paired) < 6:
                    continue
                x = [float(row["weights"][objective]) for row in paired]
                y = [float(row["criteria_violations"][criterion]) for row in paired]
                correlation = _spearman(x, y)
                if correlation is not None:
                    correlations.append(correlation)
            if len(correlations) < minimum_contexts:
                continue
            center = round(float(median(correlations)), 6)
            if abs(center) < minimum_abs_correlation:
                continue
            direction = -1.0 if center < 0.0 else 1.0
            consistency = round(fmean(
                1.0 if value * direction > 0.0 else 0.0
                for value in correlations
            ), 6)
            if consistency < minimum_sign_consistency:
                continue
            result.append(
                {
                    "criterion": criterion,
                    "objective": objective,
                    "median_spearman": center,
                    "sign_consistency": consistency,
                    "context_count": len(correlations),
                    "higher_weight_associated_with": (
                        "lower_violation" if center < 0.0 else "higher_violation"
                    ),
                }
            )
    result.sort(
        key=lambda value: (
            value["criterion"],
            -abs(float(value["median_spearman"])),
            value["objective"],
        )
    )
    return result


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    return _pearson(_average_ranks(x), _average_ranks(y))


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_mean = fmean(x)
    y_mean = fmean(y)
    x_centered = [value - x_mean for value in x]
    y_centered = [value - y_mean for value in y]
    denominator = math.sqrt(
        math.fsum(value * value for value in x_centered)
        * math.fsum(value * value for value in y_centered)
    )
    if denominator <= 0.0:
        return None
    return math.fsum(
        left * right for left, right in zip(x_centered, y_centered)
    ) / denominator


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ContractError("cannot calculate an empty quantile")
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(float(values[lower]), 6)
    fraction = position - lower
    return round(
        float(values[lower] * (1.0 - fraction) + values[upper] * fraction),
        6,
    )


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)
