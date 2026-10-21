

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Iterable, Literal

from radoncgym.metrics.episode import EpisodeSummary


MatchKey = tuple[str, str, str, int, int, int | None, int]
ContextFullBudgetKey = tuple[str, str, int, int, int]
PairingScope = Literal["exact", "context_full_budget"]


def _match_key(summary: EpisodeSummary) -> MatchKey:
    return (
        summary.patient_id,
        summary.prediction_set_id,
        summary.task_id,
        summary.max_steps,
        summary.max_full_calls,
        summary.max_cheap_calls,
        summary.seed,
    )


@dataclass(frozen=True, slots=True)
class PairedEpisodeDelta:
    pass

    patient_id: str
    prediction_set_id: str
    task_id: str
    max_steps: int
    max_full_calls: int
    seed: int
    baseline_episode_id: str
    candidate_episode_id: str
    objective_improvement: float
    constraint_improvement: float | None
    pairing_scope: PairingScope = "exact"
    baseline_task_id: str | None = None
    candidate_task_id: str | None = None
    baseline_max_steps: int | None = None
    candidate_max_steps: int | None = None
    baseline_max_cheap_calls: int | None = None
    candidate_max_cheap_calls: int | None = None
    baseline_full_calls_attempted: int | None = None
    candidate_full_calls_attempted: int | None = None

    @property
    def candidate_wins(self) -> bool:
        return self.objective_improvement > 0.0

    @property
    def candidate_harms(self) -> bool:
        return self.objective_improvement < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        } | {
            "candidate_wins": self.candidate_wins,
            "candidate_harms": self.candidate_harms,
        }


def pair_episode_summaries(
    *,
    baseline: Iterable[EpisodeSummary],
    candidate: Iterable[EpisodeSummary],
    pairing_scope: PairingScope = "exact",
) -> tuple[PairedEpisodeDelta, ...]:
    pass

    if pairing_scope not in {"exact", "context_full_budget"}:
        raise ValueError(f"unsupported pairing scope: {pairing_scope}")
    baseline_index = _unique_index(baseline, "baseline", pairing_scope)
    candidate_index = _unique_index(candidate, "candidate", pairing_scope)
    if baseline_index.keys() != candidate_index.keys():
        missing_candidate = sorted(baseline_index.keys() - candidate_index.keys())
        missing_baseline = sorted(candidate_index.keys() - baseline_index.keys())
        raise ValueError(
            "paired comparison keys differ: "
            f"missing_candidate={missing_candidate}, "
            f"missing_baseline={missing_baseline}"
        )

    pairs = []
    for key in sorted(baseline_index):
        baseline_item = baseline_index[key]
        candidate_item = candidate_index[key]
        if pairing_scope == "context_full_budget" and (
            baseline_item.full_calls_attempted
            != candidate_item.full_calls_attempted
        ):
            raise ValueError(
                "context_full_budget pairing requires equal consumed full calls"
            )
        if baseline_item.best_full_objective is None:
            raise ValueError(f"baseline has no solved full objective for {key}")
        if candidate_item.best_full_objective is None:
            raise ValueError(f"candidate has no solved full objective for {key}")
        baseline_constraint = baseline_item.best_full_constraint_sum
        candidate_constraint = candidate_item.best_full_constraint_sum
        pairs.append(
            PairedEpisodeDelta(
                patient_id=baseline_item.patient_id,
                prediction_set_id=baseline_item.prediction_set_id,
                task_id=(
                    baseline_item.task_id
                    if pairing_scope == "exact"
                    else "cross_task_equal_full_budget"
                ),
                max_steps=candidate_item.max_steps,
                max_full_calls=baseline_item.max_full_calls,
                seed=baseline_item.seed,
                baseline_episode_id=baseline_item.episode_id,
                candidate_episode_id=candidate_item.episode_id,
                objective_improvement=(
                    baseline_item.best_full_objective
                    - candidate_item.best_full_objective
                ),
                constraint_improvement=(
                    None
                    if baseline_constraint is None or candidate_constraint is None
                    else baseline_constraint - candidate_constraint
                ),
                pairing_scope=pairing_scope,
                baseline_task_id=baseline_item.task_id,
                candidate_task_id=candidate_item.task_id,
                baseline_max_steps=baseline_item.max_steps,
                candidate_max_steps=candidate_item.max_steps,
                baseline_max_cheap_calls=baseline_item.max_cheap_calls,
                candidate_max_cheap_calls=candidate_item.max_cheap_calls,
                baseline_full_calls_attempted=baseline_item.full_calls_attempted,
                candidate_full_calls_attempted=candidate_item.full_calls_attempted,
            )
        )
    return tuple(pairs)


def aggregate_paired_deltas(
    deltas: Iterable[PairedEpisodeDelta],
) -> dict[str, Any]:
    pass

    values = tuple(deltas)
    if not values:
        raise ValueError("at least one paired delta is required")
    improvements = [value.objective_improvement for value in values]
    by_patient: dict[str, list[float]] = {}
    for value in values:
        by_patient.setdefault(value.patient_id, []).append(
            value.objective_improvement
        )
    patient_means = {
        patient: mean(patient_values)
        for patient, patient_values in sorted(by_patient.items())
    }
    return {
        "paired_contexts": len(values),
        "patients": len(patient_means),
        "mean_context_improvement": mean(improvements),
        "median_context_improvement": median(improvements),
        "context_win_rate": sum(value > 0 for value in improvements) / len(values),
        "context_harm_rate": sum(value < 0 for value in improvements) / len(values),
        "worst_context_improvement": min(improvements),
        "mean_patient_cluster_improvement": mean(patient_means.values()),
        "median_patient_cluster_improvement": median(patient_means.values()),
        "patient_cluster_win_rate": (
            sum(value > 0 for value in patient_means.values()) / len(patient_means)
        ),
        "patient_cluster_harm_rate": (
            sum(value < 0 for value in patient_means.values()) / len(patient_means)
        ),
        "patient_cluster_means": patient_means,
    }


def _unique_index(
    summaries: Iterable[EpisodeSummary],
    name: str,
    pairing_scope: PairingScope,
) -> dict[MatchKey | ContextFullBudgetKey, EpisodeSummary]:
    index: dict[MatchKey | ContextFullBudgetKey, EpisodeSummary] = {}
    for summary in summaries:
        key = (
            _match_key(summary)
            if pairing_scope == "exact"
            else _context_full_budget_key(summary)
        )
        if key in index:
            raise ValueError(f"duplicate {name} comparison key: {key}")
        index[key] = summary
    if not index:
        raise ValueError(f"{name} summaries are empty")
    return index


def _context_full_budget_key(summary: EpisodeSummary) -> ContextFullBudgetKey:
    return (
        summary.patient_id,
        summary.prediction_set_id,
        summary.max_full_calls,
        summary.full_calls_attempted,
        summary.seed,
    )
