from __future__ import annotations

from dataclasses import dataclass

from radoncgym.core.models import HistoryEntry, ObjectiveWeights, SolverStatus


@dataclass(frozen=True, slots=True)
class PaperProtocol:
    horizon: int = 16
    full_solve_budget: int = 3
    all_ones_anchor_excluded_from_budget: bool = True

    @property
    def terminal_full_turns(self) -> tuple[int, ...]:
        start = self.horizon - self.full_solve_budget + 1
        return tuple(range(start, self.horizon + 1))


@dataclass(frozen=True, slots=True)
class FinalSelection:
    weights: ObjectiveWeights
    step_index: int | None
    used_all_ones_fallback: bool


def select_final_full_step(history: tuple[HistoryEntry, ...]) -> int | None:
    candidates: list[tuple[float, int]] = []
    for entry in history:
        if (
            not entry.accepted
            or entry.outcome is None
            or entry.outcome.full is None
            or entry.outcome.full.status is not SolverStatus.SOLVED
        ):
            continue
        value = entry.outcome.full.metrics.get("mean_protocol_violation_gy")
        if value is None:
            value = entry.outcome.full.metrics.get("objective")
        if value is not None:
            candidates.append((float(value), entry.step_index))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def select_final_weights(history: tuple[HistoryEntry, ...]) -> FinalSelection:
    step_index = select_final_full_step(history)
    if step_index is None:
        return FinalSelection(
            weights=ObjectiveWeights.all_ones(),
            step_index=None,
            used_all_ones_fallback=True,
        )
    entry = next(item for item in history if item.step_index == step_index)
    if entry.outcome is None:
        raise AssertionError("selected full-solve step has no outcome")
    return FinalSelection(
        weights=entry.outcome.executed_weights,
        step_index=step_index,
        used_all_ones_fallback=False,
    )
