

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Any

from radoncgym.core.constants import OBJECTIVE_NAMES
from radoncgym.core.errors import ContractError
from radoncgym.core.models import ObjectiveWeights
from radoncgym.planning.objectives import objective_name_to_control


MEAN_ABSOLUTE_EPSILON = 1e-5


@dataclass(frozen=True, slots=True)
class InversePlanningDualRow:
    objective_name: str
    control_name: str
    raw_dual: float
    inverse_weight: float
    lower_bound: float
    upper_bound: float
    upper_bound_fraction: float
    materially_active: bool
    bound_violation: float
    bound_violation_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_name": self.objective_name,
            "control_name": self.control_name,
            "raw_dual": self.raw_dual,
            "inverse_weight": self.inverse_weight,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "upper_bound_fraction": self.upper_bound_fraction,
            "materially_active": self.materially_active,
            "bound_violation": self.bound_violation,
            "bound_violation_fraction": self.bound_violation_fraction,
        }


@dataclass(frozen=True, slots=True)
class InversePlanningDualSummary:
    rows: tuple[InversePlanningDualRow, ...]
    inverse_weight_by_control: tuple[tuple[str, float], ...]
    normalized_share_by_control: tuple[tuple[str, float], ...]
    active_rows_by_control: tuple[tuple[str, int], ...]
    total_inverse_weight: float
    entropy: float
    dominant_control: str
    materially_active_rows: int
    materially_active_controls: int
    maximum_bound_violation: float
    maximum_bound_violation_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantics": "equivalent_conventional_inverse_planning_dual_weights",
            "rows": [row.to_dict() for row in self.rows],
            "inverse_weight_by_control": dict(self.inverse_weight_by_control),
            "normalized_share_by_control": dict(self.normalized_share_by_control),
            "active_rows_by_control": dict(self.active_rows_by_control),
            "total_inverse_weight": self.total_inverse_weight,
            "entropy": self.entropy,
            "dominant_control": self.dominant_control,
            "materially_active_rows": self.materially_active_rows,
            "materially_active_controls": self.materially_active_controls,
            "maximum_bound_violation": self.maximum_bound_violation,
            "maximum_bound_violation_fraction": (
                self.maximum_bound_violation_fraction
            ),
        }


def interpret_absolute_mean_duals(
    objective_artifact: str | Path,
    weights: ObjectiveWeights,
    *,
    activity_multiple: float = 10.0,
) -> InversePlanningDualSummary:
    pass

    if activity_multiple <= 1:
        raise ContractError("dual activity multiple must be greater than one")
    path = Path(objective_artifact)
    if not path.is_file():
        raise FileNotFoundError(path)
    multiplier_by_control = weights.to_dict()
    rows: list[InversePlanningDualRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for index, raw in enumerate(csv.reader(handle), start=1):
            if len(raw) != 4:
                raise ContractError(
                    f"objective artifact row {index} must contain four fields"
                )
            name, dual_raw, _optimized, _input = raw
            control = objective_name_to_control(name)
            if control is None or control not in multiplier_by_control:
                raise ContractError(f"unmapped objective name: {name}")
            try:
                dual = float(dual_raw)
            except ValueError as error:
                raise ContractError(
                    f"objective artifact row {index} has a nonnumeric dual"
                ) from error
            upper = float(multiplier_by_control[control])
            lower = MEAN_ABSOLUTE_EPSILON * upper
            interpreted = min(max(dual, lower), upper)
            violation = max(lower - dual, dual - upper, 0.0)
            rows.append(
                InversePlanningDualRow(
                    objective_name=name,
                    control_name=control,
                    raw_dual=dual,
                    inverse_weight=interpreted,
                    lower_bound=lower,
                    upper_bound=upper,
                    upper_bound_fraction=interpreted / upper,
                    materially_active=interpreted > activity_multiple * lower,
                    bound_violation=violation,
                    bound_violation_fraction=violation / upper,
                )
            )
    if not rows:
        raise ContractError("objective artifact contains no rows")

    by_control = {name: 0.0 for name in OBJECTIVE_NAMES}
    active_by_control = {name: 0 for name in OBJECTIVE_NAMES}
    for row in rows:
        by_control[row.control_name] += row.inverse_weight
        active_by_control[row.control_name] += int(row.materially_active)
    total = sum(by_control.values())
    if total <= 0:
        raise ContractError("inverse-planning dual weight total must be positive")
    share = {name: value / total for name, value in by_control.items()}
    entropy = -sum(value * math.log(value) for value in share.values() if value > 0)
    dominant = max(share, key=share.__getitem__)
    return InversePlanningDualSummary(
        rows=tuple(rows),
        inverse_weight_by_control=tuple(by_control.items()),
        normalized_share_by_control=tuple(share.items()),
        active_rows_by_control=tuple(active_by_control.items()),
        total_inverse_weight=total,
        entropy=entropy,
        dominant_control=dominant,
        materially_active_rows=sum(row.materially_active for row in rows),
        materially_active_controls=sum(value > 0 for value in active_by_control.values()),
        maximum_bound_violation=max(row.bound_violation for row in rows),
        maximum_bound_violation_fraction=max(
            row.bound_violation_fraction for row in rows
        ),
    )
