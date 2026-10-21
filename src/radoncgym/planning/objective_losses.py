

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any

from radoncgym.core.constants import OBJECTIVE_NAMES
from radoncgym.core.errors import ContractError
from radoncgym.core.models import ObjectiveWeights
from radoncgym.planning.objectives import objective_name_to_control


@dataclass(frozen=True, slots=True)
class ObjectiveLossRow:
    objective_name: str
    control_name: str
    input_objective: float
    optimized_objective: float
    multiplier: float
    positive_excess: float
    below_input_improvement: float
    weighted_contribution: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_name": self.objective_name,
            "control_name": self.control_name,
            "input_objective": self.input_objective,
            "optimized_objective": self.optimized_objective,
            "multiplier": self.multiplier,
            "positive_excess": self.positive_excess,
            "below_input_improvement": self.below_input_improvement,
            "weighted_contribution": self.weighted_contribution,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveLossSummary:
    rows: tuple[ObjectiveLossRow, ...]
    positive_excess_by_control: tuple[tuple[str, float], ...]
    weighted_contribution_by_control: tuple[tuple[str, float], ...]
    reconstructed_total: float
    reconstructed_mean: float
    saved_gap_value: float | None
    absolute_gap_error: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantics": "absolute_mean_one_sided_predicted_dose_mimic",
            "rows": [row.to_dict() for row in self.rows],
            "positive_excess_by_control": dict(self.positive_excess_by_control),
            "weighted_contribution_by_control": dict(
                self.weighted_contribution_by_control
            ),
            "reconstructed_total": self.reconstructed_total,
            "reconstructed_mean": self.reconstructed_mean,
            "saved_gap_value": self.saved_gap_value,
            "absolute_gap_error": self.absolute_gap_error,
        }


def reconstruct_absolute_mean_losses(
    objective_artifact: str | Path,
    weights: ObjectiveWeights,
    *,
    gap_artifact: str | Path | None = None,
) -> ObjectiveLossSummary:
    pass

    path = Path(objective_artifact)
    if not path.is_file():
        raise FileNotFoundError(path)
    weight_map = weights.to_dict()
    rows: list[ObjectiveLossRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for index, raw in enumerate(csv.reader(handle), start=1):
            if len(raw) != 4:
                raise ContractError(
                    f"objective artifact row {index} must contain four fields"
                )
            name, _dual_value, optimized_raw, input_raw = raw
            control = objective_name_to_control(name)
            if control is None or control not in weight_map:
                raise ContractError(f"unmapped objective name: {name}")
            try:
                optimized = float(optimized_raw)
                input_value = float(input_raw)
            except ValueError as error:
                raise ContractError(
                    f"objective artifact row {index} contains nonnumeric values"
                ) from error
            multiplier = float(weight_map[control])
            positive = max(optimized - input_value, 0.0)
            improvement = max(input_value - optimized, 0.0)
            contribution = multiplier * (positive - 1e-5 * improvement)
            rows.append(
                ObjectiveLossRow(
                    objective_name=name,
                    control_name=control,
                    input_objective=input_value,
                    optimized_objective=optimized,
                    multiplier=multiplier,
                    positive_excess=positive,
                    below_input_improvement=improvement,
                    weighted_contribution=contribution,
                )
            )
    if not rows:
        raise ContractError("objective artifact contains no rows")
    positive_by_control = {name: 0.0 for name in OBJECTIVE_NAMES}
    contribution_by_control = {name: 0.0 for name in OBJECTIVE_NAMES}
    for row in rows:
        positive_by_control[row.control_name] += row.positive_excess
        contribution_by_control[row.control_name] += row.weighted_contribution
    total = sum(row.weighted_contribution for row in rows)
    mean = total / len(rows)
    saved_gap = _load_gap(gap_artifact) if gap_artifact is not None else None
    return ObjectiveLossSummary(
        rows=tuple(rows),
        positive_excess_by_control=tuple(positive_by_control.items()),
        weighted_contribution_by_control=tuple(contribution_by_control.items()),
        reconstructed_total=total,
        reconstructed_mean=mean,
        saved_gap_value=saved_gap,
        absolute_gap_error=None if saved_gap is None else abs(mean - saved_gap),
    )


def objective_artifacts_from_dose(dose_artifact: str | Path) -> tuple[Path, Path]:
    dose = Path(dose_artifact)
    text = str(dose)
    if "/plan-dose/" not in text:
        raise ContractError("dose artifact is not inside a plan-dose directory")
    return (
        Path(text.replace("/plan-dose/", "/plan-weights/", 1)),
        Path(text.replace("/plan-dose/", "/plan-gap/", 1)),
    )


def _load_gap(path_value: str | Path) -> float:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or len(rows[0]) != 2 or rows[0][0] != "gap value":
        raise ContractError("gap artifact has an invalid first row")
    try:
        return float(rows[0][1])
    except ValueError as error:
        raise ContractError("gap artifact value is not numeric") from error
