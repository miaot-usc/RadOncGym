

from __future__ import annotations

from radoncgym.core.models import FullIncumbentRule, MetricVector


TARGET_MARKER = "PTV"
SERIAL_OR_MAX_DOSE_STRUCTURES = ("Brainstem", "SpinalCord", "Mandible")


def constraint_priority_group(name: str) -> str:
    pass

    if TARGET_MARKER in name:
        return "target"
    if any(structure in name for structure in SERIAL_OR_MAX_DOSE_STRUCTURES):
        return "serial_or_max_dose_oar"
    return "parallel_or_mean_dose_oar"


def constraint_priority_sums(
    constraints: MetricVector,
) -> tuple[float, float, float]:
    pass

    totals = {
        "target": 0.0,
        "serial_or_max_dose_oar": 0.0,
        "parallel_or_mean_dose_oar": 0.0,
    }
    for name, value in constraints.items:
        totals[constraint_priority_group(name)] += max(0.0, float(value))
    return (
        totals["target"],
        totals["serial_or_max_dose_oar"],
        totals["parallel_or_mean_dose_oar"],
    )


def full_incumbent_key(
    *,
    rule: FullIncumbentRule,
    constraints: MetricVector,
    objective: float,
    step_index: int,
) -> tuple[float, ...]:
    pass

    if rule is FullIncumbentRule.PRIORITY_LEXICOGRAPHIC:
        target, serial, parallel = constraint_priority_sums(constraints)
        return (target, serial, parallel, objective, float(step_index))
    if rule is FullIncumbentRule.FEASIBILITY_THEN_QUALITY:
        return (
            constraints.sum(),
            objective,
            0.0,
            0.0,
            float(step_index),
        )
    return (objective, 0.0, 0.0, 0.0, float(step_index))
