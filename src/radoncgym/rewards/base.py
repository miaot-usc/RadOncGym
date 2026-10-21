

from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol

from radoncgym.core.models import Observation, StepOutcome


class RewardFunction(Protocol):
    pass

    version: str

    def __call__(
        self,
        before: Observation,
        outcome: StepOutcome | None,
        accepted: bool,
    ) -> float:
        pass


class VisibleMetricSource(str, Enum):
    pass

    CHEAP = "cheap"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class ZeroReward:
    pass

    version: str = "ZeroReward-v0"

    def __call__(
        self,
        before: Observation,
        outcome: StepOutcome | None,
        accepted: bool,
    ) -> float:
        del before, outcome, accepted
        return 0.0


@dataclass(frozen=True, slots=True)
class NegativeVisibleMetricReward:
    pass

    metric_name: str = "objective"

    @property
    def version(self) -> str:
        return f"NegativeVisibleMetricReward-v0:{self.metric_name}"

    def __call__(
        self,
        before: Observation,
        outcome: StepOutcome | None,
        accepted: bool,
    ) -> float:
        del before
        if not accepted or outcome is None:
            return 0.0
        result = outcome.full if outcome.full is not None else outcome.cheap
        value = result.metrics.get(self.metric_name)
        if value is None:
            return 0.0
        return -float(value)


@dataclass(frozen=True, slots=True)
class VisibleIncumbentImprovementReward:
    pass

    source: VisibleMetricSource = VisibleMetricSource.CHEAP
    metric_name: str = "predicted_clinical_improvement_from_all_ones"
    baseline_value: float | None = 0.0
    invalid_action_penalty: float = 0.0
    higher_is_better: bool = True

    @property
    def version(self) -> str:
        baseline = "none" if self.baseline_value is None else repr(self.baseline_value)
        return ":".join(
            (
                "VisibleIncumbentImprovementReward-v0",
                self.source.value,
                self.metric_name,
                f"baseline={baseline}",
                f"invalid={self.invalid_action_penalty!r}",
                f"higher={str(self.higher_is_better).lower()}",
            )
        )

    def __post_init__(self) -> None:
        if self.baseline_value is not None and not math.isfinite(self.baseline_value):
            raise ValueError("baseline_value must be finite when provided")
        if not math.isfinite(self.invalid_action_penalty):
            raise ValueError("invalid_action_penalty must be finite")
        if not self.metric_name:
            raise ValueError("metric_name must be non-empty")

    def __call__(
        self,
        before: Observation,
        outcome: StepOutcome | None,
        accepted: bool,
    ) -> float:
        if not accepted:
            return float(self.invalid_action_penalty)
        if outcome is None:
            return 0.0
        current_result = (
            outcome.cheap if self.source is VisibleMetricSource.CHEAP else outcome.full
        )
        if current_result is None:
            return 0.0
        current = current_result.metrics.get(self.metric_name)
        if current is None:
            return 0.0

        previous_values: list[float] = []
        for entry in before.history:
            if not entry.accepted or entry.outcome is None:
                continue
            result = (
                entry.outcome.cheap
                if self.source is VisibleMetricSource.CHEAP
                else entry.outcome.full
            )
            if result is None:
                continue
            value = result.metrics.get(self.metric_name)
            if value is not None:
                previous_values.append(float(value))

        if self.baseline_value is not None:
            previous_values.append(float(self.baseline_value))
        if not previous_values:
            return 0.0
        if self.higher_is_better:
            previous_best = max(previous_values)
            next_best = max(previous_best, float(current))
            return next_best - previous_best
        previous_best = min(previous_values)
        next_best = min(previous_best, float(current))
        return previous_best - next_best
