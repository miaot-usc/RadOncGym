

from __future__ import annotations

from collections import deque
from importlib.metadata import PackageNotFoundError, version
import math
from typing import Any

import numpy as np

from radoncgym.agents.baselines import FullCallStrategy, _requires_full
from radoncgym.core.constants import OBJECTIVE_NAMES
from radoncgym.core.errors import ContractError
from radoncgym.core.metric_schema import clinical_improvement, returned_clinical_improvement
from radoncgym.core.models import CandidateAction, ObjectiveWeights, Observation
from radoncgym.trajectories.schema import PolicyProvenance


class CMAESAgent:
    pass

    def __init__(
        self,
        *,
        population_size: int = 11,
        sigma: float = 0.45,
        objective_source: str = "returned",
        full_call_strategy: FullCallStrategy = FullCallStrategy.FRONTLOAD,
        include_all_ones: bool = False,
        failure_penalty: float = 1_000_000.0,
    ) -> None:
        if population_size < 2:
            raise ContractError("CMA-ES population_size must be at least two")
        if not 0.0 < sigma <= 2.0 or not math.isfinite(sigma):
            raise ContractError("CMA-ES sigma must be finite and lie in (0, 2]")
        if objective_source not in {"cheap", "full", "returned"}:
            raise ContractError("CMA-ES objective_source must be cheap, full, or returned")
        if failure_penalty <= 0 or not math.isfinite(failure_penalty):
            raise ContractError("CMA-ES failure penalty must be finite and positive")
        self.population_size = population_size
        self.sigma = sigma
        self.objective_source = objective_source
        self.full_call_strategy = FullCallStrategy(full_call_strategy)
        if self.full_call_strategy is FullCallStrategy.ADAPTIVE_PROMOTION:
            raise ContractError("CMA-ES does not support adaptive_promotion")
        self.include_all_ones = include_all_ones
        self.failure_penalty = failure_penalty
        self._optimizer: Any | None = None
        self._candidate_queue: deque[np.ndarray] = deque()
        self._step_to_point: dict[int, np.ndarray] = {}
        self._generation_points: list[np.ndarray] = []
        self._generation_values: list[float] = []
        self._processed_history = 0

    @property
    def provenance(self) -> PolicyProvenance:
        try:
            package_version = version("cma")
        except PackageNotFoundError:
            package_version = "missing"
        return PolicyProvenance(
            family="evolutionary_search",
            name=(
                f"pycma:source={self.objective_source}:pop={self.population_size}:"
                f"sigma={self.sigma:g}:full={self.full_call_strategy.value}:"
                f"all_ones_anchor={str(self.include_all_ones).lower()}:"
                f"failure_penalty={self.failure_penalty:g}"
            ),
            version=f"adapter-1.2.0+pycma-{package_version}",
        )

    def reset(self, observation: Observation, *, seed: int) -> None:
        if observation.history:
            raise ContractError("agent reset requires an initial observation")
        if (
            self.objective_source == "full"
            and not _requires_full(observation)
            and self.full_call_strategy is FullCallStrategy.NEVER
        ):
            raise ContractError(
                "full-objective CMA-ES cannot use a never-full strategy"
            )
        self._optimizer = _make_optimizer(
            population_size=self.population_size,
            sigma=self.sigma,
            seed=seed,
        )
        self._candidate_queue.clear()
        self._step_to_point.clear()
        self._generation_points.clear()
        self._generation_values.clear()
        self._processed_history = 0





        self._candidate_queue.extend(
            np.asarray(point, dtype=float)
            for point in self._optimizer.ask(self.population_size)
        )

    def act(self, observation: Observation) -> CandidateAction:
        if self._optimizer is None:
            raise ContractError("agent reset must be called before act")
        self._ingest_new_outcomes(observation)
        if self._is_verification_phase(observation):
            candidate = self._best_unverified_cheap_candidate(observation)
            if candidate is not None:
                self._step_to_point[observation.budget_state.steps_used] = (
                    np.log(np.asarray(candidate.values)) / math.log(4.0)
                )
                return CandidateAction(
                    weights=candidate,
                    request_full_verifier=True,
                )

        step = observation.budget_state.steps_used
        if self.include_all_ones and not observation.history:
            return CandidateAction(
                weights=ObjectiveWeights.all_ones(),
                request_full_verifier=self._request_full(observation),
            )
        if not self._candidate_queue:
            points = self._optimizer.ask(self.population_size)
            self._candidate_queue.extend(np.asarray(point, dtype=float) for point in points)
        point = self._candidate_queue.popleft()
        self._step_to_point[step] = point.copy()
        return CandidateAction(
            weights=_point_to_weights(point),
            request_full_verifier=self._request_full(observation),
        )

    def _ingest_new_outcomes(self, observation: Observation) -> None:
        for entry in observation.history[self._processed_history :]:
            point = self._step_to_point.pop(entry.step_index, None)
            if point is None:
                continue
            value = self.failure_penalty
            if entry.accepted and entry.outcome is not None:
                if self.objective_source == "returned":
                    metric = returned_clinical_improvement(entry.outcome)
                elif self.objective_source == "cheap":
                    metric = clinical_improvement(entry.outcome.cheap)
                elif entry.outcome.full is not None:
                    metric = clinical_improvement(entry.outcome.full)
                else:
                    metric = None
                if metric is not None and math.isfinite(metric):
                    value = -float(metric)
            self._generation_points.append(point)
            self._generation_values.append(value)
        self._processed_history = len(observation.history)
        if len(self._generation_points) == self.population_size:
            if self._optimizer is None:
                raise AssertionError("CMA-ES optimizer disappeared during episode")
            self._optimizer.tell(
                self._generation_points,
                self._generation_values,
            )
            self._generation_points.clear()
            self._generation_values.clear()
        elif len(self._generation_points) > self.population_size:
            raise ContractError("CMA-ES generation exceeded its declared population")

    def _request_full(self, observation: Observation) -> bool:
        if _requires_full(observation):
            return True
        if self.full_call_strategy is FullCallStrategy.NEVER:
            return False
        if self.full_call_strategy is FullCallStrategy.FRONTLOAD:
            return (
                observation.budget_state.full_calls_attempted
                < observation.budget_config.max_full_calls
            )
        return self._is_verification_phase(observation)

    def _is_verification_phase(self, observation: Observation) -> bool:
        if _requires_full(observation):
            return False
        if self.full_call_strategy is not FullCallStrategy.CHEAP_THEN_VERIFY:
            return False
        verification_steps = min(
            observation.budget_config.max_full_calls,
            observation.budget_config.max_steps,
        )
        return (
            observation.budget_state.steps_used
            >= observation.budget_config.max_steps - verification_steps
        )

    @staticmethod
    def _best_unverified_cheap_candidate(
        observation: Observation,
    ) -> ObjectiveWeights | None:
        verified = {ObjectiveWeights.all_ones()}
        verified.update(
            entry.outcome.executed_weights
            for entry in observation.history
            if entry.accepted
            and entry.outcome is not None
            and entry.outcome.full is not None
        )
        ranked: list[tuple[float, int, ObjectiveWeights]] = []
        for entry in observation.history:
            if not entry.accepted or entry.outcome is None:
                continue
            weights = entry.outcome.executed_weights
            improvement = clinical_improvement(entry.outcome.cheap)
            if weights not in verified and improvement is not None:
                ranked.append((improvement, -entry.step_index, weights))
        if not ranked:
            return None
        return max(ranked, key=lambda item: (item[0], item[1]))[2]


def _point_to_weights(point: np.ndarray) -> ObjectiveWeights:
    clipped = np.clip(np.asarray(point, dtype=float), -1.0, 1.0)







    values = np.round(np.power(4.0, clipped), decimals=8)
    return ObjectiveWeights(tuple(float(value) for value in values))


def _make_optimizer(*, population_size: int, sigma: float, seed: int) -> Any:
    pass

    try:
        import cma
    except ImportError as error:
        raise ContractError(
            "CMA-ES requires the optional 'optimize' dependency"
        ) from error
    random_state = np.random.RandomState(int(seed))
    return cma.CMAEvolutionStrategy(
        np.zeros(len(OBJECTIVE_NAMES), dtype=float),
        sigma,
        {




            "CMA_eigenmethod": cma.utilities.math.eig,
            "bounds": [-1.0, 1.0],
            "popsize": population_size,
            "randn": random_state.randn,


            "seed": np.nan,
            "verbose": -9,
        },
    )
