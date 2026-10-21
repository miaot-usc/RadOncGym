

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from pathlib import Path
import string

import numpy as np

from radoncgym.core.constants import MAX_OBJECTIVE_WEIGHT, MIN_OBJECTIVE_WEIGHT
from radoncgym.core.errors import ContractError
from radoncgym.core.metric_schema import clinical_improvement
from radoncgym.core.models import (
    CandidateAction,
    FidelityControl,
    ObjectiveWeights,
    Observation,
    PlanningContext,
)
from radoncgym.trajectories.schema import PolicyProvenance


class FullCallStrategy(str, Enum):
    pass

    FRONTLOAD = "frontload"
    CHEAP_THEN_VERIFY = "cheap_then_verify"
    ADAPTIVE_PROMOTION = "adaptive_promotion"
    NEVER = "never"


def _requires_full(observation: Observation) -> bool:
    return observation.task_config.fidelity_control is FidelityControl.FULL_ONLY


class AllOnesAgent:
    pass

    provenance = PolicyProvenance(
        family="static_control",
        name="all_ones",
        version="1.0.0",
    )

    def reset(self, observation: Observation, *, seed: int) -> None:
        del seed
        if observation.history:
            raise ContractError("agent reset requires an initial observation")

    def act(self, observation: Observation) -> CandidateAction:
        if observation.history:
            return CandidateAction(stop=True)
        request_full = (
            _requires_full(observation)
            or observation.budget_config.max_full_calls > 0
        )
        return CandidateAction(
            weights=ObjectiveWeights.all_ones(),
            request_full_verifier=request_full,
        )


class CandidateReplayAgent:
    pass

    def __init__(
        self,
        candidates: tuple[ObjectiveWeights, ...],
        *,
        manifest_sha256: str,
    ) -> None:
        if not candidates:
            raise ContractError("candidate replay requires at least one candidate")
        if len(set(candidates)) != len(candidates):
            raise ContractError("candidate replay weights must be unique")
        if len(manifest_sha256) != 64 or any(
            character not in string.hexdigits for character in manifest_sha256
        ):
            raise ContractError("candidate replay manifest SHA-256 is invalid")
        self.candidates = candidates
        self.manifest_sha256 = manifest_sha256

    @property
    def provenance(self) -> PolicyProvenance:
        return PolicyProvenance(
            family="fidelity_calibration",
            name=f"candidate_replay:n={len(self.candidates)}",
            version="1.0.0",
            checkpoint_sha256=self.manifest_sha256,
        )

    def reset(self, observation: Observation, *, seed: int) -> None:
        del seed
        if observation.history:
            raise ContractError("agent reset requires an initial observation")
        if len(self.candidates) > observation.budget_config.max_steps:
            raise ContractError("candidate replay exceeds the episode step budget")
        if len(self.candidates) > observation.budget_config.max_full_calls:
            raise ContractError("candidate replay exceeds the episode full-call budget")

    def act(self, observation: Observation) -> CandidateAction:
        index = observation.budget_state.steps_used
        if index >= len(self.candidates):
            return CandidateAction(stop=True)
        return CandidateAction(
            weights=self.candidates[index],
            request_full_verifier=True,
        )


def load_candidate_replay_agent(
    path: str | Path,
    context: PlanningContext,
) -> CandidateReplayAgent:
    pass

    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw)
    if payload.get("schema_version") != "candidate-replay-manifest-0.1.0":
        raise ContractError("unsupported candidate replay manifest schema")
    records = payload.get("contexts")
    if not isinstance(records, list):
        raise ContractError("candidate replay contexts must be a list")
    matched = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("patient_id") == context.patient_id
        and record.get("prediction_set_id") == context.prediction_set_id
        and record.get("planning_formulation") == context.planning_formulation
    ]
    if len(matched) != 1:
        raise ContractError(
            "candidate replay manifest must match exactly one planning context"
        )
    raw_candidates = matched[0].get("candidates")
    if not isinstance(raw_candidates, list):
        raise ContractError("candidate replay candidates must be a list")
    candidates = tuple(
        ObjectiveWeights.from_mapping(candidate)
        for candidate in raw_candidates
        if isinstance(candidate, dict)
    )
    if len(candidates) != len(raw_candidates):
        raise ContractError("candidate replay weights must be mappings")
    return CandidateReplayAgent(candidates, manifest_sha256=manifest_sha256)


class LogUniformRandomAgent:
    pass

    def __init__(
        self,
        *,
        full_call_strategy: FullCallStrategy = FullCallStrategy.CHEAP_THEN_VERIFY,
    ) -> None:
        self.full_call_strategy = FullCallStrategy(full_call_strategy)
        self._rng: np.random.Generator | None = None

    @property
    def provenance(self) -> PolicyProvenance:
        return PolicyProvenance(
            family="sampling",
            name=f"log_uniform_random:{self.full_call_strategy.value}",
            version="1.0.0",
        )

    def reset(self, observation: Observation, *, seed: int) -> None:
        if observation.history:
            raise ContractError("agent reset requires an initial observation")
        self._rng = np.random.default_rng(seed)

    def act(self, observation: Observation) -> CandidateAction:
        if self._rng is None:
            raise ContractError("agent reset must be called before act")
        if _requires_full(observation):
            return self._new_random_action(request_full=True)
        if self.full_call_strategy is FullCallStrategy.NEVER:
            return self._new_random_action(request_full=False)
        if self.full_call_strategy is FullCallStrategy.FRONTLOAD:
            request_full = (
                observation.budget_state.full_calls_attempted
                < observation.budget_config.max_full_calls
            )
            return self._new_random_action(request_full=request_full)
        return self._cheap_then_verify(observation)

    def _cheap_then_verify(self, observation: Observation) -> CandidateAction:
        config = observation.budget_config
        verification_steps = min(config.max_full_calls, config.max_steps)
        screening_steps = config.max_steps - verification_steps
        if observation.budget_state.steps_used < screening_steps:
            return self._new_random_action(request_full=False)

        verified = {ObjectiveWeights.all_ones()}
        verified.update(
            entry.outcome.executed_weights
            for entry in observation.history
            if entry.accepted
            and entry.outcome is not None
            and entry.outcome.full is not None
        )
        ranked = []
        for entry in observation.history:
            if not entry.accepted or entry.outcome is None:
                continue
            weights = entry.outcome.executed_weights
            if weights in verified:
                continue
            improvement = clinical_improvement(entry.outcome.cheap)
            if improvement is not None:
                ranked.append((improvement, -entry.step_index, weights))
        if ranked:
            selected = max(ranked, key=lambda item: (item[0], item[1]))[2]
        else:
            selected = self._sample_weights()
        return CandidateAction(weights=selected, request_full_verifier=True)

    def _new_random_action(self, *, request_full: bool) -> CandidateAction:
        return CandidateAction(
            weights=self._sample_weights(),
            request_full_verifier=request_full,
        )

    def _sample_weights(self) -> ObjectiveWeights:
        if self._rng is None:
            raise ContractError("agent reset must be called before sampling")
        values = np.exp(
            self._rng.uniform(
                math.log(MIN_OBJECTIVE_WEIGHT),
                math.log(MAX_OBJECTIVE_WEIGHT),
                size=12,
            )
        )
        return ObjectiveWeights(tuple(float(value) for value in values))


class SobolSearchAgent:
    pass

    def __init__(
        self,
        *,
        full_call_strategy: FullCallStrategy = FullCallStrategy.CHEAP_THEN_VERIFY,
        include_all_ones: bool = False,
    ) -> None:
        self.full_call_strategy = FullCallStrategy(full_call_strategy)
        self.include_all_ones = include_all_ones
        self._candidates: tuple[ObjectiveWeights, ...] = ()

    @property
    def provenance(self) -> PolicyProvenance:
        return PolicyProvenance(
            family="low_discrepancy_search",
            name=(
                f"sobol_log_weights:{self.full_call_strategy.value}:"
                f"all_ones_anchor={str(self.include_all_ones).lower()}"
            ),
            version="1.0.0",
        )

    def reset(self, observation: Observation, *, seed: int) -> None:
        if observation.history:
            raise ContractError("agent reset requires an initial observation")
        try:
            from scipy.stats import qmc
        except ImportError as error:
            raise ContractError(
                "Sobol search requires the optional 'optimize' dependency"
            ) from error
        count = observation.budget_config.max_steps
        anchor_count = int(self.include_all_ones)
        draw_count = max(0, count - anchor_count)
        points: list[ObjectiveWeights] = []
        if self.include_all_ones:
            points.append(ObjectiveWeights.all_ones())
        if draw_count:
            exponent = int(math.ceil(math.log2(draw_count)))
            unit = qmc.Sobol(d=12, scramble=True, seed=seed).random_base2(exponent)
            log_minimum = math.log(MIN_OBJECTIVE_WEIGHT)
            log_span = math.log(MAX_OBJECTIVE_WEIGHT) - log_minimum
            for row in unit[:draw_count]:
                values = np.exp(log_minimum + row * log_span)
                points.append(
                    ObjectiveWeights(tuple(float(value) for value in values))
                )
        self._candidates = tuple(points)

    def act(self, observation: Observation) -> CandidateAction:
        if not self._candidates:
            raise ContractError("agent reset must be called before act")
        if _requires_full(observation):
            return self._next_candidate(observation, request_full=True)
        if self.full_call_strategy is FullCallStrategy.NEVER:
            return self._next_candidate(observation, request_full=False)
        if self.full_call_strategy is FullCallStrategy.FRONTLOAD:
            request_full = (
                observation.budget_state.full_calls_attempted
                < observation.budget_config.max_full_calls
            )
            return self._next_candidate(observation, request_full=request_full)
        return self._cheap_then_verify(observation)

    def _next_candidate(
        self,
        observation: Observation,
        *,
        request_full: bool,
    ) -> CandidateAction:
        index = min(
            observation.budget_state.steps_used,
            len(self._candidates) - 1,
        )
        return CandidateAction(
            weights=self._candidates[index],
            request_full_verifier=request_full,
        )

    def _cheap_then_verify(self, observation: Observation) -> CandidateAction:
        config = observation.budget_config
        verification_steps = min(config.max_full_calls, config.max_steps)
        screening_steps = config.max_steps - verification_steps
        if observation.budget_state.steps_used < screening_steps:
            return self._next_candidate(observation, request_full=False)

        verified = {ObjectiveWeights.all_ones()}
        verified.update(
            entry.outcome.executed_weights
            for entry in observation.history
            if entry.accepted
            and entry.outcome is not None
            and entry.outcome.full is not None
        )
        ranked = []
        for entry in observation.history:
            if not entry.accepted or entry.outcome is None:
                continue
            weights = entry.outcome.executed_weights
            if weights in verified:
                continue
            improvement = clinical_improvement(entry.outcome.cheap)
            if improvement is not None:
                ranked.append((improvement, -entry.step_index, weights))
        if ranked:
            selected = max(ranked, key=lambda item: (item[0], item[1]))[2]
            return CandidateAction(
                weights=selected,
                request_full_verifier=True,
            )
        return self._next_candidate(observation, request_full=True)


class CoordinatePatternSearchAgent:
    pass

    def __init__(
        self,
        *,
        initial_step: float = 0.25,
        shrink_factor: float = 0.5,
        objective_source: str = "full",
        full_call_strategy: FullCallStrategy = FullCallStrategy.FRONTLOAD,
    ) -> None:
        if not 0.0 < initial_step <= 1.0 or not math.isfinite(initial_step):
            raise ContractError("coordinate initial_step must lie in (0, 1]")
        if not 0.0 < shrink_factor <= 1.0 or not math.isfinite(shrink_factor):
            raise ContractError("coordinate shrink_factor must lie in (0, 1]")
        if objective_source not in {"cheap", "full"}:
            raise ContractError("coordinate objective_source must be cheap or full")
        self.initial_step = initial_step
        self.shrink_factor = shrink_factor
        self.objective_source = objective_source
        self.full_call_strategy = FullCallStrategy(full_call_strategy)

    @property
    def provenance(self) -> PolicyProvenance:
        return PolicyProvenance(
            family="local_search",
            name=(
                f"coordinate_pattern_log_weights:source={self.objective_source}:"
                f"step={self.initial_step:g}:shrink={self.shrink_factor:g}:"
                f"full={self.full_call_strategy.value}"
            ),
            version="1.0.0",
        )

    def reset(self, observation: Observation, *, seed: int) -> None:
        del seed
        if observation.history:
            raise ContractError("agent reset requires an initial observation")
        if (
            self.objective_source == "full"
            and not _requires_full(observation)
            and self.full_call_strategy is not FullCallStrategy.FRONTLOAD
        ):
            raise ContractError(
                "full-objective coordinate search requires full-only or frontload"
            )

    def act(self, observation: Observation) -> CandidateAction:
        if not observation.history:
            return CandidateAction(
                weights=ObjectiveWeights.all_ones(),
                request_full_verifier=self._request_full(observation),
            )
        if self._is_verification_phase(observation):
            selected = self._best_unverified_cheap(observation)
            if selected is not None:
                return CandidateAction(
                    weights=selected,
                    request_full_verifier=True,
                )

        best = self._best_weights(observation)
        proposal_index = max(0, observation.budget_state.steps_used - 1)
        dimension = (proposal_index // 2) % len(best.values)
        direction = 1.0 if proposal_index % 2 == 0 else -1.0
        sweep = proposal_index // (2 * len(best.values))
        step = self.initial_step * self.shrink_factor**sweep
        point = np.log(np.asarray(best.values, dtype=float)) / math.log(4.0)
        point[dimension] = np.clip(
            point[dimension] + direction * step,
            -1.0,
            1.0,
        )
        weights = ObjectiveWeights(
            tuple(float(value) for value in np.power(4.0, point))
        )
        return CandidateAction(
            weights=weights,
            request_full_verifier=self._request_full(observation),
        )

    def _best_weights(self, observation: Observation) -> ObjectiveWeights:
        ranked: list[tuple[float, int, ObjectiveWeights]] = []
        for entry in observation.history:
            if not entry.accepted or entry.outcome is None:
                continue
            if self.objective_source == "cheap":
                metric = clinical_improvement(entry.outcome.cheap)
            elif entry.outcome.full is not None:
                metric = clinical_improvement(entry.outcome.full)
            else:
                metric = None
            if metric is not None:
                ranked.append(
                    (float(metric), -entry.step_index, entry.outcome.executed_weights)
                )
        if not ranked:
            return ObjectiveWeights.all_ones()
        return max(ranked, key=lambda item: (item[0], item[1]))[2]

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
    def _best_unverified_cheap(
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
            metric = clinical_improvement(entry.outcome.cheap)
            weights = entry.outcome.executed_weights
            if metric is not None and weights not in verified:
                ranked.append((float(metric), -entry.step_index, weights))
        if not ranked:
            return None
        return max(ranked, key=lambda item: (item[0], item[1]))[2]
