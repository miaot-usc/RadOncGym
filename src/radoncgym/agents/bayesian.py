

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import math
import numpy as np

from radoncgym.agents.baselines import FullCallStrategy, _requires_full
from radoncgym.core.constants import OBJECTIVE_NAMES
from radoncgym.core.errors import ContractError
from radoncgym.core.metric_schema import (
    absolute_clinical_objective,
    clinical_improvement,
    returned_clinical_improvement,
)
from radoncgym.core.models import CandidateAction, ObjectiveWeights, Observation, SolverStatus
from radoncgym.trajectories.schema import PolicyProvenance


class BoTorchGPAgent:
    pass

    def __init__(
        self,
        *,
        initial_design_size: int = 4,
        objective_source: str = "returned",
        full_call_strategy: FullCallStrategy = FullCallStrategy.FRONTLOAD,
        raw_samples: int = 256,
        num_restarts: int = 16,
        fit_maxiter: int = 200,
        failure_penalty: float = 1_000_000.0,
        promotion_min_improvement: float = 0.0,
        external_all_ones_anchor: bool = True,
        all_ones_clinical_objective: float | None = None,
    ) -> None:
        if initial_design_size < 2:
            raise ContractError("BO initial_design_size must be at least two")
        if objective_source not in {"cheap", "full", "returned"}:
            raise ContractError("BO objective_source must be cheap, full, or returned")
        if all_ones_clinical_objective is not None and (
            not math.isfinite(all_ones_clinical_objective)
            or all_ones_clinical_objective < 0.0
        ):
            raise ContractError("all-ones clinical objective must be finite and nonnegative")
        for name, value in (
            ("raw_samples", raw_samples),
            ("num_restarts", num_restarts),
            ("fit_maxiter", fit_maxiter),
        ):
            if value <= 0:
                raise ContractError(f"BO {name} must be positive")
        if failure_penalty <= 0 or not math.isfinite(failure_penalty):
            raise ContractError("BO failure penalty must be finite and positive")
        if (
            promotion_min_improvement < 0
            or not math.isfinite(promotion_min_improvement)
        ):
            raise ContractError(
                "BO promotion minimum improvement must be finite and nonnegative"
            )
        self.initial_design_size = initial_design_size
        self.objective_source = objective_source
        self.full_call_strategy = FullCallStrategy(full_call_strategy)
        self.raw_samples = raw_samples
        self.num_restarts = num_restarts
        self.fit_maxiter = fit_maxiter
        self.failure_penalty = failure_penalty
        self.promotion_min_improvement = float(promotion_min_improvement)
        self.external_all_ones_anchor = bool(external_all_ones_anchor)
        self.all_ones_clinical_objective = all_ones_clinical_objective
        self._seed: int | None = None
        self._initial_points: np.ndarray | None = None

    @property
    def provenance(self) -> PolicyProvenance:
        def package(name: str) -> str:
            try:
                return version(name)
            except PackageNotFoundError:
                return "missing"

        return PolicyProvenance(
            family="bayesian_optimization",
            name=(
                f"botorch_exact_gp_logei:source={self.objective_source}:"
                f"initial={self.initial_design_size}:raw={self.raw_samples}:"
                f"restarts={self.num_restarts}:fit_maxiter={self.fit_maxiter}:"
                f"full={self.full_call_strategy.value}:"
                f"promotion_min={self.promotion_min_improvement:g}:"
                f"external_anchor={str(self.external_all_ones_anchor).lower()}:"
                f"absolute_anchor={self.all_ones_clinical_objective}:"
                f"failure_penalty={self.failure_penalty:g}"
            ),
            version=(
                f"adapter-1.1.0+botorch-{package('botorch')}+"
                f"gpytorch-{package('gpytorch')}+torch-{package('torch')}"
            ),
        )

    def reset(self, observation: Observation, *, seed: int) -> None:
        if observation.history:
            raise ContractError("agent reset requires an initial observation")
        if (
            self.full_call_strategy is FullCallStrategy.ADAPTIVE_PROMOTION
            and self.all_ones_clinical_objective is None
        ):
            raise ContractError("adaptive promotion requires the precomputed all-ones clinical objective")
        if (
            self.objective_source == "full"
            and not _requires_full(observation)
            and self.full_call_strategy is not FullCallStrategy.FRONTLOAD
        ):
            raise ContractError(
                "full-objective BO requires full-only or frontload full calls"
            )
        try:
            import torch
            import botorch
        except ImportError as error:
            raise ContractError(
                "Bayesian optimization requires the optional 'bayesian' dependencies"
            ) from error
        self._seed = int(seed)
        engine = torch.quasirandom.SobolEngine(
            dimension=len(OBJECTIVE_NAMES),
            scramble=True,
            seed=self._seed,
        )
        random_count = max(
            0,
            self.initial_design_size - 1,
        )
        random_points = (
            engine.draw(random_count).double().cpu().numpy()
            if random_count
            else np.empty((0, len(OBJECTIVE_NAMES)), dtype=float)
        )
        self._initial_points = (
            random_points
            if self.external_all_ones_anchor
            else np.vstack(
                (np.full((1, len(OBJECTIVE_NAMES)), 0.5), random_points)
            )
        )

    def act(self, observation: Observation) -> CandidateAction:
        if self._seed is None or self._initial_points is None:
            raise ContractError("agent reset must be called before act")
        if self._is_verification_phase(observation):
            candidate = self._best_unverified_cheap(observation)
            if candidate is not None:
                return CandidateAction(weights=candidate, request_full_verifier=True)
        if self.full_call_strategy is FullCallStrategy.ADAPTIVE_PROMOTION:
            candidate = self._adaptive_promotion_candidate(observation)
            if candidate is not None:
                return CandidateAction(
                    weights=candidate,
                    request_full_verifier=True,
                )

        evaluated = self._training_data(observation)
        if len(observation.history) < len(self._initial_points):
            point = self._initial_points[len(observation.history)]
        elif len(evaluated) >= 2:
            point = self._propose_with_gp(evaluated, len(observation.history))
        else:
            point = self._deterministic_fallback(len(observation.history))
        return CandidateAction(
            weights=_unit_point_to_weights(point),
            request_full_verifier=self._request_full(observation),
        )

    def _training_data(
        self,
        observation: Observation,
    ) -> list[tuple[np.ndarray, float]]:
        data: list[tuple[np.ndarray, float]] = []
        if self.external_all_ones_anchor:
            data.append(
                (
                    np.full(len(OBJECTIVE_NAMES), 0.5, dtype=float),
                    0.0,
                )
            )
        for entry in observation.history:
            if not entry.accepted or entry.outcome is None:
                continue
            if self.objective_source == "returned":
                metric = returned_clinical_improvement(entry.outcome)
            elif self.objective_source == "cheap":
                metric = clinical_improvement(entry.outcome.cheap)
            elif entry.outcome.full is not None:
                metric = clinical_improvement(entry.outcome.full)
            else:
                metric = None
            value = (
                -self.failure_penalty
                if metric is None or not math.isfinite(metric)
                else float(metric)
            )
            point = (
                np.log(np.asarray(entry.outcome.executed_weights.values))
                / math.log(4.0)
                + 1.0
            ) / 2.0
            data.append((point, value))
        return data

    def _propose_with_gp(
        self,
        data: list[tuple[np.ndarray, float]],
        step_index: int,
    ) -> np.ndarray:
        import torch
        from botorch.acquisition import LogExpectedImprovement
        from botorch.exceptions.errors import ModelFittingError
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.optim.fit import fit_gpytorch_mll_torch
        from botorch.optim import optimize_acqf
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood

        train_x = torch.tensor(
            np.stack([point for point, _ in data]), dtype=torch.double
        )
        raw_objective = torch.tensor(
            [value for _, value in data], dtype=torch.double
        ).unsqueeze(-1)
        mean = raw_objective.mean()
        scale = raw_objective.std(unbiased=False).clamp_min(1e-8)
        train_y = (raw_objective - mean) / scale
        def new_model() -> tuple[SingleTaskGP, ExactMarginalLogLikelihood]:
            covariance = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=len(OBJECTIVE_NAMES))
            )
            candidate_model = SingleTaskGP(
                train_x,
                train_y,
                covar_module=covariance,
            )
            candidate_mll = ExactMarginalLogLikelihood(
                candidate_model.likelihood,
                candidate_model,
            )
            return candidate_model, candidate_mll

        model, mll = new_model()
        try:
            fit_gpytorch_mll(
                mll,
                optimizer_kwargs={"options": {"maxiter": self.fit_maxiter}},
            )
        except ModelFittingError:



            model, mll = new_model()
            try:
                fit_gpytorch_mll(
                    mll,
                    optimizer=fit_gpytorch_mll_torch,
                    optimizer_kwargs={"step_limit": self.fit_maxiter},
                )
            except (ModelFittingError, RuntimeError):
                return self._deterministic_fallback(step_index)
        acquisition = LogExpectedImprovement(model, best_f=train_y.max())
        bounds = torch.stack(
            (
                torch.zeros(len(OBJECTIVE_NAMES), dtype=torch.double),
                torch.ones(len(OBJECTIVE_NAMES), dtype=torch.double),
            )
        )
        torch.manual_seed(self._seed + step_index)
        candidate, _ = optimize_acqf(
            acquisition,
            bounds=bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"maxiter": 200},
        )
        point = candidate.detach().cpu().numpy().reshape(-1)
        if np.min(np.linalg.norm(train_x.cpu().numpy() - point, axis=1)) < 1e-6:
            return self._deterministic_fallback(step_index)
        return point

    def _deterministic_fallback(self, step_index: int) -> np.ndarray:
        import torch

        if self._seed is None:
            raise AssertionError("BO seed disappeared")
        engine = torch.quasirandom.SobolEngine(
            dimension=len(OBJECTIVE_NAMES),
            scramble=True,
            seed=self._seed + 10_000,
        )
        return engine.draw(step_index + 1)[-1].double().cpu().numpy()

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
        if self.full_call_strategy is FullCallStrategy.ADAPTIVE_PROMOTION:
            return self._must_spend_remaining_full(observation)
        return self._is_verification_phase(observation)

    @staticmethod
    def _must_spend_remaining_full(observation: Observation) -> bool:
        remaining_steps = (
            observation.budget_config.max_steps
            - observation.budget_state.steps_used
        )
        remaining_full = (
            observation.budget_config.max_full_calls
            - observation.budget_state.full_calls_attempted
        )
        return remaining_full > 0 and remaining_steps <= remaining_full

    def _adaptive_promotion_candidate(
        self,
        observation: Observation,
    ) -> ObjectiveWeights | None:
        pass

        if _requires_full(observation):
            return None
        if (
            observation.budget_state.full_calls_attempted
            >= observation.budget_config.max_full_calls
        ):
            return None
        anchor = ObjectiveWeights.all_ones()
        baseline = self.all_ones_clinical_objective
        if baseline is None:
            raise ContractError("adaptive promotion requires the precomputed all-ones clinical objective")
        full_rows: list[tuple[ObjectiveWeights, float, float]] = [
            (anchor, 0.0, 0.0)
        ]
        attempted: set[ObjectiveWeights] = {anchor}
        for entry in observation.history:
            if (
                not entry.accepted
                or entry.outcome is None
                or entry.outcome.full is None
            ):
                continue
            weights = entry.outcome.executed_weights
            attempted.add(weights)
            if entry.outcome.full.status is not SolverStatus.SOLVED:
                continue
            cheap_value = clinical_improvement(entry.outcome.cheap)
            full_objective = absolute_clinical_objective(entry.outcome.full)
            full_value = (
                None if full_objective is None else baseline - full_objective
            )
            if (
                cheap_value is None
                or full_value is None
                or not math.isfinite(cheap_value)
                or not math.isfinite(full_value)
            ):
                continue
            full_rows.append((weights, float(cheap_value), float(full_value)))
        bias = float(
            np.mean([full_value - cheap_value for _, cheap_value, full_value in full_rows])
        )
        best_full = max(full_value for _, _, full_value in full_rows)
        ranked: list[tuple[float, int, ObjectiveWeights]] = []
        for entry in observation.history:
            if not entry.accepted or entry.outcome is None:
                continue
            weights = entry.outcome.executed_weights
            if weights in attempted or entry.outcome.full is not None:
                continue
            cheap_value = clinical_improvement(entry.outcome.cheap)
            if cheap_value is None or not math.isfinite(cheap_value):
                continue
            predicted_full = float(cheap_value) + bias
            ranked.append((predicted_full, entry.step_index, weights))
        if not ranked:
            return None
        predicted_full, _, weights = max(
            ranked,
            key=lambda item: (item[0], -item[1]),
        )
        if self._must_spend_remaining_full(observation):
            return weights
        if predicted_full - self.promotion_min_improvement > best_full:
            return weights
        return None

    def _is_verification_phase(self, observation: Observation) -> bool:
        if _requires_full(observation):
            return False
        if self.full_call_strategy is not FullCallStrategy.CHEAP_THEN_VERIFY:
            return False
        count = min(
            observation.budget_config.max_full_calls,
            observation.budget_config.max_steps,
        )
        verification_start = observation.budget_config.max_steps - count
        return observation.budget_state.steps_used >= verification_start

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


class BoTorchSAASAgent(BoTorchGPAgent):
    pass

    def __init__(
        self,
        *,
        initial_design_size: int = 4,
        objective_source: str = "returned",
        full_call_strategy: FullCallStrategy = FullCallStrategy.FRONTLOAD,
        raw_samples: int = 256,
        num_restarts: int = 16,
        nuts_warmup_steps: int = 128,
        nuts_num_samples: int = 64,
        nuts_thinning: int = 4,
        max_tree_depth: int = 6,
        mc_samples: int = 256,
        failure_penalty: float = 1_000_000.0,
        external_all_ones_anchor: bool = True,
        all_ones_clinical_objective: float | None = None,
    ) -> None:
        super().__init__(
            initial_design_size=initial_design_size,
            objective_source=objective_source,
            full_call_strategy=full_call_strategy,
            raw_samples=raw_samples,
            num_restarts=num_restarts,
            fit_maxiter=1,
            failure_penalty=failure_penalty,
            external_all_ones_anchor=external_all_ones_anchor,
            all_ones_clinical_objective=all_ones_clinical_objective,
        )
        for name, value in (
            ("nuts_warmup_steps", nuts_warmup_steps),
            ("nuts_num_samples", nuts_num_samples),
            ("nuts_thinning", nuts_thinning),
            ("max_tree_depth", max_tree_depth),
            ("mc_samples", mc_samples),
        ):
            if value <= 0:
                raise ContractError(f"SAAS-BO {name} must be positive")
        if nuts_num_samples < nuts_thinning:
            raise ContractError(
                "SAAS-BO nuts_num_samples must be at least nuts_thinning"
            )
        self.nuts_warmup_steps = nuts_warmup_steps
        self.nuts_num_samples = nuts_num_samples
        self.nuts_thinning = nuts_thinning
        self.max_tree_depth = max_tree_depth
        self.mc_samples = mc_samples

    @property
    def provenance(self) -> PolicyProvenance:
        def package(name: str) -> str:
            try:
                return version(name)
            except PackageNotFoundError:
                return "missing"

        return PolicyProvenance(
            family="bayesian_optimization",
            name=(
                f"botorch_saas_qlogei:source={self.objective_source}:"
                f"initial={self.initial_design_size}:raw={self.raw_samples}:"
                f"restarts={self.num_restarts}:warmup={self.nuts_warmup_steps}:"
                f"samples={self.nuts_num_samples}:thin={self.nuts_thinning}:"
                f"depth={self.max_tree_depth}:mc={self.mc_samples}:"
                f"full={self.full_call_strategy.value}:"
                f"failure_penalty={self.failure_penalty:g}"
            ),
            version=(
                f"adapter-1.0.0+botorch-{package('botorch')}+"
                f"gpytorch-{package('gpytorch')}+pyro-{package('pyro-ppl')}+"
                f"torch-{package('torch')}"
            ),
        )

    def _propose_with_gp(
        self,
        data: list[tuple[np.ndarray, float]],
        step_index: int,
    ) -> np.ndarray:
        import torch
        from botorch.acquisition.logei import qLogExpectedImprovement
        from botorch.fit import fit_fully_bayesian_model_nuts
        from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
        from botorch.optim import optimize_acqf
        from botorch.sampling.normal import SobolQMCNormalSampler

        train_x = torch.tensor(
            np.stack([point for point, _ in data]), dtype=torch.double
        )
        raw_objective = torch.tensor(
            [value for _, value in data], dtype=torch.double
        ).unsqueeze(-1)
        mean = raw_objective.mean()
        scale = raw_objective.std(unbiased=False).clamp_min(1e-8)
        train_y = (raw_objective - mean) / scale
        model = SaasFullyBayesianSingleTaskGP(train_x, train_y)
        torch.manual_seed(self._seed + step_index)
        fit_fully_bayesian_model_nuts(
            model,
            max_tree_depth=self.max_tree_depth,
            warmup_steps=self.nuts_warmup_steps,
            num_samples=self.nuts_num_samples,
            thinning=self.nuts_thinning,
            disable_progbar=True,
        )
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.mc_samples]),
            seed=self._seed + step_index,
        )
        acquisition = qLogExpectedImprovement(
            model,
            best_f=train_y.max(),
            sampler=sampler,
        )
        bounds = torch.stack(
            (
                torch.zeros(len(OBJECTIVE_NAMES), dtype=torch.double),
                torch.ones(len(OBJECTIVE_NAMES), dtype=torch.double),
            )
        )
        candidate, _ = optimize_acqf(
            acquisition,
            bounds=bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"maxiter": 200},
        )
        point = candidate.detach().cpu().numpy().reshape(-1)
        if np.min(np.linalg.norm(train_x.cpu().numpy() - point, axis=1)) < 1e-6:
            return self._deterministic_fallback(step_index)
        return point


def _unit_point_to_weights(point: np.ndarray) -> ObjectiveWeights:
    clipped = np.clip(np.asarray(point, dtype=float), 0.0, 1.0)
    values = np.power(4.0, clipped * 2.0 - 1.0)
    return ObjectiveWeights(tuple(float(value) for value in values))
