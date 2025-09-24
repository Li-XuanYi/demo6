from __future__ import annotations

from collections import deque
from typing import Any, TYPE_CHECKING
from warnings import warn

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Import other BayesOpt components
from bayes_opt import acquisition
from bayes_opt.parameter import wrap_kernel
from bayes_opt.target_space import TargetSpace
from bayes_opt.observer import _Tracker

# Try importing event and logger; define fallback if not present
try:
    from bayes_opt.event import DEFAULT_EVENTS, Events
    from bayes_opt.logger import _get_default_logger
except ImportError:
    # Define Events and DEFAULT_EVENTS
    class Events:
        OPTIMIZATION_START = "OPTIMIZATION_START"
        OPTIMIZATION_STEP = "OPTIMIZATION_STEP"
        OPTIMIZATION_END = "OPTIMIZATION_END"
    DEFAULT_EVENTS = [Events.OPTIMIZATION_START, Events.OPTIMIZATION_STEP, Events.OPTIMIZATION_END]
    # Define a simple printing logger
    class _Printer(_Tracker):
        """Logger for printing optimization progress."""
        def update(self, event: str, instance: BayesianOptimization) -> None:  # type: ignore[name-defined]
            if event == Events.OPTIMIZATION_START:
                print("开始优化…")
            elif event == Events.OPTIMIZATION_STEP:
                # Get current iteration and best result so far
                current_iter = self._iterations
                best = instance.max
                if best is not None:
                    best_error = -best["target"]
                    best_params = {k: round(v, 4) for k, v in best["params"].items()}
                    print(f"迭代{current_iter} 完成，当前最佳误差: {best_error:.4f}, 最佳参数: {best_params}")
            elif event == Events.OPTIMIZATION_END:
                print("优化结束。")
    def _get_default_logger(verbose: int, is_constrained: bool) -> _Printer:
        # If verbose > 0, use the printer (we ignore is_constrained for simplicity)
        if verbose:
            return _Printer()
        else:
            # If verbose is 0, return a logger that does nothing (here we still return _Printer but it will only print on events)
            return _Printer()

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from numpy.random import RandomState
    from numpy.typing import NDArray
    from scipy.optimize import NonlinearConstraint
    from bayes_opt.acquisition import AcquisitionFunction
    from bayes_opt.constraint import ConstraintModel
    from bayes_opt.domain_reduction import DomainTransformer
    from bayes_opt.parameter import BoundsMapping, ParamsType
    Float = np.floating[Any]

class Observable:
    """Simple observable class to handle event subscriptions."""
    def __init__(self, events: Iterable[Any]) -> None:
        self._events = {event: {} for event in events}  # map event -> {subscriber: callback}

    def subscribe(self, event: Any, subscriber: Any, callback: Callable[..., Any] | None = None) -> None:
        if callback is None:
            callback = subscriber.update  # default to subscriber's update method
        self._events[event][subscriber] = callback

    def dispatch(self, event: Any) -> None:
        for callback in self._events[event].values():
            callback(event, self)

class BayesianOptimization(Observable):
    """
    Bayesian Optimization class for maximizing a target function.
    If use_llm=True, uses an LLM-based sampler and surrogate for suggestions.
    """
    def __init__(
        self,
        f: Callable[..., float] | None,
        pbounds: Mapping[str, tuple[float, float]],
        acquisition_function: AcquisitionFunction | None = None,
        constraint: NonlinearConstraint | None = None,
        random_state: int | RandomState | None = None,
        verbose: int = 2,
        bounds_transformer: DomainTransformer | None = None,
        allow_duplicate_points: bool = False,
        use_llm: bool = False,
        llm_sampler: Any = None,
        llm_surrogate: Any = None
    ):
        super().__init__(events=DEFAULT_EVENTS)  # initialize Observable with default events

        self._random_state = np.random.RandomState(random_state) if random_state is not None else np.random.RandomState()
        self._allow_duplicate_points = allow_duplicate_points
        self._queue: deque[ParamsType] = deque()

        # Acquisition function selection (GP-UCB default for unconstrained, EI for constrained if not provided)
        if acquisition_function is None:
            if constraint is None:
                self._acquisition_function = acquisition.UpperConfidenceBound(kappa=2.576, random_state=self._random_state)
            else:
                self._acquisition_function = acquisition.ExpectedImprovement(xi=0.01, random_state=self._random_state)
        else:
            self._acquisition_function = acquisition_function

        # Initialize target space (with optional constraint model)
        if constraint is None:
            self._space = TargetSpace(f, pbounds, random_state=random_state, allow_duplicate_points=self._allow_duplicate_points)
            self.is_constrained = False
        else:
            from bayes_opt.constraint import ConstraintModel  # ensure ConstraintModel is available
            constraint_model = ConstraintModel(constraint.fun, constraint.lb, constraint.ub, random_state=random_state)
            self._space = TargetSpace(f, pbounds, constraint=constraint_model, random_state=random_state, allow_duplicate_points=self._allow_duplicate_points)
            self.is_constrained = True

        # Internal Gaussian Process surrogate model for acquisition function
        self._gp = GaussianProcessRegressor(
            kernel=wrap_kernel(Matern(nu=2.5), transform=self._space.kernel_transform),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            if not isinstance(self._bounds_transformer, DomainTransformer):
                raise TypeError("The bounds_transformer must be an instance of DomainTransformer")
            self._bounds_transformer.initialize(self._space)

        # LLM integration
        self.use_llm = use_llm
        if self.use_llm:
            if llm_sampler is None or llm_surrogate is None:
                raise ValueError("LLMSampler and LLMSurrogate must be provided when use_llm is True.")
            self.llm_sampler = llm_sampler
            self.llm_surrogate = llm_surrogate

        self._sorting_warning_already_shown = False

    def max(self) -> dict[str, Any] | None:
        """Return the highest target value found and the corresponding parameters."""
        return self._space.max()

    @property
    def res(self) -> list[dict[str, Any]]:
        """Return all evaluation results as a list of dicts (with 'target' and 'params')."""
        return self._space.res()

    def probe(self, params: ParamsType, lazy: bool = True) -> None:
        """Evaluate the target function at the given parameters."""
        # Allow passing numpy array of parameters
        if isinstance(params, np.ndarray) and not self._sorting_warning_already_shown:
            warn("Registering an np.ndarray directly. Parameter order follows pbounds dictionary order.", stacklevel=1)
            self._sorting_warning_already_shown = True
            params = self._space.array_to_params(params)
        if lazy:
            # Queue the point to evaluate later in maximize()
            self._queue.append(params)
        else:
            # Evaluate immediately and register
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def suggest(self) -> dict[str, float]:
        """Suggest the next point to sample (either via LLM or GP acquisition)."""
        if self._space.empty:
            # If no point has been sampled yet, pick a random point
            return self._space.array_to_params(self._space.random_sample(random_state=self._random_state))
        if self.use_llm:
            # --- LLAMBO-inspired suggestion flow using LLM ---
            # 1. Use LLM Sampler to generate candidate points
            candidate_points = self.llm_sampler.generate_candidates(
                target_space=self._space,
                n_candidates=20  # number of candidates to generate
            )
            if not candidate_points:
                print("LLM Sampler未能生成有效候选点，改用随机点。")
                return self._space.array_to_params(self._space.random_sample(random_state=self._random_state))
            # 2. Use LLM Surrogate to evaluate candidates and select the best
            next_point = self.llm_surrogate.select_best_candidate(
                target_space=self._space,
                candidate_points=candidate_points
            )
            if next_point is None:
                print("LLM Surrogate未能选择候选点，改用随机点。")
                return self._space.array_to_params(self._space.random_sample(random_state=self._random_state))
            return next_point
        else:
            # --- Standard GP-based suggestion flow ---
            suggestion = self._acquisition_function.suggest(
                gp=self._gp,
                target_space=self._space,
                fit_gp=True
            )
            return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points: int) -> None:
        """Ensure there are `init_points` random points to start with."""
        if not self._queue and self._space.empty:
            init_points = max(init_points, 1)
        for _ in range(init_points):
            sample = self._space.random_sample(random_state=self._random_state)
            self._queue.append(self._space.array_to_params(sample))

    def _prime_subscriptions(self) -> None:
        """Subscribe default logger if no other subscribers have been added."""
        if not any(len(subs) for subs in self._events.values()):
            _logger = _get_default_logger(self._verbose, self.is_constrained)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def maximize(self, init_points: int = 5, n_iter: int = 25) -> None:
        """Run Bayesian Optimization to maximize the target function."""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        # Evaluate the initial random points
        self._prime_queue(init_points)
        iteration = 0
        while self._queue or iteration < n_iter:
            try:
                # If there are queued points (initial random points), evaluate them first
                x_probe = self._queue.popleft()
            except IndexError:
                # Queue empty, use acquisition/LLM to get a suggestion
                x_probe = self.suggest()
                iteration += 1
            # Evaluate the function at x_probe
            self.probe(x_probe, lazy=False)
            # If using dynamic bounds adjustment
            if self._bounds_transformer and iteration > 0:
                self.set_bounds(self._bounds_transformer.transform(self._space))
        self.dispatch(Events.OPTIMIZATION_END)
