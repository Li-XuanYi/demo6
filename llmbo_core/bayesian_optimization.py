# llmbo_core/bayesian_optimization.py
from __future__ import annotations
from collections import deque
from typing import Any, Iterable, Callable, TYPE_CHECKING
from warnings import warn

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# 复用你项目中的模块（不是 pip 的）
from bayes_opt import acquisition
from bayes_opt.parameter import wrap_kernel
from bayes_opt.target_space import TargetSpace

try:
    from bayes_opt.event import DEFAULT_EVENTS, Events
    from bayes_opt.logger import _get_default_logger
except Exception:
    # 兜底：若你项目中没有 event/logger（一般都有），就用轻量打印
    class Events:
        OPTIMIZATION_START = "OPTIMIZATION_START"
        OPTIMIZATION_STEP = "OPTIMIZATION_STEP"
        OPTIMIZATION_END = "OPTIMIZATION_END"
    DEFAULT_EVENTS = [Events.OPTIMIZATION_START, Events.OPTIMIZATION_STEP, Events.OPTIMIZATION_END]

    class _Printer:
        def update(self, event, instance):  # noqa: D401
            if event == Events.OPTIMIZATION_START:
                print("开始优化…")
            elif event == Events.OPTIMIZATION_STEP:
                if instance.max is not None:
                    best = instance.max
                    print(f"迭代步完成 | 当前best目标: {best['target']:.6f} | 参数: {best['params']}")
            elif event == Events.OPTIMIZATION_END:
                print("优化结束。")

    def _get_default_logger(verbose: int, is_constrained: bool):
        return _Printer()

if TYPE_CHECKING:
    from numpy.random import RandomState
    from numpy.typing import NDArray
    from collections.abc import Mapping
    from bayes_opt.acquisition import AcquisitionFunction
    from bayes_opt.domain_reduction import DomainTransformer
    from bayes_opt.parameter import ParamsType

    Float = np.floating[Any]


class Observable:
    """极简事件系统，兼容 bayes_opt.logger 的订阅接口。"""
    def __init__(self, events: Iterable[Any]) -> None:
        self._events = {event: {} for event in events}

    def subscribe(self, event: Any, subscriber: Any, callback: Callable[..., Any] | None = None) -> None:
        if callback is None:
            callback = subscriber.update
        self._events[event][subscriber] = callback

    def dispatch(self, event: Any) -> None:
        for cb in self._events[event].values():
            cb(event, self)


class BayesianOptimization(Observable):
    """
    自定义版本，支持:
      - use_llm: 是否启用 LLM 辅助建议
      - llm_sampler: 负责生成候选
      - llm_surrogate: 负责从候选中选出最优
    其余行为尽量与原版保持一致。
    """
    def __init__(
        self,
        f: Callable[..., float] | None,
        pbounds: Mapping[str, tuple[float, float]],
        acquisition_function: AcquisitionFunction | None = None,
        random_state: int | RandomState | None = None,
        verbose: int = 2,
        bounds_transformer: DomainTransformer | None = None,
        allow_duplicate_points: bool = False,
        use_llm: bool = False,
        llm_sampler: Any | None = None,
        llm_surrogate: Any | None = None,
    ):
        super().__init__(DEFAULT_EVENTS)

        self._random_state = np.random.RandomState(random_state) if random_state is not None else np.random.RandomState()
        self._allow_duplicate_points = allow_duplicate_points
        self._queue: deque[ParamsType] = deque()

        # 目标空间
        self._space = TargetSpace(
            f, pbounds, random_state=random_state, allow_duplicate_points=self._allow_duplicate_points
        )
        self.is_constrained = False  # 如需约束，可扩展

        # GP 代理
        self._gp = GaussianProcessRegressor(
            kernel=wrap_kernel(Matern(nu=2.5), transform=self._space.kernel_transform),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        # 采集函数
        self._acq = acquisition.UpperConfidenceBound(kappa=2.576, random_state=self._random_state) \
            if acquisition_function is None else acquisition_function

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer is not None:
            self._bounds_transformer.initialize(self._space)

        # LLM 集成
        self.use_llm = bool(use_llm)
        self.llm_sampler = llm_sampler
        self.llm_surrogate = llm_surrogate
        if self.use_llm and (self.llm_sampler is None or self.llm_surrogate is None):
            raise ValueError("use_llm=True 时必须提供 llm_sampler 和 llm_surrogate")

        self._sorting_warning_already_shown = False
        self._prime_subscriptions()

    # --- 便捷访问 ---
    @property
    def space(self) -> TargetSpace:
        return self._space

    @property
    def acquisition_function(self) -> AcquisitionFunction:
        return self._acq

    @property
    def max(self) -> dict[str, Any] | None:
        return self._space.max()

    @property
    def res(self) -> list[dict[str, Any]]:
        return self._space.res()

    # --- 注册/探测 ---
    def register(self, params: ParamsType, target: float) -> None:
        self._space.register(params, target)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params: ParamsType, lazy: bool = True) -> None:
        if isinstance(params, np.ndarray) and not self._sorting_warning_already_shown:
            warn("传入了 np.ndarray；参数顺序按 pbounds 顺序解析。", stacklevel=1)
            self._sorting_warning_already_shown = True
            params = self._space.array_to_params(params)
        if lazy:
            self._queue.append(params)
        else:
            self._space.probe(params)  # 会调用目标函数并自动 register
            self.dispatch(Events.OPTIMIZATION_STEP)

    # --- 建议下一个点 ---
    def _suggest_via_acq(self) -> dict[str, float]:
        # 拟合 GP
        X = self._space.params
        Y = self._space.target
        if len(Y) == 0:
            # 随机点
            x = self._space.random_sample(random_state=self._random_state)
            return self._space.array_to_params(x)
        self._gp.fit(X, Y)
        # 采集函数建议（使用你项目里的 acquisition.suggest 接口）
        suggestion = self._acq.suggest(gp=self._gp, target_space=self._space, fit_gp=False)
        return self._space.array_to_params(suggestion)

    def _suggest_via_llm(self) -> dict[str, float]:
        # 1) LLM 生成候选
        print("LLM Sampler: 正在生成候选点...")
        candidates = self.llm_sampler.generate_candidates(self._space, n_candidates=20)
        if not candidates:
            print("LLM Sampler 未能生成有效候选点，回退到高斯过程采集函数。")
            return self._suggest_via_acq()
        # 2) LLM 选择最佳
        print(f"LLM Surrogate: 正在从 {len(candidates)} 个候选点中选择...")
        chosen = self.llm_surrogate.select_best_candidate(self._space, candidates)
        if chosen is None:
            print("LLM Surrogate 未能选择候选点，回退到高斯过程采集函数。")
            return self._suggest_via_acq()
        print("LLM 成功建议下一个点。")
        return chosen

    def suggest(self) -> dict[str, float]:
        if self._space.empty:
            x = self._space.random_sample(random_state=self._random_state)
            return self._space.array_to_params(x)
        if self.use_llm:
            return self._suggest_via_llm()
        return self._suggest_via_acq()

    # --- 初始化/订阅 ---
    def _prime_queue(self, init_points: int) -> None:
        if not self._queue and self._space.empty:
            init_points = max(init_points, 1)
        for _ in range(init_points):
            x = self._space.random_sample(random_state=self._random_state)
            self._queue.append(self._space.array_to_params(x))

    def _prime_subscriptions(self) -> None:
        # 若无订阅者，则挂上默认 logger
        if not any(len(v) for v in getattr(self, "_events", {}).values()):
            logger = _get_default_logger(self._verbose, self.is_constrained)
            self.subscribe(Events.OPTIMIZATION_START, logger)
            self.subscribe(Events.OPTIMIZATION_STEP, logger)
            self.subscribe(Events.OPTIMIZATION_END, logger)

    # --- 主循环 ---
    def maximize(self, init_points: int = 5, n_iter: int = 25) -> None:
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)

        it = 0
        while self._queue or it < n_iter:
            try:
                params = self._queue.popleft()
            except IndexError:
                params = self.suggest()
                it += 1
            self.probe(params, lazy=False)

            if self._bounds_transformer is not None and it > 0:
                # 需要的话更新搜索域
                self._bounds_transformer.transform(self._space)

        self.dispatch(Events.OPTIMIZATION_END)
