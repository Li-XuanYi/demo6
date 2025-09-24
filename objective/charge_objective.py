# objective/charge_objective.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from policies.pw_current_fixed import build_piecewise_current_A
from battery_sim.spme_runner import run_spme_charge, ChargeResult
from objective.chebyshev_scalarizer import ChebyshevScalarizer

@dataclass
class ChargeObjectiveConfig:
    K: int = 3
    seg_duration_s: float = 300.0
    j_min: float = 10.0
    j_max: float = 60.0
    weights: tuple[float, float, float] = (1/3, 1/3, 1/3)
    eta: float = 1e-3
    with_aging: bool = True
    base_penalty: float = 1000.0
    target_soc: float = 0.8
    init_points: int = 5   # 用于冻结归一化范围的初始样本数量

class ChargeObjective:
    def __init__(self, cfg: ChargeObjectiveConfig):
        self.cfg = cfg
        self.scalarizer = ChebyshevScalarizer(weights=cfg.weights, eta=cfg.eta)

    def _clip_j(self, j: np.ndarray) -> np.ndarray:
        return np.clip(j, self.cfg.j_min, self.cfg.j_max)

    def evaluate_params(self, j_segments: np.ndarray, update_scalarizer: bool = True) -> tuple[float, dict]:
        j_segments = self._clip_j(np.asarray(j_segments, dtype=float))
        t_knots, I_segments_A = build_piecewise_current_A(j_segments, seg_duration_s=self.cfg.seg_duration_s)

        # ★ 关键：仿真时长 = 协议总时长 + 缓冲，避免永远跑不满导致 SOC 不达标
        total_T = float(self.cfg.seg_duration_s) * float(self.cfg.K)
        t_end_max = total_T + 300.0  # 额外 5 分钟缓冲

        result: ChargeResult = run_spme_charge(
            piecewise_current_A=(t_knots, I_segments_A),
            t_end_max=t_end_max,
            with_aging=self.cfg.with_aging,
            diagnose=False
        )

        info = {
            "feasible": result.feasible,
            "reason": result.reason,
            "metrics": {},
            "series": result
        }

        if not result.feasible or result.t_final is None:
            last_soc = result.soc[-1] if result.soc.size > 0 else 0.0
            soc_shortfall = max(0.0, self.cfg.target_soc - last_soc)
            penalty = self.cfg.base_penalty + soc_shortfall * 5000
            info["metrics"] = {"t_final": np.inf, "T_peak": np.inf, "aging": np.inf}
            # 调试打印（可注释）
            print(f"[PENALTY] reason={result.reason}, last_soc={last_soc:.3f}, penalty={penalty:.1f}")
            return -penalty, info

        f1 = result.t_final
        f2 = result.T_peak
        f3 = max(result.aging_final, 0.0)
        g = self.scalarizer.scalarize(f1, f2, f3, update=update_scalarizer)
        info["metrics"] = {"t_final": f1, "T_peak": f2, "aging": f3}
        return -g, info

    def build_bo_target(self):
        names = [f"j{k+1}" for k in range(self.cfg.K)]
        call_count = 0
        init_N = self.cfg.init_points

        def _target_func(**kwargs):
            nonlocal call_count
            j_vec = np.array([kwargs[name] for name in names], dtype=float)
            update_norm = True if call_count < init_N else False
            val, _ = self.evaluate_params(j_vec, update_scalarizer=update_norm)
            call_count += 1
            return val

        return _target_func, names
