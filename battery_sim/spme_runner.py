# demo6/battery_sim/spme_runner.py
from __future__ import annotations
import numpy as np
import pybamm

try:
    from scipy.integrate import cumulative_trapezoid as cumtrap
except ImportError:
    def cumtrap(y, x, initial=0.0):
        y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
        if y.size < 2: return np.array([initial], dtype=float)
        dx = np.diff(x); mid = 0.5 * (y[:-1] + y[1:]) * dx
        out = np.empty(y.size, dtype=float); out[0] = initial; out[1:] = np.cumsum(mid)
        return out

class ChargeResult:
    def __init__(self, feasible: bool, reason: str, t, V, T, I, soc, aging_metric,
                 t_final: float | None, T_peak: float | None, aging_final: float | None):
        self.feasible = feasible; self.reason = reason
        self.t = np.array(t) if t is not None and len(t) > 0 else np.array([0.])
        self.V = np.array(V) if V is not None and len(V) > 0 else np.array([0.])
        self.T = np.array(T) if T is not None and len(T) > 0 else np.array([0.])
        self.I = np.array(I) if I is not None and len(I) > 0 else np.array([0.])
        self.soc = np.array(soc) if soc is not None and len(soc) > 0 else np.array([0.])
        self.aging = np.array(aging_metric) if aging_metric is not None and len(aging_metric) > 0 else np.array([0.])
        self.t_final = float(t_final) if t_final is not None else np.nan
        self.T_peak = float(T_peak) if T_peak is not None else np.nan
        self.aging_final = float(aging_final) if aging_final is not None else np.nan

def build_current_interp(times: np.ndarray, currents_A: np.ndarray, extend_to: float | None = None):
    times = np.asarray(times, dtype=float); currents_A = np.asarray(currents_A, dtype=float)
    K = len(currents_A)
    if len(times) != K + 1: raise ValueError("时间节点数量必须比电流段数量多一个")
    if not np.all(np.diff(times) > 1e-9): raise ValueError("时间节点必须严格递增")
    total_T = times[-1] - times[0]; eps = max(1e-9 * max(total_T, 1.0), 1e-12)
    t_grid, i_grid = [times[0]], [currents_A[0]]
    for k in range(1, K):
        t_before = max(times[k] - eps, t_grid[-1] + eps)
        t_grid.extend([t_before, times[k]])
        i_grid.extend([currents_A[k - 1], currents_A[k]])
    t_grid.append(times[-1] if times[-1] > t_grid[-1] else t_grid[-1] + eps)
    i_grid.append(currents_A[-1])
    if extend_to is not None and extend_to > t_grid[-1]:
        t_grid.append(extend_to); i_grid.append(currents_A[-1])
    return pybamm.Interpolant(np.array(t_grid), np.array(i_grid), pybamm.t)

def _set_initial_soc(params: pybamm.ParameterValues, soc_start: float, diagnose: bool):
    """
    兼容不同参数集的 SOC 键名，把初始 SOC 确认写进去。
    """
    cand_keys = [
        "Initial State of Charge",      # PyBaMM 常用
        "Initial SoC",
        "Initial state of charge",
        "Initial SOC",
    ]
    used_key = None
    for k in cand_keys:
        if k in params:
            params.update({k: float(soc_start)}, check_already_exists=False)
            used_key = k
            break
    if used_key is None:
        # 新建一个 PyBaMM 能识别的键名（多数模型用这个）
        params.update({"Initial State of Charge": float(soc_start)}, check_already_exists=False)
        used_key = "Initial State of Charge"
    if diagnose:
        print(f"[DIAG] 初始SOC已写入参数: '{used_key}' = {soc_start:.3f}")

def run_spme_charge(
    piecewise_current_A: tuple[np.ndarray, np.ndarray],
    t_end_max: float | None = None,       # ← 允许 None，自动对齐协议总时长
    soc_start: float = 0.2,
    soc_target: float = 0.8,
    v_lim: float = 4.1,
    T_lim: float = 313.15,
    with_aging: bool = False,
    diagnose: bool = False,
) -> ChargeResult:
    options = {"thermal": "lumped", "SEI": "ec reaction limited" if with_aging else "none"}
    model = pybamm.lithium_ion.SPMe(options)
    params = pybamm.ParameterValues("Chen2020")

    t_knots, I_segments = piecewise_current_A
    # PyBaMM 充电约定：放电为正，充电为负
    current_fun = build_current_interp(t_knots, -np.abs(I_segments),
                                       extend_to=t_knots[-1] if t_end_max is None else t_end_max)
    # ★★ 先把初始 SOC 用正确的键名写进去，然后再 process_model
    params.update({"Current function [A]": current_fun}, check_already_exists=False)
    _set_initial_soc(params, soc_start, diagnose=diagnose)

    geometry = model.default_geometry
    params.process_model(model)
    params.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    # ★ 如果未显式给 t_end_max，则用协议总时长；并给一点缓冲
    if t_end_max is None:
        total_T = float(t_knots[-1] - t_knots[0])
        t_end_max = total_T + 300.0

    solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-6, dt_max=20.0)
    t_eval = np.linspace(0.0, t_end_max, 600)
    solution = None
    try:
        if diagnose: print("[DIAG] 正在调用 CasadiSolver...")
        solution = solver.solve(model, t_eval)
    except Exception as e1:
        if diagnose: print(f"[DIAG] CasadiSolver 失败: {e1}. 回退 ScipySolver…")
        try:
            solver2 = pybamm.ScipySolver(atol=1e-6, rtol=1e-6)
            solution = solver2.solve(model, t_eval)
        except Exception as e2:
            if diagnose: print(f"[DIAG] ScipySolver 同样失败: {e2}")
            return ChargeResult(False, f"solver_error: {e2}", None,None,None,None,None,None,None,None,None)

    if solution is None or solution.t is None or len(solution.t) < 2:
        return ChargeResult(False, "solver_returned_empty_solution", None,None,None,None,None,None,None,None,None)

    t = solution.t
    V = solution["Terminal voltage [V]"].entries
    T = solution["X-averaged cell temperature [K]"].entries
    I = solution["Current [A]"].entries

    Q_n = float(params["Nominal cell capacity [A.h]"])
    Ah_cum = cumtrap(-I, t, initial=0.0) / 3600.0
    soc = np.clip(soc_start + Ah_cum / Q_n, 0.0, 1.0)

    aging_pct = np.zeros_like(t)
    if with_aging and "Loss of lithium inventory [%]" in solution:
        aging_pct = solution["Loss of lithium inventory [%]"].entries

    # 后处理裁剪
    start_idx = 1
    idx_soc = np.where(soc[start_idx:] >= soc_target)[0] + start_idx
    idx_v = np.where(V[start_idx:] >= v_lim)[0] + start_idx
    idx_T = np.where(T[start_idx:] >= T_lim)[0] + start_idx

    cut_idx, reason, feasible = len(t) - 1, "ok", True
    if len(idx_v) > 0: cut_idx = min(cut_idx, idx_v[0]); reason = "voltage_limit_exceeded"; feasible = False
    if len(idx_T) > 0: cut_idx = min(cut_idx, idx_T[0]); reason = "temperature_limit_exceeded"; feasible = False
    if len(idx_soc) > 0:
        cut_idx = min(cut_idx, idx_soc[0])
    else:
        feasible = False
        if reason == "ok": reason = "did_not_reach_target_soc"

    sl = slice(0, cut_idx + 1)
    res_t, res_V, res_T, res_I, res_soc, res_aging = t[sl], V[sl], T[sl], I[sl], soc[sl], aging_pct[sl]

    t_final = res_t[-1] if res_t.size > 0 else 0.0
    T_peak = np.max(res_T) if res_T.size > 0 else np.nan
    aging_final = res_aging[-1] if res_aging.size > 0 else 0.0

    if diagnose:
        print(f"[DIAG] 仿真结果: feasible={feasible}, reason={reason}, "
              f"t_end={t_final:.1f}s, V_peak={np.max(res_V) if res_V.size>0 else np.nan:.3f}V, "
              f"T_peak={T_peak:.2f}K, SOC_last={res_soc[-1] if res_soc.size>0 else np.nan:.3f}")

    return ChargeResult(feasible, reason, res_t, res_V, res_T, res_I, res_soc, res_aging,
                        t_final, T_peak, aging_final)
def debug_run(j_segments_Apm2=None, seg_duration_s: float = 600.0):
    """
    调试函数：用给定的分段电流密度跑一次仿真。
    :param j_segments_Apm2: list/array of 电流密度 [A/m^2]
    :param seg_duration_s: 每段持续时间 [s]
    """
    import numpy as np
    from policies.pw_current_fixed import build_piecewise_current_A

    if j_segments_Apm2 is None:
        j_segments_Apm2 = [30, 25, 20]   # 默认测试值

    j_segments_Apm2 = np.asarray(j_segments_Apm2, dtype=float)
    t_knots, I_segments = build_piecewise_current_A(j_segments_Apm2, seg_duration_s)

    total_T = float(seg_duration_s) * len(j_segments_Apm2)
    res = run_spme_charge(
        (t_knots, I_segments),
        t_end_max=total_T + 300.0,
        soc_start=0.2,
        soc_target=0.8,
        with_aging=True,
        diagnose=True
    )

    print(f"[DEBUG_RUN] feasible={res.feasible}, reason={res.reason}, "
          f"t_end={res.t_final}, SOC_last={res.soc[-1] if res.soc.size>0 else None:.3f}")
    return res
