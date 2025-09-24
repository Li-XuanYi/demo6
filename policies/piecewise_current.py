# policies/piecewise_current.py
# -*- coding: utf-8 -*-
"""
分段恒流策略（K段），每段电流上限 [30, 75] A/m^2
同时提供把 A/m^2 转成 A 的函数（用 Chen2020 的极片面积）
"""
from __future__ import annotations
import numpy as np
import pybamm


def area_from_params() -> float:
    """从 Chen2020 参数集中读取电极面积 [m2]。"""
    p = pybamm.ParameterValues("Chen2020")
    # 一般参数名为 "Electrode area [m2]"；若无，采用常见近似
    if "Electrode area [m2]" in p:
        return float(p["Electrode area [m2]"])
    # 兜底：常见 21700 圆柱单体极片面积数量级 ~ 0.01~0.1 m^2，取一个保守值
    return 0.05


def current_density_to_A(j_A_per_m2: float) -> float:
    """电流密度 [A/m^2] 转电流 [A]"""
    A = area_from_params()
    return float(j_A_per_m2 * A)


def build_piecewise_current_A(
    j_segments_Apm2: np.ndarray,
    seg_duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构造 K 段恒流（单位 A）:
    - 每段持续 seg_duration_s 秒
    - j_segments_Apm2: shape (K,) 每段的电流密度（A/m^2），保证在[30,75]
    返回:
    - t_knots: shape (K+1,)
    - I_segments_A: shape (K,)
    """
    K = len(j_segments_Apm2)
    t_knots = np.linspace(0.0, seg_duration_s * K, K + 1)
    I_segments_A = np.array([current_density_to_A(j) for j in j_segments_Apm2], dtype=float)
    return t_knots, I_segments_A
