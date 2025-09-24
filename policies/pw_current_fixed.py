# policies/pw_current_fixed.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import numpy as np

# 可配置的极片有效面积 (m^2)，默认 0.05；如有更精确面积，可用环境变量覆盖：
#   Windows: set ELECTRODE_AREA_M2=0.065
#   Linux/Mac: export ELECTRODE_AREA_M2=0.065
_DEFAULT_AREA_M2 = float(os.getenv("ELECTRODE_AREA_M2", "0.05"))

def get_effective_area_m2() -> float:
    return _DEFAULT_AREA_M2

def current_density_to_A(j_A_per_m2: float, area_m2: float | None = None) -> float:
    A = get_effective_area_m2() if area_m2 is None else float(area_m2)
    return float(j_A_per_m2 * A)

def build_piecewise_current_A(
    j_segments_Apm2: np.ndarray,
    seg_duration_s: float,
    area_m2: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    输入：
      - j_segments_Apm2: shape (K,) 每段电流密度 [A/m^2]（外层会裁剪到 [30,75]）
      - seg_duration_s: 每段持续时间（秒）
    返回：
      - t_knots: shape (K+1,) 分段时间节点 (0 → K*seg_duration_s)
      - I_segments_A: shape (K,) 各段恒流 [A]
    """
    K = len(j_segments_Apm2)
    t_knots = np.linspace(0.0, seg_duration_s * K, K + 1)
    I_segments_A = np.array(
        [current_density_to_A(j, area_m2=area_m2) for j in j_segments_Apm2],
        dtype=float
    )
    return t_knots, I_segments_A
