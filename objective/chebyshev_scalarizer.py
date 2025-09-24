# objective/chebyshev_scalarizer.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class RunningMinMax:
    f1_min: float = +np.inf
    f1_max: float = -np.inf
    f2_min: float = +np.inf
    f2_max: float = -np.inf
    f3_min: float = +np.inf
    f3_max: float = -np.inf

    def update(self, f1: float, f2: float, f3: float) -> None:
        self.f1_min = min(self.f1_min, f1); self.f1_max = max(self.f1_max, f1)
        self.f2_min = min(self.f2_min, f2); self.f2_max = max(self.f2_max, f2)
        self.f3_min = min(self.f3_min, f3); self.f3_max = max(self.f3_max, f3)

    def normalize(self, f1: float, f2: float, f3: float) -> tuple[float, float, float]:
        def norm(v, vmin, vmax):
            if vmax <= vmin + 1e-12:
                return 0.5
            return (v - vmin) / (vmax - vmin)
        return (
            norm(f1, self.f1_min, self.f1_max),
            norm(f2, self.f2_min, self.f2_max),
            norm(f3, self.f3_min, self.f3_max),
        )

class ChebyshevScalarizer:
    def __init__(self, weights=(1/3, 1/3, 1/3), eta: float = 1e-3) -> None:
        self.w = np.array(weights, dtype=float) / np.sum(weights)
        self.eta = float(eta)
        self.running = RunningMinMax()

    def scalarize(self, f1: float, f2: float, f3: float, update: bool = True) -> float:
        if update:
            self.running.update(f1, f2, f3)
        f1n, f2n, f3n = self.running.normalize(f1, f2, f3)
        vals = np.array([f1n, f2n, f3n])
        return float(np.max(self.w * vals) + self.eta * np.sum(self.w * vals))
