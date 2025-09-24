# objective/chebyshev_scalarizer.py - FIXED VERSION
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class RunningMinMax:
    """Track min/max values for normalization."""
    f1_min: float = +np.inf
    f1_max: float = -np.inf
    f2_min: float = +np.inf
    f2_max: float = -np.inf
    f3_min: float = +np.inf
    f3_max: float = -np.inf
    
    def update(self, f1: float, f2: float, f3: float) -> None:
        """Update running min/max values."""
        self.f1_min = min(self.f1_min, f1)
        self.f1_max = max(self.f1_max, f1)
        self.f2_min = min(self.f2_min, f2)
        self.f2_max = max(self.f2_max, f2)
        self.f3_min = min(self.f3_min, f3)
        self.f3_max = max(self.f3_max, f3)
    
    def normalize(self, f1: float, f2: float, f3: float) -> tuple[float, float, float]:
        """Normalize values to [0, 1] range."""
        def norm(v, vmin, vmax):
            if abs(vmax - vmin) < 1e-12:
                return 0.5
            return (v - vmin) / (vmax - vmin)
        
        return (
            norm(f1, self.f1_min, self.f1_max),
            norm(f2, self.f2_min, self.f2_max),
            norm(f3, self.f3_min, self.f3_max)
        )

class ChebyshevScalarizer:
    """Chebyshev scalarization for multi-objective optimization."""
    
    def __init__(self, weights=(0.4, 0.3, 0.3), eta: float = 1e-3):
        self.w = np.array(weights, dtype=float)
        self.w = self.w / np.sum(self.w)  # Normalize weights
        self.eta = float(eta)
        self.running = RunningMinMax()
        self.n_updates = 0
    
    def scalarize(self, f1: float, f2: float, f3: float, update: bool = True) -> float:
        """
        Scalarize multiple objectives using Chebyshev method.
        
        f1: charging time (minutes) - minimize
        f2: peak temperature (Celsius) - minimize
        f3: aging percentage - minimize
        """
        if update:
            self.running.update(f1, f2, f3)
            self.n_updates += 1
        
        # Only normalize after we have enough samples
        if self.n_updates >= 3:
            f1n, f2n, f3n = self.running.normalize(f1, f2, f3)
        else:
            # Use raw values initially
            f1n = f1 / 60.0  # Normalize to ~[0, 1] range
            f2n = f2 / 50.0  # Assume max temp ~50°C
            f3n = f3 / 1.0   # Aging already in percentage
        
        # Chebyshev scalarization
        vals = np.array([f1n, f2n, f3n])
        chebyshev_term = np.max(self.w * vals)
        linear_term = self.eta * np.sum(self.w * vals)
        
        return float(chebyshev_term + linear_term)