# objective/chebyshev_scalarizer.py - COMPREHENSIVE FIX
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class RunningStats:
    """Track min/max/mean values for robust normalization."""
    f1_min: float = +np.inf
    f1_max: float = -np.inf
    f2_min: float = +np.inf
    f2_max: float = -np.inf
    f3_min: float = +np.inf
    f3_max: float = -np.inf
    
    # Track means for better normalization
    f1_sum: float = 0.0
    f2_sum: float = 0.0
    f3_sum: float = 0.0
    count: int = 0
    
    def update(self, f1: float, f2: float, f3: float) -> None:
        """Update running statistics."""
        self.f1_min = min(self.f1_min, f1)
        self.f1_max = max(self.f1_max, f1)
        self.f2_min = min(self.f2_min, f2)
        self.f2_max = max(self.f2_max, f2)
        self.f3_min = min(self.f3_min, f3)
        self.f3_max = max(self.f3_max, f3)
        
        self.f1_sum += f1
        self.f2_sum += f2
        self.f3_sum += f3
        self.count += 1
    
    def get_ranges(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Get (min, max) ranges for each objective."""
        return (
            (self.f1_min, self.f1_max),
            (self.f2_min, self.f2_max),
            (self.f3_min, self.f3_max)
        )
    
    def get_means(self) -> tuple[float, float, float]:
        """Get mean values for each objective."""
        if self.count == 0:
            return 0.0, 0.0, 0.0
        return (
            self.f1_sum / self.count,
            self.f2_sum / self.count,
            self.f3_sum / self.count
        )
    
    def normalize(self, f1: float, f2: float, f3: float) -> tuple[float, float, float]:
        """Normalize objectives to [0, 1] range with robust handling."""
        
        def robust_normalize(val: float, vmin: float, vmax: float) -> float:
            """Robust normalization with fallback strategies."""
            if np.isinf(vmin) or np.isinf(vmax) or abs(vmax - vmin) < 1e-12:
                # No range observed yet or degenerate range
                return 0.5  # Neutral value
            
            # Clip to observed range and normalize
            val_clipped = np.clip(val, vmin, vmax)
            return (val_clipped - vmin) / (vmax - vmin)
        
        f1_norm = robust_normalize(f1, self.f1_min, self.f1_max)
        f2_norm = robust_normalize(f2, self.f2_min, self.f2_max)
        f3_norm = robust_normalize(f3, self.f3_min, self.f3_max)
        
        return f1_norm, f2_norm, f3_norm


class ChebyshevScalarizer:
    """
    Improved Chebyshev scalarization for multi-objective optimization.
    
    Implements the augmented Chebyshev method:
    g(f) = max_i{w_i * f_i_normalized} + eta * sum_i{w_i * f_i_normalized}
    
    Where:
    - f1: charging time (minutes) - minimize
    - f2: peak temperature (Celsius) - minimize  
    - f3: aging percentage - minimize
    """
    
    def __init__(self, weights=(0.4, 0.3, 0.3), eta: float = 1e-3):
        # Normalize and validate weights
        weights_array = np.array(weights, dtype=float)
        if len(weights_array) != 3:
            raise ValueError("Exactly 3 weights required for 3 objectives")
        if np.any(weights_array <= 0):
            raise ValueError("All weights must be positive")
        
        self.w = weights_array / np.sum(weights_array)  # Normalize to sum=1
        self.eta = float(eta)
        self.stats = RunningStats()
        self.n_updates = 0
        
        # Reference point (ideal objectives - all zeros after normalization)
        self.ideal_point = np.array([0.0, 0.0, 0.0])
        
        print(f"ChebyshevScalarizer initialized: weights={self.w}, eta={self.eta}")
    
    def scalarize(self, f1: float, f2: float, f3: float, update: bool = True) -> float:
        """
        Scalarize multiple objectives using augmented Chebyshev method.
        
        Args:
            f1: charging time (minutes) - to minimize
            f2: peak temperature (Celsius) - to minimize
            f3: aging percentage - to minimize
            update: whether to update normalization statistics
            
        Returns:
            Scalarized objective value (lower is better)
        """
        
        # Validate inputs
        if not all(np.isfinite([f1, f2, f3])):
            print(f"WARNING: Non-finite objective values: f1={f1}, f2={f2}, f3={f3}")
            return 1e6  # Large penalty for invalid values
        
        if update:
            self.stats.update(f1, f2, f3)
            self.n_updates += 1
            
            if self.n_updates <= 3:
                print(f"Scalarizer update #{self.n_updates}: f=({f1:.2f}, {f2:.2f}, {f3:.4f})")
                if self.n_updates == 3:
                    ranges = self.stats.get_ranges()
                    print(f"Initial objective ranges: {ranges}")
        
        # Choose normalization strategy based on available data
        if self.n_updates >= 5:  # Use statistical normalization
            f1_norm, f2_norm, f3_norm = self.stats.normalize(f1, f2, f3)
        else:  # Use heuristic normalization for initial points
            # Reasonable expected ranges for battery charging
            f1_norm = np.clip(f1 / 60.0, 0, 1)    # Expect 0-60 min charging
            f2_norm = np.clip((f2 - 25) / 25.0, 0, 1)  # Expect 25-50°C
            f3_norm = np.clip(f3 / 2.0, 0, 1)     # Expect 0-2% aging
        
        # Apply Chebyshev scalarization formula
        f_norm = np.array([f1_norm, f2_norm, f3_norm])
        
        # Weighted deviations from ideal point (0,0,0)
        weighted_deviations = self.w * f_norm
        
        # Chebyshev term (minimax)
        chebyshev_term = np.max(weighted_deviations)
        
        # Linear term (weighted sum)
        linear_term = self.eta * np.sum(weighted_deviations)
        
        # Final scalarized value
        g = chebyshev_term + linear_term
        
        # Debug output for early evaluations
        if self.n_updates <= 5 or self.n_updates % 10 == 0:
            print(f"  Scalarization detail: f_raw=({f1:.2f},{f2:.2f},{f3:.4f}) "
                  f"-> f_norm=({f1_norm:.3f},{f2_norm:.3f},{f3_norm:.3f}) -> g={g:.6f}")
        
        return float(g)
    
    def get_status(self) -> dict:
        """Get scalarizer status for debugging."""
        return {
            "n_updates": self.n_updates,
            "weights": self.w.tolist(),
            "eta": self.eta,
            "objective_ranges": self.stats.get_ranges() if self.n_updates > 0 else None,
            "objective_means": self.stats.get_means() if self.n_updates > 0 else None
        }