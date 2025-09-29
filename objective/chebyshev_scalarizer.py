# objective/chebyshev_scalarizer.py - ROBUST NaN HANDLING
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
    
    f1_sum: float = 0.0
    f2_sum: float = 0.0
    f3_sum: float = 0.0
    count: int = 0
    
    # Track realistic expected ranges for battery charging
    # These serve as fallback when no data is available yet
    f1_expected_range: tuple[float, float] = (10.0, 60.0)  # 10-60 minutes
    f2_expected_range: tuple[float, float] = (25.0, 50.0)  # 25-50°C
    f3_expected_range: tuple[float, float] = (0.001, 2.0)  # 0.001-2% aging
    
    def update(self, f1: float, f2: float, f3: float) -> None:
        """Update running statistics with validation."""
        # Only update with finite values
        if np.isfinite(f1):
            self.f1_min = min(self.f1_min, f1)
            self.f1_max = max(self.f1_max, f1)
            self.f1_sum += f1
            self.count += 1
        
        if np.isfinite(f2):
            self.f2_min = min(self.f2_min, f2)
            self.f2_max = max(self.f2_max, f2)
            self.f2_sum += f2
        
        if np.isfinite(f3):
            self.f3_min = min(self.f3_min, f3)
            self.f3_max = max(self.f3_max, f3)
            self.f3_sum += f3
    
    def get_ranges(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Get (min, max) ranges for each objective with fallback."""
        # Use observed ranges if available, otherwise use expected ranges
        r1 = (self.f1_min, self.f1_max) if np.isfinite(self.f1_min) else self.f1_expected_range
        r2 = (self.f2_min, self.f2_max) if np.isfinite(self.f2_min) else self.f2_expected_range
        r3 = (self.f3_min, self.f3_max) if np.isfinite(self.f3_min) else self.f3_expected_range
        return (r1, r2, r3)
    
    def normalize(self, f1: float, f2: float, f3: float) -> tuple[float, float, float]:
        """Normalize objectives to [0, 1] range with robust NaN handling."""
        
        def safe_normalize(val: float, vmin: float, vmax: float, expected_range: tuple[float, float]) -> float:
            """Robust normalization with multiple fallback strategies."""
            # Check for NaN/infinite input
            if not np.isfinite(val):
                return 0.5  # Neutral value for invalid input
            
            # If we have valid observed range
            if np.isfinite(vmin) and np.isfinite(vmax):
                range_val = vmax - vmin
                if range_val > 1e-10:  # Sufficient range
                    val_clipped = np.clip(val, vmin, vmax)
                    return (val_clipped - vmin) / range_val
            
            # Fallback to expected range
            exp_min, exp_max = expected_range
            val_clipped = np.clip(val, exp_min, exp_max)
            return (val_clipped - exp_min) / (exp_max - exp_min)
        
        f1_norm = safe_normalize(f1, self.f1_min, self.f1_max, self.f1_expected_range)
        f2_norm = safe_normalize(f2, self.f2_min, self.f2_max, self.f2_expected_range)
        f3_norm = safe_normalize(f3, self.f3_min, self.f3_max, self.f3_expected_range)
        
        return f1_norm, f2_norm, f3_norm


class ChebyshevScalarizer:
    """
    Robust Chebyshev scalarization with NaN handling for multi-objective optimization.
    
    Implements the augmented Chebyshev method:
    g(f) = max_i{w_i * f_i_normalized} + eta * sum_i{w_i * f_i_normalized}
    
    Objectives (all to minimize):
    - f1: charging time (minutes)
    - f2: peak temperature (Celsius)
    - f3: aging percentage
    """
    
    def __init__(self, weights=(0.4, 0.3, 0.3), eta: float = 1e-3):
        # Normalize and validate weights
        weights_array = np.array(weights, dtype=float)
        if len(weights_array) != 3:
            raise ValueError("Exactly 3 weights required for 3 objectives")
        if np.any(weights_array <= 0):
            raise ValueError("All weights must be positive")
        if np.any(~np.isfinite(weights_array)):
            raise ValueError("All weights must be finite numbers")
        
        self.w = weights_array / np.sum(weights_array)  # Normalize to sum=1
        self.eta = float(eta)
        self.stats = RunningStats()
        self.n_updates = 0
        
        # Reference point (ideal objectives - all zeros after normalization)
        self.ideal_point = np.array([0.0, 0.0, 0.0])
        
        # Large penalty value for invalid inputs
        self.large_penalty = 1e6
        
        print(f"ChebyshevScalarizer initialized: weights={self.w}, eta={self.eta}")
    
    def scalarize(self, f1: float, f2: float, f3: float, update: bool = True) -> float:
        """
        Scalarize multiple objectives with robust NaN handling.
        
        Args:
            f1: charging time (minutes) - to minimize
            f2: peak temperature (Celsius) - to minimize
            f3: aging percentage - to minimize
            update: whether to update normalization statistics
            
        Returns:
            Scalarized objective value (lower is better), always finite
        """
        
        # CRITICAL: Check for NaN/infinite inputs FIRST
        inputs_valid = all(np.isfinite([f1, f2, f3]))
        
        if not inputs_valid:
            nan_objectives = []
            if not np.isfinite(f1): nan_objectives.append("f1 (time)")
            if not np.isfinite(f2): nan_objectives.append("f2 (temp)")
            if not np.isfinite(f3): nan_objectives.append("f3 (aging)")
            
            print(f"WARNING: Non-finite objectives detected: {', '.join(nan_objectives)}")
            print(f"  f1={f1}, f2={f2}, f3={f3}")
            print(f"  Returning large penalty: {self.large_penalty}")
            
            # Do NOT update stats with invalid values
            return self.large_penalty
        
        # Validate realistic ranges (sanity check)
        if f1 < 0 or f1 > 120:  # Time should be 0-120 minutes
            print(f"WARNING: Unrealistic time value: {f1} minutes")
            return self.large_penalty * 0.5
        
        if f2 < 0 or f2 > 100:  # Temperature should be 0-100°C
            print(f"WARNING: Unrealistic temperature: {f2}°C")
            return self.large_penalty * 0.5
        
        if f3 < 0 or f3 > 10:  # Aging should be 0-10%
            print(f"WARNING: Unrealistic aging: {f3}%")
            return self.large_penalty * 0.5
        
        # Update statistics with valid data
        if update:
            self.stats.update(f1, f2, f3)
            self.n_updates += 1
            
            if self.n_updates <= 5:
                print(f"Scalarizer update #{self.n_updates}: f=({f1:.2f}, {f2:.2f}, {f3:.4f})")
                if self.n_updates == 5:
                    ranges = self.stats.get_ranges()
                    print(f"Objective ranges after 5 samples: {ranges}")
        
        # Normalize objectives (always returns finite values due to fallback)
        f1_norm, f2_norm, f3_norm = self.stats.normalize(f1, f2, f3)
        
        # Verify normalization produced finite values
        if not all(np.isfinite([f1_norm, f2_norm, f3_norm])):
            print(f"ERROR: Normalization produced non-finite values!")
            print(f"  f_raw=({f1}, {f2}, {f3}) -> f_norm=({f1_norm}, {f2_norm}, {f3_norm})")
            return self.large_penalty
        
        # Apply Chebyshev scalarization formula
        f_norm = np.array([f1_norm, f2_norm, f3_norm])
        
        # Weighted deviations from ideal point (0,0,0)
        weighted_deviations = self.w * f_norm
        
        # Verify weighted deviations are finite
        if not np.all(np.isfinite(weighted_deviations)):
            print(f"ERROR: Weighted deviations contain non-finite values!")
            return self.large_penalty
        
        # Chebyshev term (minimax)
        chebyshev_term = np.max(weighted_deviations)
        
        # Linear term (weighted sum)
        linear_term = self.eta * np.sum(weighted_deviations)
        
        # Final scalarized value
        g = chebyshev_term + linear_term
        
        # Final validation
        if not np.isfinite(g):
            print(f"ERROR: Final scalarized value is non-finite: {g}")
            print(f"  chebyshev_term={chebyshev_term}, linear_term={linear_term}")
            return self.large_penalty
        
        # Debug output for early evaluations
        if self.n_updates <= 5 or self.n_updates % 5 == 0:
            print(f"  Scalarization: f_raw=({f1:.2f},{f2:.2f},{f3:.4f}) "
                  f"-> f_norm=({f1_norm:.3f},{f2_norm:.3f},{f3_norm:.3f}) -> g={g:.6f}")
        
        return float(g)
    
    def get_status(self) -> dict:
        """Get scalarizer status for debugging."""
        return {
            "n_updates": self.n_updates,
            "weights": self.w.tolist(),
            "eta": self.eta,
            "objective_ranges": self.stats.get_ranges() if self.n_updates > 0 else None,
            "large_penalty": self.large_penalty
        }
    