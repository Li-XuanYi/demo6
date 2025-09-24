# objective/charge_objective.py - FIXED VERSION
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from policies.pw_current_fixed import build_piecewise_current_A
from battery_sim.spme_runner import run_spme_charge, ChargeResult
from objective.chebyshev_scalarizer import ChebyshevScalarizer

@dataclass
class ChargeObjectiveConfig:
    K: int = 3
    seg_duration_s: float = 600.0  # 10 minutes per segment
    j_min: float = 10.0
    j_max: float = 60.0
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
    eta: float = 1e-3
    with_aging: bool = True
    base_penalty: float = 100.0
    target_soc: float = 0.8
    init_points: int = 5

class ChargeObjective:
    def __init__(self, cfg: ChargeObjectiveConfig):
        self.cfg = cfg
        self.scalarizer = ChebyshevScalarizer(weights=cfg.weights, eta=cfg.eta)
        self.evaluation_count = 0
        self.best_so_far = float('inf')
        
    def _clip_j(self, j: np.ndarray) -> np.ndarray:
        return np.clip(j, self.cfg.j_min, self.cfg.j_max)
    
    def evaluate_params(self, j_segments: np.ndarray, update_scalarizer: bool = True) -> tuple[float, dict]:
        self.evaluation_count += 1
        j_segments = self._clip_j(np.asarray(j_segments, dtype=float))
        
        # Build current profile
        t_knots, I_segments_A = build_piecewise_current_A(
            j_segments, 
            seg_duration_s=self.cfg.seg_duration_s
        )
        
        # Total simulation time with buffer
        total_T = float(self.cfg.seg_duration_s * self.cfg.K)
        t_end_max = total_T + 600.0  # 10 minute buffer
        
        # Run simulation
        result: ChargeResult = run_spme_charge(
            piecewise_current_A=(t_knots, I_segments_A),
            t_end_max=t_end_max,
            soc_start=0.2,
            soc_target=self.cfg.target_soc,
            with_aging=self.cfg.with_aging,
            diagnose=False
        )
        
        info = {
            "feasible": result.feasible,
            "reason": result.reason,
            "metrics": {},
            "series": result
        }
        
        # Check feasibility
        if not result.feasible or result.t_final is None:
            # Calculate penalty based on how far we are from target
            last_soc = result.soc[-1] if result.soc.size > 0 else 0.2
            soc_gap = self.cfg.target_soc - last_soc
            
            # Penalty proportional to SOC gap
            penalty = self.cfg.base_penalty * (1 + 10 * soc_gap)
            
            info["metrics"] = {
                "t_final": total_T,
                "T_peak": result.T_peak if result.T_peak else 323.15,
                "aging": 1.0,
                "soc_final": last_soc
            }
            
            return -penalty, info
        
        # Extract objectives
        f1 = result.t_final / 60.0  # Convert to minutes
        f2 = result.T_peak - 273.15  # Convert to Celsius
        f3 = result.aging_final * 100  # Convert to percentage
        
        # Scalarize objectives
        g = self.scalarizer.scalarize(f1, f2, f3, update=update_scalarizer)
        
        info["metrics"] = {
            "t_final": f1,
            "T_peak": f2,
            "aging": f3,
            "soc_final": result.soc[-1]
        }
        
        # Track best
        if g < self.best_so_far:
            self.best_so_far = g
            print(f"  New best! Score: {g:.4f}, Time: {f1:.1f}min, Temp: {f2:.1f}°C, Aging: {f3:.4f}%")
        
        return -g, info  # Negative for maximization
    
    def build_bo_target(self):
        """Build target function for Bayesian Optimization."""
        names = [f"j{k+1}" for k in range(self.cfg.K)]
        call_count = 0
        init_N = self.cfg.init_points
        
        def _target_func(**kwargs):
            nonlocal call_count
            j_vec = np.array([kwargs[name] for name in names], dtype=float)
            
            # Update scalarizer only during initial points
            update_norm = call_count < init_N
            val, info = self.evaluate_params(j_vec, update_scalarizer=update_norm)
            
            call_count += 1
            
            # Print progress
            if call_count % 5 == 0:
                print(f"  Evaluation {call_count}: Current={j_vec}, Score={-val:.4f}")
            
            return val
        
        return _target_func, names
    