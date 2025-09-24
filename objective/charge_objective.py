# objective/charge_objective.py - COMPREHENSIVE FIX
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
    base_penalty: float = 1000.0  # Reduced penalty
    target_soc: float = 0.8
    init_points: int = 8  # More initial points for better scalarizer initialization

class ChargeObjective:
    def __init__(self, cfg: ChargeObjectiveConfig):
        self.cfg = cfg
        self.scalarizer = ChebyshevScalarizer(weights=cfg.weights, eta=cfg.eta)
        self.evaluation_count = 0
        self.best_so_far = float('inf')
        
        # Track objectives for debugging
        self.objective_history = []
        
    def _clip_j(self, j: np.ndarray) -> np.ndarray:
        return np.clip(j, self.cfg.j_min, self.cfg.j_max)
    
    def evaluate_params(self, j_segments: np.ndarray, update_scalarizer: bool = True) -> tuple[float, dict]:
        self.evaluation_count += 1
        j_segments = self._clip_j(np.asarray(j_segments, dtype=float))
        
        print(f"Eval #{self.evaluation_count}: j_segments = {j_segments}")
        
        # Build current profile
        try:
            t_knots, I_segments_A = build_piecewise_current_A(
                j_segments, 
                seg_duration_s=self.cfg.seg_duration_s
            )
            print(f"Current profile: t_knots = {t_knots}, I_segments = {I_segments_A}")
        except Exception as e:
            print(f"Error building current profile: {e}")
            return -self.cfg.base_penalty, {"feasible": False, "reason": "current_profile_error"}
        
        # Total simulation time with buffer
        total_T = float(self.cfg.seg_duration_s * self.cfg.K)
        t_end_max = total_T + 300.0  # 5 minute buffer
        
        # Run simulation
        try:
            result: ChargeResult = run_spme_charge(
                piecewise_current_A=(t_knots, I_segments_A),
                t_end_max=t_end_max,
                soc_start=0.2,
                soc_target=self.cfg.target_soc,
                with_aging=self.cfg.with_aging,
                diagnose=True  # Enable debugging
            )
        except Exception as e:
            print(f"Simulation error: {e}")
            return -self.cfg.base_penalty, {"feasible": False, "reason": "simulation_error"}
        
        # Extract final SOC for debugging
        final_soc = result.soc[-1] if result.soc.size > 0 else 0.0
        print(f"Simulation result: feasible={result.feasible}, reason={result.reason}, "
              f"final_SOC={final_soc:.3f}, t_final={result.t_final:.1f}s")
        
        info = {
            "feasible": result.feasible,
            "reason": result.reason,
            "metrics": {},
            "series": result,
            "j_segments": j_segments.copy()
        }
        
        # Check feasibility and extract objectives
        if not result.feasible or result.t_final is None or final_soc < 0.75:
            # More nuanced penalty calculation
            soc_gap = max(0, self.cfg.target_soc - final_soc)
            time_penalty = 1.0 if result.t_final is None else result.t_final / 3600.0
            
            # Penalty increases with SOC gap and simulation issues
            if result.reason == "voltage_limit_exceeded":
                penalty = self.cfg.base_penalty * 1.5
            elif result.reason == "temperature_limit_exceeded":
                penalty = self.cfg.base_penalty * 2.0
            else:
                penalty = self.cfg.base_penalty * (1 + 5 * soc_gap + time_penalty)
            
            info["metrics"] = {
                "t_final": result.t_final or total_T,
                "T_peak": result.T_peak or 323.15,
                "aging": result.aging_final or 1.0,
                "soc_final": final_soc,
                "penalty": penalty
            }
            
            print(f"  Infeasible: penalty = {penalty:.2f}")
            return -penalty, info
        
        # Extract objectives (all should be minimized)
        f1 = result.t_final / 60.0  # Charging time in minutes
        f2 = result.T_peak - 273.15  # Peak temperature in Celsius  
        f3 = result.aging_final * 100 if result.aging_final > 0 else 0.01  # Aging percentage
        
        print(f"  Objectives: t={f1:.2f}min, T={f2:.2f}°C, aging={f3:.4f}%")
        
        # Store objectives for analysis
        self.objective_history.append((f1, f2, f3))
        
        # Scalarize objectives using Chebyshev method
        try:
            g = self.scalarizer.scalarize(f1, f2, f3, update=update_scalarizer)
            print(f"  Scalarized objective: {g:.6f}")
        except Exception as e:
            print(f"Scalarization error: {e}")
            g = f1 + f2 + f3  # Fallback to simple sum
        
        info["metrics"] = {
            "t_final": f1,
            "T_peak": f2,
            "aging": f3,
            "soc_final": final_soc,
            "scalarized": g
        }
        
        # Track best solution
        if g < self.best_so_far:
            self.best_so_far = g
            print(f"  *** NEW BEST! Score: {g:.4f}, Time: {f1:.1f}min, Temp: {f2:.1f}°C, Aging: {f3:.4f}% ***")
        
        # Return negative for maximization (BO maximizes target)
        target_value = -g
        print(f"  Target value (for BO maximization): {target_value:.6f}")
        
        return target_value, info
    
    def build_bo_target(self):
        """Build target function for Bayesian Optimization with proper error handling."""
        param_names = [f"j{k+1}" for k in range(self.cfg.K)]
        call_count = [0]  # Use list for mutable reference
        
        def _target_func(**kwargs):
            call_count[0] += 1
            current_call = call_count[0]
            
            try:
                # Extract current densities
                j_vec = np.array([kwargs[name] for name in param_names], dtype=float)
                
                # Update scalarizer during initial exploration
                update_norm = current_call <= self.cfg.init_points
                
                # Evaluate
                val, info = self.evaluate_params(j_vec, update_scalarizer=update_norm)
                
                # Progress reporting
                if current_call % 3 == 0 or current_call <= 5:
                    print(f"\n=== EVALUATION {current_call} SUMMARY ===")
                    print(f"Current densities: {j_vec}")
                    print(f"Target value: {val:.6f}")
                    if "metrics" in info:
                        m = info["metrics"]
                        print(f"Metrics: {m}")
                    print("=" * 40)
                
                return val
                
            except Exception as e:
                print(f"ERROR in target function call {current_call}: {e}")
                import traceback
                traceback.print_exc()
                return -self.cfg.base_penalty * 2  # Large penalty for errors
        
        return _target_func, param_names