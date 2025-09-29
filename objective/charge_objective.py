# objective/charge_objective.py - ROBUST VERSION
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from policies.pw_current_fixed import build_piecewise_current_A
from battery_sim.spme_runner import run_spme_charge, ChargeResult
from objective.chebyshev_scalarizer import ChebyshevScalarizer

@dataclass
class ChargeObjectiveConfig:
    K: int = 3
    seg_duration_s: float = 400.0  # Reduced for faster convergence
    j_min: float = 15.0  # More conservative minimum
    j_max: float = 55.0  # More conservative maximum
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
    eta: float = 1e-3
    with_aging: bool = True
    base_penalty: float = 500.0
    target_soc: float = 0.8
    init_points: int = 8

class ChargeObjective:
    def __init__(self, cfg: ChargeObjectiveConfig):
        self.cfg = cfg
        self.scalarizer = ChebyshevScalarizer(weights=cfg.weights, eta=cfg.eta)
        self.evaluation_count = 0
        self.best_so_far = float('inf')
        self.failure_count = 0
        self.max_failures = 20  # Abort if too many consecutive failures
        
        # Track objectives for analysis
        self.objective_history = []
        
        print(f"ChargeObjective initialized:")
        print(f"  Stages: {cfg.K}, Duration: {cfg.seg_duration_s}s/stage")
        print(f"  Current density: [{cfg.j_min}, {cfg.j_max}] A/m²")
        print(f"  Weights: {cfg.weights}, Base penalty: {cfg.base_penalty}")
    
    def _clip_j(self, j: np.ndarray) -> np.ndarray:
        """Clip current densities to valid range."""
        clipped = np.clip(j, self.cfg.j_min, self.cfg.j_max)
        if not np.array_equal(j, clipped):
            print(f"  Clipped j from {j} to {clipped}")
        return clipped
    
    def evaluate_params(self, j_segments: np.ndarray, update_scalarizer: bool = True) -> tuple[float, dict]:
        """Evaluate charging protocol with robust error handling."""
        self.evaluation_count += 1
        j_segments = self._clip_j(np.asarray(j_segments, dtype=float))
        
        print(f"\n{'='*60}")
        print(f"EVALUATION #{self.evaluation_count}")
        print(f"{'='*60}")
        print(f"Current densities: {j_segments} A/m²")
        
        # Build current profile
        try:
            t_knots, I_segments_A = build_piecewise_current_A(
                j_segments, 
                seg_duration_s=self.cfg.seg_duration_s
            )
            print(f"Current profile: {len(I_segments_A)} segments, "
                  f"total time {t_knots[-1]:.0f}s ({t_knots[-1]/60:.1f}min)")
        except Exception as e:
            print(f"✗ ERROR building current profile: {e}")
            self.failure_count += 1
            return -self.cfg.base_penalty * 2, {"feasible": False, "reason": "current_profile_error"}
        
        # Run simulation
        total_T = float(self.cfg.seg_duration_s * self.cfg.K)
        t_end_max = total_T + 300.0
        
        try:
            result: ChargeResult = run_spme_charge(
                piecewise_current_A=(t_knots, I_segments_A),
                t_end_max=t_end_max,
                soc_start=0.2,
                soc_target=self.cfg.target_soc,
                with_aging=self.cfg.with_aging,
                use_soft_constraints=True,  # ENABLE soft constraints for smooth limit handling
                diagnose=(self.evaluation_count <= 3)  # Detailed diagnostics for first few
            )
        except Exception as e:
            print(f"✗ SIMULATION ERROR: {e}")
            self.failure_count += 1
            
            if self.failure_count > self.max_failures:
                print(f"\n⚠ TOO MANY FAILURES ({self.failure_count}), aborting...")
                raise RuntimeError("Excessive simulation failures, check battery model setup")
            
            return -self.cfg.base_penalty * 2, {"feasible": False, "reason": f"simulation_error: {e}"}
        
        # Extract final SOC for validation
        final_soc = result.soc[-1] if result.soc.size > 0 else 0.0
        
        print(f"Simulation result:")
        print(f"  Feasible: {result.feasible}, Reason: {result.reason}")
        print(f"  Final SOC: {final_soc:.1%}, Target: {self.cfg.target_soc:.1%}")
        
        info = {
            "feasible": result.feasible,
            "reason": result.reason,
            "metrics": {},
            "series": result,
            "j_segments": j_segments.copy()
        }
        
        # Handle infeasible results
        if not result.feasible or result.t_final is None or np.isnan(result.t_final) or final_soc < 0.75:
            # Calculate adaptive penalty
            soc_gap = max(0, self.cfg.target_soc - final_soc)
            time_penalty = 1.0 if result.t_final is None or np.isnan(result.t_final) else min(result.t_final / 3600.0, 2.0)
            
            # Different penalties for different failure modes
            if result.reason == "voltage_limit_exceeded":
                penalty = self.cfg.base_penalty * 1.8
            elif result.reason == "temperature_limit_exceeded":
                penalty = self.cfg.base_penalty * 2.0
            elif "solver_error" in result.reason:
                penalty = self.cfg.base_penalty * 2.5
            else:
                penalty = self.cfg.base_penalty * (1.0 + 10.0 * soc_gap + time_penalty)
            
            self.failure_count += 1
            
            # Store metrics even for failures (for analysis)
            info["metrics"] = {
                "t_final": result.t_final if result.t_final and np.isfinite(result.t_final) else total_T,
                "T_peak": result.T_peak if result.T_peak and np.isfinite(result.T_peak) else 323.15,
                "aging": result.aging_final if result.aging_final and np.isfinite(result.aging_final) else 1.0,
                "soc_final": final_soc,
                "penalty": penalty
            }
            
            print(f"✗ Infeasible solution, penalty = {penalty:.2f}")
            print(f"  Failure count: {self.failure_count}")
            return -penalty, info
        
        # Reset failure counter on success
        self.failure_count = 0
        
        # Extract objectives (all should be minimized)
        f1 = result.t_final / 60.0  # Charging time in minutes
        f2 = result.T_peak - 273.15  # Peak temperature in Celsius
        f3 = max(result.aging_final * 100, 0.001)  # Aging percentage (avoid zero)
        
        # Validate extracted values
        if not all(np.isfinite([f1, f2, f3])):
            print(f"✗ ERROR: Non-finite objectives: f1={f1}, f2={f2}, f3={f3}")
            penalty = self.cfg.base_penalty * 1.5
            info["metrics"] = {"penalty": penalty}
            return -penalty, info
        
        print(f"✓ Feasible solution:")
        print(f"  f1 (time): {f1:.2f} min")
        print(f"  f2 (temp): {f2:.2f} °C")
        print(f"  f3 (aging): {f3:.4f} %")
        
        # Store for analysis
        self.objective_history.append((f1, f2, f3))
        
        # Scalarize objectives
        try:
            g = self.scalarizer.scalarize(f1, f2, f3, update=update_scalarizer)
            
            if not np.isfinite(g):
                print(f"✗ ERROR: Scalarization returned non-finite value: {g}")
                return -self.cfg.base_penalty, info
            
            print(f"  Scalarized objective: g = {g:.6f}")
            
        except Exception as e:
            print(f"✗ Scalarization error: {e}")
            return -self.cfg.base_penalty, info
        
        info["metrics"] = {
            "t_final": f1,
            "T_peak": f2,
            "aging": f3,
            "soc_final": final_soc,
            "scalarized": g
        }
        
        # Track best solution
        if g < self.best_so_far:
            improvement = ((self.best_so_far - g) / self.best_so_far * 100) if self.best_so_far < float('inf') else 0
            self.best_so_far = g
            print(f"\n{'*'*60}")
            print(f"★★★ NEW BEST SOLUTION ★★★")
            print(f"Score: {g:.6f} (improvement: {improvement:.1f}%)")
            print(f"Time: {f1:.1f} min, Temp: {f2:.1f}°C, Aging: {f3:.4f}%")
            print(f"Current profile: {j_segments}")
            print(f"{'*'*60}\n")
        
        # Return negative for BO maximization
        target_value = -g
        print(f"Target value for BO: {target_value:.6f}")
        print(f"{'='*60}\n")
        
        return target_value, info
    
    def build_bo_target(self):
        """Build target function for Bayesian Optimization."""
        param_names = [f"j{k+1}" for k in range(self.cfg.K)]
        
        def _target_func(**kwargs):
            """Wrapper function for BO with error handling."""
            try:
                # Extract current densities from kwargs
                j_vec = np.array([kwargs[name] for name in param_names], dtype=float)
                
                # Update scalarizer during initial exploration
                update_norm = self.evaluation_count < self.cfg.init_points
                
                # Evaluate
                val, info = self.evaluate_params(j_vec, update_scalarizer=update_norm)
                
                # Validate return value
                if not np.isfinite(val):
                    print(f"WARNING: Target function returned non-finite value: {val}")
                    return -self.cfg.base_penalty
                
                return val
                
            except KeyboardInterrupt:
                print("\n\nOptimization interrupted by user.")
                raise
            except Exception as e:
                print(f"\n✗ CRITICAL ERROR in target function: {e}")
                import traceback
                traceback.print_exc()
                return -self.cfg.base_penalty * 3
        
        return _target_func, param_names
    