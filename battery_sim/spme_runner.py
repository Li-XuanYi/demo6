# battery_sim/spme_runner.py - COMPLETE FIX for interpolation bounds error
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


def set_initial_soc(params, soc_initial: float, diagnose: bool = False):
    """Set initial SOC by adjusting stoichiometry parameters."""
    if diagnose:
        print(f"[DIAG] Setting initial SOC to {soc_initial}")
    
    if not isinstance(soc_initial, (int, float)):
        raise TypeError("soc_initial must be numeric")
    
    if not (0.0 <= soc_initial <= 1.0):
        raise ValueError(f"soc_initial must be between 0 and 1, got {soc_initial}")
    
    try:
        c_n_max = params["Maximum concentration in negative electrode [mol.m-3]"]
        c_p_max = params["Maximum concentration in positive electrode [mol.m-3]"]
        
        if diagnose:
            print(f"[DIAG] Max concentrations: c_n_max={c_n_max}, c_p_max={c_p_max}")
        
        # Calculate stoichiometry based on SOC
        x_n_min, x_n_max = 0.01, 0.99
        x_p_min, x_p_max = 0.99, 0.01
        
        x_n = x_n_min + soc_initial * (x_n_max - x_n_min)
        x_p = x_p_max + soc_initial * (x_p_min - x_p_max)
        
        if diagnose:
            print(f"[DIAG] Calculated stoichiometries: x_n={x_n:.3f}, x_p={x_p:.3f}")
        
        params.update({
            "Initial concentration in negative electrode [mol.m-3]": c_n_max * x_n,
            "Initial concentration in positive electrode [mol.m-3]": c_p_max * x_p
        })
        
        if diagnose:
            print("[DIAG] Successfully updated initial concentrations")
            
    except KeyError as e:
        raise KeyError(f"Could not find required parameter: {e}")


def run_spme_charge(
    piecewise_current_A: tuple[np.ndarray, np.ndarray],
    t_end_max: float | None = None,
    soc_start: float = 0.2,
    soc_target: float = 0.8,
    v_lim: float = 4.1,
    T_lim: float = 313.15,
    with_aging: bool = False,
    diagnose: bool = False,
) -> ChargeResult:
    """
    Run SPMe charging simulation using PyBaMM Experiment framework.
    
    This approach completely avoids interpolation bounds errors by using
    event-based simulation instead of interpolant-based current functions.
    
    Args:
        piecewise_current_A: Tuple of (time_knots, current_segments)
            - time_knots: array of shape (K+1,) with stage boundaries in seconds
            - current_segments: array of shape (K,) with current in Amperes
        t_end_max: Maximum simulation time (not needed with Experiment, kept for compatibility)
        soc_start: Initial state of charge (0.0-1.0)
        soc_target: Target state of charge (0.0-1.0)
        v_lim: Maximum voltage limit [V]
        T_lim: Maximum temperature limit [K]
        with_aging: Whether to include SEI aging model
        diagnose: Print diagnostic information
        
    Returns:
        ChargeResult with simulation data and feasibility status
    """
    if diagnose:
        print("[DIAG] Starting SPMe charging simulation with Experiment framework")
    
    try:
        t_knots, I_segments = piecewise_current_A
        
        if diagnose:
            print(f"[DIAG] Time knots: {t_knots}")
            print(f"[DIAG] Current segments: {I_segments} A")
        
        # Build experiment string for multi-stage constant current
        experiment_steps = []
        durations = np.diff(t_knots)  # Duration of each stage
        
        for i, (current, duration) in enumerate(zip(I_segments, durations)):
            # PyBaMM uses positive current for charging
            current_abs = abs(float(current))
            duration_s = float(duration)
            
            # Build step with termination conditions
            # Use "Rest" instead of voltage limit to avoid initial condition issues
            step = f"Charge at {current_abs:.6f} A for {duration_s:.1f} seconds"
            experiment_steps.append(step)
            
            if diagnose:
                print(f"[DIAG] Stage {i+1}: {step}")
        
        # CRITICAL FIX: Wrap steps in parentheses to indicate a cycle
        # This tells PyBaMM these steps are part of a continuous protocol
        experiment = pybamm.Experiment(
            [tuple(experiment_steps)],  # Wrap in tuple to indicate a cycle
            termination=f"{v_lim} V"    # Use voltage as experiment termination
        )
        
        # Set up model with options
        options = {
            "thermal": "lumped",
            "SEI": "ec reaction limited" if with_aging else "none"
        }
        model = pybamm.lithium_ion.SPMe(options)
        params = pybamm.ParameterValues("Chen2020")
        
        # IMPORTANT: Do NOT manually set initial concentrations here
        # Let PyBaMM's simulation.solve(initial_soc=...) handle it correctly
        
        # CRITICAL: Use IDAKLUSolver which handles experiments better than CasadiSolver
        try:
            solver = pybamm.IDAKLUSolver(
                rtol=1e-6,
                atol=1e-8,
                root_tol=1e-6
            )
            solver_name = "IDAKLUSolver"
            if diagnose:
                print(f"[DIAG] Using {solver_name}")
        except:
            # Fallback to CasadiSolver with safe mode for experiments
            solver = pybamm.CasadiSolver(
                mode="safe",
                rtol=1e-5,
                atol=1e-7,
                dt_max=60
            )
            solver_name = "CasadiSolver"
            if diagnose:
                print(f"[DIAG] Using {solver_name} (IDAKLU not available)")
        
        # Create and run simulation
        simulation = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=params,
            solver=solver
        )
        
        if diagnose:
            print(f"[DIAG] Running simulation with initial SOC = {soc_start}")
        
        # Solve with initial SOC - PyBaMM will handle initial conditions correctly
        solution = simulation.solve(initial_soc=soc_start)
        
        if solution is None or not hasattr(solution, 't'):
            if diagnose:
                print("[DIAG] Simulation returned empty solution")
            return ChargeResult(
                False, "solver_returned_empty_solution",
                None, None, None, None, None, None, None, None, None
            )
        
        if diagnose:
            print(f"[DIAG] Simulation completed with {len(solution.t)} time points")
        
        # Extract solution data
        t = solution["Time [s]"].entries
        V = solution["Terminal voltage [V]"].entries
        T = solution["X-averaged cell temperature [K]"].entries
        I = solution["Current [A]"].entries
        
        # Calculate SOC from charge throughput
        Q_n = float(params["Nominal cell capacity [A.h]"])
        Ah_cum = cumtrap(-I, t, initial=0.0) / 3600.0
        soc = np.clip(soc_start + Ah_cum / Q_n, 0.0, 1.0)
        
        # Extract aging data
        aging_pct = np.zeros_like(t)
        if with_aging:
            try:
                if "Loss of lithium inventory [%]" in solution:
                    aging_pct = solution["Loss of lithium inventory [%]"].entries
                elif "Total lithium lost [mol]" in solution:
                    Q_Li_init = float(params["Initial lithium in negative electrode [mol]"])
                    Li_lost = solution["Total lithium lost [mol]"].entries
                    aging_pct = (Li_lost / Q_Li_init) * 100
            except:
                if diagnose:
                    print("[DIAG] Could not extract aging data")
        
        # Check feasibility conditions
        feasible = True
        reason = "completed_successfully"
        
        # Find where we reached target SOC
        idx_target_soc = np.where(soc >= soc_target)[0]
        
        # Check voltage violations
        idx_v_violation = np.where(V > v_lim)[0]
        
        # Check temperature violations
        idx_T_violation = np.where(T > T_lim)[0]
        
        # Determine cutoff index
        cutoff_idx = len(t) - 1
        
        if len(idx_v_violation) > 0:
            cutoff_idx = min(cutoff_idx, idx_v_violation[0])
            reason = "voltage_limit_exceeded"
            feasible = False
            if diagnose:
                print(f"[DIAG] Voltage limit exceeded at t={t[idx_v_violation[0]]:.1f}s")
        
        if len(idx_T_violation) > 0:
            cutoff_idx = min(cutoff_idx, idx_T_violation[0])
            reason = "temperature_limit_exceeded"
            feasible = False
            if diagnose:
                print(f"[DIAG] Temperature limit exceeded at t={t[idx_T_violation[0]]:.1f}s")
        
        if len(idx_target_soc) > 0:
            cutoff_idx = min(cutoff_idx, idx_target_soc[0])
            if reason == "completed_successfully":
                reason = "reached_target_soc"
            if diagnose:
                print(f"[DIAG] Reached target SOC at t={t[idx_target_soc[0]]:.1f}s")
        else:
            if reason == "completed_successfully":
                reason = "did_not_reach_target_soc"
                feasible = False
                if diagnose:
                    print(f"[DIAG] Did not reach target SOC. Final SOC: {soc[-1]:.3f}")
        
        # Slice results to cutoff point
        sl = slice(0, cutoff_idx + 1)
        res_t = t[sl]
        res_V = V[sl]
        res_T = T[sl]
        res_I = I[sl]
        res_soc = soc[sl]
        res_aging = aging_pct[sl]
        
        # Calculate final metrics
        t_final = res_t[-1] if len(res_t) > 0 else 0.0
        T_peak = np.max(res_T) if len(res_T) > 0 else np.nan
        aging_final = res_aging[-1] if len(res_aging) > 0 else 0.0
        
        if diagnose:
            final_soc = res_soc[-1] if len(res_soc) > 0 else 0.0
            print(f"[DIAG] Results: feasible={feasible}, reason={reason}")
            print(f"[DIAG] Final: t={t_final:.1f}s, V={res_V[-1]:.3f}V, "
                  f"T={T_peak:.1f}K ({T_peak-273.15:.1f}°C), SOC={final_soc:.3f}")
        
        return ChargeResult(
            feasible, reason,
            res_t, res_V, res_T, res_I, res_soc, res_aging,
            t_final, T_peak, aging_final
        )
        
    except Exception as e:
        if diagnose:
            print(f"[DIAG] Unexpected error in run_spme_charge: {e}")
            import traceback
            traceback.print_exc()
        
        return ChargeResult(
            False, f"unexpected_error: {str(e)}",
            None, None, None, None, None, None, None, None, None
        )


def debug_run(j_segments_Apm2=None, seg_duration_s: float = 600.0):
    """
    Debug function: run simulation with given current density segments.
    """
    from policies.pw_current_fixed import build_piecewise_current_A

    if j_segments_Apm2 is None:
        j_segments_Apm2 = [30, 25, 20]

    print(f"Debug run with j_segments: {j_segments_Apm2} A/m², duration: {seg_duration_s} s")
    
    j_segments_Apm2 = np.asarray(j_segments_Apm2, dtype=float)
    t_knots, I_segments = build_piecewise_current_A(j_segments_Apm2, seg_duration_s)

    total_T = float(seg_duration_s) * len(j_segments_Apm2)
    res = run_spme_charge(
        (t_knots, I_segments),
        t_end_max=total_T + 300.0,
        soc_start=0.2,
        soc_target=0.8,
        with_aging=False,
        diagnose=True
    )

    print(f"\n[DEBUG_RUN] feasible={res.feasible}, reason={res.reason}")
    if res.t_final and not np.isnan(res.t_final):
        final_soc = res.soc[-1] if res.soc.size > 0 else 0.0
        print(f"[DEBUG_RUN] t_end={res.t_final:.1f}s, SOC_last={final_soc:.3f}")
    else:
        print("[DEBUG_RUN] Simulation failed to complete")
    
    return res
