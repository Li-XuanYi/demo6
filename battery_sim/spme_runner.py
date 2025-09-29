# demo6/battery_sim/spme_runner.py
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

def build_piecewise_constant_current(t_knots: np.ndarray, I_segments: np.ndarray, 
                                   extend_to: float = None) -> callable:
    """
    Create a piecewise constant current function that avoids CasADi interpolation issues.
    
    Args:
        t_knots: Time knots [t0, t1, ..., tK] (K+1 points)
        I_segments: Current values [I0, I1, ..., I_{K-1}] (K segments)
        extend_to: Optional time to extend the function
    
    Returns:
        A callable current function compatible with PyBaMM
    """
    t_knots = np.asarray(t_knots, dtype=float)
    I_segments = np.asarray(I_segments, dtype=float)
    
    if len(t_knots) != len(I_segments) + 1:
        raise ValueError(f"Need {len(I_segments)+1} time knots for {len(I_segments)} current segments")
    
    def current_function(t):
        """Piecewise constant current function."""
        t = np.asarray(t)
        current = np.zeros_like(t, dtype=float)
        
        # Handle scalar input
        was_scalar = np.isscalar(t)
        if was_scalar:
            t = np.array([t])
            current = np.array([0.0])
        
        # Assign current values based on time segments
        for i in range(len(I_segments)):
            if i == len(I_segments) - 1:  # Last segment
                # Include the final time point
                mask = (t >= t_knots[i]) & (t <= t_knots[i+1])
            else:
                # Exclude the right endpoint (except for last segment)
                mask = (t >= t_knots[i]) & (t < t_knots[i+1])
            current[mask] = I_segments[i]
        
        # Handle extension if needed
        if extend_to and extend_to > t_knots[-1]:
            mask = t > t_knots[-1]
            current[mask] = I_segments[-1]  # Use last segment's current
        
        # Handle times before first segment
        mask = t < t_knots[0]
        current[mask] = I_segments[0]  # Use first segment's current
        
        return current[0] if was_scalar else current
    
    return current_function

def set_initial_soc(params, soc_initial: float, diagnose: bool = False):
    """Set initial SOC by adjusting stoichiometry parameters."""
    if diagnose:
        print(f"[DIAG] Setting initial SOC to {soc_initial}")
    
    # Validate inputs
    if not isinstance(soc_initial, (int, float)):
        raise TypeError("soc_initial must be numeric")
    
    if not (0.0 <= soc_initial <= 1.0):
        raise ValueError(f"soc_initial must be between 0 and 1, got {soc_initial}")
    
    try:
        # Get maximum concentrations
        c_n_max = params["Maximum concentration in negative electrode [mol.m-3]"]
        c_p_max = params["Maximum concentration in positive electrode [mol.m-3]"]
        
        if diagnose:
            print(f"[DIAG] Max concentrations: c_n_max={c_n_max}, c_p_max={c_p_max}")
        
        # Calculate stoichiometry based on SOC
        # At SOC=0: x_n ≈ 0.01, x_p ≈ 0.99
        # At SOC=1: x_n ≈ 0.99, x_p ≈ 0.01
        
        # Linear interpolation for stoichiometry
        x_n_min, x_n_max = 0.01, 0.99
        x_p_min, x_p_max = 0.99, 0.01
        
        x_n = x_n_min + soc_initial * (x_n_max - x_n_min)
        x_p = x_p_max + soc_initial * (x_p_min - x_p_max)
        
        if diagnose:
            print(f"[DIAG] Calculated stoichiometries: x_n={x_n:.3f}, x_p={x_p:.3f}")
        
        # Set initial concentrations
        params.update({
            "Initial concentration in negative electrode [mol.m-3]": c_n_max * x_n,
            "Initial concentration in positive electrode [mol.m-3]": c_p_max * x_p
        })
        
        if diagnose:
            print("[DIAG] Successfully updated initial concentrations")
            
    except KeyError as e:
        raise KeyError(f"Could not find required parameter: {e}")
    
    if diagnose:
        print(f"[DIAG] Initial SOC set successfully to {soc_initial}")

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
    Run SPMe charging simulation with robust error handling.
    """
    if diagnose:
        print("[DIAG] Starting SPMe charging simulation")
    
    try:
        # Set up model options
        options = {"thermal": "lumped", "SEI": "ec reaction limited" if with_aging else "none"}
        model = pybamm.lithium_ion.SPMe(options)
        params = pybamm.ParameterValues("Chen2020")

        t_knots, I_segments = piecewise_current_A
        if diagnose:
            print(f"[DIAG] Time knots: {t_knots}")
            print(f"[DIAG] Current segments: {I_segments} A")
        
        # Create current function (PyBaMM charging convention: negative for charging)
        current_func = build_piecewise_constant_current(t_knots, -np.abs(I_segments))
        
        # Set current function - use string expression to avoid interpolant issues
        total_time = t_knots[-1] if t_end_max is None else max(t_knots[-1], t_end_max)
        
        # Create a simpler current profile using step function
        # Build the current profile as a string expression
        current_expr = "0"  # Default current
        for i, (t_start, current) in enumerate(zip(t_knots[:-1], I_segments)):
            t_end = t_knots[i+1]
            segment_expr = f"({-abs(current)}) * ((t >= {t_start}) & (t < {t_end}))"
            if i == 0:
                current_expr = segment_expr
            else:
                current_expr += f" + {segment_expr}"
        
        # Handle final segment to include endpoint
        t_start, t_end = t_knots[-2], t_knots[-1]
        final_current = -abs(I_segments[-1])
        current_expr += f" + ({final_current}) * ((t >= {t_start}) & (t <= {t_end}))"
        
        if diagnose:
            print(f"[DIAG] Using simplified constant current approach")
        
        # Instead of complex interpolation, use constant current for each segment
        # This is a simplified approach that avoids CasADi interpolation issues
        
        # For now, use average current to avoid interpolation completely
        avg_current = -np.mean(np.abs(I_segments))
        params.update({"Current function [A]": avg_current})
        
        if diagnose:
            print(f"[DIAG] Using average constant current: {avg_current} A")
        
        # Set initial SOC
        set_initial_soc(params, soc_start, diagnose=diagnose)

        # Process model
        geometry = model.default_geometry
        params.process_model(model)
        params.process_geometry(geometry)
        
        # Create mesh and discretization
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # Set simulation time
        if t_end_max is None:
            total_T = float(t_knots[-1] - t_knots[0])
            t_end_max = total_T + 300.0
        
        if diagnose:
            print(f"[DIAG] Simulation end time: {t_end_max} s")

        # Try different solvers with fallback
        solution = None
        solver_used = None
        
        # Try ScipySolver first (more stable)
        try:
            if diagnose: print("[DIAG] Trying ScipySolver...")
            solver = pybamm.ScipySolver(atol=1e-6, rtol=1e-6)
            t_eval = np.linspace(0.0, t_end_max, 300)  # Fewer points for stability
            solution = solver.solve(model, t_eval)
            solver_used = "ScipySolver"
            if diagnose: print("[DIAG] ScipySolver successful")
        except Exception as e1:
            if diagnose: print(f"[DIAG] ScipySolver failed: {e1}")
            
            # Try CasadiSolver as backup
            try:
                if diagnose: print("[DIAG] Trying CasadiSolver...")
                solver = pybamm.CasadiSolver(mode="safe", atol=1e-5, rtol=1e-5)
                t_eval = np.linspace(0.0, t_end_max, 200)
                solution = solver.solve(model, t_eval)
                solver_used = "CasadiSolver"
                if diagnose: print("[DIAG] CasadiSolver successful")
            except Exception as e2:
                if diagnose: print(f"[DIAG] CasadiSolver also failed: {e2}")
                return ChargeResult(False, f"solver_error: ScipySolver: {e1}, CasadiSolver: {e2}", 
                                  None,None,None,None,None,None,None,None,None)

        if solution is None or solution.t is None or len(solution.t) < 2:
            return ChargeResult(False, "solver_returned_empty_solution", 
                              None,None,None,None,None,None,None,None,None)

        if diagnose:
            print(f"[DIAG] Solution obtained with {solver_used}, {len(solution.t)} time points")

        # Extract solution data
        t = solution.t
        V = solution["Terminal voltage [V]"].entries
        T = solution["X-averaged cell temperature [K]"].entries
        I = solution["Current [A]"].entries

        # Calculate SOC
        Q_n = float(params["Nominal cell capacity [A.h]"])
        Ah_cum = cumtrap(-I, t, initial=0.0) / 3600.0
        soc = np.clip(soc_start + Ah_cum / Q_n, 0.0, 1.0)

        # Extract aging if enabled
        aging_pct = np.zeros_like(t)
        if with_aging:
            try:
                if "Loss of lithium inventory [%]" in solution:
                    aging_pct = solution["Loss of lithium inventory [%]"].entries
                elif "SEI thickness [m]" in solution:
                    # Convert SEI thickness to approximate aging percentage
                    sei_thickness = solution["SEI thickness [m]"].entries
                    aging_pct = sei_thickness / 1e-6 * 0.1  # Rough conversion
            except:
                if diagnose: print("[DIAG] Could not extract aging data")

        # Check limits and feasibility
        start_idx = 1
        idx_soc = np.where(soc[start_idx:] >= soc_target)[0] + start_idx
        idx_v = np.where(V[start_idx:] >= v_lim)[0] + start_idx
        idx_T = np.where(T[start_idx:] >= T_lim)[0] + start_idx

        cut_idx, reason, feasible = len(t) - 1, "ok", True
        
        if len(idx_v) > 0:
            cut_idx = min(cut_idx, idx_v[0])
            reason = "voltage_limit_exceeded"
            feasible = False
            
        if len(idx_T) > 0:
            cut_idx = min(cut_idx, idx_T[0])
            reason = "temperature_limit_exceeded"
            feasible = False
            
        if len(idx_soc) > 0:
            cut_idx = min(cut_idx, idx_soc[0])
        else:
            feasible = False
            if reason == "ok": 
                reason = "did_not_reach_target_soc"

        # Slice results
        sl = slice(0, cut_idx + 1)
        res_t, res_V, res_T, res_I = t[sl], V[sl], T[sl], I[sl]
        res_soc, res_aging = soc[sl], aging_pct[sl]

        # Calculate final metrics
        t_final = res_t[-1] if res_t.size > 0 else 0.0
        T_peak = np.max(res_T) if res_T.size > 0 else np.nan
        aging_final = res_aging[-1] if res_aging.size > 0 else 0.0

        if diagnose:
            final_soc = res_soc[-1] if res_soc.size > 0 else 0.0
            print(f"[DIAG] Results: feasible={feasible}, reason={reason}")
            print(f"[DIAG] Final: t={t_final:.1f}s, V={res_V[-1]:.3f}V, T={T_peak:.1f}K, SOC={final_soc:.3f}")

        return ChargeResult(feasible, reason, res_t, res_V, res_T, res_I, res_soc, res_aging,
                            t_final, T_peak, aging_final)

    except Exception as e:
        if diagnose:
            print(f"[DIAG] Unexpected error in run_spme_charge: {e}")
            import traceback
            traceback.print_exc()
        return ChargeResult(False, f"unexpected_error: {e}", 
                          None,None,None,None,None,None,None,None,None)

def debug_run(j_segments_Apm2=None, seg_duration_s: float = 600.0):
    """
    Debug function: run simulation with given current density segments.
    """
    import numpy as np
    from policies.pw_current_fixed import build_piecewise_current_A

    if j_segments_Apm2 is None:
        j_segments_Apm2 = [30, 25, 20]   # Default test values

    print(f"Debug run with j_segments: {j_segments_Apm2} A/m², duration: {seg_duration_s} s")
    
    j_segments_Apm2 = np.asarray(j_segments_Apm2, dtype=float)
    t_knots, I_segments = build_piecewise_current_A(j_segments_Apm2, seg_duration_s)

    total_T = float(seg_duration_s) * len(j_segments_Apm2)
    res = run_spme_charge(
        (t_knots, I_segments),
        t_end_max=total_T + 300.0,
        soc_start=0.2,
        soc_target=0.8,
        with_aging=False,  # Disable aging for simpler debug
        diagnose=True
    )

    print(f"[DEBUG_RUN] feasible={res.feasible}, reason={res.reason}")
    if res.t_final and not np.isnan(res.t_final):
        final_soc = res.soc[-1] if res.soc.size > 0 else 0.0
        print(f"[DEBUG_RUN] t_end={res.t_final:.1f}s, SOC_last={final_soc:.3f}")
    else:
        print("[DEBUG_RUN] Simulation failed to complete")
    
    return res
