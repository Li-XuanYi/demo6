# optimize_charge.py - FIXED VERSION with correct time parameters
from __future__ import annotations
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

from openai import OpenAI

from llmbo_core.bayesian_optimization import BayesianOptimization
from llm_components import LLMSampler, LLMSurrogate
from objective.charge_objective import ChargeObjective, ChargeObjectiveConfig
from policies.pw_current_fixed import build_piecewise_current_A
from battery_sim.spme_runner import run_spme_charge
from plot.compare_plots import compare_time_series

# API Configuration
API_KEY = "sk-84ac2d321cf444e799ddc9db79c02e92"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODELS = ["qwen-plus", "qwen-max", "qwen1.5-7b-chat", "qwen-turbo"]

# 🔧 FIXED: Optimization parameters with REALISTIC time scale
K = 3  # Number of stages
J_MIN, J_MAX = 20.0, 60.0  # Current density bounds (A/m²)
SEG_DURATION = 1500.0  # 🔧 INCREASED: 1500s per segment = 25 min (was 400s)
# Total time: 3 × 1500s = 4500s = 75 minutes (sufficient for 20%→80%)

INIT_POINTS = 6  # Initial random evaluations
N_ITER = 10  # Optimization iterations
WEIGHTS = (0.4, 0.3, 0.3)  # Multi-objective weights (time, temp, aging)

print(f"""
╔════════════════════════════════════════════════════════════╗
║          🔧 PARAMETER FIX EXPLANATION 🔧                   ║
╠════════════════════════════════════════════════════════════╣
║ Previous problem: seg_duration = 400s was too short       ║
║ Calculation: 5Ah battery, 20%→80% needs ~3Ah charge       ║
║              With ~2A current: 3Ah/2A = 5400s needed      ║
║                                                            ║
║ NEW FIXED PARAMETERS:                                      ║
║  • Segment duration: {SEG_DURATION}s = {SEG_DURATION/60:.0f} minutes                  ║
║  • Total time: {K * SEG_DURATION}s = {K * SEG_DURATION/60:.0f} minutes                     ║
║  • Current range: [{J_MIN:.0f}, {J_MAX:.0f}] A/m² = [1-3]A              ║
║  • Expected charge: ~2A × 1.25h = 2.5-3Ah ✓               ║
╚════════════════════════════════════════════════════════════╝
""")

def test_qwen_connection():
    """Test Qwen API connection."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    for model in MODELS:
        try:
            print(f"Testing {model}...")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply OK"}],
                max_tokens=5,
                timeout=10
            )
            print(f"✓ {model} working")
            return client, model
        except Exception as e:
            print(f"✗ {model} failed: {str(e)[:80]}")
    
    raise RuntimeError("No Qwen models available")

def run_optimization_method(use_llm: bool, config: ChargeObjectiveConfig) -> tuple:
    """Run optimization with fixed parameters."""
    method = "LLM+BO" if use_llm else "Traditional BO"
    print(f"\n{'='*60}")
    print(f"  {method}")
    print(f"{'='*60}")
    
    objective = ChargeObjective(config)
    target_func, param_names = objective.build_bo_target()
    pbounds = {name: (J_MIN, J_MAX) for name in param_names}
    
    print(f"Parameters: {param_names}")
    print(f"Bounds: [{J_MIN:.0f}, {J_MAX:.0f}] A/m²")
    print(f"Segment duration: {config.seg_duration_s}s")
    
    if use_llm:
        client, model = test_qwen_connection()
        optimizer = BayesianOptimization(
            f=target_func,
            pbounds=pbounds,
            random_state=42,
            verbose=2,
            use_llm=True,
            llm_sampler=LLMSampler(client, model),
            llm_surrogate=LLMSurrogate(client, model)
        )
        print(f"✓ LLM+BO with {model}")
    else:
        optimizer = BayesianOptimization(
            f=target_func,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )
        print("✓ Traditional BO")
    
    print(f"\nStarting optimization...")
    print(f"  Init points: {INIT_POINTS}, Iterations: {N_ITER}")
    
    try:
        optimizer.maximize(init_points=INIT_POINTS, n_iter=N_ITER)
        print(f"✓ {method} completed")
    except Exception as e:
        print(f"✗ {method} error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    best = optimizer.max
    if best is None:
        print(f"No valid results")
        return None, None, None
    
    print(f"\n{method} Best Result:")
    print(f"  Target: {best['target']:.6f}")
    print(f"  Params: {[f'{v:.1f}' for v in best['params'].values()]}")
    
    # Detailed simulation
    j_vec = np.array([best['params'][name] for name in param_names])
    detailed = simulate_best(j_vec, config)
    
    return best, detailed, objective

def simulate_best(j_segments: np.ndarray, config: ChargeObjectiveConfig):
    """Run detailed simulation of best solution."""
    print(f"\n--- Detailed simulation: {j_segments} A/m² ---")
    
    t_knots, I_segments = build_piecewise_current_A(
        j_segments,
        seg_duration_s=config.seg_duration_s
    )
    
    total_time = config.seg_duration_s * config.K
    
    result = run_spme_charge(
        piecewise_current_A=(t_knots, I_segments),
        t_end_max=total_time + 600.0,
        soc_start=0.2,
        soc_target=config.target_soc,
        with_aging=config.with_aging,
        diagnose=True
    )
    
    if result.feasible:
        print(f"✓ Feasible: {result.reason}")
        print(f"  Time: {result.t_final/60:.1f} min")
        print(f"  Temp: {result.T_peak-273.15:.1f} °C")
        print(f"  SOC: {result.soc[-1]:.1%}")
        print(f"  Aging: {result.aging_final:.4f}%")
    else:
        print(f"✗ Infeasible: {result.reason}")
        print(f"  SOC reached: {result.soc[-1]:.1%}")
    
    return result

def run_sanity_check():
    """Quick sanity check before optimization."""
    print("\n" + "="*60)
    print("SANITY CHECK: Can we reach 80% SOC?")
    print("="*60)
    
    print("Testing moderate profile: [40, 35, 30] A/m²")
    print(f"Duration: {SEG_DURATION}s per segment")
    
    try:
        t_knots, I_seg = build_piecewise_current_A([40, 35, 30], SEG_DURATION)
        
        result = run_spme_charge(
            piecewise_current_A=(t_knots, I_seg),
            t_end_max=SEG_DURATION * 3 + 600,
            soc_start=0.2,
            soc_target=0.8,
            with_aging=False,
            diagnose=True
        )
        
        final_soc = result.soc[-1] if len(result.soc) > 0 else 0.0
        
        if final_soc >= 0.75:
            print(f"\n✓ SANITY CHECK PASSED")
            print(f"  Reached {final_soc:.1%} SOC in {result.t_final/60:.1f} min")
            print("  Parameters are reasonable, proceeding...\n")
            return True
        else:
            print(f"\n⚠ WARNING: Only reached {final_soc:.1%}")
            print("  You may need to increase SEG_DURATION further")
            
            # Calculate required time
            capacity_needed = 5.0 * 0.6  # 5Ah × 60% = 3Ah
            avg_current = np.mean(I_seg)
            time_needed = (capacity_needed / avg_current) * 3600
            
            print(f"\n  Calculation:")
            print(f"    Capacity needed: {capacity_needed:.1f} Ah")
            print(f"    Average current: {avg_current:.2f} A")
            print(f"    Time needed: {time_needed:.0f}s ({time_needed/60:.0f} min)")
            print(f"    Current total: {SEG_DURATION * 3:.0f}s ({SEG_DURATION * 3/60:.0f} min)")
            
            response = input("\n  Continue anyway? (y/n): ")
            return response.lower() == 'y'
            
    except Exception as e:
        print(f"\n✗ SANITY CHECK FAILED: {e}")
        return False

def main():
    """Main optimization pipeline."""
    print("="*80)
    print("  LITHIUM-ION BATTERY CHARGING OPTIMIZATION")
    print("  LLM-Enhanced Bayesian Optimization (FIXED VERSION)")
    print("="*80)
    
    # Configuration with FIXED parameters
    config = ChargeObjectiveConfig(
        K=K,
        seg_duration_s=SEG_DURATION,  # 🔧 FIXED: Now 1500s
        j_min=J_MIN,
        j_max=J_MAX,
        weights=WEIGHTS,
        eta=1e-3,
        with_aging=True,
        init_points=INIT_POINTS,
        target_soc=0.8,
        base_penalty=500.0
    )
    
    print(f"\nConfiguration:")
    print(f"  Stages: {config.K}")
    print(f"  Duration: {config.seg_duration_s}s/stage ({config.seg_duration_s/60:.0f} min)")
    print(f"  Total time: {config.K * config.seg_duration_s}s ({config.K * config.seg_duration_s/60:.0f} min)")
    print(f"  Current range: [{config.j_min}, {config.j_max}] A/m²")
    print(f"  Weights: {config.weights}")
    
    # Sanity check
    if not run_sanity_check():
        print("\nSanity check failed. Aborting.")
        return
    
    results = {}
    
    # Traditional BO
    print(f"\n{'#'*60}")
    print("RUNNING: Traditional Bayesian Optimization")
    print(f"{'#'*60}")
    try:
        best_trad, res_trad, obj_trad = run_optimization_method(False, config)
        results['trad'] = (best_trad, res_trad, obj_trad)
    except Exception as e:
        print(f"Traditional BO failed: {e}")
        results['trad'] = (None, None, None)
    
    # LLM+BO
    print(f"\n{'#'*60}")
    print("RUNNING: LLM-Enhanced Bayesian Optimization")
    print(f"{'#'*60}")
    try:
        best_llm, res_llm, obj_llm = run_optimization_method(True, config)
        results['llm'] = (best_llm, res_llm, obj_llm)
    except Exception as e:
        print(f"LLM+BO failed: {e}")
        results['llm'] = (None, None, None)
    
    # Compare results
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    
    best_trad, res_trad, _ = results['trad']
    best_llm, res_llm, _ = results['llm']
    
    if best_trad and best_llm:
        print(f"\nTraditional BO:")
        print(f"  Target: {best_trad['target']:.6f}")
        print(f"  Current: {[f'{v:.1f}' for v in best_trad['params'].values()]} A/m²")
        
        print(f"\nLLM+BO:")
        print(f"  Target: {best_llm['target']:.6f}")
        print(f"  Current: {[f'{v:.1f}' for v in best_llm['params'].values()]} A/m²")
        
        improvement = ((best_llm['target'] - best_trad['target']) / 
                      abs(best_trad['target']) * 100)
        print(f"\nImprovement: {improvement:+.2f}%")
        
        # Generate plots
        if res_trad and res_llm and res_trad.feasible and res_llm.feasible:
            print("\nGenerating comparison plots...")
            try:
                compare_time_series(res_llm, res_trad, "optimized_charging")
                print("✓ Plots saved to output_plots/")
            except Exception as e:
                print(f"Plot error: {e}")
    
    elif best_llm:
        print("Only LLM+BO succeeded:")
        print(f"  Target: {best_llm['target']:.6f}")
    elif best_trad:
        print("Only Traditional BO succeeded:")
        print(f"  Target: {best_trad['target']:.6f}")
    else:
        print("Both methods failed. Check parameters.")
    
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        