# optimize_charge.py - COMPREHENSIVE FIX
from __future__ import annotations
import numpy as np
import sys
import os
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')

# Ensure matplotlib uses non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Configure matplotlib
try:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.facecolor'] = 'white'
except Exception:
    pass

from llmbo_core.bayesian_optimization import BayesianOptimization
from llm_components import LLMSampler, LLMSurrogate
from objective.charge_objective import ChargeObjective, ChargeObjectiveConfig
from policies.pw_current_fixed import build_piecewise_current_A
from battery_sim.spme_runner import run_spme_charge, debug_run
from plot.compare_plots import compare_time_series

# Configuration
API_KEY = "sk-84ac2d321cf444e799ddc9db79c02e92"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODELS = ["qwen-plus", "qwen-max", "qwen1.5-7b-chat", "qwen-turbo"]

# Optimization parameters
K = 3  # Number of charging stages
J_MIN, J_MAX = 15.0, 55.0  # Current density bounds (A/m^2) - more conservative
INIT_POINTS = 8  # Initial random evaluations
N_ITER = 12  # Optimization iterations  
WEIGHTS = (0.4, 0.3, 0.3)  # Multi-objective weights (time, temp, aging)

def test_qwen_connection():
    """Test connection to Qwen models and return working client."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    for model in MODELS:
        try:
            print(f"Testing {model}...")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello, respond with just 'OK'"}],
                max_tokens=5,
                timeout=10
            )
            response_text = response.choices[0].message.content.strip()
            print(f"✓ {model} working: '{response_text}'")
            return client, model
            
        except Exception as e:
            print(f"✗ {model} failed: {str(e)[:100]}")
    
    raise RuntimeError("No Qwen models available. Check API key and network.")

def run_optimization_method(use_llm: bool, config: ChargeObjectiveConfig) -> tuple:
    """Run single optimization method (LLM+BO or Traditional BO)."""
    method_name = "LLM+BO" if use_llm else "Traditional BO"
    print(f"\n{'='*20} {method_name} {'='*20}")
    
    # Create fresh objective instance
    objective = ChargeObjective(config)
    
    # Build target function and parameter bounds
    target_func, param_names = objective.build_bo_target()
    pbounds = {name: (J_MIN, J_MAX) for name in param_names}
    
    print(f"Parameter bounds: {pbounds}")
    print(f"Optimization config: init_points={INIT_POINTS}, n_iter={N_ITER}")
    
    # Create optimizer
    if use_llm:
        client, model = test_qwen_connection()
        optimizer = BayesianOptimization(
            f=target_func,
            pbounds=pbounds,
            random_state=42,  # Fixed seed for reproducibility
            verbose=2,
            use_llm=True,
            llm_sampler=LLMSampler(client, model),
            llm_surrogate=LLMSurrogate(client, model)
        )
        print(f"✓ LLM+BO optimizer created with model: {model}")
    else:
        optimizer = BayesianOptimization(
            f=target_func,
            pbounds=pbounds,
            random_state=42,
            verbose=2,
            use_llm=False
        )
        print("✓ Traditional BO optimizer created")
    
    # Run optimization
    print(f"\nStarting {method_name} optimization...")
    try:
        optimizer.maximize(init_points=INIT_POINTS, n_iter=N_ITER)
        print(f"✓ {method_name} optimization completed successfully")
    except Exception as e:
        print(f"ERROR during {method_name} optimization: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    # Extract results
    best_result = optimizer.max
    if best_result is None:
        print(f"No valid results from {method_name}")
        return None, None, None
    
    print(f"\n{method_name} Results:")
    print(f"  Best target value: {best_result['target']:.6f}")
    print(f"  Best parameters: {best_result['params']}")
    
    # Generate detailed simulation with best parameters
    j_vec = np.array([best_result['params'][name] for name in param_names])
    detailed_result = simulate_best_solution(j_vec, config)
    
    return best_result, detailed_result, objective

def simulate_best_solution(j_segments: np.ndarray, config: ChargeObjectiveConfig):
    """Simulate the best solution in detail for analysis."""
    print(f"\n--- Detailed simulation with j = {j_segments} ---")
    
    # Build current profile
    t_knots, I_segments_A = build_piecewise_current_A(
        j_segments, 
        seg_duration_s=config.seg_duration_s
    )
    
    total_time = float(config.seg_duration_s * config.K)
    
    # Run detailed simulation
    result = run_spme_charge(
        piecewise_current_A=(t_knots, I_segments_A),
        t_end_max=total_time + 300.0,
        soc_start=0.2,
        soc_target=config.target_soc,
        with_aging=config.with_aging,
        diagnose=True
    )
    
    # Report detailed metrics
    if result.feasible:
        print(f"✓ Solution feasible: {result.reason}")
        print(f"  Final time: {result.t_final:.1f}s ({result.t_final/60:.1f}min)")
        print(f"  Peak temperature: {result.T_peak:.2f}K ({result.T_peak-273.15:.1f}°C)")
        print(f"  Final SOC: {result.soc[-1]*100:.1f}%")
        print(f"  Battery aging: {result.aging_final:.4f}%")
    else:
        print(f"✗ Solution infeasible: {result.reason}")
        print(f"  Stopped at: {result.t_final:.1f}s, SOC: {result.soc[-1]*100:.1f}%")
    
    return result

def run_diagnostic_test(config: ChargeObjectiveConfig):
    """Run diagnostic test to verify system functionality."""
    print("\n" + "="*60)
    print("DIAGNOSTIC TEST")
    print("="*60)
    
    print("Testing basic simulation with moderate parameters: [30, 25, 20] A/m²")
    
    try:
        test_result = debug_run(
            j_segments_Apm2=[30, 25, 20],
            seg_duration_s=config.seg_duration_s
        )
        
        if test_result.feasible:
            print("✓ Basic simulation working correctly")
            print(f"  Test metrics: time={test_result.t_final:.1f}s, "
                  f"SOC={test_result.soc[-1]*100:.1f}%, "
                  f"T_peak={test_result.T_peak-273.15:.1f}°C")
        else:
            print(f"⚠ Basic simulation issues: {test_result.reason}")
        
        return True
        
    except Exception as e:
        print(f"✗ Diagnostic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main optimization pipeline with comprehensive error handling."""
    print("="*80)
    print("LITHIUM-ION BATTERY CHARGING OPTIMIZATION")
    print("LLM-Enhanced Bayesian Optimization with Chebyshev Scalarization")
    print("="*80)
    
    # Create configuration
    config = ChargeObjectiveConfig(
        K=K,
        seg_duration_s=400.0,  # Reduced to 400s per segment for faster simulation
        j_min=J_MIN,
        j_max=J_MAX,
        weights=WEIGHTS,
        eta=1e-3,
        with_aging=True,
        init_points=INIT_POINTS,
        target_soc=0.8,
        base_penalty=500.0  # Reasonable penalty
    )
    
    print(f"Configuration:")
    print(f"  Stages: {config.K}")
    print(f"  Duration per stage: {config.seg_duration_s}s")
    print(f"  Current density range: [{config.j_min}, {config.j_max}] A/m²")
    print(f"  Objective weights: {config.weights}")
    
    # Run diagnostic test
    if not run_diagnostic_test(config):
        print("Diagnostic test failed. Please check your setup.")
        return
    
    # Initialize results storage
    results = {}
    
    # Run Traditional BO
    print(f"\n{'='*25} TRADITIONAL BO {'='*25}")
    try:
        best_trad, result_trad, obj_trad = run_optimization_method(use_llm=False, config=config)
        results['traditional'] = (best_trad, result_trad, obj_trad)
    except Exception as e:
        print(f"Traditional BO failed: {e}")
        results['traditional'] = (None, None, None)
    
    # Run LLM+BO
    print(f"\n{'='*28} LLM+BO {'='*28}")
    try:
        best_llm, result_llm, obj_llm = run_optimization_method(use_llm=True, config=config)
        results['llm'] = (best_llm, result_llm, obj_llm)
    except Exception as e:
        print(f"LLM+BO failed: {e}")
        results['llm'] = (None, None, None)
    
    # Compare results
    print(f"\n{'='*25} RESULTS COMPARISON {'='*25}")
    
    trad_best, trad_result, trad_obj = results['traditional']
    llm_best, llm_result, llm_obj = results['llm']
    
    if trad_best and llm_best:
        print(f"Traditional BO:")
        print(f"  Target: {trad_best['target']:.6f}")
        print(f"  Parameters: {[f'{v:.1f}' for v in trad_best['params'].values()]}")
        
        print(f"LLM+BO:")
        print(f"  Target: {llm_best['target']:.6f}") 
        print(f"  Parameters: {[f'{v:.1f}' for v in llm_best['params'].values()]}")
        
        improvement = ((llm_best['target'] - trad_best['target']) / abs(trad_best['target']) * 100)
        print(f"LLM+BO improvement: {improvement:+.2f}%")
        
        # Generate comparison plots
        if trad_result and llm_result and trad_result.feasible and llm_result.feasible:
            print("\nGenerating comparison plots...")
            try:
                compare_time_series(llm_result, trad_result, save_prefix="optimization_results")
                print("✓ Plots generated successfully")
            except Exception as e:
                print(f"Plot generation failed: {e}")
        else:
            print("Cannot generate plots - insufficient feasible results")
    
    elif llm_best:
        print("Only LLM+BO produced valid results:")
        print(f"  Target: {llm_best['target']:.6f}")
        print(f"  Parameters: {[f'{v:.1f}' for v in llm_best['params'].values()]}")
    
    elif trad_best:
        print("Only Traditional BO produced valid results:")
        print(f"  Target: {trad_best['target']:.6f}")
        print(f"  Parameters: {[f'{v:.1f}' for v in trad_best['params'].values()]}")
    
    else:
        print("Both optimization methods failed to find valid solutions.")
        print("Consider:")
        print("  - Adjusting parameter bounds")
        print("  - Reducing segment duration")
        print("  - Checking battery model parameters")
    
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()