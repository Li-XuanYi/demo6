# optimize_charge.py - FIXED VERSION
from __future__ import annotations
import numpy as np
from openai import OpenAI
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Set Chinese font
try:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
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

K = 3  # 3 segments
PB_MIN, PB_MAX = 10.0, 60.0  # Current density range (A/m^2)
INIT_POINTS = 5  # Initial random points
N_ITER = 15  # Optimization iterations
WEIGHTS = (0.4, 0.3, 0.3)  # Weights for objectives

def pick_qwen_model():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    for m in MODELS:
        try:
            print(f"Testing model: {m}...")
            _ = client.chat.completions.create(
                model=m, 
                messages=[{"role": "user", "content": "ping"}], 
                max_tokens=4
            )
            print(f"Model {m} connected successfully!")
            return client, m
        except Exception as e:
            print(f"Model {m} failed: {e}")
    raise RuntimeError("No Qwen models available")

def run_bo_once(use_llm: bool, objective: ChargeObjective):
    target_func, names = objective.build_bo_target()
    pbounds = {name: (PB_MIN, PB_MAX) for name in names}
    
    if use_llm:
        client, model = pick_qwen_model()
        opt = BayesianOptimization(
            f=target_func, 
            pbounds=pbounds, 
            random_state=42,  # Fixed seed for reproducibility
            verbose=2,
            use_llm=True, 
            llm_sampler=LLMSampler(client, model), 
            llm_surrogate=LLMSurrogate(client, model)
        )
    else:
        opt = BayesianOptimization(
            f=target_func, 
            pbounds=pbounds, 
            random_state=42, 
            verbose=2, 
            use_llm=False
        )

    print(f"\n--- Starting {'LLM+BO' if use_llm else 'Traditional BO'} ---")
    opt.maximize(init_points=INIT_POINTS, n_iter=N_ITER)
    print(f"--- {'LLM+BO' if use_llm else 'Traditional BO'} completed ---")
    return opt, names

def params_to_series(names: list[str], params: dict, cfg: ChargeObjectiveConfig):
    print(f"\nRe-simulating with optimal parameters: {params}")
    j_vec = np.array([params[name] for name in names])
    t_knots, I_segments_A = build_piecewise_current_A(
        j_vec, 
        seg_duration_s=cfg.seg_duration_s
    )
    total_T = float(cfg.seg_duration_s) * float(cfg.K)
    
    return run_spme_charge(
        piecewise_current_A=(t_knots, I_segments_A),
        t_end_max=total_T + 300.0,
        soc_start=0.2,
        soc_target=0.8,
        with_aging=cfg.with_aging,
        diagnose=True
    )

if __name__ == "__main__":
    # FIXED: Use 600s per segment for reasonable simulation
    cfg = ChargeObjectiveConfig(
        K=3,
        seg_duration_s=600.0,  # 10 minutes per segment (30 minutes total)
        j_min=10.0,
        j_max=60.0,
        weights=WEIGHTS,
        with_aging=True,
        init_points=INIT_POINTS,
        target_soc=0.8,
        base_penalty=1000.0
    )
    
    print("\n>>> Running diagnostic test with [30, 25, 20] A/m^2...")
    test_result = debug_run(
        j_segments_Apm2=[30, 25, 20], 
        seg_duration_s=cfg.seg_duration_s
    )
    print(">>> Diagnostic complete.")
    
    # Run LLM+BO
    print("\n==================== LLM+BO ====================")
    obj_llm = ChargeObjective(cfg)
    opt_llm, names = run_bo_once(use_llm=True, objective=obj_llm)
    best_llm = opt_llm.max
    
    # Run Traditional BO
    print("\n==================== Traditional BO ====================")
    obj_base = ChargeObjective(cfg)
    opt_base, _ = run_bo_once(use_llm=False, objective=obj_base)
    best_base = opt_base.max
    
    if best_llm and best_base:
        res_llm = params_to_series(names, best_llm["params"], cfg)
        res_base = params_to_series(names, best_base["params"], cfg)
        
        print("\n==================== RESULTS COMPARISON ====================")
        print(f"LLM+BO best: Target={-best_llm['target']:.4f}")
        print(f"  Parameters: {best_llm['params']}")
        print(f"  Metrics: Time={res_llm.t_final:.1f}s, "
              f"T_peak={res_llm.T_peak:.2f}K, "
              f"Aging={res_llm.aging_final:.4f}%")
        
        print(f"\nTraditional BO best: Target={-best_base['target']:.4f}")
        print(f"  Parameters: {best_base['params']}")
        print(f"  Metrics: Time={res_base.t_final:.1f}s, "
              f"T_peak={res_base.T_peak:.2f}K, "
              f"Aging={res_base.aging_final:.4f}%")
        
        compare_time_series(res_llm, res_base, save_prefix="compare")
        print("\nComparison plots saved.")
    else:
        print("\nOptimization failed to find valid solutions.")