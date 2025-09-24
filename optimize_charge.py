# demo6/optimize_charge.py (最终修正版)
from __future__ import annotations
import numpy as np
from openai import OpenAI
import inspect
import matplotlib

# 尝试设置中文字体
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    print("中文字体 'SimHei' 设置成功。")
except Exception:
    print("未能设置中文字体，图像标签可能显示不正常。")

from llmbo_core.bayesian_optimization import BayesianOptimization
from llm_components import LLMSampler, LLMSurrogate
from objective.charge_objective import ChargeObjective, ChargeObjectiveConfig
from policies.pw_current_fixed import build_piecewise_current_A
from battery_sim.spme_runner import run_spme_charge, debug_run
from plot.compare_plots import compare_time_series

# --- 配置 ---
API_KEY = "sk-84ac2d321cf444e799ddc9db79c02e92" # 请务必替换为您的真实密钥
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODELS = ["qwen-plus", "qwen-max", "qwen1.5-7b-chat", "qwen-turbo"] # 将plus和max放前面

K = 3  # 3段恒流
PB_MIN, PB_MAX = 10.0, 60.0  # 电流密度范围 (A/m^2)
INIT_POINTS = 5  # 初始随机点
N_ITER = 15      # 优化迭代次数
WEIGHTS = (0.4, 0.3, 0.3)  # 稍微侧重充电时间

def pick_qwen_model():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    for m in MODELS:
        try:
            print(f"尝试连接模型: {m}...")
            _ = client.chat.completions.create(model=m, messages=[{"role": "user", "content": "ping"}], max_tokens=4)
            print(f"模型 {m} 连接成功！将使用此模型。")
            return client, m
        except Exception as e:
            print(f"模型 {m} 连接失败: {e}")
    raise RuntimeError("所有Qwen模型均无法连接，请检查API key/网络。")

def run_bo_once(use_llm: bool, objective: ChargeObjective):
    target_func, names = objective.build_bo_target()
    pbounds = {name: (PB_MIN, PB_MAX) for name in names}
    
    if use_llm:
        client, model = pick_qwen_model()
        opt = BayesianOptimization(
            f=target_func, pbounds=pbounds, random_state=1, verbose=2,
            use_llm=True, llm_sampler=LLMSampler(client, model), llm_surrogate=LLMSurrogate(client, model)
        )
    else:
        opt = BayesianOptimization(
            f=target_func, pbounds=pbounds, random_state=1, verbose=2, use_llm=False
        )

    print(f"\n--- 开始 {'LLM+BO' if use_llm else '传统BO'} 优化 (init_points={INIT_POINTS}, n_iter={N_ITER}) ---")
    opt.maximize(init_points=INIT_POINTS, n_iter=N_ITER)
    print(f"--- {'LLM+BO' if use_llm else '传统BO'} 优化结束 ---")
    return opt, names



def params_to_series(names: list[str], params: dict, cfg: ChargeObjectiveConfig):
    print(f"\n正在使用最优参数重新仿真: {params}")
    import numpy as np
    from policies.pw_current_fixed import build_piecewise_current_A
    from battery_sim.spme_runner import run_spme_charge
    j_vec = np.array([params[name] for name in names])
    t_knots, I_segments_A = build_piecewise_current_A(j_vec, seg_duration_s=cfg.seg_duration_s)
    total_T = float(cfg.seg_duration_s) * float(cfg.K)
    return run_spme_charge(
        piecewise_current_A=(t_knots, I_segments_A),
        with_aging=cfg.with_aging,
        t_end_max=total_T + 300.0,  # ★ 与 BO 过程一致
        diagnose=True
    )

if __name__ == "__main__":
    cfg = ChargeObjectiveConfig(
        K=3,
        seg_duration_s=3600.0,  # ★ 每段1小时
        weights=WEIGHTS,
        with_aging=True,
        init_points=5          # 与 BO init_points 对齐
    )
    
    print("\n>>> 运行前诊断：使用一个中等策略 [30, 25, 20] A/m^2 进行测试...")
    debug_run(j_segments_Apm2=[30, 25, 20], seg_duration_s=cfg.seg_duration_s)
    print(">>> 诊断结束, 准备开始正式优化流程。")

    # 运行 LLM+BO
    print("\n==================== 运行 LLM+BO ====================")
    obj_llm = ChargeObjective(cfg)
    opt_llm, names = run_bo_once(use_llm=True, objective=obj_llm)
    best_llm = opt_llm.max

    # 运行 传统BO
    print("\n==================== 运行 传统BO ====================")
    obj_base = ChargeObjective(cfg)
    opt_base, _ = run_bo_once(use_llm=False, objective=obj_base)
    best_base = opt_base.max

    if best_llm and best_base:
        res_llm = params_to_series(names, best_llm["params"], cfg)
        res_base = params_to_series(names, best_base["params"], cfg)

        print("\n\n==================== 最终结果对比 ====================")
        print(f"LLM+BO 最优解: {best_llm}")
        print(f"传统BO 最优解: {best_base}")
        
        print(f"\nLLM+BO 最终性能指标: "
              f"充电时间={res_llm.t_final:.1f}s ({res_llm.t_final/60:.1f}min), "
              f"峰值温度={res_llm.T_peak:.2f}K, "
              f"老化={res_llm.aging_final:.4f}%, "
              f"可行性={res_llm.feasible} ({res_llm.reason})")
        
        print(f"传统BO 最终性能指标: "
              f"充电时间={res_base.t_final:.1f}s ({res_base.t_final/60:.1f}min), "
              f"峰值温度={res_base.T_peak:.2f}K, "
              f"老化={res_base.aging_final:.4f}%, "
              f"可行性={res_base.feasible} ({res_base.reason})")

        compare_time_series(res_llm, res_base, save_path="compare_curves.png")
        print("\n对比图像已保存至: compare_curves.png")
    else:
        print("\n优化过程未能找到有效解，无法进行对比和绘图。请检查仿真参数和LLM连接。")