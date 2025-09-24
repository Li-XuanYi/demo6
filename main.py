import os
import numpy as np
from openai import OpenAI

# Import our modules
from llmbo_core.bayesian_optimization import BayesianOptimization
from llm_components import LLMSampler, LLMSurrogate
from battery_model import simulate_spm

# Qwen API key and base URL for Alibaba Cloud (ensure the key is kept secure in practice)
API_KEY = "sk-84ac2d321cf444e799ddc9db79c02e92"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Initialize OpenAI-compatible client for Qwen
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Determine an available Qwen model to use
models_to_try = ["qwen-turbo", "qwen-plus", "qwen-max", "qwen1.5-7b-chat"]
chosen_model = None
for model_name in models_to_try:
    try:
        print(f"尝试调用模型: {model_name}")
        # Test a simple query to see if this model works
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=5
        )
        print(f"{model_name} 调用成功！回复: {response.choices[0].message.content}\n")
        chosen_model = model_name
        break
    except Exception as e:
        print(f"{model_name} 调用失败: {e}\n")
if chosen_model is None:
    raise RuntimeError("没有可用的 Qwen 模型，请检查 API 密钥和网络。")

# Define the parameter bounds for optimization (target parameters to calibrate)
pbounds = {
    "ambient_temperature_K": (290.0, 310.0),
    "cation_transference_number": (0.2, 0.6),
}



# Generate synthetic "target" data using default parameters (this simulates an experimental dataset)
print("正在生成目标数据…")
default_params = {"Ambient temperature [K]": 298.15, "Cation transference number": 0.4}
target_time, target_voltage = simulate_spm(default_params)

# Define the objective function for optimization
def target_func(ambient_temperature_K, cation_transference_number):
    # 这里用“干净键名”接参，再映射回 PyBaMM 期望的参数名
    sim_params = {
        "Ambient temperature [K]": float(ambient_temperature_K),
        "Cation transference number": float(cation_transference_number),
    }
    sim_time, sim_voltage = simulate_spm(sim_params)

    # 对齐到 target_time 并计算 RMSE（与你之前一致）
    if len(sim_time) != len(target_time):
        sim_voltage_aligned = np.interp(target_time, sim_time, sim_voltage, right=sim_voltage[-1])
    else:
        sim_voltage_aligned = sim_voltage

    error = np.sqrt(np.mean((sim_voltage_aligned - target_voltage) ** 2))
    return -error  # 仍然“最大化负误差”=“最小化误差”

# Set up Bayesian Optimization with LLM integration
use_llm = True  # set False if you want to run without LLM guidance
if use_llm:
    llm_sampler = LLMSampler(client, chosen_model)
    llm_surrogate = LLMSurrogate(client, chosen_model)
    optimizer = BayesianOptimization(f=target_func, pbounds=pbounds, random_state=0,
                                     verbose=2, use_llm=True,
                                     llm_sampler=llm_sampler, llm_surrogate=llm_surrogate)
else:
    optimizer = BayesianOptimization(f=target_func, pbounds=pbounds, random_state=0, verbose=2)

# Run the optimization process
optimizer.maximize(init_points=5, n_iter=15)

# Get and print the best result
best_result = optimizer.max
if best_result is not None:
    best_params = best_result["params"]
    best_error = -best_result["target"]
    print("\n贝叶斯优化完成！")
    print(f"最优参数: { {k: round(v,4) for k,v in best_params.items()} }, 最小误差: {best_error:.4f}")
else:
    print("\n优化未找到任何结果。")
