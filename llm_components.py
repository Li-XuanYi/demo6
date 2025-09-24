# demo6/llm_components.py
import re
import numpy as np

class LLMSampler:
    """使用 LLM 建议新的候选参数集 (LLMBO Warm Start & Candidate Sampling思想)"""
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    def generate_candidates(self, target_space, n_candidates: int = 10) -> list[dict]:
        param_names = list(target_space.keys)
        bounds = target_space.bounds

        # 构建一个更专业的Prompt，模仿LLMBO论文中的专家知识注入
        prompt_lines = [
            "You are an expert in electrochemistry and battery management systems.",
            "Your task is to assist in a Bayesian Optimization process to find the optimal multi-stage constant current charging strategy for a lithium-ion battery.",
            "A higher 'target' value is better, as it represents a better trade-off between multiple objectives.",
            "\n--- Optimization Objectives & Constraints ---",
            "1. Minimize Charging Time (to reach 80% SOC).",
            "2. Minimize Peak Temperature (must stay below 313.15 K).",
            "3. Minimize Battery Aging (SEI layer growth).",
            "4. Hard Constraint: Terminal voltage must not exceed 4.1V.",
            f"\n--- Decision Variables: Current Densities for {len(param_names)} stages ---"
        ]
        for i, name in enumerate(param_names):
            prompt_lines.append(f"- {name}: Current density for stage {i+1}, range [{bounds[i, 0]:.1f}, {bounds[i, 1]:.1f}] A/m^2.")

        # 添加历史数据
        if target_space.res():
            prompt_lines.append("\n--- Past Trials (higher target is better) ---")
            sorted_history = sorted(target_space.res(), key=lambda r: r['target'], reverse=True)
            for record in sorted_history[:10]:
                params_str = ", ".join(f"{k}={v:.2f}" for k, v in record["params"].items())
                prompt_lines.append(f"- Parameters: {params_str} -> Target: {record['target']:.4f}")

        prompt_lines.append(
            "\n--- Your Task ---"
            f"Based on the physical principles and past data, suggest {n_candidates} new, diverse, and physically plausible combinations of current densities ({', '.join(param_names)}) that could lead to a higher target value."
        )
        prompt_lines.append("Provide suggestions as comma-separated numeric values on separate lines, without any other text. For example:\n45.0, 50.0, 30.0")

        prompt = "\n".join(prompt_lines)

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant and an expert in battery modeling."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.8  # 提高温度以增加多样性
            )
            content = completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLMSampler 调用模型失败: {e}")
            return []

        # 解析LLM输出
        candidates = []
        for line in content.splitlines():
            line_clean = re.sub(r'^[^\d-]*', '', line).strip()
            if not line_clean: continue

            nums = re.findall(r"-?\d+(?:\.\d+)?", line_clean)
            if len(nums) != len(param_names): continue

            try:
                values = [float(x) for x in nums]
                cand = {name: val for name, val in zip(param_names, values)}
                
                # 严格检查边界
                out_of_bounds = any(
                    not (bounds[i, 0] <= values[i] <= bounds[i, 1])
                    for i in range(len(param_names))
                )
                if not out_of_bounds:
                    candidates.append(cand)
            except (ValueError, IndexError):
                continue

        return candidates[:n_candidates]


class LLMSurrogate:
    """使用 LLM 评估候选点并选择最佳 (LLMBO Enhanced Surrogate思想)"""
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    def select_best_candidate(self, target_space, candidate_points: list[dict]) -> dict | None:
        if not candidate_points: return None
        history = target_space.res()

        # 构建一个更结构化的Prompt
        prompt_lines = [
            "You are an expert battery optimization assistant.",
            "Your task is to analyze past experimental data and select the most promising new charging strategy from a list of candidates.",
            "The goal is to maximize a 'target' value, which represents the best balance of fast charging, low temperature, and minimal aging.",
            "\n--- Past Trials (higher target is better) ---"
        ]
        sorted_history = sorted(history, key=lambda r: r['target'], reverse=True)
        for i, record in enumerate(sorted_history[:15], start=1):
            params = record["params"]
            param_str = "; ".join([f"{name}={val:.2f}" for name, val in params.items()])
            prompt_lines.append(f"Trial {i}: {param_str} -> Target = {record['target']:.4f}")

        prompt_lines.append("\n--- Candidate Strategies to Evaluate ---")
        for j, cand in enumerate(candidate_points, start=1):
            param_str = "; ".join([f"{name}={val:.2f}" for name, val in cand.items()])
            prompt_lines.append(f"Candidate {j}: {param_str}")

        prompt_lines.append(
            "\n--- Your Decision ---"
            "Based on the past trials, which candidate number do you predict will achieve the highest target value? "
            "Consider the trade-offs: aggressive currents (high values) charge faster but increase temperature and aging. "
            "A good strategy often starts aggressively and then tapers down."
            "Respond with the number of the best candidate only (e.g., '3')."
        )
        prompt = "\n".join(prompt_lines)

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable battery modeling assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.0 # 需要确定性选择
            )
            content = completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLMSurrogate 调用模型失败: {e}")
            return None

        # 解析回应
        match = re.search(r"\d+", content)
        if not match: return None

        best_index = int(match.group())
        if not (1 <= best_index <= len(candidate_points)): return None

        return candidate_points[best_index - 1]