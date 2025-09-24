# llm_components.py - COMPREHENSIVE LLMBO IMPLEMENTATION
import re
import numpy as np
from typing import List, Dict, Optional, Tuple

class LLMSampler:
    """
    Enhanced LLM-based candidate generation following LLMBO principles.
    Implements physics-informed warm start and candidate sampling.
    """
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name
        self.domain_knowledge = self._initialize_domain_knowledge()
    
    def _initialize_domain_knowledge(self) -> Dict:
        """Initialize electrochemical domain knowledge for battery charging."""
        return {
            "physical_principles": [
                "Lower current densities reduce temperature rise but increase charging time",
                "Higher initial currents enable fast charging but must decrease to prevent overheating", 
                "Optimal protocols often use decreasing current stages (CC-CV behavior)",
                "Current density typically ranges 10-60 A/m² for safe lithium-ion charging",
                "Temperature must remain below 40°C (313K) to prevent thermal runaway",
                "Voltage should not exceed 4.1V to prevent lithium plating"
            ],
            "successful_patterns": [
                "Aggressive start, conservative finish (e.g., 50→35→25 A/m²)",
                "Moderate constant current (e.g., 35→35→30 A/m²)",
                "Conservative safe charging (e.g., 25→20→15 A/m²)"
            ],
            "parameter_correlations": [
                "High initial current → Lower subsequent currents needed",
                "Temperature rise correlates with current magnitude and duration",
                "Aging increases with both current level and temperature"
            ]
        }
    
    def generate_candidates(self, target_space, n_candidates: int = 15) -> List[Dict]:
        """Generate physics-informed candidate points using LLM reasoning."""
        param_names = list(target_space.keys)
        bounds = target_space.bounds
        n_params = len(param_names)
        
        print(f"\nLLM Sampler: Generating {n_candidates} candidates for {n_params} parameters")
        
        # Build comprehensive prompt with domain expertise
        prompt_sections = [
            "You are a leading expert in lithium-ion battery electrochemistry and optimization.",
            "Task: Design multi-stage constant current charging protocols to optimize:",
            "  • Minimize charging time (reach 80% SOC quickly)",
            "  • Minimize peak temperature (stay below 40°C)", 
            "  • Minimize battery aging (SEI layer growth)",
            "  • Respect safety limits: Voltage ≤ 4.1V, Temperature ≤ 40°C",
            "",
            "Physical Constraints and Knowledge:",
            *[f"  • {principle}" for principle in self.domain_knowledge["physical_principles"]],
            "",
            f"Decision Variables ({n_params} charging stages):"
        ]
        
        # Add parameter descriptions
        for i, name in enumerate(param_names):
            lb, ub = bounds[i, 0], bounds[i, 1]
            prompt_sections.append(
                f"  • {name}: Current density [A/m²] for stage {i+1}, range [{lb:.1f}, {ub:.1f}]"
            )
        
        # Add historical performance data
        history = target_space.res()
        if history:
            prompt_sections.extend([
                "",
                "Historical Performance Data (target = -scalarized_objective, higher is better):"
            ])
            
            # Show best and worst performers for learning
            sorted_history = sorted(history, key=lambda x: x['target'], reverse=True)
            for i, record in enumerate(sorted_history[:8]):  # Show top 8
                params_str = ", ".join([f"{v:.1f}" for v in record['params'].values()])
                rank_desc = "★ BEST" if i == 0 else "☆ GOOD" if i < 3 else "○ REF"
                prompt_sections.append(
                    f"  {rank_desc}: [{params_str}] → target = {record['target']:.4f}"
                )
            
            # Also show a few poor performers for contrast
            if len(sorted_history) > 10:
                prompt_sections.append("  Poor performers for reference:")
                for record in sorted_history[-3:]:
                    params_str = ", ".join([f"{v:.1f}" for v in record['params'].values()])
                    prompt_sections.append(
                        f"  ✗ POOR: [{params_str}] → target = {record['target']:.4f}"
                    )
        
        # Add generation instructions
        prompt_sections.extend([
            "",
            "Generate Instructions:",
            f"Based on electrochemical principles and historical data, suggest {n_candidates} diverse",
            "and physically sound combinations of current densities.",
            "",
            "Strategy Guidelines:",
            "1. Consider decreasing current patterns (CC-CV-like behavior)",
            "2. Balance aggressive charging with temperature management", 
            "3. Explore both conservative and aggressive approaches",
            "4. Ensure all values are within specified bounds",
            "",
            "Output Format: Provide exactly one parameter set per line as comma-separated numbers:",
            f"Example: {', '.join(['45.0'] * n_params)}",
            "",
            "Your Suggestions:"
        ])
        
        prompt = "\n".join(prompt_sections)
        
        # Call LLM with controlled parameters
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert battery engineer with deep knowledge of electrochemical optimization and charging protocols."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7,  # Balanced creativity and consistency
                top_p=0.9
            )
            content = completion.choices[0].message.content.strip()
            print(f"LLM response length: {len(content)} characters")
            
        except Exception as e:
            print(f"ERROR: LLM API call failed: {e}")
            return self._generate_fallback_candidates(bounds, n_candidates, param_names)
        
        # Parse LLM response
        candidates = self._parse_llm_response(content, param_names, bounds, n_candidates)
        
        print(f"Successfully parsed {len(candidates)} candidates from LLM")
        return candidates
    
    def _parse_llm_response(self, content: str, param_names: List[str], 
                           bounds: np.ndarray, n_candidates: int) -> List[Dict]:
        """Parse LLM response into valid parameter candidates."""
        candidates = []
        n_params = len(param_names)
        
        # Extract numeric sequences from each line
        for line_num, line in enumerate(content.splitlines(), 1):
            if line_num > n_candidates + 5:  # Stop if too many lines
                break
                
            # Clean the line and extract numbers
            clean_line = re.sub(r'^[^\d\-]*', '', line.strip())
            if not clean_line:
                continue
            
            # Extract all numbers from the line
            numbers = re.findall(r'-?\d+(?:\.\d+)?', clean_line)
            
            if len(numbers) != n_params:
                continue  # Skip lines with wrong number of parameters
            
            try:
                values = [float(x) for x in numbers]
                
                # Validate bounds
                valid = True
                for i, val in enumerate(values):
                    if not (bounds[i, 0] <= val <= bounds[i, 1]):
                        valid = False
                        break
                
                if valid:
                    candidate = {name: val for name, val in zip(param_names, values)}
                    candidates.append(candidate)
                    
            except (ValueError, IndexError):
                continue
        
        # If we didn't get enough candidates, fill with physics-informed defaults
        if len(candidates) < n_candidates // 2:
            additional = self._generate_physics_based_candidates(
                bounds, n_candidates - len(candidates), param_names
            )
            candidates.extend(additional)
        
        return candidates[:n_candidates]
    
    def _generate_physics_based_candidates(self, bounds: np.ndarray, 
                                         n_needed: int, param_names: List[str]) -> List[Dict]:
        """Generate physics-based candidates when LLM parsing fails."""
        candidates = []
        n_params = len(param_names)
        
        # Strategy 1: Decreasing current profiles (CC-CV like)
        for i in range(min(n_needed // 3, 5)):
            # Start high, decrease gradually
            start_current = np.random.uniform(bounds[0, 1] * 0.7, bounds[0, 1])
            decay_factor = 0.7 + 0.2 * np.random.random()
            
            values = []
            current_val = start_current
            for j in range(n_params):
                current_val = max(current_val * decay_factor, bounds[j, 0])
                values.append(np.clip(current_val, bounds[j, 0], bounds[j, 1]))
            
            candidate = {name: val for name, val in zip(param_names, values)}
            candidates.append(candidate)
        
        # Strategy 2: Moderate constant current
        for i in range(min(n_needed // 3, 3)):
            base_current = np.random.uniform(bounds[0, 0] + 0.3 * (bounds[0, 1] - bounds[0, 0]),
                                           bounds[0, 0] + 0.7 * (bounds[0, 1] - bounds[0, 0]))
            values = [base_current + np.random.uniform(-5, 5) for _ in range(n_params)]
            values = [np.clip(v, bounds[i, 0], bounds[i, 1]) for i, v in enumerate(values)]
            
            candidate = {name: val for name, val in zip(param_names, values)}
            candidates.append(candidate)
        
        # Strategy 3: Conservative safe charging
        remaining = n_needed - len(candidates)
        for i in range(remaining):
            values = [np.random.uniform(bounds[j, 0], bounds[j, 0] + 0.4 * (bounds[j, 1] - bounds[j, 0]))
                     for j in range(n_params)]
            candidate = {name: val for name, val in zip(param_names, values)}
            candidates.append(candidate)
        
        return candidates
    
    def _generate_fallback_candidates(self, bounds: np.ndarray, n_candidates: int, 
                                    param_names: List[str]) -> List[Dict]:
        """Generate fallback candidates when LLM completely fails."""
        print("Generating fallback candidates with physics-based heuristics")
        return self._generate_physics_based_candidates(bounds, n_candidates, param_names)


class LLMSurrogate:
    """
    Enhanced LLM-based candidate evaluation following LLMBO principles.
    Implements contextual reasoning and domain knowledge integration.
    """
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name
    
    def select_best_candidate(self, target_space, candidate_points: List[Dict]) -> Optional[Dict]:
        """Select the most promising candidate using LLM reasoning."""
        if not candidate_points:
            return None
            
        history = target_space.res()
        if not history:
            # No history available, select based on physics principles
            return self._select_by_physics_principles(candidate_points)
        
        print(f"\nLLM Surrogate: Evaluating {len(candidate_points)} candidates")
        
        # Build sophisticated evaluation prompt
        prompt_sections = [
            "You are a battery optimization expert analyzing charging strategies.",
            "Goal: Select the most promising charging protocol from candidates.",
            "Optimization targets: ⚡ Fast charging + 🌡️ Low temperature + 🔋 Minimal aging",
            "",
            "Historical Performance Analysis:"
        ]
        
        # Show performance trends
        sorted_history = sorted(history, key=lambda x: x['target'], reverse=True)
        for i, record in enumerate(sorted_history[:12]):
            params = list(record['params'].values())
            param_str = " → ".join([f"{p:.1f}" for p in params])
            
            # Analyze the pattern
            trend = "Decreasing" if len(params) > 1 and params[0] > params[-1] else \
                   "Increasing" if len(params) > 1 and params[0] < params[-1] else "Constant"
            level = "High" if np.mean(params) > 40 else "Medium" if np.mean(params) > 25 else "Low"
            
            status = "🏆 EXCELLENT" if i == 0 else "✅ GOOD" if i < 4 else "📊 REFERENCE"
            prompt_sections.append(
                f"  {status}: {param_str} | {trend}, {level} | Score: {record['target']:.4f}"
            )
        
        # Add candidate evaluation section
        prompt_sections.extend([
            "",
            "Candidate Strategies to Evaluate:",
            ""
        ])
        
        for j, candidate in enumerate(candidate_points, 1):
            params = list(candidate.values())
            param_str = " → ".join([f"{p:.1f}" for p in params])
            
            # Analyze candidate characteristics  
            trend = "Decreasing" if len(params) > 1 and params[0] > params[-1] else \
                   "Increasing" if len(params) > 1 and params[0] < params[-1] else "Constant"
            level = "High" if np.mean(params) > 40 else "Medium" if np.mean(params) > 25 else "Low"
            
            prompt_sections.append(f"  {j}. {param_str} | {trend}, {level}")
        
        # Add expert reasoning instructions
        prompt_sections.extend([
            "",
            "Expert Analysis Framework:",
            "• Historical patterns show which strategies perform best",
            "• Decreasing current profiles often balance speed and safety effectively", 
            "• High initial currents enable fast charging but require careful management",
            "• Temperature management becomes critical with aggressive protocols",
            "• Consider both immediate performance and long-term battery health",
            "",
            "Select the candidate number (1-{}) that best balances all objectives based on".format(len(candidate_points)),
            "the historical evidence and electrochemical principles.",
            "",
            "Your selection (number only):"
        ])
        
        prompt = "\n".join(prompt_sections)
        
        # Call LLM for candidate evaluation
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert battery engineer with extensive experience in charging optimization and electrochemical analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0.1  # Low temperature for consistent selection
            )
            content = completion.choices[0].message.content.strip()
            print(f"LLM selection response: '{content}'")
            
        except Exception as e:
            print(f"ERROR: LLM candidate selection failed: {e}")
            return self._select_fallback_candidate(candidate_points, history)
        
        # Parse selection
        match = re.search(r'\b(\d+)\b', content)
        if match:
            selection = int(match.group(1))
            if 1 <= selection <= len(candidate_points):
                selected = candidate_points[selection - 1]
                print(f"LLM selected candidate #{selection}: {selected}")
                return selected
        
        print("Failed to parse LLM selection, using fallback")
        return self._select_fallback_candidate(candidate_points, history)
    
    def _select_by_physics_principles(self, candidates: List[Dict]) -> Dict:
        """Select candidate based on physics principles when no history available."""
        print("No history available, selecting by physics principles")
        
        # Prefer decreasing current profiles as they mimic CC-CV behavior
        scores = []
        for candidate in candidates:
            params = list(candidate.values())
            
            # Score based on decreasing trend (CC-CV like)
            trend_score = 0
            if len(params) > 1:
                for i in range(len(params) - 1):
                    if params[i] >= params[i + 1]:
                        trend_score += 1
                trend_score /= (len(params) - 1)
            
            # Score based on reasonable starting current
            start_score = 1.0 if 25 <= params[0] <= 50 else 0.5
            
            # Combined physics score
            physics_score = 0.7 * trend_score + 0.3 * start_score
            scores.append(physics_score)
        
        best_idx = np.argmax(scores)
        selected = candidates[best_idx]
        print(f"Physics-based selection: {selected} (score: {scores[best_idx]:.3f})")
        return selected
    
    def _select_fallback_candidate(self, candidates: List[Dict], history: List) -> Dict:
        """Fallback candidate selection using simple heuristics."""
        if not history:
            return self._select_by_physics_principles(candidates)
        
        # Find the best historical pattern and select most similar candidate
        best_historical = max(history, key=lambda x: x['target'])
        best_params = list(best_historical['params'].values())
        
        # Calculate similarity to best historical solution
        similarities = []
        for candidate in candidates:
            cand_params = list(candidate.values())
            # Normalized euclidean distance
            diff = np.array(cand_params) - np.array(best_params)
            similarity = 1 / (1 + np.linalg.norm(diff) / len(diff))
            similarities.append(similarity)
        
        best_idx = np.argmax(similarities)
        selected = candidates[best_idx]
        print(f"Fallback selection (most similar to best historical): {selected}")
        return selected