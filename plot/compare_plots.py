# plot/compare_plots.py - FIXED VERSION
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def compare_time_series(res_llm, res_base, save_prefix="compare"):
    """Generate comparison plots with proper initialization and data handling."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('LLM+BO vs Traditional BO Comparison', fontsize=16)
    
    # Convert time to minutes for better readability
    t_llm_min = res_llm.t / 60.0
    t_base_min = res_base.t / 60.0
    
    # Plot 1: Voltage
    ax = axes[0, 0]
    ax.plot(t_llm_min, res_llm.V, 'b-', label="LLM+BO", linewidth=2)
    ax.plot(t_base_min, res_base.V, 'r--', label="Traditional BO", linewidth=2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Terminal Voltage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Current
    ax = axes[0, 1]
    ax.plot(t_llm_min, -res_llm.I, 'b-', label="LLM+BO", linewidth=2)  # Negative for charging
    ax.plot(t_base_min, -res_base.I, 'r--', label="Traditional BO", linewidth=2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Charging Current")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: SOC
    ax = axes[0, 2]
    ax.plot(t_llm_min, res_llm.soc * 100, 'b-', label="LLM+BO", linewidth=2)
    ax.plot(t_base_min, res_base.soc * 100, 'r--', label="Traditional BO", linewidth=2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("SOC (%)")
    ax.set_title("State of Charge")
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Temperature
    ax = axes[1, 0]
    ax.plot(t_llm_min, res_llm.T - 273.15, 'b-', label="LLM+BO", linewidth=2)  # Convert to Celsius
    ax.plot(t_base_min, res_base.T - 273.15, 'r--', label="Traditional BO", linewidth=2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Cell Temperature")
    ax.axhline(y=40, color='k', linestyle=':', alpha=0.5, label='Limit (40°C)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Aging
    ax = axes[1, 1]
    ax.plot(t_llm_min, res_llm.aging, 'b-', label="LLM+BO", linewidth=2)
    ax.plot(t_base_min, res_base.aging, 'r--', label="Traditional BO", linewidth=2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Aging (%)")
    ax.set_title("Battery Aging (SEI Growth)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary metrics
    ax = axes[1, 2]
    ax.axis('off')
    
    # Create comparison table
    metrics_text = f"""
    Performance Comparison:
    
    LLM+BO:
    • Charging Time: {res_llm.t_final/60:.1f} min
    • Peak Temp: {res_llm.T_peak-273.15:.1f}°C
    • Final Aging: {res_llm.aging_final:.4f}%
    • Final SOC: {res_llm.soc[-1]*100:.1f}%
    
    Traditional BO:
    • Charging Time: {res_base.t_final/60:.1f} min
    • Peak Temp: {res_base.T_peak-273.15:.1f}°C
    • Final Aging: {res_base.aging_final:.4f}%
    • Final SOC: {res_base.soc[-1]*100:.1f}%
    """
    ax.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved as {save_prefix}_comparison.png")