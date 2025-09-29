# plot/compare_plots.py - ENGLISH VERSION WITH PROPER SAVE
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# English font configuration
plt.style.use('default')
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
except Exception as e:
    print(f"Warning: Could not set matplotlib rcParams: {e}")

def validate_and_prepare_data(result, label="Result"):
    """Validate and prepare simulation result data."""
    print(f"\n=== Validating {label} ===")
    
    if result is None:
        raise ValueError(f"{label} is None")
    
    required_attrs = ['t', 'V', 'T', 'I', 'soc', 'aging']
    for attr in required_attrs:
        if not hasattr(result, attr):
            raise ValueError(f"{label} missing attribute: {attr}")
        
        data = getattr(result, attr)
        if data is None or len(data) == 0:
            print(f"WARNING: {label}.{attr} is empty")
        else:
            print(f"{label}.{attr}: shape={np.array(data).shape}, range=[{np.min(data):.3f}, {np.max(data):.3f}]")
    
    # Convert to numpy arrays
    t = np.asarray(result.t, dtype=float)
    V = np.asarray(result.V, dtype=float)
    T = np.asarray(result.T, dtype=float)
    I = np.asarray(result.I, dtype=float)
    soc = np.asarray(result.soc, dtype=float)
    aging = np.asarray(result.aging, dtype=float)
    
    # Check lengths
    lengths = [len(arr) for arr in [t, V, T, I, soc, aging]]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        print(f"WARNING: Trimming to minimum length {min_len}")
        t, V, T, I, soc, aging = [arr[:min_len] for arr in [t, V, T, I, soc, aging]]
    
    if len(t) < 2:
        raise ValueError(f"{label} has insufficient data points: {len(t)}")
    
    print(f"{label} validation complete: {len(t)} points")
    return t, V, T, I, soc, aging

def compare_time_series(res_llm, res_base, save_prefix="compare"):
    """Generate comprehensive comparison plots in English."""
    
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS (ENGLISH)")
    print("="*70)
    
    try:
        # Validate data
        t_llm, V_llm, T_llm, I_llm, soc_llm, aging_llm = validate_and_prepare_data(res_llm, "LLM+BO")
        t_base, V_base, T_base, I_base, soc_base, aging_base = validate_and_prepare_data(res_base, "Traditional BO")
        
        # Convert to minutes
        t_llm_min = t_llm / 60.0
        t_base_min = t_base / 60.0
        
        # Create figure
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle('Battery Charging Optimization: LLM+BO vs Traditional BO', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Define layout
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, 
                             left=0.08, right=0.95, top=0.92, bottom=0.08)
        
        # 1. Terminal Voltage
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t_llm_min, V_llm, 'b-', label="LLM+BO", linewidth=2.5, alpha=0.85)
        ax1.plot(t_base_min, V_base, 'r--', label="Traditional BO", linewidth=2.5, alpha=0.85)
        ax1.axhline(y=4.1, color='darkred', linestyle=':', alpha=0.7, linewidth=2, label='Voltage Limit (4.1V)')
        ax1.set_xlabel("Time (min)", fontsize=11)
        ax1.set_ylabel("Voltage (V)", fontsize=11)
        ax1.set_title("Terminal Voltage Evolution", fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([3.6, 4.2])
        
        # 2. Charging Current
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t_llm_min, -I_llm, 'b-', label="LLM+BO", linewidth=2.5, alpha=0.85)
        ax2.plot(t_base_min, -I_base, 'r--', label="Traditional BO", linewidth=2.5, alpha=0.85)
        ax2.set_xlabel("Time (min)", fontsize=11)
        ax2.set_ylabel("Current (A)", fontsize=11)
        ax2.set_title("Charging Current Profile", fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        # 3. State of Charge
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(t_llm_min, soc_llm * 100, 'b-', label="LLM+BO", linewidth=2.5, alpha=0.85)
        ax3.plot(t_base_min, soc_base * 100, 'r--', label="Traditional BO", linewidth=2.5, alpha=0.85)
        ax3.axhline(y=80, color='green', linestyle=':', alpha=0.7, linewidth=2, label='Target SOC (80%)')
        ax3.set_xlabel("Time (min)", fontsize=11)
        ax3.set_ylabel("State of Charge (%)", fontsize=11)
        ax3.set_title("SOC Progress", fontsize=12, fontweight='bold')
        ax3.set_ylim([15, 85])
        ax3.legend(fontsize=9, loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cell Temperature
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(t_llm_min, T_llm - 273.15, 'b-', label="LLM+BO", linewidth=2.5, alpha=0.85)
        ax4.plot(t_base_min, T_base - 273.15, 'r--', label="Traditional BO", linewidth=2.5, alpha=0.85)
        ax4.axhline(y=40, color='orange', linestyle=':', alpha=0.7, linewidth=2, label='Safety Limit (40°C)')
        ax4.set_xlabel("Time (min)", fontsize=11)
        ax4.set_ylabel("Temperature (°C)", fontsize=11)
        ax4.set_title("Cell Temperature", fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9, loc='best')
        ax4.grid(True, alpha=0.3)
        
        # 5. Battery Aging
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(t_llm_min, aging_llm * 100, 'b-', label="LLM+BO", linewidth=2.5, alpha=0.85)
        ax5.plot(t_base_min, aging_base * 100, 'r--', label="Traditional BO", linewidth=2.5, alpha=0.85)
        ax5.set_xlabel("Time (min)", fontsize=11)
        ax5.set_ylabel("Capacity Loss (%)", fontsize=11)
        ax5.set_title("Battery Aging (SEI Growth)", fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9, loc='best')
        ax5.grid(True, alpha=0.3)
        
        # 6. Power Profile
        ax6 = fig.add_subplot(gs[1, 2])
        power_llm = V_llm * (-I_llm)
        power_base = V_base * (-I_base)
        ax6.plot(t_llm_min, power_llm, 'b-', label="LLM+BO", linewidth=2.5, alpha=0.85)
        ax6.plot(t_base_min, power_base, 'r--', label="Traditional BO", linewidth=2.5, alpha=0.85)
        ax6.set_xlabel("Time (min)", fontsize=11)
        ax6.set_ylabel("Charging Power (W)", fontsize=11)
        ax6.set_title("Charging Power", fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9, loc='best')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(bottom=0)
        
        # 7. Performance Comparison Table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Extract metrics
        llm_time = t_llm[-1] / 60.0 if len(t_llm) > 0 else 0
        llm_temp = np.max(T_llm) - 273.15 if len(T_llm) > 0 else 0
        llm_soc = soc_llm[-1] * 100 if len(soc_llm) > 0 else 0
        llm_aging = aging_llm[-1] * 100 if len(aging_llm) > 0 else 0
        llm_energy = np.trapz(V_llm * (-I_llm), t_llm) / 3600.0 if len(t_llm) > 1 else 0  # Wh
        
        base_time = t_base[-1] / 60.0 if len(t_base) > 0 else 0
        base_temp = np.max(T_base) - 273.15 if len(T_base) > 0 else 0
        base_soc = soc_base[-1] * 100 if len(soc_base) > 0 else 0
        base_aging = aging_base[-1] * 100 if len(aging_base) > 0 else 0
        base_energy = np.trapz(V_base * (-I_base), t_base) / 3600.0 if len(t_base) > 1 else 0  # Wh
        
        # Calculate improvements
        time_imp = ((base_time - llm_time) / base_time * 100) if base_time > 0 else 0
        temp_imp = ((base_temp - llm_temp) / base_temp * 100) if base_temp > 0 else 0
        aging_imp = ((base_aging - llm_aging) / base_aging * 100) if base_aging > 0 else 0
        energy_imp = ((llm_energy - base_energy) / base_energy * 100) if base_energy > 0 else 0
        
        # Create comparison table
        table_text = f"""
╔══════════════════════════╦═══════════════╦═══════════════════╦════════════════╗
║      Performance         ║    LLM+BO     ║  Traditional BO   ║  Improvement   ║
╠══════════════════════════╬═══════════════╬═══════════════════╬════════════════╣
║ Charging Time            ║  {llm_time:6.1f} min   ║    {base_time:6.1f} min    ║   {time_imp:+6.1f}%      ║
║ Peak Temperature         ║  {llm_temp:6.1f} °C    ║    {base_temp:6.1f} °C     ║   {temp_imp:+6.1f}%      ║
║ Final SOC                ║  {llm_soc:6.1f} %     ║    {base_soc:6.1f} %      ║      -         ║
║ Battery Aging            ║  {llm_aging:6.4f} %    ║    {base_aging:6.4f} %     ║   {aging_imp:+6.1f}%      ║
║ Charging Energy          ║  {llm_energy:6.2f} Wh   ║    {base_energy:6.2f} Wh    ║   {energy_imp:+6.1f}%      ║
║ Feasible Solution        ║  {res_llm.feasible}          ║    {res_base.feasible}           ║      -         ║
╚══════════════════════════╩═══════════════╩═══════════════════╩════════════════╝

Key Insights:
• LLM+BO leverages domain knowledge for physics-informed optimization
• Soft constraints enable smooth current reduction near safety limits  
• Multi-objective Chebyshev scalarization balances speed, safety, and longevity
• Charging protocols adapt dynamically to voltage and temperature feedback

Optimization Status:
• LLM+BO: {res_llm.reason}
• Traditional BO: {res_base.reason}

Generated using PyBaMM SPMe model with Chen2020 parameters
        """
        
        ax7.text(0.05, 0.95, table_text, transform=ax7.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Save figure
        save_dir = "output_plots"
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"{save_prefix}_comparison.png"
        filepath = os.path.join(save_dir, filename)
        
        print(f"\nSaving plot to: {filepath}")
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Plot saved successfully!")
        
        # Also save as PDF for publications
        pdf_path = filepath.replace('.png', '.pdf')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"✓ PDF version saved: {pdf_path}")
        
        plt.close(fig)
        
        print("\n" + "="*70)
        print("PLOT GENERATION COMPLETE")
        print("="*70)
        
        return filepath
        
    except Exception as e:
        print(f"\n✗ ERROR generating plots: {e}")
        import traceback
        traceback.print_exc()
        return None