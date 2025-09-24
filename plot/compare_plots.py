# plot/compare_plots.py
from __future__ import annotations
import matplotlib
import matplotlib.pyplot as plt

try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

def compare_time_series(res_llm, res_base, save_prefix="compare"):
    # Voltage
    plt.figure()
    plt.plot(res_llm.t, res_llm.V, label="LLM+BO")
    plt.plot(res_base.t, res_base.V, label="传统BO", linestyle="--")
    plt.xlabel("时间 (s)"); plt.ylabel("电压 V (V)")
    plt.legend(); plt.grid(True)
    plt.savefig(f"{save_prefix}_voltage.png", dpi=200); plt.close()

    # Current
    plt.figure()
    plt.plot(res_llm.t, res_llm.I, label="LLM+BO")
    plt.plot(res_base.t, res_base.I, label="传统BO", linestyle="--")
    plt.xlabel("时间 (s)"); plt.ylabel("电流 I (A)")
    plt.legend(); plt.grid(True)
    plt.savefig(f"{save_prefix}_current.png", dpi=200); plt.close()

    # SOC
    plt.figure()
    plt.plot(res_llm.t, res_llm.soc, label="LLM+BO")
    plt.plot(res_base.t, res_base.soc, label="传统BO", linestyle="--")
    plt.xlabel("时间 (s)"); plt.ylabel("SOC")
    plt.ylim(0, 1.0)
    plt.legend(); plt.grid(True)
    plt.savefig(f"{save_prefix}_SOC.png", dpi=200); plt.close()

    # Temperature
    plt.figure()
    plt.plot(res_llm.t, res_llm.T, label="LLM+BO")
    plt.plot(res_base.t, res_base.T, label="传统BO", linestyle="--")
    plt.xlabel("时间 (s)"); plt.ylabel("温度 T (K)")
    plt.legend(); plt.grid(True)
    plt.savefig(f"{save_prefix}_temperature.png", dpi=200); plt.close()

    # Aging
    plt.figure()
    plt.plot(res_llm.t, res_llm.aging, label="LLM+BO")
    plt.plot(res_base.t, res_base.aging, label="传统BO", linestyle="--")
    plt.xlabel("时间 (s)"); plt.ylabel("老化 (%)")
    plt.legend(); plt.grid(True)
    plt.savefig(f"{save_prefix}_aging.png", dpi=200); plt.close()
