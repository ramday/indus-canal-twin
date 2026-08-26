"""
Simulation script demonstrating tail-end starvation under unmetered diversion.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
import matplotlib.pyplot as plt
from src.network import CanalNetwork

def main():
    os.makedirs("assets", exist_ok=True)
    net = CanalNetwork()

    print("Solving steady-state equilibrium & running 24-hour simulation...")
    time_hrs, depths, h_eq = net.simulate(duration_hours=24.0, theft_active=True)

    # Compute Moga deliveries over time
    moga1_flow = np.array([net.reaches[0].moga.discharge(h) for h in depths[0]])
    moga2_flow = np.array([net.reaches[1].moga.discharge(h) for h in depths[1]])
    moga3_flow = np.array([net.reaches[2].moga.discharge(h) for h in depths[2]])

    # Metrics computed against true steady-state equilibrium
    baseline_tail = moga3_flow[0]
    min_tail = np.min(moga3_flow)
    drop_pct = ((baseline_tail - min_tail) / baseline_tail) * 100.0

    print("\n--- EQUILIBRIUM & TRANSIENT METRICS ---")
    print(f"Equilibrium Depths (h1, h2, h3): [{h_eq[0]:.2f}m, {h_eq[1]:.2f}m, {h_eq[2]:.2f}m]")
    print(f"Moga 1 (Head) Flow: {moga1_flow[0]:.3f} m3/s")
    print(f"Moga 3 (Tail) Baseline Flow: {baseline_tail:.3f} m3/s")
    print(f"Moga 3 (Tail) Minimum Flow during Theft: {min_tail:.3f} m3/s")
    print(f"Tail-End Deprivation Index: {drop_pct:.1f}% reduction")

    # Diagnostic Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Depth Plot
    ax1.plot(time_hrs, depths[0], label="Reach 1 (Head)", color="#1f77b4", linewidth=2)
    ax1.plot(time_hrs, depths[1], label="Reach 2 (Middle - Theft Site)", color="#ff7f0e", linewidth=2)
    ax1.plot(time_hrs, depths[2], label="Reach 3 (Tail)", color="#d62728", linestyle="--", linewidth=2)
    ax1.axvspan(8, 16, color="red", alpha=0.15, label="Illegal Diversion Active (2.5 m³/s)")
    ax1.set_ylabel("Water Depth $h$ (meters)", fontsize=11)
    ax1.set_title("Canal Pool Depths: Steady State with Mid-Stream Diversion Shock", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Moga Deliveries Plot
    ax2.plot(time_hrs, moga1_flow, label="Moga 1 (Head Farmer)", color="#1f77b4", linewidth=2)
    ax2.plot(time_hrs, moga2_flow, label="Moga 2 (Middle Farmer)", color="#ff7f0e", linewidth=2)
    ax2.plot(time_hrs, moga3_flow, label="Moga 3 (Tail Farmer - Deprived)", color="#d62728", linestyle="--", linewidth=2)
    ax2.axvspan(8, 16, color="red", alpha=0.15)
    ax2.set_xlabel("Simulation Time (Hours)", fontsize=11)
    ax2.set_ylabel("Moga Discharge $Q_m$ (m³/s)", fontsize=11)
    ax2.set_title("Tail-End Starvation Metric (Colonial Fixed-Gate Operation)", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("assets/tail_starvation.png", dpi=300)
    print("Updated clean equilibrium plot saved to assets/tail_starvation.png")
    plt.show()

if __name__ == "__main__":
    main()
    