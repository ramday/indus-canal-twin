"""
Policy Comparison: Colonial Static Allocation vs. Continuous Telemetry-Driven Closed-Loop Governance.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.network import CanalNetwork
from src.detector import CUSUMDetector
from src.optimizer import EquityOptimizer

GATE_MIN_METERS = 0.20
GATE_MAX_METERS = 1.80

def run_continuous_closed_loop_simulation(
    net: CanalNetwork,
    duration_hours: float = 24.0,
    inflow: float = 14.0,
    baseline_gates: list = [1.2, 0.85, 0.75],
    control_step_sec: float = 600.0,      # 10-minute control interval
    physical_max_slew: float = 0.030      # 3.0 cm / 10 min motor slew limit
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    h_eq = net.find_steady_state(inflow, baseline_gates)
    optimizer = EquityOptimizer(net, tail_floor_ratio=0.85)
    detector = CUSUMDetector(net, slack_k=0.8, threshold_h=6.0, filter_alpha=0.15)

    q_targets = [
        net.reaches[0].moga.discharge(h_eq[0]),
        net.reaches[1].moga.discharge(h_eq[1]),
        net.reaches[2].moga.discharge(h_eq[2])
    ]

    total_steps = int((duration_hours * 3600.0) / control_step_sec)
    current_h_true = list(h_eq)
    current_w = [baseline_gates[1], baseline_gates[2]]

    # Online State Tracking Buffers
    h_filt = list(h_eq)
    s_pos = 0.0
    theft_filtered = 0.0
    theft_smoothing_beta = 0.10  # Disturbance tracking filter

    t_all, h_all = [], []

    print("\n" + "=" * 94)
    print(f"{'Time (h)':<10} | {'CUSUM S+':<10} | {'Est Theft (m3/s)':<18} | {'Gate w1 (m)':<12} | {'Gate w2 (m)':<12} | {'Status':<15}")
    print("=" * 94)

    for step in range(total_steps):
        t_start = step * control_step_sec
        t_end = (step + 1) * control_step_sec
        t_hrs = t_end / 3600.0

        # 1. IoT Sensor Telemetry (with sensor noise)
        noisy_h = [
            max(0.0, current_h_true[i] + float(np.random.normal(0, 0.012)))
            for i in range(3)
        ]

        # 2. Online Recursive Low-Pass Filter
        prev_h2_filt = h_filt[1]
        for i in range(3):
            h_filt[i] = detector.alpha * noisy_h[i] + (1.0 - detector.alpha) * h_filt[i]

        # 3. Step Residual using Active Gate Positions
        active_gate_openings = [baseline_gates[0], current_w[0], current_w[1]]
        step_residual = detector.compute_step_residual(
            h1_curr=h_filt[0],
            h2_curr=h_filt[1],
            h2_prev=prev_h2_filt,
            gate_openings=active_gate_openings,
            dt_seconds=control_step_sec
        )

        # 4. Continuous Disturbance Tracking Filter
        raw_theft_est = max(0.0, step_residual)
        theft_filtered = theft_smoothing_beta * raw_theft_est + (1.0 - theft_smoothing_beta) * theft_filtered

        # 5. CUSUM Anomaly Tracking
        s_pos = max(0.0, s_pos + (step_residual - detector.mu_0 - detector.slack_k))
        alarm_active = (s_pos >= detector.threshold_h) or (theft_filtered >= 0.80)
        
        if theft_filtered < 0.30 and s_pos < 2.0:
            alarm_active = False

        status_str = "MITIGATING" if alarm_active else "NORMAL"

        # 6. Steady-State Hydraulic Target Generation
        effective_theft = theft_filtered if alarm_active else 0.0
        target_w = optimizer.compute_equitable_gate_targets(
            inflow=inflow,
            estimated_theft=effective_theft,
            baseline_h=h_eq,
            baseline_gates=baseline_gates
        )

        # Diagnostic logging
        if step % 9 == 0 or (step > 0 and step % 9 == 1 and alarm_active):
            print(f"{t_hrs:<10.2f} | {s_pos:<10.2f} | {theft_filtered:<18.3f} | {current_w[0]:<12.3f} | {current_w[1]:<12.3f} | {status_str:<15}")

        # 7. Smooth Slew-Rate Limited Actuator Tracking
        for j in range(2):
            delta = target_w[j] - current_w[j]
            slew_step = np.clip(delta, -physical_max_slew, physical_max_slew)
            current_w[j] = float(np.clip(current_w[j] + slew_step, GATE_MIN_METERS, GATE_MAX_METERS))

        # 8. Physical Simulation Integration Step
        gate_schedule = [baseline_gates[0], current_w[0], current_w[1]]
        ode_step = lambda t, h: net.system_derivatives(t, h, gate_schedule, inflow, theft_active=True)
        sol = solve_ivp(ode_step, (t_start, t_end), current_h_true, method="RK45", max_step=60.0)
        current_h_true = sol.y[:, -1].tolist()

        t_all.extend(sol.t.tolist())
        h_all.append(sol.y)

    print("=" * 94 + "\n")

    time_grid = np.linspace(0.0, duration_hours, 600)
    depths_interp = np.zeros((3, len(time_grid)))
    raw_t_hrs = np.array(t_all) / 3600.0
    raw_h = np.hstack(h_all)

    for i in range(3):
        depths_interp[i] = np.interp(time_grid, raw_t_hrs, raw_h[i])

    return time_grid, depths_interp, q_targets


def main():
    os.makedirs("assets", exist_ok=True)

    # 1. Baseline: Colonial Static Allocation
    net_static = CanalNetwork()
    print("Running Baseline: Colonial Static Allocation...")
    t_hrs, depths_static, _ = net_static.simulate(duration_hours=24.0, theft_active=True)

    # 2. Continuous Closed-Loop Controller
    net_ctrl = CanalNetwork()
    print("Running Continuous Telemetry-Driven Equity Controller...")
    _, depths_ctrl, q_targets = run_continuous_closed_loop_simulation(net_ctrl, duration_hours=24.0)

    tail_static = np.array([net_static.reaches[2].moga.discharge(h) for h in depths_static[2]])
    tail_ctrl = np.array([net_ctrl.reaches[2].moga.discharge(h) for h in depths_ctrl[2]])

    optimizer = EquityOptimizer(net_static)

    gini_static = [
        optimizer.calculate_gini_index([
            net_static.reaches[0].moga.discharge(depths_static[0, k]),
            net_static.reaches[1].moga.discharge(depths_static[1, k]),
            net_static.reaches[2].moga.discharge(depths_static[2, k])
        ])
        for k in range(len(t_hrs))
    ]

    gini_ctrl = [
        optimizer.calculate_gini_index([
            net_ctrl.reaches[0].moga.discharge(depths_ctrl[0, k]),
            net_ctrl.reaches[1].moga.discharge(depths_ctrl[1, k]),
            net_ctrl.reaches[2].moga.discharge(depths_ctrl[2, k])
        ])
        for k in range(len(t_hrs))
    ]

    base_tail = tail_static[0]
    deficit_static = ((base_tail - np.min(tail_static)) / base_tail) * 100.0
    deficit_ctrl = ((base_tail - np.min(tail_ctrl)) / base_tail) * 100.0

    print("--- RIGOROUS CLOSED-LOOP BENCHMARK ---")
    print(f"Colonial Static -> Tail-End Deficit: {deficit_static:.1f}% | Peak Gini: {np.max(gini_static):.3f}")
    print(f"Closed-Loop Opt -> Tail-End Deficit: {deficit_ctrl:.1f}% | Peak Gini: {np.max(gini_ctrl):.3f}")
    print(f"Tail Security Restored: +{deficit_static - deficit_ctrl:.1f}% supply preserved")

    # Diagnostic Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # Subplot 1: Tail-End Delivery
    ax1.plot(t_hrs, tail_static, label=f"Colonial Static ({deficit_static:.1f}% Deficit)", color="#d62728", linestyle="--", linewidth=2.2)
    ax1.plot(t_hrs, tail_ctrl, label=f"Dynamic Closed Loop ({deficit_ctrl:.1f}% Deficit)", color="#2ca02c", linewidth=2.5)
    ax1.axhline(base_tail, color="black", linestyle=":", alpha=0.6, label="Sanctioned Target Quota")
    ax1.axvspan(8, 16, color="red", alpha=0.12, label="Unmetered Theft (2.5 m³/s)")
    ax1.set_ylabel("Moga 3 Delivery $Q_{m,3}$ (m³/s)", fontsize=11)
    ax1.set_title("Tail-End Farmer Delivery: Static Colonial vs. Dynamic Closed-Loop Governance", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Hydraulic Gini Index
    ax2.plot(t_hrs, gini_static, label=f"Colonial Gini (Peak: {np.max(gini_static):.3f})", color="#d62728", linestyle="--", linewidth=2)
    ax2.plot(t_hrs, gini_ctrl, label=f"Dynamic Controller Gini (Peak: {np.max(gini_ctrl):.3f})", color="#2ca02c", linewidth=2.2)
    ax2.axvspan(8, 16, color="red", alpha=0.12)
    ax2.set_xlabel("Simulation Time (Hours)", fontsize=11)
    ax2.set_ylabel("Gini Coefficient ($G$)", fontsize=11)
    ax2.set_title("Hydraulic Inequality Metric Across Network (Lower = More Equitable)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = "assets/policy_benchmark.png"
    plt.savefig(output_path, dpi=300)
    print(f"Closed-loop policy plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()