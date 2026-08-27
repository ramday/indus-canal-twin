"""
End-to-End Cyber-Physical Simulation: Ingests noisy IoT telemetry and detects water theft via CUSUM.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
import matplotlib.pyplot as plt
from src.network import CanalNetwork
from src.telemetry import CanalTelemetry
from src.detector import CUSUMDetector

def main():
    os.makedirs("assets", exist_ok=True)
    gate_openings = [1.2, 0.85, 0.75]
    inflow_barrage = 14.0
    theft_window = (8.0, 16.0)

    # 1. Physical Simulation Core
    net = CanalNetwork()
    print("Running physical hydrodynamic simulation...")
    t_cont, depths_cont, _ = net.simulate(
        duration_hours=24.0,
        inflow=inflow_barrage,
        gate_openings=gate_openings,
        theft_active=True
    )

    # 2. IoT Telemetry Generation (5-minute sampling + noise + packet drop)
    telemetry = CanalTelemetry(sampling_interval_seconds=300.0, sensor_noise_std=0.015, packet_loss_rate=0.02)
    t_sample, h_meas = telemetry.sample_network(t_cont, depths_cont)

    # 3. Residual & CUSUM Detection Layer
    detector = CUSUMDetector(network=net, slack_k=0.8, threshold_h=6.0)
    _, residuals = detector.compute_residuals(t_sample, h_meas, gate_openings, inflow_barrage)
    results = detector.detect_anomalies(residuals)

    # 4. Compute Audit Metrics
    alarm_triggered = results["alarm_triggered"]
    first_alarm_idx = results["first_alarm_idx"]

    print("\n--- WATER THEFT DETECTION AUDIT ---")
    if alarm_triggered:
        alarm_time = t_sample[first_alarm_idx]
        actual_theft_start = theft_window[0]
        latency_minutes = (alarm_time - actual_theft_start) * 60.0
        print(f"Status: ANOMALY CONFIRMED (Theft Detected)")
        print(f"Theft Activated At:     {actual_theft_start:.2f} hrs (08:00 AM)")
        print(f"Alarm Triggered At:     {alarm_time:.2f} hrs")
        print(f"Detection Latency:      {latency_minutes:.1f} minutes")
        print(f"Peak CUSUM Statistic:   {np.max(results['s_pos']):.2f} (Threshold H = {results['threshold']})")
    else:
        print("Status: MISSED DETECTION (No alarm triggered)")

    # 5. Publication-Grade Diagnostic Plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # Plot 1: Sensor Readings (Continuous vs Noisy Telemetry)
    ax1.plot(t_cont, depths_cont[1], label="True Reach 2 Depth (Continuous)", color="#1f77b4", linewidth=2)
    ax1.scatter(t_sample, h_meas[1], label="Sampled IoT Sensor Telemetry (5-min)", color="#ff7f0e", s=12, alpha=0.7)
    ax1.axvspan(theft_window[0], theft_window[1], color="red", alpha=0.12, label="Illegal Diversion (2.5 m³/s)")
    ax1.set_ylabel("Water Depth $h_2$ (m)", fontsize=10)
    ax1.set_title("Reach 2 IoT Depth Telemetry with Sensor Noise & Sampling Discretization", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Plot 2: Mass Conservation Residual
    ax2.plot(t_sample, residuals, label="Mass-Balance Residual $r(t)$", color="#2ca02c", linewidth=1.5)
    ax2.axhline(0.0, color="black", linestyle="--", alpha=0.5)
    ax2.axhline(2.5, color="red", linestyle=":", label="Nominal Theft Magnitude (2.5 m³/s)")
    ax2.axvspan(theft_window[0], theft_window[1], color="red", alpha=0.12)
    ax2.set_ylabel("Residual $r_k$ (m³/s)", fontsize=10)
    ax2.set_title("Reach 2 Mass-Conservation Residual Stream", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Plot 3: CUSUM Test Statistic & Decision Boundary
    ax3.plot(t_sample, results["s_pos"], label="CUSUM Statistic $S^+(t)$", color="#9467bd", linewidth=2)
    ax3.axhline(results["threshold"], color="red", linestyle="--", linewidth=1.8, label=f"Decision Threshold $H = {results['threshold']}$")
    ax3.axvspan(theft_window[0], theft_window[1], color="red", alpha=0.12)
    if alarm_triggered:
        ax3.scatter(
            [t_sample[first_alarm_idx]],
            [results["s_pos"][first_alarm_idx]],
            color="red",
            s=120,
            zorder=5,
            label=f"First Alarm ({t_sample[first_alarm_idx]:.2f}h)"
        )
    ax3.set_xlabel("Simulation Time (Hours)", fontsize=11)
    ax3.set_ylabel("CUSUM Score $S^+$", fontsize=10)
    ax3.set_title("CUSUM Change-Point Detection (Evidence Accumulation)", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = "assets/theft_detection_audit.png"
    plt.savefig(output_path, dpi=300)
    print(f"Diagnostic plot successfully saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()