"""
Unit tests for the CUSUM statistical anomaly detector and IoT telemetry module.
"""
import numpy as np
from src.network import CanalNetwork
from src.telemetry import CanalTelemetry
from src.detector import CUSUMDetector

def test_no_false_alarm_under_normal_operation():
    """Confirms that in normal steady-state operation, no false alarms are triggered."""
    net = CanalNetwork()
    gate_openings = [1.2, 0.85, 0.75]
    inflow = 14.0

    t_cont, depths_cont, _ = net.simulate(duration_hours=12.0, inflow=inflow, gate_openings=gate_openings, theft_active=False)
    telemetry = CanalTelemetry(sampling_interval_seconds=300.0, sensor_noise_std=0.015, packet_loss_rate=0.0)
    t_sample, h_meas = telemetry.sample_network(t_cont, depths_cont)

    detector = CUSUMDetector(network=net, slack_k=0.8, threshold_h=6.0)
    _, residuals = detector.compute_residuals(t_sample, h_meas, gate_openings, inflow)
    results = detector.detect_anomalies(residuals)

    assert not results["alarm_triggered"], "False alarm triggered during normal steady-state operation"

def test_theft_detection_deterministic():
    """Confirms that a single unmetered diversion is deterministically flagged."""
    net = CanalNetwork()
    gate_openings = [1.2, 0.85, 0.75]
    inflow = 14.0

    t_cont, depths_cont, _ = net.simulate(duration_hours=24.0, inflow=inflow, gate_openings=gate_openings, theft_active=True)
    telemetry = CanalTelemetry(sampling_interval_seconds=300.0, sensor_noise_std=0.015, packet_loss_rate=0.0)
    t_sample, h_meas = telemetry.sample_network(t_cont, depths_cont)

    detector = CUSUMDetector(network=net, slack_k=0.8, threshold_h=6.0)
    _, residuals = detector.compute_residuals(t_sample, h_meas, gate_openings, inflow)
    results = detector.detect_anomalies(residuals)

    assert results["alarm_triggered"], "Detector failed to flag active diversion"
    alarm_time = t_sample[results["first_alarm_idx"]]
    assert 8.0 <= alarm_time <= 10.0, f"Detection latency out of bounds: {alarm_time:.2f}h"

def test_multi_event_theft_detection():
    """
    Confirms that CUSUM accumulator reset enables sequential detection
    of two separate, independent theft events occurring hours apart.
    """
    net = CanalNetwork()
    detector = CUSUMDetector(network=net, slack_k=0.8, threshold_h=6.0)

    # Construct synthetic residual stream with two distinct 2-hour theft pulses:
    # Event 1 at t = 2h to 4h (steps 24 to 48)
    # Event 2 at t = 10h to 12h (steps 120 to 144)
    n_steps = 200
    residuals = np.random.normal(0.0, 0.1, n_steps)
    residuals[24:48] += 2.5   # Event 1
    residuals[120:144] += 2.5 # Event 2

    results = detector.detect_anomalies(residuals, reset_on_alarm=True)

    # Must detect both events separately
    alarms = results["all_alarms"]
    assert len(alarms) >= 2, "Failed to detect multiple sequential theft events"

    event1_detected = any(24 <= idx <= 36 for idx in alarms)
    event2_detected = any(120 <= idx <= 132 for idx in alarms)

    assert event1_detected, "Event 1 was not detected"
    assert event2_detected, "Event 2 was missed due to accumulator saturation"