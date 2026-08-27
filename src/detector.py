"""
Algorithmic Water Theft Detection: Computes mass-balance residuals and runs CUSUM change-point tracking.
Supports both batch simulation evaluation and online step-by-step recursive telemetry updates.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
from src.network import CanalNetwork

class CUSUMDetector:
    """
    Two-sided Cumulative Sum (CUSUM) statistical anomaly detector for hydraulic networks.
    Tracks persistent positive shifts in the mass-conservation residual:
    r_k = Q_in,k - Q_out,k - Q_moga,k - Q_seep,k - A_s * (dh / dt)_k
    """
    def __init__(
        self,
        network: CanalNetwork,
        slack_k: float = 0.8,       # Slack parameter K (m^3/s): half of expected anomaly magnitude
        threshold_h: float = 6.0,   # Decision threshold H: controls false alarm vs. detection latency
        nominal_mean: float = 0.0,
        filter_alpha: float = 0.20  # Low-pass filter smoothing factor (0 < alpha <= 1)
    ):
        self.net = network
        self.slack_k = slack_k
        self.threshold_h = threshold_h
        self.mu_0 = nominal_mean
        self.alpha = filter_alpha

    def smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Applies a first-order exponential low-pass filter to a full sensor time series."""
        smoothed = np.zeros_like(signal)
        smoothed[0] = signal[0]
        for k in range(1, len(signal)):
            smoothed[k] = self.alpha * signal[k] + (1.0 - self.alpha) * smoothed[k - 1]
        return smoothed

    def compute_step_residual(
        self,
        h1_curr: float,
        h2_curr: float,
        h2_prev: float,
        gate_openings: List[float],
        dt_seconds: float
    ) -> float:
        """
        Pure online single-step mass balance residual computation for Reach 2.
        Avoids batch recomputation causality leaks by evaluating only current step parameters.
        """
        w0, w1, w2 = gate_openings
        reach2 = self.net.reaches[1]

        q_in = self.net.gates[1].discharge(h1_curr, w1)
        q_out = self.net.gates[2].discharge(h2_curr, w2)
        q_moga = reach2.moga.discharge(h2_curr)
        q_seep = reach2.seepage_loss(h2_curr)

        avg_h2 = 0.5 * (h2_prev + h2_curr)
        surf_area = reach2.surface_area(avg_h2)
        dh_dt = (h2_curr - h2_prev) / dt_seconds
        storage_rate = surf_area * dh_dt

        return float((q_in - q_out - q_moga - q_seep) - storage_rate)

    def compute_residuals(
        self,
        t_sample_hrs: np.ndarray,
        h_measured: np.ndarray,
        gate_openings: List[float],
        inflow_barrage: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch evaluation: Filters noisy telemetry and computes mass conservation residuals
        for static post-hoc scenario analysis and unit testing.
        """
        dt_seconds = (t_sample_hrs[1] - t_sample_hrs[0]) * 3600.0
        n_steps = len(t_sample_hrs)
        residuals = np.zeros(n_steps)

        # 1. Low-pass filter depth telemetry
        h_filt = np.zeros_like(h_measured)
        for i in range(h_measured.shape[0]):
            h_filt[i] = self.smooth_signal(h_measured[i])

        w0, w1, w2 = gate_openings
        reach2 = self.net.reaches[1]

        for k in range(1, n_steps):
            h1_curr = h_filt[0, k]
            h2_prev = h_filt[1, k - 1]
            h2_curr = h_filt[1, k]

            # 2. Estimated boundary flows
            q_in_est = self.net.gates[1].discharge(h1_curr, w1)
            q_out_est = self.net.gates[2].discharge(h2_curr, w2)
            q_moga_est = reach2.moga.discharge(h2_curr)
            q_seep_est = reach2.seepage_loss(h2_curr)

            # 3. Filtered storage rate
            avg_h2 = 0.5 * (h2_prev + h2_curr)
            surf_area = reach2.surface_area(avg_h2)
            dh_dt = (h2_curr - h2_prev) / dt_seconds
            storage_rate = surf_area * dh_dt

            # 4. Mass-Balance Residual
            residuals[k] = (q_in_est - q_out_est - q_moga_est - q_seep_est) - storage_rate

        residuals[0] = residuals[1]
        return t_sample_hrs, residuals

    def detect_anomalies(self, residuals: np.ndarray, reset_on_alarm: bool = False) -> Dict[str, Any]:
        """
        Executes CUSUM recursion S_k^+ = max(0, S_{k-1}^+ + (r_k - mu_0 - K)).
        When reset_on_alarm is True, resets S_k^+ to 0 upon crossing H to enable
        sequential multi-event detection.
        """
        n_steps = len(residuals)
        s_pos = np.zeros(n_steps)
        alarm_indices = []

        for k in range(1, n_steps):
            accumulated = s_pos[k - 1] + (residuals[k] - self.mu_0 - self.slack_k)
            s_pos[k] = max(0.0, accumulated)

            if s_pos[k] >= self.threshold_h:
                alarm_indices.append(k)
                if reset_on_alarm:
                    s_pos[k] = 0.0

        alarm_triggered = len(alarm_indices) > 0
        first_alarm_idx = alarm_indices[0] if alarm_triggered else None

        return {
            "s_pos": s_pos,
            "alarm_triggered": alarm_triggered,
            "first_alarm_idx": first_alarm_idx,
            "all_alarms": alarm_indices,
            "total_alarms": len(alarm_indices),
            "threshold": self.threshold_h
        }