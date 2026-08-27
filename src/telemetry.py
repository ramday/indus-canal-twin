"""
IoT Telemetry module: Synthesizes noisy discrete sensor measurements from continuous canal physics.
"""
from typing import Dict, Tuple
import numpy as np
from scipy.interpolate import interp1d

class CanalTelemetry:
    """
    Simulates physical IoT sensors (Ultrasonic Water Level Recorders & Gate Encoders).
    Samples continuous states at discrete intervals (dt_sample) and injects Gaussian measurement noise.
    """
    def __init__(
        self,
        sampling_interval_seconds: float = 300.0,  # 5-minute sampling interval
        sensor_noise_std: float = 0.015,           # 1.5 cm standard deviation on ultrasonic level
        packet_loss_rate: float = 0.02,            # 2% random telemetry packet drop
        random_seed: int = 42
    ):
        self.dt = sampling_interval_seconds
        self.noise_std = sensor_noise_std
        self.loss_rate = packet_loss_rate
        self.rng = np.random.default_rng(random_seed)

    def sample_network(
        self,
        time_hours: np.ndarray,
        depths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resamples continuous simulation results to discrete telemetry time-series with noise and dropouts.
        
        Args:
            time_hours: Array of continuous time in hours from solve_ivp.
            depths: (3, N) matrix of continuous reach depths (h1, h2, h3).
        Returns:
            t_sample_hrs: Uniformly spaced telemetry timestamps in hours.
            h_measured: (3, M) matrix of noisy, sampled water level readings (meters).
        """
        t_total_seconds = time_hours[-1] * 3600.0
        t_sample_sec = np.arange(0, t_total_seconds + self.dt, self.dt)
        t_sample_hrs = t_sample_sec / 3600.0

        num_reaches = depths.shape[0]
        h_measured = np.zeros((num_reaches, len(t_sample_sec)))

        for i in range(num_reaches):
            # Interpolate continuous trajectory onto discrete sampling points
            interpolator = interp1d(time_hours * 3600.0, depths[i], kind='linear')
            discrete_h = interpolator(t_sample_sec)

            # Add zero-mean Gaussian measurement noise
            noise = self.rng.normal(loc=0.0, scale=self.noise_std, size=len(discrete_h))
            h_noisy = discrete_h + noise

            # Simulate sporadic sensor packet dropouts (forward-fill imputation)
            if self.loss_rate > 0.0:
                drop_mask = self.rng.random(size=len(h_noisy)) < self.loss_rate
                for k in range(1, len(h_noisy)):
                    if drop_mask[k]:
                        h_noisy[k] = h_noisy[k - 1]  # Hold last valid reading

            h_measured[i] = np.clip(h_noisy, 0.0, None)

        return t_sample_hrs, h_measured