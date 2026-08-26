"""
Multi-reach canal network assembling interconnected hydraulic pools and ODEs.
Includes smooth physical transitions to eliminate numerical interpolation ringing.
"""
from typing import List, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from src.structures import SluiceGate
from src.reach import CanalReach

class CanalNetwork:
    """Simulates a 3-reach contiguous canal network (Head -> Middle -> Tail)."""
    def __init__(self):
        self.reaches = [
            CanalReach("Reach 1 (Head)", length=6000.0, bed_width=6.0, side_slope=1.5),
            CanalReach("Reach 2 (Middle)", length=5000.0, bed_width=5.5, side_slope=1.5),
            CanalReach("Reach 3 (Tail)", length=4000.0, bed_width=5.0, side_slope=1.5)
        ]
        self.gates = [
            SluiceGate(gate_width=4.0, cd=0.62),   # Gate 0: Headworks Barrage Gate
            SluiceGate(gate_width=3.5, cd=0.60),   # Gate 1: Cross-Regulator 1 (R1 -> R2)
            SluiceGate(gate_width=3.0, cd=0.60)    # Gate 2: Cross-Regulator 2 (R2 -> R3)
        ]

    def _smooth_theft_profile(
        self,
        t: float,
        theft_rate: float = 2.5,
        theft_window: Tuple[float, float] = (8.0, 16.0),
        ramp_tau: float = 300.0  # 5-minute physical ramp-up time constant
    ) -> float:
        """
        Computes a C^infinity smooth logistic transition for water theft,
        eliminating numerical RK45 polynomial ringing artifacts.
        """
        t_start = theft_window[0] * 3600.0
        t_end = theft_window[1] * 3600.0

        # Safe sigmoid calculation to prevent numerical overflow
        def sigmoid(val):
            val_clipped = np.clip(val, -50.0, 50.0)
            return 1.0 / (1.0 + np.exp(-val_clipped))

        turn_on = sigmoid((t - t_start) / ramp_tau)
        turn_off = sigmoid((t - t_end) / ramp_tau)
        return float(theft_rate * turn_on * (1.0 - turn_off))

    def system_derivatives(
        self,
        t: float,
        h_states: List[float],
        gate_openings: List[float],
        inflow_barrage: float,
        theft_active: bool = False,
        theft_rate: float = 2.5,
        theft_window: Tuple[float, float] = (8.0, 16.0)
    ) -> List[float]:
        """Computes dh/dt for each reach based on mass conservation."""
        h1, h2, h3 = h_states
        w0, w1, w2 = gate_openings

        # 1. Gate Discharges
        Q_in1 = inflow_barrage
        Q_gate1 = self.gates[1].discharge(h1, w1)
        Q_gate2 = self.gates[2].discharge(h2, w2)
        Q_tail_spill = 0.6 * max(0.0, h3 - 0.5) ** 1.5

        # 2. Mogas & Seepage
        Q_moga1 = self.reaches[0].moga.discharge(h1)
        Q_moga2 = self.reaches[1].moga.discharge(h2)
        Q_moga3 = self.reaches[2].moga.discharge(h3)

        Q_s1 = self.reaches[0].seepage_loss(h1)
        Q_s2 = self.reaches[1].seepage_loss(h2)
        Q_s3 = self.reaches[2].seepage_loss(h3)

        # 3. Smooth Physical Theft Profile
        Q_theft = 0.0
        if theft_active:
            Q_theft = self._smooth_theft_profile(t, theft_rate, theft_window)

        # 4. ODEs: dh/dt
        dh1_dt = (Q_in1 - Q_gate1 - Q_moga1 - Q_s1) / self.reaches[0].surface_area(h1)
        dh2_dt = (Q_gate1 - Q_gate2 - Q_moga2 - Q_s2 - Q_theft) / self.reaches[1].surface_area(h2)
        dh3_dt = (Q_gate2 - Q_tail_spill - Q_moga3 - Q_s3) / self.reaches[2].surface_area(h3)

        return [dh1_dt, dh2_dt, dh3_dt]

    def find_steady_state(self, inflow: float = 14.0, gate_openings: List[float] = [1.2, 0.85, 0.75]) -> np.ndarray:
        """Calculates equilibrium depths where dh/dt = 0."""
        loss_fn = lambda h: self.system_derivatives(0.0, h, gate_openings, inflow, theft_active=False)
        sol = root(loss_fn, x0=[1.5, 1.4, 2.5], method='hybr')
        return sol.x

    def simulate(
        self,
        duration_hours: float = 24.0,
        inflow: float = 14.0,
        gate_openings: List[float] = [1.2, 0.85, 0.75],
        theft_active: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs the simulation with a maximum step size to ensure smooth trajectory."""
        h_eq = self.find_steady_state(inflow, gate_openings)
        t_span = (0.0, duration_hours * 3600.0)
        t_eval = np.linspace(0.0, duration_hours * 3600.0, 600)

        ode_wrapper = lambda t, h: self.system_derivatives(
            t, h, gate_openings, inflow, theft_active=theft_active
        )
        
        # max_step=120s forces RK45 to sample accurately through transition windows
        sol = solve_ivp(ode_wrapper, t_span, h_eq, t_eval=t_eval, method="RK45", max_step=120.0)
        return sol.t / 3600.0, sol.y, h_eq