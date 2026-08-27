"""
Dynamic Equity Optimizer: Provides both numerical constrained optimization (SLSQP)
and steady-state hydraulic feedforward target generation for equitable canal governance.
"""
from typing import List, Tuple
import numpy as np
from scipy.optimize import minimize
from src.network import CanalNetwork

class EquityOptimizer:
    """
    Equitable water allocation engine:
    1. calculate_gini_index: Quantifies delivery disparity across mogas.
    2. optimize_gate_openings: Numerical SLSQP constrained optimizer.
    3. compute_equitable_gate_targets: Analytical hydraulic inversion for stable setpoints.
    """
    def __init__(
        self,
        network: CanalNetwork,
        actuator_penalty: float = 0.10,
        tail_floor_ratio: float = 0.85,    # Minimum 85% delivery floor for tail
        tail_penalty_weight: float = 8.0   # Asymmetric penalty weight for tail deprivation
    ):
        self.net = network
        self.lambda_w = actuator_penalty
        self.tail_floor = tail_floor_ratio
        self.gamma_tail = tail_penalty_weight
        self.w_bounds = [(0.20, 1.80), (0.20, 1.80)]  # Physical gate travel bounds (meters)

    def calculate_gini_index(self, deliveries: List[float]) -> float:
        """Computes the standard Gini inequality coefficient across moga discharges."""
        q = np.array(deliveries, dtype=float)
        if np.all(q <= 1e-6) or np.sum(q) <= 1e-6:
            return 0.0
        n = len(q)
        diff_sum = np.sum(np.abs(q[:, None] - q[None, :]))
        return float(diff_sum / (2.0 * n * np.sum(q)))

    def optimize_gate_openings(
        self,
        current_depths: List[float],
        target_deliveries: List[float],
        previous_openings: List[float] = [0.85, 0.75],
        inflow: float = 14.0,
        estimated_theft: float = 0.0,
        dt_horizon: float = 600.0
    ) -> Tuple[List[float], float]:
        """
        Solves constrained nonlinear program (SLSQP) to balance deliveries:
        min_{w1, w2} [ Variance(Ratios) + gamma * Tail_Deficit_Penalty + lambda * Actuator_Movement ]
        """
        h1, h2, h3 = current_depths
        q_targets = np.array(target_deliveries, dtype=float)
        w_prev = np.array(previous_openings, dtype=float)

        def objective(w_vars):
            w1, w2 = w_vars

            # 1. Forward predict gate flows
            q_g1 = self.net.gates[1].discharge(h1, w1)
            q_g2 = self.net.gates[2].discharge(h2, w2)

            # 2. Forward predict 1-step storage derivatives
            dh1 = (inflow - q_g1 - self.net.reaches[0].moga.discharge(h1) - self.net.reaches[0].seepage_loss(h1)) / self.net.reaches[0].surface_area(h1)
            dh2 = (q_g1 - q_g2 - self.net.reaches[1].moga.discharge(h2) - self.net.reaches[1].seepage_loss(h2) - estimated_theft) / self.net.reaches[1].surface_area(h2)
            dh3 = (q_g2 - (0.6 * max(0.0, h3 - 0.5) ** 1.5) - self.net.reaches[2].moga.discharge(h3) - self.net.reaches[2].seepage_loss(h3)) / self.net.reaches[2].surface_area(h3)

            h1_next = max(0.1, h1 + dh1 * dt_horizon)
            h2_next = max(0.1, h2 + dh2 * dt_horizon)
            h3_next = max(0.1, h3 + dh3 * dt_horizon)

            # 3. Predicted moga satisfaction ratios
            q_pred = np.array([
                self.net.reaches[0].moga.discharge(h1_next),
                self.net.reaches[1].moga.discharge(h2_next),
                self.net.reaches[2].moga.discharge(h3_next)
            ])
            ratios = q_pred / q_targets
            mean_ratio = np.mean(ratios)

            # 4. Multi-objective penalty terms
            variance_cost = np.sum((ratios - mean_ratio) ** 2)
            tail_deficit = max(0.0, self.tail_floor - ratios[2])
            tail_penalty = self.gamma_tail * (tail_deficit ** 2)
            actuator_cost = self.lambda_w * np.sum((w_vars - w_prev) ** 2)

            return variance_cost + tail_penalty + actuator_cost

        res = minimize(
            objective,
            x0=w_prev,
            method="SLSQP",
            bounds=self.w_bounds,
            options={"ftol": 1e-4, "maxiter": 80}
        )

        optimal_w = [float(res.x[0]), float(res.x[1])]
        gini = self.calculate_gini_index([
            self.net.reaches[0].moga.discharge(h1),
            self.net.reaches[1].moga.discharge(h2),
            self.net.reaches[2].moga.discharge(h3)
        ])
        return optimal_w, gini

    def compute_equitable_gate_targets(
        self,
        inflow: float,
        estimated_theft: float,
        baseline_h: List[float],
        baseline_gates: List[float] = [1.2, 0.85, 0.75]
    ) -> List[float]:
        """
        Solves analytical steady-state hydraulic balance to find target gate openings (w1*, w2*).
        Guarantees stability without numerical solver oscillations.
        """
        if estimated_theft <= 0.20:
            return [baseline_gates[1], baseline_gates[2]]

        q_targets = [
            self.net.reaches[0].moga.discharge(baseline_h[0]),
            self.net.reaches[1].moga.discharge(baseline_h[1]),
            self.net.reaches[2].moga.discharge(baseline_h[2])
        ]

        theft_fraction = min(0.35, estimated_theft / inflow)
        target_ratio = max(self.tail_floor, 1.0 - theft_fraction)

        q_m1_des = q_targets[0] * target_ratio
        q_m2_des = q_targets[1] * target_ratio
        q_m3_des = q_targets[2] * target_ratio

        g = 9.81

        # 1. Reach 3 Required Head and Flow
        moga3 = self.net.reaches[2].moga
        b3 = getattr(moga3, 'b', getattr(moga3, 'b_moga', 0.8))
        sill3 = getattr(moga3, 'sill', getattr(moga3, 'h_sill', 0.5))
        cd_b3 = moga3.cd * b3 * np.sqrt(2.0 / 3.0 * g)
        h3_req = sill3 + (q_m3_des / cd_b3) ** (2.0 / 3.0)

        q_tail_spill = 0.6 * max(0.0, h3_req - 0.5) ** 1.5
        q_seep3 = self.net.reaches[2].seepage_loss(h3_req)
        q_in_reach3_req = q_m3_des + q_tail_spill + q_seep3

        # 2. Reach 2 Required Head and Flow
        moga2 = self.net.reaches[1].moga
        b2 = getattr(moga2, 'b', getattr(moga2, 'b_moga', 0.8))
        sill2 = getattr(moga2, 'sill', getattr(moga2, 'h_sill', 0.5))
        cd_b2 = moga2.cd * b2 * np.sqrt(2.0 / 3.0 * g)
        h2_req = sill2 + (q_m2_des / cd_b2) ** (2.0 / 3.0)

        q_seep2 = self.net.reaches[1].seepage_loss(h2_req)
        q_in_reach2_req = q_in_reach3_req + q_m2_des + q_seep2 + estimated_theft

        # 3. Reach 1 Required Head
        moga1 = self.net.reaches[0].moga
        b1 = getattr(moga1, 'b', getattr(moga1, 'b_moga', 0.8))
        sill1 = getattr(moga1, 'sill', getattr(moga1, 'h_sill', 0.5))
        cd_b1 = moga1.cd * b1 * np.sqrt(2.0 / 3.0 * g)
        h1_req = sill1 + (q_m1_des / cd_b1) ** (2.0 / 3.0)

        # 4. Invert Torricelli Orifice Equation
        gate1 = self.net.gates[1]
        b_g1 = getattr(gate1, 'b', getattr(gate1, 'b_gate', 3.5))
        cd_g1 = gate1.cd * b_g1 * np.sqrt(2.0 * g * max(0.2, h1_req))
        w1_target = q_in_reach2_req / cd_g1

        gate2 = self.net.gates[2]
        b_g2 = getattr(gate2, 'b', getattr(gate2, 'b_gate', 3.5))
        cd_g2 = gate2.cd * b_g2 * np.sqrt(2.0 * g * max(0.2, h2_req))
        w2_target = q_in_reach3_req / cd_g2

        return [
            float(np.clip(w1_target, self.w_bounds[0][0], self.w_bounds[0][1])),
            float(np.clip(w2_target, self.w_bounds[1][0], self.w_bounds[1][1]))
        ]