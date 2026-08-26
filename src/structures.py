"""
Hydraulic boundary structures: Sluice Gates (underflow orifice) and Mogas (broad-crested weirs).
"""
import numpy as np

class SluiceGate:
    """
    Automated or manual underflow sluice gate.
    Governed by Torricelli-based orifice equation: Q = Cd * b * w * sqrt(2 * g * h_up)
    """
    def __init__(self, gate_width: float = 3.5, cd: float = 0.60):
        self.b = gate_width          # Gate width (meters)
        self.cd = cd                 # Discharge coefficient (dimensionless, typically 0.58 - 0.65)
        self.g = 9.81                # Gravitational acceleration (m/s^2)

    def discharge(self, h_upstream: float, opening: float) -> float:
        """
        Calculate free underflow discharge through the gate orifice.
        
        Args:
            h_upstream: Water depth immediately upstream of the gate (m).
            opening: Gate opening height 'w' (m).
        Returns:
            Discharge Q in m^3/s.
        """
        w = max(0.0, float(opening))
        head = max(0.0, float(h_upstream))
        if head == 0.0 or w == 0.0:
            return 0.0
        return self.cd * self.b * w * np.sqrt(2.0 * self.g * head)


class MogaOutlet:
    """
    Proportional modular outlet (APM / Open Flume / Broad-Crested Weir).
    Delivers flow to watercourses based on upstream head above the outlet sill.
    Governed by: Q = Cd * b * sqrt(2/3 * g) * (h - h_sill)^(1.5)
    """
    def __init__(self, crest_width: float = 0.40, sill_height: float = 0.20, cd: float = 0.62):
        self.b = crest_width         # Throat / crest width (meters)
        self.sill = sill_height      # Elevation of moga crest above canal bed (meters)
        self.cd = cd                 # Discharge coefficient
        self.g = 9.81

    def discharge(self, h_canal: float) -> float:
        """
        Calculate moga delivery discharge into the farmer watercourse.
        
        Args:
            h_canal: Water depth in the parent canal reach (m).
        Returns:
            Discharge Q in m^3/s.
        """
        head_over_crest = max(0.0, float(h_canal) - self.sill)
        if head_over_crest == 0.0:
            return 0.0
        # Standard broad-crested weir formulation
        return self.cd * self.b * np.sqrt(2.0 / 3.0 * self.g) * (head_over_crest ** 1.5)