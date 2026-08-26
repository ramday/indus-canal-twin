"""
Lumped-parameter canal reach representing 1D mass conservation and geometry.
"""
import numpy as np
from src.structures import MogaOutlet

class CanalReach:
    """
    Trapezoidal canal pool representing storage, surface area, seepage, and outlet extraction.
    """
    def __init__(
        self,
        name: str,
        length: float = 5000.0,
        bed_width: float = 6.0,
        side_slope: float = 1.5,
        seepage_rate: float = 0.00005,
        moga_crest_width: float = 0.40,
        moga_sill_height: float = 0.20
    ):
        self.name = name
        self.length = length                # Reach length (meters)
        self.bed_width = bed_width          # Bed width 'B' (meters)
        self.side_slope = side_slope        # Side slope 'z' (1 vertical : z horizontal)
        self.k_seep = seepage_rate          # Seepage loss coefficient (m/s per wetted perimeter)
        self.moga = MogaOutlet(crest_width=moga_crest_width, sill_height=moga_sill_height)

    def top_width(self, h: float) -> float:
        """Top water surface width T = B + 2 * z * h"""
        return self.bed_width + 2.0 * self.side_slope * max(0.0, float(h))

    def surface_area(self, h: float) -> float:
        """Top water surface area A_s = Length * Top_Width"""
        return self.length * self.top_width(h)

    def wetted_perimeter(self, h: float) -> float:
        """Wetted perimeter P = B + 2 * h * sqrt(1 + z^2)"""
        h_eff = max(0.0, float(h))
        return self.bed_width + 2.0 * h_eff * np.sqrt(1.0 + self.side_slope ** 2)

    def seepage_loss(self, h: float) -> float:
        """Total reach seepage volumetric loss Q_seep = k_seep * P * Length (m^3/s)"""
        return self.k_seep * self.wetted_perimeter(h) * self.length