"""
Integration test verifying global volume conservation across the canal network.
"""
import numpy as np
from scipy.integrate import simpson
from src.network import CanalNetwork

def test_global_mass_conservation_steady_state():
    """
    Over steady-state operation with no theft, total inflow volume must equal
    total outflow volume (Mogas + Seepage + Spill) with < 1.0% numerical error.
    """
    net = CanalNetwork()
    duration_hrs = 12.0
    inflow_rate = 14.0  # m3/s constant

    time_hrs, depths, _ = net.simulate(duration_hours=duration_hrs, inflow=inflow_rate, theft_active=False)
    time_seconds = time_hrs * 3600.0

    # Total inflow volume (m^3)
    total_inflow_vol = inflow_rate * (duration_hrs * 3600.0)

    # Compute outflow rates over time
    moga_sum = np.array([
        net.reaches[0].moga.discharge(h1) + 
        net.reaches[1].moga.discharge(h2) + 
        net.reaches[2].moga.discharge(h3)
        for h1, h2, h3 in zip(depths[0], depths[1], depths[2])
    ])

    seep_sum = np.array([
        net.reaches[0].seepage_loss(h1) + 
        net.reaches[1].seepage_loss(h2) + 
        net.reaches[2].seepage_loss(h3)
        for h1, h2, h3 in zip(depths[0], depths[1], depths[2])
    ])

    tail_spill = np.array([0.6 * max(0.0, h3 - 0.5) ** 1.5 for h3 in depths[2]])

    total_outflow_rate = moga_sum + seep_sum + tail_spill

    # Integrate outflow rates over time using Simpson's rule
    total_outflow_vol = simpson(total_outflow_rate, x=time_seconds)

    # Compute storage change in trapezoidal reaches: delta_V = Sum(Area * delta_h)
    storage_change = 0.0
    for i in range(3):
        avg_area = net.reaches[i].surface_area(depths[i][0])
        delta_h = depths[i][-1] - depths[i][0]
        storage_change += avg_area * delta_h

    # Mass balance residual
    volume_error = abs(total_inflow_vol - (total_outflow_vol + storage_change))
    relative_error_pct = (volume_error / total_inflow_vol) * 100.0

    print(f"\nMass Balance Check -> Total In: {total_inflow_vol:.1f} m3 | Total Out + dS: {total_outflow_vol + storage_change:.1f} m3")
    print(f"Numerical Error: {relative_error_pct:.4f}%")

    assert relative_error_pct < 1.0, f"Mass conservation violated with error: {relative_error_pct}%"