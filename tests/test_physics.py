"""
Unit tests for boundary structures and reach geometry.
"""
import pytest
from src.structures import SluiceGate, MogaOutlet
from src.reach import CanalReach

def test_gate_zero_flow_when_closed():
    gate = SluiceGate(gate_width=3.5)
    assert gate.discharge(h_upstream=2.5, opening=0.0) == 0.0

def test_moga_zero_flow_below_sill():
    moga = MogaOutlet(sill_height=0.30)
    # If canal depth is below sill height, flow must be 0
    assert moga.discharge(h_canal=0.20) == 0.0

def test_moga_monotonic_increase():
    moga = MogaOutlet(sill_height=0.20)
    q_low = moga.discharge(h_canal=1.5)
    q_high = moga.discharge(h_canal=2.5)
    assert q_high > q_low > 0.0

def test_reach_surface_area_expansion():
    reach = CanalReach("Test Reach", length=1000.0, bed_width=5.0, side_slope=1.5)
    # Area at h=2m: 1000 * (5 + 2 * 1.5 * 2) = 1000 * 11 = 11,000 m^2
    assert reach.surface_area(2.0) == 11000.0