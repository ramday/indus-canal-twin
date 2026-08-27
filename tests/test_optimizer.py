"""
Unit tests for the Gini inequality calculation and dynamic equity optimizer.
"""
from src.network import CanalNetwork
from src.optimizer import EquityOptimizer

def test_gini_coefficient_properties():
    net = CanalNetwork()
    optimizer = EquityOptimizer(net)

    # 1. Perfectly equal distribution -> Gini must be 0.0
    equal_deliveries = [1.5, 1.5, 1.5]
    assert optimizer.calculate_gini_index(equal_deliveries) == 0.0

    # 2. Unequal distribution -> Gini must be positive
    unequal_deliveries = [2.5, 1.5, 0.5]
    gini = optimizer.calculate_gini_index(unequal_deliveries)
    assert 0.0 < gini < 1.0

def test_optimizer_bounds_enforcement():
    net = CanalNetwork()
    optimizer = EquityOptimizer(net)
    
    # Depressed reach depths
    test_depths = [1.4, 0.7, 1.8]
    targets = [0.93, 0.64, 2.14]
    
    w_opt, _ = optimizer.optimize_gate_openings(test_depths, targets)
    
    # Assert gate openings remain within mechanical travel bounds (0.2m to 2.0m)
    assert 0.20 <= w_opt[0] <= 2.00
    assert 0.20 <= w_opt[1] <= 2.00