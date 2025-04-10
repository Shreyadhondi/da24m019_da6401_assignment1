# tests/test_optimizer.py

import numpy as np
from Optimizer import Optimizer_step

def test_optimizer_step_sgd_shapes():
    w = {'W': np.random.randn(2, 2)}
    b = {'B': np.random.randn(2, 1)}
    dw = {'W': np.random.randn(2, 2)}
    db = {'B': np.random.randn(2, 1)}
    w_new, b_new = Optimizer_step("sgd", 1, w, b, dw, db, 0.01, 1, 0)
    assert w_new['W'].shape == (2, 2)
    assert b_new['B'].shape == (2, 1)
