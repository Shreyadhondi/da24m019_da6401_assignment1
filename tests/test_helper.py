import numpy as np
import pytest
from Helper_Functions import (
    update_grad, SGD_step, momentum_velocity_update, momentum_step,
    NAG_step, rmsprop_update_grad, rmsprop_step, adam_update_momentum,
    adam_bias_correction, adam_step, Nadam_step
)

@pytest.fixture
def dummy_dicts():
    dw = {"W_layer_1": np.ones((2, 2))}
    db = {"B_layer_1": np.ones((2, 1))}
    grad_W = {"W_layer_1": np.ones((2, 2)) * 2}
    grad_B = {"B_layer_1": np.ones((2, 1)) * 2}
    W = {"W_layer_1": np.ones((2, 2))}
    B = {"B_layer_1": np.ones((2, 1))}
    v_w = {"W_layer_1": np.zeros((2, 2))}
    v_b = {"B_layer_1": np.zeros((2, 1))}
    m_w = {"W_layer_1": np.zeros((2, 2))}
    m_b = {"B_layer_1": np.zeros((2, 1))}
    return dw, db, grad_W, grad_B, W, B, v_w, v_b, m_w, m_b

def test_update_grad(dummy_dicts):
    dw, db, grad_W, grad_B, _, _, _, _, _, _ = dummy_dicts
    updated_dw, updated_db = update_grad(dw.copy(), grad_W, db.copy(), grad_B)
    assert np.all(updated_dw["W_layer_1"] == 3)
    assert np.all(updated_db["B_layer_1"] == 3)

def test_SGD_step(dummy_dicts):
    dw, db, _, _, W, B, _, _, _, _ = dummy_dicts
    new_W, new_B = SGD_step(W.copy(), B.copy(), dw, db, eta=0.01, batch_size=2, weight_decay=0.1)
    assert new_W["W_layer_1"].shape == (2, 2)
    assert new_B["B_layer_1"].shape == (2, 1)

def test_momentum_velocity_update(dummy_dicts):
    dw, db, _, _, _, _, v_w, v_b, _, _ = dummy_dicts
    v_w, v_b = momentum_velocity_update(v_w, v_b, momentum=0.9, dw=dw, db=db, eta=0.01, batch_size=2)
    assert v_w["W_layer_1"].shape == (2, 2)

def test_momentum_step(dummy_dicts):
    _, _, _, _, W, B, v_w, v_b, _, _ = dummy_dicts
    W_new, B_new = momentum_step(W.copy(), B.copy(), v_w, v_b, eta=0.01, batch_size=2, weight_decay=0.1)
    assert W_new["W_layer_1"].shape == (2, 2)

def test_NAG_step(dummy_dicts):
    dw, db, _, _, W, B, v_w, v_b, _, _ = dummy_dicts
    W_new, B_new = NAG_step(W.copy(), B.copy(), momentum=0.9, v_w=v_w, v_b=v_b, dw=dw, db=db, eta=0.01, batch_size=2, weight_decay=0.1)
    assert W_new["W_layer_1"].shape == (2, 2)

def test_rmsprop_update_grad(dummy_dicts):
    dw, db, _, _, _, _, v_w, v_b, _, _ = dummy_dicts
    v_w_new, v_b_new = rmsprop_update_grad(v_w.copy(), v_b.copy(), beta=0.9, dw=dw, db=db, batch_size=2)
    assert v_w_new["W_layer_1"].shape == (2, 2)

def test_rmsprop_step(dummy_dicts):
    dw, db, _, _, W, B, v_w, v_b, _, _ = dummy_dicts
    new_W, new_B = rmsprop_step(W.copy(), B.copy(), v_w, v_b, eps=1e-8, dw=dw, db=db, eta=0.01, batch_size=2, weight_decay=0.1)
    assert new_W["W_layer_1"].shape == (2, 2)

def test_adam_update_momentum(dummy_dicts):
    dw, db, _, _, _, _, _, _, m_w, m_b = dummy_dicts
    m_w_new, m_b_new = adam_update_momentum(m_w.copy(), m_b.copy(), beta1=0.9, dw=dw, db=db, batch_size=2)
    assert m_w_new["W_layer_1"].shape == (2, 2)

def test_adam_bias_correction(dummy_dicts):
    _, _, _, _, _, _, _, _, m_w, m_b = dummy_dicts
    m_w_hat, m_b_hat = adam_bias_correction(m_w, m_b, t=1, beta=0.9)
    assert m_w_hat["W_layer_1"].shape == (2, 2)

def test_adam_step(dummy_dicts):
    dw, db, _, _, W, B, v_w, v_b, m_w, m_b = dummy_dicts
    m_w_hat, m_b_hat = adam_bias_correction(m_w, m_b, t=1, beta=0.9)
    v_w_hat, v_b_hat = adam_bias_correction(v_w, v_b, t=1, beta=0.9)
    W_new, B_new = adam_step(W.copy(), B.copy(), v_w_hat, v_b_hat, m_w_hat, m_b_hat, eps=1e-8, eta=0.01, batch_size=2, weight_decay=0.1)
    assert W_new["W_layer_1"].shape == (2, 2)

def test_Nadam_step(dummy_dicts):
    dw, db, _, _, W, B, v_w, v_b, m_w, m_b = dummy_dicts
    v_w_hat, v_b_hat = adam_bias_correction(v_w, v_b, t=1, beta=0.9)
    m_w_hat, m_b_hat = adam_bias_correction(m_w, m_b, t=1, beta=0.9)
    W_new, B_new = Nadam_step(
        t=1, W_dict=W.copy(), B_dict=B.copy(),
        dw=dw, db=db,
        eta=0.01, batch_size=2,
        v_w_hat=v_w_hat, v_b_hat=v_b_hat,
        m_w_hat=m_w_hat, m_b_hat=m_b_hat,
        eps=1e-8, beta1=0.9, weight_decay=0.1
    )
    assert W_new["W_layer_1"].shape == (2, 2)
