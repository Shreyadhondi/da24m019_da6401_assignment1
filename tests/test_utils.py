# tests/test_utils.py

import numpy as np
from Utils import create_HL_list, Xavier_weight_init, ReLu, sigmoid, activation_func, output_func, loss_func_CEL, loss_func, L2_Loss, unisonShuffleDataset

def test_create_HL_list():
    assert create_HL_list(3, 64) == [64, 64, 64]

def test_xavier_weight_init_shape():
    w = Xavier_weight_init(128, 64)
    assert w.shape == (128, 64)

def test_relu_output():
    x = np.array([[-1.0], [0.0], [2.0]])
    assert np.allclose(ReLu(x), np.array([[0.], [0.], [2.]]))

def test_sigmoid_output():
    x = np.array([[0.0]])
    assert np.allclose(sigmoid(x), np.array([[0.5]]))

def test_activation_func_relu():
    x = np.array([[1.0], [-2.0]])
    relu_out = activation_func(x, "ReLu")
    assert np.all(relu_out >= 0)

def test_output_func_softmax():
    x = np.array([[2.0], [1.0], [0.1]])
    out = output_func(x)
    assert np.isclose(np.sum(out), 1.0)

def test_loss_func_CEL_valid():
    actual = np.array([[0], [1]])
    pred = np.array([[0.1], [0.9]])
    loss = loss_func_CEL(actual, pred)
    assert loss > 0

def test_unison_shuffle():
    a = np.array([[1], [2], [3]])
    b = np.array([[4], [5], [6]])
    a_s, b_s = unisonShuffleDataset(a, b)
    assert sorted(a_s.flatten()) == [1, 2, 3]
    assert sorted(b_s.flatten()) == [4, 5, 6]
