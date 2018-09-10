from unittest import TestCase
from deepNN import *


class deepNNtest(TestCase):
    def test_initialize_parameters(self):
        parameters = initialize_parameters(3, 2, 1)
        W1 = np.array([[0.01624345, -0.00611756, -0.00528172], [-0.01072969, 0.00865408, -0.02301539]])
        np.testing.assert_almost_equal(parameters["W1"], W1)
        b1 = np.array([[0.], [0.]])
        np.testing.assert_almost_equal(parameters["b1"], b1)
        W2 = np.array([[0.01744812, -0.00761207]])
        np.testing.assert_almost_equal(parameters["W2"], W2)
        b2 = np.array([[0.]])
        np.testing.assert_almost_equal(parameters["b2"], b2)

    def test_initialize_parameters_deep(self):
        parameters = initialize_parameters_deep([5, 4, 3])
        W1 = np.array([[0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
                       [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
                       [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034],
                       [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]])
        np.testing.assert_almost_equal(parameters["W1"], W1)
        b1 = np.array([[0.], [0.], [0.], [0.]])
        np.testing.assert_almost_equal(parameters["b1"], b1)
        W2 = np.array([[-0.01185047, -0.0020565, 0.01486148, 0.00236716],
                       [-0.01023785, -0.00712993, 0.00625245, -0.00160513],
                       [-0.00768836, -0.00230031, 0.00745056, 0.01976111]])
        np.testing.assert_almost_equal(parameters["W2"], W2)
        b2 = np.array([[0.], [0.], [0.]])
        np.testing.assert_almost_equal(parameters["b2"], b2)

    def test_linear_forward(self):
        A, W, b = linear_forward_test_case()
        Z, linear_cache = linear_forward(A, W, b)
        Ze = np.array([[3.26295337, -1.23429987]])
        np.testing.assert_allclose(Z, Ze)

    def test_linear_activation_forward(self):
        A_prev, W, b = linear_activation_forward_test_case()

        A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
        Ae_SIGMOID = np.array([[0.96890023, 0.11013289]])
        np.testing.assert_allclose(A, Ae_SIGMOID)
        print("With sigmoid: A = " + str(A))

        A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
        Ae_RELU = np.array([[3.43896131, 0.]])
        np.testing.assert_allclose(A, Ae_RELU)

    def test_L_model_forward_2hidden(self):
        X, parameters = L_model_forward_test_case_2hidden()
        AL, caches = L_model_forward(X, parameters)
        ALe = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])
        np.testing.assert_allclose(AL, ALe)
        np.testing.assert_almost_equal(len(caches), 3)

    def test_compute_cost(self):
        Y, AL = compute_cost_test_case()
        np.testing.assert_equal(compute_cost(AL, Y), 0.41493159961539694)

    def test_linear_backward(self):
        dZ, linear_cache = linear_backward_test_case()
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        dA_prevE = np.array([[0.51822968, -0.19517421],
                             [-0.40506361, 0.15255393],
                             [2.37496825, -0.89445391]])
        np.testing.assert_almost_equal(dA_prev, dA_prevE)
        dWe = np.array([[-0.10076895, 1.40685096, 1.64992505]])
        np.testing.assert_almost_equal(dW, dWe)
        dbe = np.array([[0.50629448]])
        np.testing.assert_almost_equal(db, dbe)

    def test_linear_activation_backward(self):
        dAL, linear_activation_cache = linear_activation_backward_test_case()

        dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation="sigmoid")
        dA_preve = np.array([[0.11017994, 0.01105339], [0.09466817, 0.00949723], [-0.05743092, -0.00576154]])
        dWe = np.array([[0.10266786, 0.09778551, -0.01968084]])
        dbe = np.array([[-0.05729622]])
        np.testing.assert_almost_equal(dA_prev, dA_preve)
        np.testing.assert_almost_equal(dW, dWe)
        np.testing.assert_almost_equal(db, dbe)

        dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation="relu")
        dA_preve = np.array([[0.44090989, 0.], [0.37883606, 0.], [-0.2298228, 0.]])
        dWe = np.array([[0.44513824, 0.37371418, -0.10478989]])
        dbe = np.array([[-0.20837892]])
        np.testing.assert_almost_equal(dA_prev, dA_preve)
        np.testing.assert_almost_equal(dW, dWe)
        np.testing.assert_almost_equal(db, dbe)

    def test_L_model_backward(self):
        AL, Y_assess, caches = L_model_backward_test_case()
        grads = L_model_backward(AL, Y_assess, caches)
        dW1 = np.array([[0.41010002, 0.07807203, 0.13798444, 0.10502167],
                        [0., 0., 0., 0.],
                        [0.05283652, 0.01005865, 0.01777766, 0.0135308]])
        np.testing.assert_almost_equal(grads["dW1"], dW1)

        db1 = np.array([[-0.22007063], [0.], [-0.02835349]])
        np.testing.assert_almost_equal(grads["db1"], db1)

        dA1 = np.array([[0.12913162, -0.44014127],
                        [-0.14175655, 0.48317296],
                        [0.01663708, -0.05670698]])
        np.testing.assert_almost_equal(grads["dA1"], dA1)

    def test_update_parameters(self):
        parameters, grads = update_parameters_test_case()
        parameters = update_parameters(parameters, grads, 0.1)
        W1 = np.array([[-0.59562069, -0.09991781, -2.14584584, 1.82662008],
                       [-1.76569676, -0.80627147, 0.51115557, -1.18258802],
                       [-1.0535704, -0.86128581, 0.68284052, 2.20374577]])
        b1 = np.array([[-0.04659241], [-1.28888275], [0.53405496]])
        W2 = np.array([[-0.55569196, 0.0354055, 1.32964895]])
        b2 = np.array([[-0.84610769]])
        np.testing.assert_allclose(W1, parameters["W1"])
        np.testing.assert_allclose(b1, parameters["b1"])
        np.testing.assert_allclose(W2, parameters["W2"])
        np.testing.assert_allclose(b2, parameters["b2"])
