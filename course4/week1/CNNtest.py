from unittest import TestCase
from CNN import *


class CNNtest(TestCase):

    def test_zero_pad(self):
        np.random.seed(1)
        x = np.random.randn(4, 3, 3, 2)
        x_11_des = [[0.90085595, - 0.68372786], [-0.12289023, - 0.93576943], [-0.26788808, 0.53035547]]
        np.testing.assert_almost_equal(x[1, 1], x_11_des)
        x_pad = zero_pad(x, 2)
        x_pad_shape_des = (4, 7, 7, 2)
        x_pad11_des = [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]
        np.testing.assert_equal(x_pad.shape, x_pad_shape_des)
        np.testing.assert_almost_equal(x_pad[1, 1], x_pad11_des)

    def test_conv_single_step(self):
        np.random.seed(1)
        a_slice_prev = np.random.randn(4, 4, 3)
        W = np.random.randn(4, 4, 3)
        b = np.random.randn(1, 1, 1)
        Z = conv_single_step(a_slice_prev, W, b)
        Zdes = -6.99908945068
        np.testing.assert_almost_equal(Z, Zdes)

    def test_conv_forward(self):
        Z, cache_conv = self.gen_conv_forward()
        np.testing.assert_almost_equal(np.mean(Z), 0.0489952035289)
        np.testing.assert_almost_equal(Z[3, 2, 1],
                                       [-0.61490741, - 6.7439236, - 2.55153897, 1.75698377, 3.56208902, 0.53036437,
                                        5.18531798, 8.75898442])
        np.testing.assert_almost_equal(cache_conv[0][1][2][3], [-0.20075807, 0.18656139, 0.41005165])

    def gen_conv_forward(self):
        np.random.seed(1)
        A_prev = np.random.randn(10, 4, 4, 3)
        W = np.random.randn(2, 2, 3, 8)
        b = np.random.randn(1, 1, 1, 8)
        hparameters = {"pad": 2, "stride": 2}
        Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
        return Z, cache_conv

    def test_pool_forward(self):
        np.random.seed(1)
        A_prev = np.random.randn(2, 4, 4, 3)
        np.testing.assert_almost_equal(A_prev[1, 2, 2, :], [0.23009474, 0.76201118, - 0.22232814])
        hparameters = {"stride": 2, "f": 3}
        A1 = [[[[1.74481176, 0.86540763, 1.13376944]]], [[[1.13162939, 1.51981682, 2.18557541]]]]
        A, cache = pool_forward(A_prev, hparameters)
        np.testing.assert_almost_equal(A, A1)
        A2 = [[[[0.02105773, - 0.20328806, - 0.40389855]]], [[[-0.22154621, 0.51716526, 0.48155844]]]]
        A, cache = pool_forward(A_prev, hparameters, mode="average")
        np.testing.assert_almost_equal(A, A2)

    def test_conv_backward(self):
        Z, cache_conv = self.gen_conv_forward()
        dA, dW, db = conv_backward(Z, cache_conv)
        np.testing.assert_almost_equal(np.mean(dA), 1.45243777754)
        np.testing.assert_almost_equal(np.mean(dW), 1.72699145831)
        np.testing.assert_almost_equal(np.mean(db), 7.83923256462)

    def test_create_mask_from_window(self):
        np.random.seed(1)
        x = np.random.randn(2, 3)
        mask = create_mask_from_window(x)
        np.testing.assert_almost_equal(x,
                                       [[1.62434536, -0.61175641, -0.52817175], [-1.07296862, 0.86540763, -2.3015387]])
        np.testing.assert_almost_equal(mask, [[True, False, False], [False, False, False]])

    def test_distribute_value(self):
        a = distribute_value(2, (2, 2))
        np.testing.assert_almost_equal(a, [[0.5, 0.5], [0.5, 0.5]])

    def test_pool_backwards(self):
        np.random.seed(1)
        A_prev = np.random.randn(5, 5, 3, 2)
        hparameters = {"stride": 1, "f": 2}
        A, cache = pool_forward(A_prev, hparameters)
        dA = np.random.randn(5, 4, 2, 2)
        dA_prev = pool_backward(dA, cache, mode="max")
        np.testing.assert_almost_equal(np.mean(dA), 0.145713902729)
        np.testing.assert_almost_equal(dA_prev[1, 1], [[0., 0.], [5.05844394, -1.68282702], [0., 0.]])
        dA_prev = pool_backward(dA, cache, mode="average")
        np.testing.assert_almost_equal(np.mean(dA), 0.145713902729)
        np.testing.assert_almost_equal(dA_prev[1, 1],
                                       [[0.08485462, 0.2787552], [1.26461098, -0.25749373], [1.17975636, -0.53624893]])
