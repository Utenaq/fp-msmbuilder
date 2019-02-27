from __future__ import print_function

import numpy as np
import scipy.linalg
from scipy.optimize import check_grad, approx_fprime

try:
    import numdifftools as nd
except ImportError:
    print('Missig test dependency')
    print('  pip installl numdifftools')
    raise

from msmbuilder.msm import _ratematrix_PES
from msmbuilder.msm import PESContinuousTimeMSM, MarkovStateModel
from msmbuilder.example_datasets import MullerPotential, DoubleWell
from msmbuilder.example_datasets.muller import MULLER_PARAMETERS as PARAMS
from msmbuilder.cluster import NDGrid
from msmbuilder import utils
import tempfile
import shutil

random = np.random.RandomState(0)


def example_theta(n):
    return random.uniform(0, 1, size=(n - 1) + n)


def test_build_ratemat():
    # test build_ratemat
    n = 4
    theta = example_theta(n)
    K = np.zeros((n, n))
    _ratematrix_PES.build_ratemat(theta, n, K)

    # diagonal entries are negative
    assert np.all(np.diag(K) < 0)
    # off diagonal entries are non-negative
    assert np.all(np.extract(1 - np.eye(n), K) >= 0)
    # row-sums are 0
    np.testing.assert_array_almost_equal(np.sum(K, axis=1), 0)


def test_dK_dtheta_1():
    # test function `dK_dtheta_A` against the numerical gradient of
    # `build_ratemat
    n = 4
    theta = example_theta(n)

    def func(x, i, j):
        # (i,j) entry of the rate matrix
        K = np.zeros((n, n))
        _ratematrix_PES.build_ratemat(x, n, K)
        return K[i, j]

    def grad(x, i, j):
        # gradient of the (i,j) entry of the rate matrix w.r.t. theta
        dKu = np.zeros((n, n))
        g = np.zeros(len(x))
        for u in range(len(x)):
            _ratematrix_PES.dK_dtheta_ij(x, n, u, None, dKu)
            g[u] = dKu[i, j]
        return g

    for i in range(n):
        for j in range(n):
            assert check_grad(func, grad, theta, i, j) < 1e-7

def test_dK_dtheta_2():
    # test function `dK_dtheta_A` to make sure that the part that hadamards
    # the matrix against A is correct.
    n = 4
    A = random.randn(4, 4)
    theta = example_theta(n)

    for u in range(len(theta)):
        dKu = np.zeros((n, n))
        _ratematrix_PES.dK_dtheta_ij(theta, n, u, None, dKu)
        value1 = (dKu * A).sum()

        dKu = np.zeros((n, n))
        value2 = _ratematrix_PES.dK_dtheta_ij(theta, n, u, A, dKu)
        value3 = _ratematrix_PES.dK_dtheta_ij(theta, n, u, A)

        np.testing.assert_approx_equal(value1, value2)
        np.testing.assert_approx_equal(value1, value3)

def test_dK_dtheta_3():
    # test dK_dtheta_ij vs dK_dtheta_u. both return slices of the same 3D
    # tensor, so by repeated calls to both functions we can build the whole
    # tensor using both approaches and check that they're equal.

    for n in [3, 4]:
        theta = example_theta(n)

        dKuij1 = np.zeros((len(theta), n, n))
        dKuij2 = np.zeros((len(theta), n, n))

        for u in range(len(theta)):
            _ratematrix_PES.dK_dtheta_ij(theta, n, u, None, dKuij1[u])

        for i in range(n):
            for j in range(n):
                _ratematrix_PES.dK_dtheta_u(theta, n, i, j, out=dKuij2[:, i, j])

        np.testing.assert_array_almost_equal(dKuij1, dKuij2)


def test_dK_dtheta_4():
    # check that the dot product part of dK_dtheta_u works
    n = 4
    theta = example_theta(n)
    A = random.randn(len(theta), len(theta))

    for i in range(n):
        for j in range(n):
            grad = np.zeros(len(theta))
            _ratematrix_PES.dK_dtheta_u(theta, n, i, j, out=grad)
            gradprod1 = np.dot(grad, A)

            gradprod2 = np.zeros(len(theta))
            grad2 = np.zeros(len(theta))
            _ratematrix_PES.dK_dtheta_u(theta, n, i, j, out=grad2, A=A,
                                    out2=gradprod2)

            np.testing.assert_almost_equal(grad, grad2)
            np.testing.assert_almost_equal(gradprod1, gradprod2)
            np.testing.assert_almost_equal(np.dot(grad2, A), gradprod2)


def test_dK_dtheta_5():
    n = 4
    theta = np.array(
        [2.59193443e-02, 3.08837678e-03,
         2.56956907e-02, -1.48051536e+00, -1.51759911e+00,
         -1.34983215e+00, -1.22431771e+00])
    size = len(theta)

    dK1 = np.zeros((size, n, n))
    dK2 = np.zeros((size, n, n))
    dK3 = np.zeros((size, n, n))

    for u in range(size):
        _ratematrix_PES.dK_dtheta_ij(theta, n, u, A=None, out=dK1[u, :, :])
    for i in range(n):
        for j in range(n):
            _ratematrix_PES.dK_dtheta_u(theta, n, i, j, out=dK2[:, i, j])
    for i in range(n):
        for j in range(n):
            dKij = np.zeros(size)
            _ratematrix_PES.dK_dtheta_u(theta, n, i, j, out=dKij)
            dK3[:, i, j] = dKij

    np.testing.assert_almost_equal(dK1, dK2)
    np.testing.assert_almost_equal(dK1, dK3)
    np.testing.assert_almost_equal(dK2, dK3)


def test_grad_logl_1():
    # test the gradient of the `loglikelihood` against a numerical gradient
    n = 4
    t = 1
    C = random.randint(10, size=(n, n)).astype(float)
    theta0 = example_theta(n)

    def func(theta):
        return _ratematrix_PES.loglikelihood(theta, C, t=t)[0]

    def grad(theta):
        return _ratematrix_PES.loglikelihood(theta, C, t=t)[1]

    assert check_grad(func, grad, theta0) < 1e-4


def test_dw_1():
    # test the gradient of the eigenvalues of K
    n = 5
    t = 1.0
    theta0 = example_theta(n)

    def grad(theta, i):
        # gradient of the ith eigenvalue of K with respect to theta
        K = np.zeros((n, n))
        _ratematrix_PES.build_ratemat(theta, n, K)
        w, V = scipy.linalg.eig(K)
        order = np.argsort(np.real(w))

        V = np.real(np.ascontiguousarray(V[:, order]))
        U = np.ascontiguousarray(scipy.linalg.inv(V).T)

        g = np.zeros(len(theta))

        for u in range(len(theta)):
            dKu = np.zeros((n, n))
            _ratematrix_PES.dK_dtheta_ij(theta, n, u, None, dKu)
            out = np.zeros(n)
            temp = np.zeros(n)
            _ratematrix_PES.dw_du(dKu, U, V, n, temp, out)
            g[u] = out[i]
        return g

    def func(theta, i):
        # ith eigenvalue of K
        K = np.zeros((n, n))
        _ratematrix_PES.build_ratemat(theta, n, K)
        w = np.real(scipy.linalg.eigvals(K))
        w = np.sort(w)
        return w[i]

    for i in range(n):
        g1 = approx_fprime(theta0, func, 1e-7, i)
        g2 = grad(theta0, i)
        assert np.linalg.norm(g1 - g2) < 2e-6


def test_hessian_1():
    n = 3
    seqs = [
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2,
         1, 1, 1, 1, 2, 3, 3, 3, 3]]

    model = PESContinuousTimeMSM().fit(seqs)
    theta = model.theta_
    C = model.countsmat_

    hessian1 = _ratematrix_PES.hessian(theta, C)
    Hfun = nd.Jacobian(lambda x: _ratematrix_PES.loglikelihood(x, C)[1])
    hessian2 = Hfun(theta)

    # not sure what the cutoff here should be (see plot_test_hessian)
    diff = np.linalg.det(hessian1 - hessian2)
    print(hessian1)
    print(hessian2)
    print("hessian difference: %f" % diff)
    assert diff < 1e-4

    print(_ratematrix_PES.sigma_pi(-scipy.linalg.pinv(hessian1), theta, n))


def test_hessian_2():
    n = 3
    seqs = [
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2,
         2, 1, 1, 1, 1, 2, 3, 3, 3, 3]]

    model = PESContinuousTimeMSM().fit(seqs)
    print(model.timescales_)
    print(model.uncertainty_timescales())
    theta = model.theta_
    C = model.countsmat_
    print(C)

    C_flat = (C + C.T)[np.triu_indices_from(C, k=1)]
    print(C_flat)
    print('theta', theta, '\n')
    inds = np.where(theta != 0)[0]

    hessian1 = _ratematrix_PES.hessian(theta, C, inds=inds)
    hessian2 = nd.Jacobian(lambda x: _ratematrix_PES.loglikelihood(x, C)[1])(theta)
    hessian3 = nd.Hessian(lambda x: _ratematrix_PES.loglikelihood(x, C)[0])(theta)

    np.set_printoptions(precision=3)

    # H1 = hessian1[np.ix_(active, active)]
    # H2 = hessian2[np.ix_(active, active)]
    # H3 = hessian2[np.ix_(active, active)]

    print(hessian1, '\n')
    print(hessian2, '\n')
    # print(hessian3)
    print('\n')

    info1 = np.zeros((len(theta), len(theta)))
    info2 = np.zeros((len(theta), len(theta)))
    info1[np.ix_(inds, inds)] = scipy.linalg.pinv(-hessian1)
    info2[np.ix_(inds, inds)] = scipy.linalg.pinv(-hessian2[np.ix_(inds, inds)])

    print('Inverse Hessian')
    print(info1)
    print(info2)
    # print(scipy.linalg.pinv(hessian2))
    # print(scipy.linalg.pinv(hessian1)[np.ix_(last, last)])
    # print(scipy.linalg.pinv(hessian2)[np.ix_(last, last)])

    print(_ratematrix_PES.sigma_pi(info1, theta, n))
    print(_ratematrix_PES.sigma_pi(info2, theta, n))

    # print(_ratematrix_PES.sigma_pi(scipy.linalg.pinv(-hessian2), theta, n))
    # print(_ratematrix_PES.sigma_pi(scipy.linalg.pinv(-hessian3), theta, n))

    # print(np.linalg.norm(H1-H2))
    #
    # # print(hessian1.shape)
    # # print(hessian1-hessian2)
    #
    # # not sure what the cutoff here should be (see plot_test_hessian)
    # assert np.linalg.norm(hessian1-hessian2) < 1e-6


def test_hessian_3():
    grid = NDGrid(n_bins_per_feature=4, min=-np.pi, max=np.pi)
    trajs = DoubleWell(random_state=0).get_cached().trajectories
    seqs = grid.fit_transform(trajs)
    seqs = [seqs[i] for i in range(10)]

    lag_time = 10
    model = PESContinuousTimeMSM(verbose=False, lag_time=lag_time)
    model.fit(seqs)
    msm = MarkovStateModel(verbose=False, lag_time=lag_time)
    print(model.summarize())
    # print('MSM timescales\n', msm.fit(seqs).timescales_)
    print('Uncertainty K\n', model.uncertainty_K())
    print('Uncertainty eigs\n', model.uncertainty_eigenvalues())

'''
def test_fit_1():
    # call fit, compare to MSM
    sequence = [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    model = PESContinuousTimeMSM(verbose=False)
    model.fit([sequence])

    msm = MarkovStateModel(verbose=False)
    msm.fit([sequence])

    # they shouldn't be equal in general, but for this input they seem to be
    np.testing.assert_array_almost_equal(model.transmat_, msm.transmat_)
'''

def test_fit_2():
    grid = NDGrid(n_bins_per_feature=5, min=-np.pi, max=np.pi)
    trajs = DoubleWell(random_state=0).get_cached().trajectories
    seqs = grid.fit_transform(trajs)

    model = PESContinuousTimeMSM(verbose=False, lag_time=10)
    model.fit(seqs)
    t1 = np.sort(model.timescales_)
    t2 = -1 / np.sort(np.log(np.linalg.eigvals(model.transmat_))[1:])

    model = MarkovStateModel(verbose=False, lag_time=10)
    model.fit(seqs)
    t3 = np.sort(model.timescales_)

    np.testing.assert_array_almost_equal(t1, t2)
    # timescales should be similar to MSM (withing 50%)
    assert abs(t1[-1] - t3[-1]) / t1[-1] < 0.50


def test_score_1():
    grid = NDGrid(n_bins_per_feature=5, min=-np.pi, max=np.pi)
    trajs = DoubleWell(random_state=0).get_cached().trajectories
    seqs = grid.fit_transform(trajs)
    model = (PESContinuousTimeMSM(verbose=False, lag_time=10, n_timescales=3)
             .fit(seqs))
    np.testing.assert_approx_equal(model.score(seqs), model.score_)


def test_uncertainties_backward():
    n = 4
    grid = NDGrid(n_bins_per_feature=n, min=-np.pi, max=np.pi)
    trajs = DoubleWell(random_state=0).get_cached().trajectories
    seqs = grid.fit_transform(trajs)

    model = PESContinuousTimeMSM(verbose=False).fit(seqs)
    sigma_ts = model.uncertainty_timescales()
    sigma_lambda = model.uncertainty_eigenvalues()
    sigma_pi = model.uncertainty_pi()
    sigma_K = model.uncertainty_K()

    yield lambda: np.testing.assert_array_almost_equal(
        sigma_ts, [9.508936, 0.124428, 0.117638])
    yield lambda: np.testing.assert_array_almost_equal(
        sigma_lambda,
        [1.76569687e-19, 7.14216858e-05, 3.31210649e-04, 3.55556718e-04])
    yield lambda: np.testing.assert_array_almost_equal(
        sigma_pi, [0.007496, 0.006564, 0.006348, 0.007863])
    yield lambda: np.testing.assert_array_almost_equal(
        sigma_K,
        [[0.000339, 0.000339, 0., 0.],
         [0.000352, 0.000372, 0.000122, 0.],
         [0., 0.000103, 0.000344, 0.000329],
         [0., 0., 0.00029, 0.00029]])
    yield lambda: np.testing.assert_array_almost_equal(
        model.ratemat_,
        [[-0.0254, 0.0254, 0., 0.],
         [0.02636, -0.029629, 0.003269, 0.],
         [0., 0.002764, -0.030085, 0.027321],
         [0., 0., 0.024098, -0.024098]])


def test_score_2():
    ds = MullerPotential(random_state=0).get_cached().trajectories
    cluster = NDGrid(n_bins_per_feature=6,
                     min=[PARAMS['MIN_X'], PARAMS['MIN_Y']],
                     max=[PARAMS['MAX_X'], PARAMS['MAX_Y']])
    assignments = cluster.fit_transform(ds)
    test_indices = [5, 0, 4, 1, 2]
    train_indices = [3, 6, 7, 8, 9]

    model = PESContinuousTimeMSM(lag_time=3, n_timescales=1)
    temp='3.13268076e-02  1.45678356e-01  3.12558665e-01  0.10000000e+00\
  2.72951862e-02  9.64773504e-02  1.26398091e-01  1.27775726e-01\
  2.74208403e-03  9.57265955e-04  1.89498433e-01  0.10010000e+00\
  9.73644477e-02  4.16877008e-02  0.10010000e+00  1.73803374e-01\
  4.31281724e-02  0.10001000e+00  0.10010000e+00  4.03763450e-01\
  0.10001000e+00  0.10001000e+00  0.10010000e+00  2.78156537e-01\
 -7.75852152e+00 -4.77716045e+00 -2.67428479e+00 -4.33901900e+00\
 -9.23925293e+00 -6.65216281e+00 -4.88309143e+00 -4.04247463e+00\
 -4.67140081e+00 -7.95471679e+00 -6.26342874e+00 -6.02515423e+00\
 -5.64532492e+00 -5.56770596e+00 -7.66164067e+00 -6.22050765e+00\
 -2.12577068e+00 -4.84152585e+00 -9.21360166e+00 -1.43207874e+00\
 -8.55459835e-01 -9.21329384e+00 -6.99418825e+00 -2.73060233e+00\
 -6.60364249e+00'
    model.theta_= list(map(np.float64, temp.split()))

    model.fit([assignments[i] for i in train_indices])
    print('Initial theta: \n')
    print(model._initial_guess(model.countsmat_))
    test = model.score([assignments[i] for i in test_indices])
    train = model.score_
    print('train', train, 'test', test)
    print(model.optimizer_state_)
    # print(model.summarize())
    assert 1 <= test < 2
    assert 1 <= train < 2


def test_score_3():
    ds = MullerPotential(random_state=0).get_cached().trajectories
    cluster = NDGrid(n_bins_per_feature=6,
                     min=[PARAMS['MIN_X'], PARAMS['MIN_Y']],
                     max=[PARAMS['MAX_X'], PARAMS['MAX_Y']])

    assignments = cluster.fit_transform(ds)

    train_indices = [9, 4, 3, 6, 2]
    test_indices = [8, 0, 5, 7, 1]
    temp = '1.0929e-02  5.4147e-02  9.8362e-02  0.1000e+00  6.0455e-02  2.8775e-02\
  6.6456e-02  3.3957e-02  4.1484e-03  0.1000e+00  5.0847e-02  1.1516e-02\
  3.5266e-02  1.2830e-02  0.1000e+00  2.1801e-02  1.6639e-02  9.4932e-03\
  0.1000e+00  0.1000e+00  1.1050e-01  4.0076e-03  0.1000e+00  0.1000e+00\
  1.8930e-02 -7.1060e+00 -4.5787e+00 -2.4950e+00 -4.0964e+00 -7.4127e+00\
 -6.7574e+00 -4.7137e+00 -3.9530e+00 -4.5781e+00 -7.4585e+00 -6.4634e+00\
 -5.8060e+00 -5.4783e+00 -5.3519e+00 -7.4653e+00 -6.5113e+00 -2.1477e+00\
 -4.8138e+00 -9.7187e+00 -9.0358e+00 -1.4599e+00 -8.8985e-01 -8.3461e+00\
 -7.0930e+00 -2.7618e+00 -6.7421e+00'

    model = PESContinuousTimeMSM(lag_time=3, n_timescales=1, sliding_window=False,
                              ergodic_cutoff=1)
    model.theta_ = list(map(np.float64, temp.split()))
    train_data = [assignments[i] for i in train_indices]
    test_data = [assignments[i] for i in test_indices]

    model.fit(train_data)
    print(model.summarize())
    train = model.score_
    test = model.score(test_data)
    print(train, test)

'''
def test_guess():
    ds = MullerPotential(random_state=0).get_cached().trajectories
    cluster = NDGrid(n_bins_per_feature=5,
                     min=[PARAMS['MIN_X'], PARAMS['MIN_Y']],
                     max=[PARAMS['MAX_X'], PARAMS['MAX_Y']])
    assignments = cluster.fit_transform(ds)
    print(assignments)

    model1 = PESContinuousTimeMSM(guess='log')
    temp = ' 1.000e-02  3.748e-01  1.397e-01  3.621e-01  1.922e-04  1.166e-01\
    1.525e-01  2.157e-01  3.109e-01  8.061e-05  4.574e-01  2.782e-02\
  3.722e-02  1.686e-01  4.482e-04  8.358e-02  3.082e-01  7.167e-01\
  5.790e-01 -9.694e+00 -9.082e+00 -6.247e+00 -3.290e+00 -4.110e+00\
 -7.069e+00 -5.909e+00 -4.498e+00 -5.113e+00 -8.740e+00 -4.742e+00\
 -3.720e+00 -7.588e+00 -9.899e+00 -1.198e+01 -5.856e+00 -3.958e-01\
 -3.115e+00 -2.551e+00 -2.391e+00'
    model1.theta_ = list(map(np.float64, temp.split()))
    # model1.theta_ = np.random.rand(len(model1.theta_))
    print('Initial theta: \n')
    print(model1._initial_guess(model1.countsmat_))
    model1.fit(assignments)

    print(model1.optimizer_state_)

    temp = ' 1.000e-02  3.748e-01  1.397e-01  3.621e-01  1.922e-04  1.166e-01\
       1.525e-01  2.157e-01  3.109e-01  8.061e-05  4.574e-01  2.782e-02\
     3.722e-02  1.686e-01  4.482e-04  8.358e-02  3.082e-01  7.167e-01\
     5.790e-01 -9.694e+00 -9.082e+00 -6.247e+00 -3.290e+00 -4.110e+00\
    -7.069e+00 -5.909e+00 -4.498e+00 -5.113e+00 -8.740e+00 -4.742e+00\
    -3.720e+00 -7.588e+00 -9.899e+00 -1.198e+01 -5.856e+00 -3.958e-01\
    -3.115e+00 -2.551e+00 -2.391e+00'
    model2 = PESContinuousTimeMSM(guess='pseudo')
    model2.theta_ = list(map(np.float64, temp.split()))
    # model2.theta_ = np.random.rand(len(model1.theta_))
    print('Initial theta: \n')
    print(model2._initial_guess(model1.countsmat_))
    model2.fit(assignments)
    print(model2.optimizer_state_)

    diff = model1.loglikelihoods_[-1] - model2.loglikelihoods_[-1]
    assert np.abs(diff) < 1e-3
    assert np.max(np.abs(model1.ratemat_ - model2.ratemat_)) < 1e-1
'''

def test_doublewell():
    trjs = DoubleWell(random_state=0).get_cached().trajectories
    for n_states in [10, 36]:
        clusterer = NDGrid(n_bins_per_feature=n_states)
        assignments = clusterer.fit_transform(trjs)

        for sliding_window in [True, False]:
            model = PESContinuousTimeMSM(lag_time=100,
                                      sliding_window=sliding_window)
            model.fit(assignments)
            # print(model.summarize())
            print(model.optimizer_state_)
            assert model.optimizer_state_.success


def test_dump():
    # gh-713
    sequence = [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    model = PESContinuousTimeMSM(verbose=False)
    model.fit([sequence])

    d = tempfile.mkdtemp()
    try:
        utils.dump(model, '{}/cmodel'.format(d))
        m2 = utils.load('{}/cmodel'.format(d))
        np.testing.assert_array_almost_equal(model.transmat_, m2.transmat_)
    finally:
        shutil.rmtree(d)


if __name__ == '__main__':
    # test_guess()
    test_score_3()
