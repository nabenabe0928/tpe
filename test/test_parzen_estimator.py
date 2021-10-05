import numpy as np
import unittest

from optimizer.parzen_estimator.kernel import (
    UniformKernel,
)
from optimizer.parzen_estimator.parzen_estimator import (
    CategoricalParzenEstimator,
    CategoricalPriorType,
    NumericalParzenEstimator,
    NumericalPriorType
)
from util.constants import default_weights


class TestCategoricalPriorType(unittest.TestCase):
    def test_get_kernel(self) -> None:
        try:
            CategoricalPriorType.no_prior.get_kernel(n_choices=4)
        except ValueError:
            pass
        except Exception:
            raise ValueError('test_get_kernel failed.')
        else:
            raise ValueError('The error was not raised.')

        kern = CategoricalPriorType.uniform.get_kernel(n_choices=5)
        if not isinstance(kern, UniformKernel):
            raise ValueError('The prior of Categorical must be UniformKernel. but got {}'.format(
                type(kern)
            ))


class TestNumericalPriorType(unittest.TestCase):
    def test_get_kernel(self) -> None:
        try:
            NumericalPriorType.no_prior.get_kernel(lb=-1, ub=1)
        except ValueError:
            pass
        except Exception:
            raise ValueError('test_get_kernel failed.')
        else:
            raise ValueError('The error was not raised.')

        n_samples = 10000
        rng = np.random.RandomState()
        kern = NumericalPriorType.gaussian.get_kernel(lb=-np.inf, ub=np.inf)
        samples = np.array([kern.sample(rng) for _ in range(n_samples)])
        assert 0.48 <= np.count_nonzero(samples >= 0) / n_samples <= 0.52
        assert 0.14 <= np.count_nonzero(samples >= 1) / n_samples <= 0.18
        assert 0.01 <= np.count_nonzero(samples >= 2) / n_samples <= 0.04

        kern = NumericalPriorType.uniform.get_kernel(lb=0, ub=10)
        samples = np.array([kern.sample(rng) for _ in range(10000)]).astype(np.int32)
        vals, counts = np.unique(samples, return_counts=True)
        assert 0.08 <= np.count_nonzero(samples < 1) / n_samples <= 0.12
        assert 0.18 <= np.count_nonzero(samples < 2) / n_samples <= 0.22
        assert 0.28 <= np.count_nonzero(samples < 3) / n_samples <= 0.32


class TestNumericalParzenEstimator(unittest.TestCase):
    def test_init(self) -> None:
        samples_set = [
            np.array([5])
        ]
        for samples in samples_set:
            try:
                NumericalParzenEstimator(samples=samples, lb=-3.0, ub=3.0, weight_func=default_weights)
            except ValueError:
                pass
            except Exception:
                raise ValueError('test_init failed.')
            else:
                raise ValueError('The error was not raised.')

    def test_sample(self) -> None:
        pass

    def test_basis_loglikelihood(self) -> None:
        samples = np.array([0, 1, 2, 3] * 10)
        pe = NumericalParzenEstimator(samples=samples, lb=-3.0, ub=3.0, weight_func=default_weights)
        assert pe.basis_loglikelihood(np.arange(4)).shape == (41, 4)
        pe = NumericalParzenEstimator(samples=samples, lb=-3.0, ub=3.0, weight_func=default_weights,
                                      prior=NumericalPriorType.no_prior)
        assert pe.basis_loglikelihood(np.arange(4)).shape == (40, 4)

        samples = np.array([
            0.79069728, 0.31428171, 0.57129296, 0.44143764, 0.40593304, 0.4402409,
            0.95061797, 0.10177345, 0.46096913, 0.47954407, 0.44844498, 0.40064817,
            0.42593935, 0.40812766, 0.45422322, 0.41891571, 0.45336276, 0.10557477,
            0.41424797, 0.49825133, 0.44741231, 0.5678886,  0.46689275, 0.48488553
        ])
        mus = np.array([0.4372727, 0.44549973])
        pe = NumericalParzenEstimator(samples=mus, lb=0.0, ub=1.0, weight_func=default_weights)
        assert np.allclose(pe.weights, [0.33333333, 0.33333333, 0.33333333])
        assert np.allclose([b.mu for b in pe.basis], [0.43727276, 0.44549973, 0.5])
        assert np.allclose([b.sigma for b in pe.basis], [0.02666232, 0.02666232, 1.])
        ans = [
            [-85.149656, -7.93393578, -9.9276751, 2.69336498, 2.01474622, 2.69936907,
             -182.644718, -76.4638619, 2.31061955, 1.44876723, 2.61777391, 1.7621133,
             2.61522227, 2.10811042, 2.50347903, 2.46854826, 2.52347548, -74.6799935,
             2.33268846, 0.0902222508, 2.63325341, -9.29401022, 2.08848182, 1.11107839],
            [-81.1070975, -9.40491002, -8.42427136, 2.6939598, 1.60444812, 2.6861141,
             -176.751405, -80.3941832, 2.53725123, 1.89036575, 2.69946428, 1.29065361,
             2.43645611, 1.72321057, 2.65204071, 2.20849794, 2.66207906, -78.5663224,
             2.01861858, 0.748318954, 2.7029927, -7.83000504, 2.38366731, 1.61449331],
            [-0.00127465297, 0.0237321586, 0.0384364572, 0.0392630256, 0.0365535041, 0.0391922257,
             -0.0605504752, -0.0383143927, 0.0402160959, 0.0407685779, 0.0396488404, 0.036042407,
             0.0382353103, 0.0367575373, 0.0399300438, 0.0376904696, 0.0398902844, -0.0368078317,
             0.037301095, 0.0409762716, 0.0395950677, 0.0386733698, 0.0404297556, 0.0408635768]
        ]
        bll = pe.basis_loglikelihood(samples).astype(np.float32)
        err = np.abs(bll - ans) / np.abs(ans)
        assert err.sum() < 1e-4


class TestCategoricalParzenEstimator(unittest.TestCase):
    def test_init(self) -> None:
        samples_set = [
            np.array([5]),
            np.array([1.0]),
            np.array(['hoge'])
        ]
        for samples in samples_set:
            try:
                CategoricalParzenEstimator(samples=samples, n_choices=4, top=0.7, weight_func=default_weights)
            except ValueError:
                pass
            except Exception:
                raise ValueError('test_init failed.')
            else:
                raise ValueError('The error was not raised.')

    def test_sample(self) -> None:
        rng = np.random.RandomState()
        n_samples = 10000

        samples_set = [
            np.array([0]),  # [0.475, 0.175, 0.175, 0.175]
            np.array([0, 3]),  # [0.35, 0.15, 0.15, 0.35]
            np.array([0, 1, 2, 3] * 10),  # [0.25, 0.25, 0.25, 0.25]
            np.array([0]),  # [0.7, 0.1, 0.1, 0.1]
            np.array([0, 3]),  # [0.4, 0.1, 0.1, 0.4]
            np.array([0, 1, 2, 3] * 10)  # [0.25, 0.25, 0.25, 0.25]
        ]
        bounds = [
            [[0.45, 0.50], [0.15, 0.20], [0.15, 0.20], [0.15, 0.20]],
            [[0.33, 0.37], [0.13, 0.17], [0.13, 0.17], [0.33, 0.37]],
            [[0.23, 0.27], [0.23, 0.27], [0.23, 0.27], [0.23, 0.27]],
            [[0.68, 0.72], [0.08, 0.12], [0.08, 0.12], [0.08, 0.12]],
            [[0.38, 0.42], [0.08, 0.12], [0.08, 0.12], [0.38, 0.42]],
            [[0.23, 0.27], [0.23, 0.27], [0.23, 0.27], [0.23, 0.27]]
        ]
        lls = [
            np.log(np.array([0.475, 0.175, 0.175, 0.175])),
            np.log(np.array([0.35, 0.15, 0.15, 0.35])),
            np.log(np.array([0.25, 0.25, 0.25, 0.25])),
            np.log(np.array([0.7, 0.1, 0.1, 0.1])),
            np.log(np.array([0.4, 0.1, 0.1, 0.4])),
            np.log(np.array([0.25, 0.25, 0.25, 0.25])),
        ]
        priors = [CategoricalPriorType.uniform] * 3 + [CategoricalPriorType.no_prior] * 3
        for samples, bound, prior, ll in zip(samples_set, bounds, priors, lls):
            pe = CategoricalParzenEstimator(samples=samples, n_choices=4, top=0.7, weight_func=default_weights,
                                            prior=prior)
            ll_est = pe.log_likelihood(np.arange(4))
            vals, counts = np.unique(pe.sample(rng, n_samples), return_counts=True)

            if not np.allclose(ll, np.log(np.ones(4) * 0.25)):
                assert np.allclose(ll_est, ll)

            if prior == CategoricalPriorType.no_prior:
                assert np.allclose(samples, np.array([b.choice for b in pe.basis]))
            else:
                assert np.allclose(samples, np.array([b.choice for b in pe.basis])[:-1])

            for i in range(4):
                assert bound[i][0] <= counts[vals == i] / n_samples <= bound[i][1]

    def test_basis_loglikelihood(self) -> None:
        samples = np.array([0, 1, 2, 3] * 10)
        pe = CategoricalParzenEstimator(samples=samples, n_choices=4, top=0.7, weight_func=default_weights)
        assert pe.basis_loglikelihood(np.arange(4)).shape == (41, 4)
        pe = CategoricalParzenEstimator(samples=samples, n_choices=4, top=0.7, weight_func=default_weights,
                                        prior=CategoricalPriorType.no_prior)
        assert pe.basis_loglikelihood(np.arange(4)).shape == (40, 4)


if __name__ == '__main__':
    unittest.main()
