import numpy as np
import unittest

import ConfigSpace.hyperparameters as CSH

from optimizer.parzen_estimator.kernel import (
    UniformKernel,
)
from optimizer.parzen_estimator.parzen_estimator import (
    CategoricalParzenEstimator,
    CategoricalPriorType,
    NumericalParzenEstimator,
    NumericalPriorType,
    build_categorical_parzen_estimator,
    build_numerical_parzen_estimator
)
from util.constants import config2type, default_weights, NumericalHPType


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
        rng = np.random.RandomState()
        samples = np.array([0, 1, 2, 3] * 10)
        pe = NumericalParzenEstimator(samples=samples, lb=-3.0, ub=3.0, weight_func=default_weights)
        ss = pe.sample(rng, 10)
        assert ss.shape == (10,)

        samples = np.array([-1, 0, 1] * 10)
        pe = NumericalParzenEstimator(samples=samples, lb=-1.25, ub=1.25, q=0.5, weight_func=default_weights)
        ss = pe.sample(rng, 100)
        choices = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        assert np.allclose(np.unique(ss), choices)

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
        assert np.allclose([b.sigma for b in pe.basis], [0.019897270747110334, 0.019897270747110334, 1.])
        ans = [
            [-154.7543487548828, -16.10599708557129, -19.68598747253418, 2.9763262271881104, 1.7578061819076538,
             2.987107515335083, -329.816162109375, -139.15806579589844, 2.2890665531158447, 0.7415248155593872,
             2.8405940532684326, 1.3041807413101196, 2.8360159397125244, 1.9254502058029175, 2.6353659629821777,
             2.5726494789123535, 2.671271324157715, -135.9549560546875, 2.3287010192871094, -1.6978763341903687,
             2.868389129638672, -18.54818344116211, 1.890196442604065, 0.13517069816589355],
            [-147.49549865722656, -18.74729347229004, -16.986461639404297, 2.9773948192596436, 1.021071434020996,
             2.9633071422576904, -319.23406982421875, -146.2153778076172, 2.696009397506714, 1.5344642400741577,
             2.987278699874878, 0.4576236605644226, 2.5150222778320312, 1.2343206405639648, 2.902125358581543,
             2.105701208114624, 2.9201500415802, -142.93328857421875, 1.7647546529769897, -0.5161906480789185,
             2.993614435195923, -15.919400215148926, 2.4202351570129395, 1.039108395576477],
            [-0.0012746538268402219, 0.023732159286737442, 0.03843645751476288, 0.03926302492618561, 0.0365535058081150,
             0.03919222578406334, -0.06055047735571861, -0.03831439092755318, 0.04021609574556351, 0.0407685786485672,
             0.03964884206652641, 0.036042407155036926, 0.03823531046509743, 0.036757536232471466, 0.03993004187941551,
             0.03769046813249588, 0.03989028558135033, -0.036807831376791, 0.037301093339920044, 0.04097627103328705,
             0.039595067501068115, 0.03867337107658386, 0.04042975604534149, 0.04086357727646828]
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
            vals, counts = np.unique(pe.sample(rng, n_samples), return_counts=True)

            if prior == CategoricalPriorType.no_prior:
                assert np.allclose(samples, np.array([b.choice for b in pe.basis]))
            else:
                assert np.allclose(samples, np.array([b.choice for b in pe.basis])[:-1])

            for i in range(4):
                assert bound[i][0] <= counts[vals == i] / n_samples <= bound[i][1]

    def test_basis_loglikelihood(self) -> None:
        samples = np.array([0, 1, 2, 3] * 3)
        pe = CategoricalParzenEstimator(samples=samples, n_choices=4, top=0.7, weight_func=default_weights)
        bll = pe.basis_loglikelihood(np.arange(4))
        assert bll.shape == (13, 4)

        ans = [[-0.3566749393939972, -2.3025851249694824, -2.3025851249694824, -2.3025851249694824],
               [-2.3025851249694824, -0.3566749393939972, -2.3025851249694824, -2.3025851249694824],
               [-2.3025851249694824, -2.3025851249694824, -0.3566749393939972, -2.3025851249694824],
               [-2.3025851249694824, -2.3025851249694824, -2.3025851249694824, -0.3566749393939972],
               [-0.3566749393939972, -2.3025851249694824, -2.3025851249694824, -2.3025851249694824],
               [-2.3025851249694824, -0.3566749393939972, -2.3025851249694824, -2.3025851249694824],
               [-2.3025851249694824, -2.3025851249694824, -0.3566749393939972, -2.3025851249694824],
               [-2.3025851249694824, -2.3025851249694824, -2.3025851249694824, -0.3566749393939972],
               [-0.3566749393939972, -2.3025851249694824, -2.3025851249694824, -2.3025851249694824],
               [-2.3025851249694824, -0.3566749393939972, -2.3025851249694824, -2.3025851249694824],
               [-2.3025851249694824, -2.3025851249694824, -0.3566749393939972, -2.3025851249694824],
               [-2.3025851249694824, -2.3025851249694824, -2.3025851249694824, -0.3566749393939972],
               [-1.3862943649291992, -1.3862943649291992, -1.3862943649291992, -1.3862943649291992]]

        assert np.allclose(ans, bll)

        pe = CategoricalParzenEstimator(samples=samples, n_choices=4, top=0.7, weight_func=default_weights,
                                        prior=CategoricalPriorType.no_prior)
        bll = pe.basis_loglikelihood(np.arange(4))
        assert bll.shape == (12, 4)

        ans = [[-0.3566749393939972, -2.3025851249694824, -2.3025851249694824, -2.3025851249694824],
               [-2.3025851249694824, -0.3566749393939972, -2.3025851249694824, -2.3025851249694824],
               [-2.3025851249694824, -2.3025851249694824, -0.3566749393939972, -2.3025851249694824],
               [-2.3025851249694824, -2.3025851249694824, -2.3025851249694824, -0.3566749393939972],
               [-0.3566749393939972, -2.3025851249694824, -2.3025851249694824, -2.3025851249694824],
               [-2.3025851249694824, -0.3566749393939972, -2.3025851249694824, -2.3025851249694824],
               [-2.3025851249694824, -2.3025851249694824, -0.3566749393939972, -2.3025851249694824],
               [-2.3025851249694824, -2.3025851249694824, -2.3025851249694824, -0.3566749393939972],
               [-0.3566749393939972, -2.3025851249694824, -2.3025851249694824, -2.3025851249694824],
               [-2.3025851249694824, -0.3566749393939972, -2.3025851249694824, -2.3025851249694824],
               [-2.3025851249694824, -2.3025851249694824, -0.3566749393939972, -2.3025851249694824],
               [-2.3025851249694824, -2.3025851249694824, -2.3025851249694824, -0.3566749393939972]]

        assert np.allclose(ans, bll)


class TestBuildParzenEstimators(unittest.TestCase):
    def build_npe(self, vals: np.ndarray, config: NumericalHPType) -> NumericalParzenEstimator:
        config_type = config.__class__.__name__
        return build_numerical_parzen_estimator(
            vals=vals,
            config=config,
            dtype=config2type[config_type],
            weight_func=default_weights,
            is_ordinal=(config.__class__.__name__ == 'OrdinalHyperparameter')
        )

    def _check(self, lb_diff: float, ub_diff: float) -> None:
        self.assertAlmostEqual(lb_diff, 0)
        self.assertAlmostEqual(ub_diff, 0)

    def test_build_categorical_parzen_estimator(self) -> None:
        C = CSH.CategoricalHyperparameter('c1', choices=['a', 'b', 'c'])
        vals = [1]

        try:
            build_categorical_parzen_estimator(
                config=C,
                vals=vals,
                weight_func=default_weights
            )
        except ValueError:
            pass
        else:
            raise ValueError('The error was not raised.')

        s_vals = ['a', 'b', 'c']
        try:
            build_categorical_parzen_estimator(
                config=C,
                vals=s_vals,
                weight_func=default_weights
            )
        except Exception:
            raise ValueError('test_build_numerical_parzen_estimator failed.')

    def test_build_numerical_parzen_estimator(self) -> None:
        lb, ub, q = 1, 100, 0.5
        x = CSH.UniformFloatHyperparameter('x', lower=lb, upper=ub)
        vals = np.arange(1, 20)
        pe = self.build_npe(vals, x)
        assert pe.q is None
        self._check(pe.lb - lb, pe.ub - ub)

        x = CSH.UniformFloatHyperparameter('x', lower=lb, upper=ub, log=True)
        pe = self.build_npe(vals, x)

        assert pe.q is None
        self._check(pe.lb - np.log(lb), pe.ub - np.log(ub))

        x = CSH.UniformFloatHyperparameter('x', lower=lb, upper=ub, q=q)
        pe = self.build_npe(vals, x)
        assert pe.q == q
        self._check(pe.lb - (lb - 0.5 * q), pe.ub - (ub + 0.5 * q))

        x = CSH.UniformFloatHyperparameter('x', lower=lb, upper=ub, q=q, log=True)
        pe = self.build_npe(vals, x)
        assert pe.q is None
        self._check(pe.lb - np.log(lb), pe.ub - np.log(ub))

        x = CSH.UniformIntegerHyperparameter('x', lower=lb, upper=ub)
        pe = self.build_npe(vals, x)
        assert pe.q == 1
        self._check(pe.lb - (lb - 0.5), pe.ub - (ub + 0.5))

        x = CSH.UniformIntegerHyperparameter('x', lower=lb, upper=ub, log=True)
        pe = self.build_npe(vals, x)
        assert pe.q is None
        self._check(pe.lb - np.log(lb), pe.ub - np.log(ub))


if __name__ == '__main__':
    unittest.main()
