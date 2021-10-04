from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
import unittest

from src.optimizer.parzen_estimator.kernel import AitchisonAitkenKernel, GaussKernel, UniformKernel


class TestKernel(unittest.TestCase):
    def setUp(self) -> None:
        self.gk = GaussKernel(mu=0.0, sigma=1.0, lb=-np.inf, ub=np.inf)
        self.ak = AitchisonAitkenKernel(n_choices=4, top=0.7, choice=0)
        self._initialized = True

    def test_probs(self) -> None:
        probs = self.ak.probs
        assert np.allclose(probs, np.array([0.7, 0.1, 0.1, 0.1]))

    def test_kernel_invalid_assignment(self) -> None:
        for var_name in ['mu', 'sigma', 'q', 'lb', 'ub', 'norm_const']:
            try:
                setattr(self.gk, var_name, 30.0)
            except AttributeError:
                pass
            except Exception:
                raise ValueError('test_kernel_invalid_assignment failed.')
            else:
                raise ValueError('The error was not raised.')

        for var_name in ['choice', 'n_choices', 'top', 'others', 'probs']:
            try:
                setattr(self.ak, var_name, 30.0)
            except AttributeError:
                pass
            except Exception:
                raise ValueError('test_kernel_invalid_assignment failed.')
            else:
                raise ValueError('The error was not raised.')

    def test_init_domain_params(self) -> None:
        params: List[Tuple[Any, Dict[str, float], Type[Union[AttributeError, ValueError]]]] = [
            (self.gk, {'lb': -np.inf, 'ub': np.inf}, AttributeError),
            (self.ak, {'n_choices': 4, 'top': 0.7}, AttributeError),
        ]

        for target, param, error in params:
            try:
                getattr(target, '_init_domain_params')(**param)
            except error:
                pass
            except Exception:
                raise ValueError('test_init_domain_params failed.')
            else:
                raise ValueError('The error was not raised.')

        params = [
            (GaussKernel, {'mu': 0.5, 'sigma': 1.0, 'lb': 1.0, 'ub': 0.0}, ValueError),
            (GaussKernel, {'mu': 1.5, 'sigma': 1.0, 'lb': 0.0, 'ub': 1.0}, ValueError),
            (GaussKernel, {'mu': 0.5, 'sigma': -1.0, 'lb': 0.0, 'ub': 1.0}, ValueError),
            (AitchisonAitkenKernel, {'n_choices': 4, 'top': 1.7, 'choice': 0}, ValueError),
            (AitchisonAitkenKernel, {'n_choices': 1, 'top': 0.7, 'choice': 0}, ValueError),
            (AitchisonAitkenKernel, {'n_choices': 4, 'top': 0.7, 'choice': 5}, ValueError),
            (UniformKernel, {'n_choices': 1}, ValueError),
        ]

        for target, param, error in params:
            try:
                target(**param)
            except error:
                pass
            except Exception:
                raise ValueError('test_init_domain_params failed.')
            else:
                raise ValueError('The error was not raised.')

        gk = GaussKernel(mu=0.5, sigma=1.0, lb=0.0, ub=1.0)
        self.assertAlmostEqual(gk.lb, 0.0, places=1)
        self.assertAlmostEqual(gk.ub, 1.0, places=1)
        gk = GaussKernel(mu=0.5, sigma=1.0, lb=-np.inf, ub=np.inf)
        self.assertEqual(gk.lb, -np.inf)
        self.assertEqual(gk.ub, np.inf)

        ak = AitchisonAitkenKernel(n_choices=2, top=0.5, choice=0)
        self.assertEqual(ak.n_choices, 2)
        self.assertAlmostEqual(ak.top, 0.5, places=1)
        self.assertAlmostEqual(ak.others, 0.5, places=1)
        ak = AitchisonAitkenKernel(n_choices=4, top=0.7, choice=0)
        self.assertEqual(ak.n_choices, 4)
        self.assertAlmostEqual(ak.top, 0.7, places=1)
        self.assertAlmostEqual(ak.others, 0.1, places=1)

    def test_pdf(self) -> None:
        self.assertAlmostEqual(self.gk.pdf(0), 1.0 / np.sqrt(2 * np.pi), places=1)

    def test_cdf(self) -> None:
        self.assertAlmostEqual(self.gk.cdf(np.inf), 1.0, places=1)
        self.assertAlmostEqual(self.gk.cdf(0), 0.5, places=1)
        self.assertAlmostEqual(self.gk.cdf(-np.inf), 0.0, places=1)
        self.assertAlmostEqual(self.ak.cdf(0), 0.7, places=1)
        self.assertAlmostEqual(self.ak.cdf(1), 0.1, places=1)
        self.assertAlmostEqual(self.ak.cdf(2), 0.1, places=1)

    def test_sample(self) -> None:
        rng = np.random.RandomState()
        cs = np.array([self.ak.sample(rng) for _ in range(10000)])
        unique, counts = np.unique(cs, return_counts=True)

        self.assertEqual((unique - np.arange(4)).sum(), 0)
        self.assertEqual(len(counts), 4)
        self.assertGreater(counts[0], 5600)
        self.assertLess(counts[0], 8400)
        for i in range(1, 4):
            self.assertGreater(counts[i], 800)
            self.assertLess(counts[i], 1200)

        n_samples = 10000
        gk = GaussKernel(lb=-10.0, ub=0.0, mu=-3.0, sigma=1.0)
        samples = np.array([gk.sample(rng) for _ in range(n_samples)])
        assert 0.48 <= np.count_nonzero(samples >= -3) / n_samples <= 0.52
        assert 0.14 <= np.count_nonzero(samples >= -2) / n_samples <= 0.18
        assert 0.01 <= np.count_nonzero(samples >= -1) / n_samples <= 0.04
        assert 0.14 <= np.count_nonzero(samples <= -4) / n_samples <= 0.18
        assert 0.01 <= np.count_nonzero(samples <= -5) / n_samples <= 0.04

        gk = GaussKernel(lb=-2.0, ub=2.0, mu=0.0, sigma=1.0, q=0.5)
        samples = np.array([gk.sample(rng) for _ in range(n_samples)])
        vals = np.unique(samples)
        assert set(vals) == set([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])


if __name__ == '__main__':
    unittest.main()
