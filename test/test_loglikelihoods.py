import numpy as np
import unittest

from tpe.optimizer.parzen_estimator.loglikelihoods import compute_config_loglikelihoods


class TestFuncs(unittest.TestCase):
    def test_compute_config_loglikelihoods(self) -> None:
        dim, n_basis, n_samples = 2, 3, 4
        bll = np.arange(dim * n_basis * n_samples).reshape(dim, n_basis, n_samples)
        ws: np.ndarray = np.ones(n_basis) / n_basis
        ll = compute_config_loglikelihoods(basis_loglikelihoods=bll, weights=ws)
        assert ll.size == n_samples

        ans = np.array([26.90172323, 28.90172323, 30.90172323, 32.90172323])
        assert np.allclose(ans, ll)


if __name__ == '__main__':
    unittest.main()
