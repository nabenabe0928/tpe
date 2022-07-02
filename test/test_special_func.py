import unittest

import numpy as np

from tpe.utils.special_funcs import erf, exp, log, logsumexp


def test_erf() -> None:
    erf(np.random.random(10))


def test_exp() -> None:
    exp(np.random.random(10))


def test_log() -> None:
    log(np.random.random(10))


def test_logsumexp() -> None:
    assert logsumexp(np.random.random((10, 20)), axis=0, weight=1).shape == (20,)
    assert logsumexp(np.random.random((10, 20)), axis=1, weight=1).shape == (10,)


if __name__ == "__main__":
    unittest.main()
