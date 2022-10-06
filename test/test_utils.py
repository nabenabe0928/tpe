import pytest

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np
import unittest

from tpe.utils.constants import QuantileFunc, WeightFuncs
from tpe.utils.utils import get_random_sample, revert_eval_config


def test_quantile_func() -> None:
    f = QuantileFunc(alpha=0.25, choice="sqrt")
    ans = [1] * 16 + [2] * 48 + [3] * 35
    assert np.allclose(ans, [f(i) for i in range(1, 100)])

    f = QuantileFunc(alpha=0.6, choice="sqrt")
    ans = [1] * 2 + [2] * 9 + [3] * 14 + [4] * 19 + [5] * 25 + [6] * 30
    assert np.allclose(ans, [f(i) for i in range(1, 100)])

    f = QuantileFunc(alpha=0.25, choice="linear")
    ans = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5]
    assert np.allclose(ans, [f(i) for i in range(1, 20)])

    f = QuantileFunc(alpha=0.6, choice="linear")
    ans = [1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 11, 11, 12]
    assert np.allclose(ans, [f(i) for i in range(1, 20)])

    with pytest.raises(ValueError):
        f = QuantileFunc(choice="dummy")
        f(1)


def test_weight_funcs() -> None:
    with pytest.raises(ValueError):
        WeightFuncs(choice="dummy")

    size = 5
    for _ in range(5):
        loss_vals = np.random.random(size)
        order = np.argsort(loss_vals)
        sorted_loss_vals = loss_vals[order]
        order_inv = np.zeros_like(order)
        order_inv[order] = np.arange(order.size)
        kwargs = dict(size=size, sorted_loss_vals=sorted_loss_vals, order=order)

        weights = WeightFuncs.older_smaller(order)
        assert np.allclose(WeightFuncs("older-smaller")(**kwargs), weights)
        weight_for_prior = weights[-1]
        assert np.min(weights) == weight_for_prior  # prior is the smallest weight
        weights = weights[:-1]
        assert np.allclose(weights[order_inv], np.maximum.accumulate(weights[order_inv]))
        assert np.isclose(weights.sum(), 1.0 - weight_for_prior)

        weights = WeightFuncs.weaker_smaller(sorted_loss_vals)
        assert np.allclose(WeightFuncs("weaker-smaller")(**kwargs), weights)
        assert np.min(weights) == weights[-1]  # prior is the smallest weight
        assert np.allclose(weights, np.minimum.accumulate(weights))
        assert np.isclose(weights.sum(), 1.0)

        weights = WeightFuncs.expected_improvement(sorted_loss_vals)
        assert np.allclose(WeightFuncs("expected-improvement")(**kwargs, lower_group=True), weights)
        assert np.allclose(WeightFuncs("expected-improvement")(**kwargs), WeightFuncs.uniform(size))
        assert np.allclose(weights[:-1], np.minimum.accumulate(weights[:-1]))
        assert np.isclose(weights.sum(), 1.0)
        assert np.isclose(weights[-1], 1.0 / (size + 1))  # prior is the uniform weight
        size = 50
    else:
        f = WeightFuncs("expected-improvement")
        f._choice = "dummy"
        with pytest.raises(NotImplementedError):
            f(**kwargs)


class TestFuncs(unittest.TestCase):
    def setUp(self) -> None:
        self.config_space = CS.ConfigurationSpace()
        lb, ub = 1, 101
        self.config_space.add_hyperparameter(CSH.UniformFloatHyperparameter("f", lower=lb, upper=ub))
        self.config_space.add_hyperparameter(CSH.UniformFloatHyperparameter("fq", lower=lb, upper=ub, q=2))
        self.config_space.add_hyperparameter(CSH.UniformFloatHyperparameter("fql", lower=lb, upper=ub, q=0.5, log=True))
        self.config_space.add_hyperparameter(CSH.UniformFloatHyperparameter("fl", lower=lb, upper=ub, log=True))

        self.config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter("i", lower=lb, upper=ub))
        self.config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter("il", lower=lb, upper=ub, log=True))

        self.config_space.add_hyperparameter(CSH.CategoricalHyperparameter("c", choices=["x", "y", "z"]))

        self.config_space.add_hyperparameter(
            CSH.OrdinalHyperparameter("o", sequence=list(range(1, 101)), meta={"lower": 1, "upper": 100, "log": False})
        )

        self.config_space.add_hyperparameter(
            CSH.OrdinalHyperparameter("ol", sequence=[1, 10, 100], meta={"lower": 1, "upper": 100, "log": True})
        )

        self.hp_names = self.config_space.get_hyperparameter_names()
        self.is_categoricals = {
            hp_name: self.config_space.get_hyperparameter(hp_name).__class__.__name__ == "CategoricalHyperparameter"
            for hp_name in self.hp_names
        }
        self.is_ordinals = {
            hp_name: self.config_space.get_hyperparameter(hp_name).__class__.__name__ == "OrdinalHyperparameter"
            for hp_name in self.hp_names
        }

    def test_get_random_sample(self) -> None:
        rng = np.random.RandomState(0)

        for _ in range(30):
            eval_config = {
                hp_name: get_random_sample(
                    hp_name=hp_name,
                    config_space=self.config_space,
                    is_categorical=self.is_categoricals[hp_name],
                    is_ordinal=self.is_ordinals[hp_name],
                    rng=rng,
                )
                for hp_name in self.hp_names
            }
            eval_config = revert_eval_config(
                eval_config=eval_config,
                config_space=self.config_space,
                is_categoricals=self.is_categoricals,
                is_ordinals=self.is_ordinals,
                hp_names=self.hp_names,
            )

            for hp_name, value in eval_config.items():
                is_categorical = self.is_categoricals[hp_name]
                is_ordinal = self.is_ordinals[hp_name]
                config = self.config_space.get_hyperparameter(hp_name)

                if is_categorical:
                    assert value in config.choices
                elif is_ordinal:
                    assert value in config.sequence
                else:
                    q = config.q
                    assert config.lower <= value <= config.upper

                    if q is not None:
                        lb = config.lower
                        self.assertAlmostEqual(int((value - lb + 1e-12) / q), (value - lb + 1e-12) / q)

    def test_revert_eval_config(self) -> None:
        eval_config = {
            "c": 0,
            "f": 10.1,
            "fl": np.log(10.1),
            "fq": 10.1,
            "fql": np.log(10.1),
            "i": 10.1,
            "il": np.log(10.1),
            "o": 5.4,
            "ol": np.log(40),
        }

        config = revert_eval_config(
            eval_config=eval_config,
            config_space=self.config_space,
            is_categoricals=self.is_categoricals,
            is_ordinals=self.is_ordinals,
            hp_names=self.hp_names,
        )

        ans = {"c": "x", "f": 10.1, "fl": 10.1, "fq": 11.0, "fql": 10.0, "i": 10, "il": 10, "o": 5, "ol": 100}

        for a, v in zip(ans.values(), config.values()):
            if isinstance(a, float):
                self.assertAlmostEqual(float(a), float(v))
            else:
                assert a == v

        eval_config = {
            "c": 1,
            "f": 10.3,
            "fl": np.log(10.3),
            "fq": 9.8,
            "fql": np.log(10.3),
            "i": 10.3,
            "il": np.log(10.6),
            "o": 5.6,
            "ol": np.log(30),
        }

        config = revert_eval_config(
            eval_config=eval_config,
            config_space=self.config_space,
            is_categoricals=self.is_categoricals,
            is_ordinals=self.is_ordinals,
            hp_names=self.hp_names,
        )

        ans = {"c": "y", "f": 10.3, "fl": 10.3, "fq": 9.0, "fql": 10.5, "i": 10, "il": 11, "o": 6, "ol": 10}

        for a, v in zip(ans.values(), config.values()):
            if isinstance(a, float):
                self.assertAlmostEqual(float(a), float(v))
            else:
                assert a == v

    def test_get_hyperparameter(self) -> None:
        pass

    def test_get_config_space(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
