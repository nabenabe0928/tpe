import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np
import unittest

from util.utils import get_random_sample, revert_eval_config


class TestFuncs(unittest.TestCase):
    def setUp(self) -> None:
        self.config_space = CS.ConfigurationSpace()
        lb, ub = 1, 100
        self.config_space.add_hyperparameter(
            CSH.UniformFloatHyperparameter('f', lower=lb, upper=ub)
        )
        self.config_space.add_hyperparameter(
            CSH.UniformFloatHyperparameter('fq', lower=lb, upper=ub, q=0.5)
        )
        self.config_space.add_hyperparameter(
            CSH.UniformFloatHyperparameter('fql', lower=lb, upper=ub, q=0.5, log=True)
        )
        self.config_space.add_hyperparameter(
            CSH.UniformFloatHyperparameter('fl', lower=lb, upper=ub, log=True)
        )

        self.config_space.add_hyperparameter(
            CSH.UniformIntegerHyperparameter('i', lower=lb, upper=ub)
        )
        self.config_space.add_hyperparameter(
            CSH.UniformIntegerHyperparameter('il', lower=lb, upper=ub, log=True)
        )

        self.config_space.add_hyperparameter(
            CSH.CategoricalHyperparameter('c', choices=['x', 'y', 'z'])
        )

        self.hp_names = self.config_space.get_hyperparameter_names()
        self.is_categoricals = {
            hp_name: self.config_space.get_hyperparameter(hp_name).__class__.__name__ == 'CategoricalHyperparameter'
            for hp_name in self.hp_names
        }
        self.is_ordinals = {
            hp_name: self.config_space.get_hyperparameter(hp_name).__class__.__name__ == 'OrdinalHyperparameter'
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
                    rng=rng
                )
                for hp_name in self.hp_names
            }
            eval_config = revert_eval_config(
                eval_config=eval_config,
                config_space=self.config_space,
                is_categoricals=self.is_categoricals,
                is_ordinals=self.is_ordinals,
                hp_names=self.hp_names
            )

            for hp_name, value in eval_config.items():
                is_categorical = self.is_categoricals[hp_name]
                config = self.config_space.get_hyperparameter(hp_name)

                if is_categorical:
                    assert value in config.choices
                else:
                    q = config.q
                    assert config.lower <= value <= config.upper

                    if q is not None:
                        self.assertAlmostEqual(int((value + 1e-12) / q), (value + 1e-12) / q)

    def test_revert_eval_config(self) -> None:
        eval_config = {
            'c': 0,
            'f': 10.1,
            'fl': np.log(10.1),
            'fq': 10.1,
            'fql': np.log(10.1),
            'i': 10.1,
            'il': np.log(10.1)
        }

        config = revert_eval_config(
            eval_config=eval_config,
            config_space=self.config_space,
            is_categoricals=self.is_categoricals,
            is_ordinals=self.is_ordinals,
            hp_names=self.hp_names
        )

        ans = {
            'c': 'x',
            'f': 10.1,
            'fl': 10.1,
            'fq': 10.0,
            'fql': 10.0,
            'i': 10,
            'il': 10
        }

        for a, v in zip(ans.values(), config.values()):
            if isinstance(a, float):
                self.assertAlmostEqual(float(a), float(v))
            else:
                assert a == v

        eval_config = {
            'c': 1,
            'f': 10.3,
            'fl': np.log(10.3),
            'fq': 10.3,
            'fql': np.log(10.3),
            'i': 10.3,
            'il': np.log(10.6)
        }

        config = revert_eval_config(
            eval_config=eval_config,
            config_space=self.config_space,
            is_categoricals=self.is_categoricals,
            is_ordinals=self.is_ordinals,
            hp_names=self.hp_names
        )

        ans = {
            'c': 'y',
            'f': 10.3,
            'fl': 10.3,
            'fq': 10.5,
            'fql': 10.5,
            'i': 10,
            'il': 11
        }

        for a, v in zip(ans.values(), config.values()):
            if isinstance(a, float):
                self.assertAlmostEqual(float(a), float(v))
            else:
                assert a == v


if __name__ == '__main__':
    unittest.main()
