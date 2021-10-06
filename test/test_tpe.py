import os

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import unittest

from optimizer.tpe import TPE
from util.utils import get_logger
from util.constants import default_percentile


def sphere(eval_config):
    vals = np.array(list(eval_config.values()))
    return (vals ** 2).sum()


def rosen(eval_config):
    val = 0
    vals = np.array(list(eval_config.values()))
    dim = len(vals)
    for d in range(dim - 1):
        t1 = 100 * (vals[d + 1] - vals[d] ** 2) ** 2
        t2 = (vals[d] - 1) ** 2
        val += t1 + t2
    return val


def cleanup():
    os.remove('log/test.log')
    os.remove('results/test.json')
    os.remove('incumbents/opt_test.json')


class TestTPE(unittest.TestCase):
    def setUp(self) -> None:
        dim = 10
        self.cs = CS.ConfigurationSpace()
        for d in range(dim):
            self.cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f'x{d}', lower=-5, upper=5))

        self.hp_names = list(self.cs._hyperparameters.keys())
        self.logger = get_logger(file_name='test', logger_name='test')

    def test_optimize(self) -> None:
        n_experiments = 5
        losses = np.ones(n_experiments)
        max_evals = 100

        # 10D sphere with 100 evals: 0.4025 \pm 0.33 over 10 times
        # 10D rosenbrock with 100 evals: 163.46 \pm 179.60 over 10 times
        for func, threshold in zip([sphere, rosen], [0.6, 200]):
            for i in range(n_experiments):
                opt = TPE(
                    obj_func=sphere,
                    config_space=self.cs,
                    max_evals=max_evals,
                    mutation_prob=0.0,
                    resultfile='test'
                )
                _, best_loss = opt.optimize(self.logger)
                losses[i] = best_loss
                assert opt.observations['x0'].size == max_evals

            # performance test
            assert losses.mean() < threshold

        cleanup()

    def test_update_observations(self):
        max_evals = 5
        opt = TPE(
            obj_func=sphere,
            config_space=self.cs,
            max_evals=max_evals,
            mutation_prob=0.0,
            resultfile='test'
        )
        opt.optimize(self.logger)

        eval_config = {hp_name: 0.0 for hp_name in self.hp_names}
        opt._update_observations(eval_config=eval_config, loss=0)
        opt.order[0] = max_evals
        assert opt.order.size == max_evals + 1
        for hp_name in self.hp_names:
            assert opt.observations[hp_name].size == max_evals + 1
            assert opt.observations[hp_name][-1] == 0.0

        cleanup()

    def test_get_config_candidates(self):
        max_evals = 5
        opt = TPE(
            obj_func=sphere,
            config_space=self.cs,
            max_evals=max_evals,
            mutation_prob=0.0,
            resultfile='test',
            seed=0
        )
        opt.optimize(self.logger)

        opt._n_ei_candidates = 3
        config_cands = opt._get_config_candidates()
        assert len(config_cands) == len(self.hp_names)
        assert config_cands[0].size == opt.n_ei_candidates

        print(config_cands)
        ans = [[0.20237672, -2.83174195,  0.26558082],
               [1.46376308, -3.33413852, -3.83153623],
               [0.86883637, 3.13149103, -3.96901532],
               [0.43255754, 4.74879878, 0.50384635],
               [-1.17660563, -1.96940625, -3.74539893],
               [1.85808094, 0.6397104, -2.27128816],
               [-0.20105122, -1.75036364, -0.49962073],
               [-3.08933447,  3.57860764, -4.78246965],
               [2.45089419, 2.27703573, 4.88384902],
               [0.69406709, -1.57490221, -0.2671852]]

        assert np.allclose(ans, config_cands)
        cleanup()

    def test_compute_basis_loglikelihoods(self):
        max_evals = 5
        opt = TPE(
            obj_func=sphere,
            config_space=self.cs,
            max_evals=max_evals,
            mutation_prob=0.0,
            resultfile='test',
            seed=0
        )
        opt.optimize(self.logger)

        samples = np.array([-1.5, 0.0, 1.5])
        n_samples = samples.size
        bll_low, bll_up = opt._compute_basis_loglikelihoods(
            hp_name=self.hp_names[0],
            samples=samples
        )

        n_lower = default_percentile(max_evals)
        n_upper = max_evals - n_lower

        assert bll_low.shape == (n_lower + 1, n_samples)
        assert bll_up.shape == (n_upper + 1, n_samples)

        print(bll_low.tolist(), bll_up.tolist())
        cleanup()
        ans = [[-38.46301430343704, -1.7804762939669245, -9.538815304262513],
               [-2.2728572925030956, -2.2616072925030957, -2.2728572925030956]]
        assert np.allclose(bll_low, ans)

        ans = [[-1.6880016584044981, -1.8741942841009118, -2.48683367359252],
               [-1.6857409199473796, -2.141880853194597, -3.0244675502370093],
               [-4.873695287580564, -3.2997702847617463, -2.1522920457381236],
               [-3.398044539819791, -2.35545316853298, -1.7393085610413628],
               [-2.2728572925030956, -2.2616072925030957, -2.2728572925030956]]
        assert np.allclose(bll_up, ans)

    def test_compute_probability_improvement(self):
        max_evals = 5
        opt = TPE(
            obj_func=sphere,
            config_space=self.cs,
            max_evals=max_evals,
            mutation_prob=0.0,
            resultfile='test',
            seed=0
        )
        opt.optimize(self.logger)

        config_cands = [np.array([0.0]) for _ in range(len(self.hp_names))]
        ll_ratio = opt._compute_probability_improvement(config_cands)

        assert ll_ratio.size == 1
        self.assertAlmostEqual(ll_ratio[0], -0.08152714)


if __name__ == '__main__':
    unittest.main()
