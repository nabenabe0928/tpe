import os

from typing import Dict, Union

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import unittest

from optimizer.tpe import TPEOptimizer
from util.utils import get_logger
from util.constants import default_percentile_maker


def sphere(eval_config: Dict[str, float]) -> float:
    vals = np.array(list(eval_config.values()))
    vals *= vals
    return vals.sum()


def rosen(eval_config: Dict[str, float]) -> float:
    val = 0
    vals = np.array(list(eval_config.values()))
    dim = len(vals)
    for d in range(dim - 1):
        t1 = 100 * (vals[d + 1] - vals[d] ** 2) ** 2
        t2 = (vals[d] - 1) ** 2
        val += t1 + t2
    return val


def mix_func(eval_config: Dict[str, Union[str, float]]) -> float:
    func_dict = {'cosine': np.cos, 'sine': np.sin}
    vals = np.array([v for v in eval_config.values() if isinstance(v, float)])
    vals *= vals
    return 1 - func_dict[eval_config['func']](vals.sum()) ** 2


def cleanup() -> None:
    os.remove('log/test.log')
    os.remove('results/test.json')
    os.remove('incumbents/opt_test.json')


class TestTreeStructuredParzenEstimator(unittest.TestCase):
    def setUp(self) -> None:
        dim = 10
        self.cs_cat = CS.ConfigurationSpace()
        self.cs = CS.ConfigurationSpace()
        self.cs_cat.add_hyperparameter(CSH.CategoricalHyperparameter('func', choices=['sine', 'cosine']))
        for d in range(dim):
            self.cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f'x{d}', lower=-5, upper=5))
            if d < dim - 1:
                self.cs_cat.add_hyperparameter(CSH.UniformFloatHyperparameter(f'x{d}', lower=-5, upper=5))
            else:
                self.cs_cat.add_hyperparameter(CSH.OrdinalHyperparameter(
                        f'x{d}',
                        sequence=list(range(-5, 6)),
                        meta={'lower': -5, 'upper': 5}
                    )
                )

        self.hp_names = list(self.cs._hyperparameters.keys())
        self.hp_names_cat = list(self.cs_cat._hyperparameters.keys())
        self.logger = get_logger(file_name='test', logger_name='test')

    def test_update_observations(self) -> None:
        max_evals, metric_name = 20, 'loss'
        opt = TPEOptimizer(
            obj_func=mix_func,
            config_space=self.cs_cat,
            max_evals=max_evals,
            mutation_prob=0.0,
            metric_name=metric_name,
            resultfile='test'
        )
        opt.optimize(self.logger)

        eval_config = {hp_name: 0.0 for hp_name in self.hp_names_cat}
        eval_config['func'] = 'cosine'
        opt.tpe.update_observations(eval_config=eval_config, loss=0)
        assert opt.tpe.sorted_observations[metric_name][0] == 0.0
        assert opt.tpe.sorted_observations[metric_name].size == max_evals + 1
        for hp_name in self.hp_names:
            assert opt.tpe.observations[hp_name].size == max_evals + 1
            assert opt.tpe.observations[hp_name][-1] == 0.0
            assert opt.tpe.sorted_observations[hp_name].size == max_evals + 1
            assert opt.tpe.sorted_observations[hp_name][0] == 0.0

        min_value = np.inf
        # The sorted_observations must be sorted
        for v in opt.tpe.sorted_observations[metric_name]:
            assert min_value >= v
            min_value = min(min_value, np.inf)

        cleanup()

    def test_set_prior_observations(self) -> None:
        metric_name = 'loss'
        opt = TPEOptimizer(
            obj_func=sphere,
            config_space=self.cs,
            max_evals=10,
            mutation_prob=0.0,
            metric_name=metric_name,
            resultfile='test'
        )
        ob = {}
        n_samples = 10
        rng = np.random.RandomState(0)
        ob[metric_name] = rng.random(n_samples)
        for hp_name in self.hp_names:
            ob[hp_name] = rng.random(n_samples)

        order = np.argsort(ob[metric_name])
        opt.tpe.set_prior_observations(ob)

        for hp_name in ob.keys():
            cnt = np.count_nonzero(opt.tpe.observations[hp_name] != ob[hp_name])
            assert cnt == 0

        sorted_ob = {k: x[order] for k, x in ob.items()}
        for hp_name in ob.keys():
            cnt = np.count_nonzero(opt.tpe.sorted_observations[hp_name] != sorted_ob[hp_name])
            assert cnt == 0

    def test_get_config_candidates(self) -> None:
        max_evals = 5
        opt = TPEOptimizer(
            obj_func=sphere,
            config_space=self.cs,
            max_evals=max_evals,
            mutation_prob=0.0,
            resultfile='test',
            seed=0
        )
        opt.optimize(self.logger)

        opt.tpe._n_ei_candidates = 3
        config_cands = opt.tpe.get_config_candidates()
        assert len(config_cands) == len(self.hp_names)
        assert config_cands[0].size == opt.tpe.n_ei_candidates

        ans = [[0.5135034241147703, 0.4448090311534248, 0.5723466512713232],
               [-1.121131319925264, 2.952173904420133, 2.2211060220605847],
               [0.3840503560941645, 0.05691106986981964, 3.2489156267078263],
               [0.6223354593074132, 0.42882383641275457, 0.6089624088247492],
               [4.406721260474024, -0.7441693444777707, -0.8200358837154326],
               [0.6573366338079927, 0.060519759177582566, 1.1315348442999413],
               [-0.48810672560449186, -0.7276334971302549, -0.6697272781333232],
               [3.856861350505235, -2.8167004005339753, 1.535239936783603],
               [-0.5896751970671259, 4.873942440726309, 1.2756781156033692],
               [-2.3906083300437304, -0.3234225699230498, -1.5374589912835055]]

        assert np.allclose(ans, config_cands)

        max_evals = 5
        opt = TPEOptimizer(
            obj_func=mix_func,
            config_space=self.cs_cat,
            max_evals=max_evals,
            mutation_prob=0.0,
            resultfile='test',
            seed=0
        )
        opt.optimize(self.logger)

        opt.tpe._n_ei_candidates = 5
        config_cands = opt.tpe.get_config_candidates()
        assert len(config_cands) == len(self.hp_names_cat)
        assert config_cands[0].size == opt.tpe.n_ei_candidates

        ans = [
            [0, 0, 1, 1, 1],
            [-0.4292062936377955, -0.5132230470889889, -0.14657438123122993, -0.17073356719780441, 3.6297139379162733],
            [3.330241948294638, 1.463115677574857, 4.406721260474024, 2.947887888286006, -0.8200358837154326],
            [1.1315348442999413, 0.23325333601086667, 1.1247585066034038, 0.3634786818265787, 1.7541576737760938],
            [1.645260846074287, -2.8167004005339753, 0.8557998084937999, -1.7809823117239185, 1.5013505165128298],
            [1.198382157298241, 1.4749412997717923, 1.528431200120444, -0.3234225699230498, -2.478268504181421],
            [-1.2633949745100692, 3.131491033668064, 0.329362518021425, 3.3028457767583435, -3.979432364310714],
            [0.07049671291066206, -2.1000631288626677, 0.1714814018182706, 0.2814734641974796, 0.1702124142878986],
            [-2.271288155092453, 0.22316506530894623, -4.001861660854562, 2.77700863464274, 4.17987337950341],
            [-3.0893344667008367, -2.97643654651492, -4.782469654709937, -1.8349970380379617, -4.9489943835675785],
            [2.1763029445963484, 0.8846172861038786, 2.1671251872872914, 1.2972697690185329, 4.492348642293887]
        ]

        assert np.allclose(ans, config_cands)
        for i in range(opt.tpe.n_ei_candidates):
            eval_config = {hp: a[-1] for a, hp in zip(ans, self.hp_names_cat)}
            ord_v = opt._revert_eval_config(eval_config)[self.hp_names_cat[-1]]
            assert ord_v in self.cs_cat.get_hyperparameter(self.hp_names_cat[-1]).sequence

        cleanup()

    def test_compute_basis_loglikelihoods(self) -> None:
        max_evals = 5
        opt = TPEOptimizer(
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
        bll_low, bll_up = opt.tpe._compute_basis_loglikelihoods(
            hp_name=self.hp_names[0],
            samples=samples
        )

        percentile_func = default_percentile_maker()
        n_lower = percentile_func(opt.tpe.observations['loss'])
        n_upper = max_evals - n_lower

        assert bll_low.shape == (n_lower + 1, n_samples)
        assert bll_up.shape == (n_upper + 1, n_samples)

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

    def test_compute_probability_improvement(self) -> None:
        max_evals = 5
        opt = TPEOptimizer(
            obj_func=sphere,
            config_space=self.cs,
            max_evals=max_evals,
            mutation_prob=0.0,
            resultfile='test',
            seed=0
        )
        opt.optimize(self.logger)

        config_cands = [np.array([0.0]) for _ in range(len(self.hp_names))]
        ll_ratio = opt.tpe.compute_probability_improvement(config_cands)

        assert ll_ratio.size == 1
        self.assertAlmostEqual(ll_ratio[0], -0.08152714)


class TestTPEOptimizer(unittest.TestCase):
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
                opt = TPEOptimizer(
                    obj_func=sphere,
                    config_space=self.cs,
                    max_evals=max_evals,
                    mutation_prob=0.0,
                    resultfile='test'
                )
                _, best_loss = opt.optimize(self.logger)
                losses[i] = best_loss
                assert opt.tpe.observations['x0'].size == max_evals

            # performance test
            assert losses.mean() < threshold

        cleanup()


if __name__ == '__main__':
    unittest.main()
