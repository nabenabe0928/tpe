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
    assert isinstance(eval_config['func'], str)
    func_name: str = eval_config['func']
    return 1 - func_dict[func_name](vals.sum()) ** 2


def cleanup() -> None:
    os.remove('log/test.log')
    os.remove('results/test.json')
    os.remove('incumbents/test.json')


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
            metric_name=metric_name,
            resultfile='test'
        )
        opt.optimize(self.logger)

        eval_config = {hp_name: 0.0 for hp_name in self.hp_names_cat}
        eval_config['func'] = 'cosine'  # type: ignore
        opt.tpe.update_observations(eval_config=eval_config, loss=0)
        assert opt.tpe._sorted_observations[metric_name][0] == 0.0
        assert opt.tpe._sorted_observations[metric_name].size == max_evals + 1
        for hp_name in self.hp_names:
            assert opt.tpe.observations[hp_name].size == max_evals + 1
            assert opt.tpe.observations[hp_name][-1] == 0.0
            assert opt.tpe._sorted_observations[hp_name].size == max_evals + 1
            assert opt.tpe._sorted_observations[hp_name][0] == 0.0

        min_value = np.inf
        # The sorted_observations must be sorted
        for v in opt.tpe._sorted_observations[metric_name]:
            assert min_value >= v
            min_value = min(min_value, np.inf)

        cleanup()

    @unittest.skip("Deprecated set prior observations at Feb 10 2022, check the commits before.")
    def test_set_prior_observations(self) -> None:
        metric_name = 'loss'
        opt = TPEOptimizer(
            obj_func=sphere,
            config_space=self.cs,
            max_evals=10,
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
            cnt = np.count_nonzero(opt.tpe._sorted_observations[hp_name] != sorted_ob[hp_name])
            assert cnt == 0

    def test_get_config_candidates(self) -> None:
        max_evals = 5
        opt = TPEOptimizer(
            obj_func=sphere,
            config_space=self.cs,
            max_evals=max_evals,
            resultfile='test',
            seed=0
        )
        opt.optimize(self.logger)

        opt.tpe._n_ei_candidates = 3
        config_cands = opt.tpe.get_config_candidates()
        assert len(config_cands) == len(self.hp_names)
        assert config_cands[0].size == opt.tpe._n_ei_candidates

        ans = [[4.8431215412066475, -1.8158257273119596, -3.744716909802062],
               [2.7519831557632024, 0.2620535804840036, 0.45758517301446067],
               [-1.871838500258336, 1.7538653266921242, -4.625055394360726],
               [-4.380743016111864, 0.2554040035802357, 0.5688739967604989],
               [-0.6816800079423969, -0.9672547770712823, -0.8441738465209289],
               [1.27499304070942, -0.2818222833865487, 1.6739088876074935],
               [-3.643343119343121, -3.536278564808815, -1.3278425612003888],
               [0.5194539579613895, 1.2898291075741066, 2.2535688848821986],
               [2.6414792686135344, 3.77160611108241, 4.524089680601648],
               [-4.031769469731796, 2.082749780768603, -0.7739934294417338]]
        assert np.allclose(ans, config_cands)

        max_evals = 5
        opt = TPEOptimizer(
            obj_func=mix_func,
            config_space=self.cs_cat,
            max_evals=max_evals,
            resultfile='test',
            seed=0
        )
        opt.optimize(self.logger)

        opt.tpe._n_ei_candidates = 5
        config_cands = opt.tpe.get_config_candidates()
        assert len(config_cands) == len(self.hp_names_cat)
        assert config_cands[0].size == opt.tpe._n_ei_candidates

        ans = [
            [0, 1, 0, 0, 0],
            [1.4404357116087798, -0.11849219656659751, 1.2167501649282841, 4.438632327454257, -0.23055211452546834],
            [3.769269749952829, 0.33438928508628574, 3.7135798107341955, -2.421495142426032, 4.576692069111804],
            [1.549474256969163, 0.24259104747226518, -3.479121493261526, 1.5634896910398006, -3.8732681740795227],
            [1.6027760417483585, 1.702971261665361, -1.906559792126414, 2.064860340169603, -2.2593965569142895],
            [0.9505559670000259, 1.8186404615462093, 1.3102614209223287, -0.2818222833865487, 2.531417847363128],
            [-3.643343119343121, -3.536278564808815, -1.3278425612003888, 0.9950722847078508, 1.3255688597976647],
            [0.5202169959079599, 4.02341641177549, 0.28569109611261634, -3.1155253212737266, 0.5616534222974544],
            [3.77160611108241, 2.5233863422088034, 4.721489975165859, 3.3155801154568216, -2.546080917850404],
            [1.2691209270361992, 4.019893634447016, -4.136189807597473, -2.680033709513804, -1.550100930908342],
            [-0.4574809909962204, 0.49898414237597755, 1.1495177677542232, 2.873876518676211, 3.604272671140195]
        ]

        assert np.allclose(ans, config_cands)
        for i in range(opt.tpe._n_ei_candidates):
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
        ans = [[-69.22714006071928, -3.3599748109147747, -17.290848337801528],
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
            resultfile='test',
            seed=0
        )
        opt.optimize(self.logger)

        config_cands = [np.array([0.0]) for _ in range(len(self.hp_names))]
        ll_ratio = opt.tpe.compute_probability_improvement(config_cands)

        assert ll_ratio.size == 1
        self.assertAlmostEqual(ll_ratio[0], 0.43775751678142427)


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
