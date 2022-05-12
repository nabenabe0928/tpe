import os
import pytest

from typing import Dict, Tuple, Union

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import unittest

from tpe.optimizer.tpe import TPEOptimizer, TreeStructuredParzenEstimator
from tpe.utils.utils import get_logger


def sphere(eval_config: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    vals = np.array(list(eval_config.values()))
    vals *= vals
    return {"loss": vals.sum()}, 0.0


def rosen(eval_config: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    val = 0
    vals = np.array(list(eval_config.values()))
    dim = len(vals)
    for d in range(dim - 1):
        t1 = 100 * (vals[d + 1] - vals[d] ** 2) ** 2
        t2 = (vals[d] - 1) ** 2
        val += t1 + t2
    return {"loss": val}, 0.0


def mix_func(eval_config: Dict[str, Union[str, float]]) -> Tuple[Dict[str, float], float]:
    func_dict = {"cosine": np.cos, "sine": np.sin}
    vals = np.array([v for v in eval_config.values() if isinstance(v, float)])
    vals *= vals
    assert isinstance(eval_config["func"], str)
    func_name: str = eval_config["func"]
    return {"loss": 1 - func_dict[func_name](vals.sum()) ** 2}, 0.0


def cleanup() -> None:
    os.remove("log/test.log")
    os.remove("results/test.json")
    os.remove("incumbents/test.json")


def test_get_min_bandwidth() -> None:
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameters(
        [
            CSH.UniformFloatHyperparameter("x0", 1, 5, log=True, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformFloatHyperparameter("x1", 1, 5, log=True),
            CSH.UniformFloatHyperparameter("x2", 1, 5, log=True, q=0.5, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformFloatHyperparameter("x3", 1, 5, log=True, q=0.5),
            CSH.UniformFloatHyperparameter("x4", 1, 5, q=0.5, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformFloatHyperparameter("x5", 1, 5, q=0.5),
            CSH.UniformFloatHyperparameter("x6", 1, 5, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformFloatHyperparameter("x7", 1, 5),
            CSH.UniformIntegerHyperparameter("x8", 1, 5, log=True, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformIntegerHyperparameter("x9", 1, 5, log=True),
            CSH.UniformIntegerHyperparameter("x10", 1, 5, log=True, q=2, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformIntegerHyperparameter("x11", 1, 5, log=True, q=2),
            CSH.UniformIntegerHyperparameter("x12", 1, 5, q=2, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformIntegerHyperparameter("x13", 1, 5, q=2),
            CSH.UniformIntegerHyperparameter("x14", 1, 5, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformIntegerHyperparameter("x15", 1, 5),
            CSH.OrdinalHyperparameter("x16", sequence=[1, 2, 3, 4, 5], meta={"lower": 1, "upper": 5}),
            CSH.OrdinalHyperparameter(
                "x17", sequence=[1, 2, 3, 4, 5], meta={"lower": 1, "upper": 5, "min_bandwidth_factor": 0.1}
            ),
        ]
    )
    tpe = TreeStructuredParzenEstimator(
        config_space,
        percentile_func=lambda x: 2,
        n_ei_candidates=24,
        metric_name="loss",
        runtime_name="iter_time",
        seed=0,
        min_bandwidth_factor=0.01,
        top=0.8,
    )
    for d, ans in enumerate(
        [0.1, 0.01, 0.1, 0.01, 0.1, 1 / 9, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 1 / 3, 0.1, 1 / 5, 1 / 5, 0.1]
    ):
        assert tpe._get_min_bandwidth_factor(f"x{d}") == ans


class TestTreeStructuredParzenEstimator(unittest.TestCase):
    def setUp(self) -> None:
        dim = 10
        self.cs_cat = CS.ConfigurationSpace()
        self.cs = CS.ConfigurationSpace()
        self.cs_cat.add_hyperparameter(CSH.CategoricalHyperparameter("func", choices=["sine", "cosine"]))
        for d in range(dim):
            self.cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-5, upper=5))
            if d < dim - 1:
                self.cs_cat.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-5, upper=5))
            else:
                self.cs_cat.add_hyperparameter(
                    CSH.OrdinalHyperparameter(f"x{d}", sequence=list(range(-5, 6)), meta={"lower": -5, "upper": 5})
                )

        self.hp_names = list(self.cs._hyperparameters.keys())
        self.hp_names_cat = list(self.cs_cat._hyperparameters.keys())

    def test_insert_observations(self) -> None:
        metric_name = "loss"
        opt = TPEOptimizer(
            obj_func=sphere,
            config_space=self.cs,
            max_evals=10,
            metric_name=metric_name,
            resultfile="test",
            min_bandwidth_factor=1e-2,
            top=0.8,
        )
        opt.optimize(logger_name="test")
        try:
            opt._tpe_samplers[metric_name]._insert_observations("x0", insert_loc=1, val=0, is_categorical=False)
        except ValueError:
            pass
        else:
            raise RuntimeError("Did not yield ValueError")

    def test_update_observations(self) -> None:
        max_evals, metric_name, runtime_name = 20, "loss", "test_run"
        opt = TPEOptimizer(
            obj_func=mix_func,
            config_space=self.cs_cat,
            max_evals=max_evals,
            metric_name=metric_name,
            runtime_name=runtime_name,
            resultfile="test",
            min_bandwidth_factor=1e-2,
            top=0.8,
        )
        opt.optimize(logger_name="test")

        eval_config = {hp_name: 0.0 for hp_name in self.hp_names_cat}
        eval_config["func"] = "cosine"  # type: ignore
        opt._tpe_samplers[metric_name].update_observations(eval_config=eval_config, loss=0, runtime=0)
        assert opt._tpe_samplers[metric_name]._sorted_observations[metric_name][0] == 0.0
        assert opt._tpe_samplers[metric_name]._sorted_observations[runtime_name][0] == 0.0
        assert opt._tpe_samplers[metric_name]._sorted_observations[metric_name].size == max_evals + 1
        for hp_name in self.hp_names:
            assert opt._tpe_samplers[metric_name].observations[hp_name].size == max_evals + 1
            assert opt._tpe_samplers[metric_name].observations[hp_name][-1] == 0.0
            assert opt._tpe_samplers[metric_name]._sorted_observations[hp_name].size == max_evals + 1
            assert opt._tpe_samplers[metric_name]._sorted_observations[hp_name][0] == 0.0

        min_value = np.inf
        # The sorted_observations must be sorted
        for v in opt._tpe_samplers[metric_name]._sorted_observations[metric_name]:
            assert min_value >= v
            min_value = min(min_value, np.inf)

        cleanup()

    def test_apply_knowledge_augmentation(self) -> None:
        metric_name = "loss"
        opt = TPEOptimizer(
            obj_func=sphere,
            config_space=self.cs,
            max_evals=10,
            metric_name=metric_name,
            resultfile="test",
            min_bandwidth_factor=1e-2,
            top=0.8,
        )
        ob = {}
        n_samples = 10
        rng = np.random.RandomState(0)
        ob[metric_name] = rng.random(n_samples)
        for hp_name in self.hp_names:
            ob[hp_name] = rng.random(n_samples)

        order = np.argsort(ob[metric_name])
        opt._tpe_samplers[metric_name].apply_knowledge_augmentation(ob)  # type: ignore

        for hp_name in ob.keys():
            cnt = np.count_nonzero(opt._tpe_samplers[metric_name].observations[hp_name] != ob[hp_name])
            assert cnt == 0

        sorted_ob = {k: x[order] for k, x in ob.items()}
        for hp_name in ob.keys():
            cnt = np.count_nonzero(opt._tpe_samplers[metric_name]._sorted_observations[hp_name] != sorted_ob[hp_name])
            assert cnt == 0

        with pytest.raises(ValueError):
            opt._tpe_samplers[metric_name].apply_knowledge_augmentation(ob)  # type: ignore

    def test_get_config_candidates(self) -> None:
        max_evals = 5
        opt = TPEOptimizer(
            obj_func=sphere,
            config_space=self.cs,
            max_evals=max_evals,
            resultfile="test",
            seed=0,
            min_bandwidth_factor=1e-2,
            top=0.8,
        )
        opt.optimize(logger_name="test")
        metric_name = opt._metric_name

        opt._tpe_samplers[metric_name]._n_ei_candidates = 3
        config_cands = opt._tpe_samplers[metric_name].get_config_candidates()
        assert len(config_cands) == len(self.hp_names)
        assert config_cands[0].size == opt._tpe_samplers[metric_name]._n_ei_candidates

        ans = [
            [4.8431215412066475, -1.8158257273119596, -3.744716909802062],
            [2.7519831557632024, 0.2620535804840036, 0.45758517301446067],
            [-1.871838500258336, 1.7538653266921242, -4.625055394360726],
            [-4.380743016111864, 0.2554040035802357, 0.5688739967604989],
            [-0.6816800079423969, -0.9672547770712823, -0.8441738465209289],
            [1.27499304070942, -0.2818222833865487, 1.6739088876074935],
            [-3.643343119343121, -3.536278564808815, -1.3278425612003888],
            [0.5194539579613895, 1.2898291075741066, 2.2535688848821986],
            [2.6414792686135344, 3.77160611108241, 4.524089680601648],
            [-4.031769469731796, 2.082749780768603, -0.7739934294417338],
        ]
        assert np.allclose(ans, config_cands)

        max_evals = 5
        opt = TPEOptimizer(
            obj_func=mix_func,
            config_space=self.cs_cat,
            max_evals=max_evals,
            resultfile="test",
            seed=0,
            min_bandwidth_factor=1e-2,
            top=0.8,
        )
        opt.optimize(logger_name="test")

        opt._tpe_samplers[metric_name]._n_ei_candidates = 5
        config_cands = opt._tpe_samplers[metric_name].get_config_candidates()
        assert len(config_cands) == len(self.hp_names_cat)
        assert config_cands[0].size == opt._tpe_samplers[metric_name]._n_ei_candidates

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
            [-0.4574809909962204, 0.49898414237597755, 1.1495177677542232, 2.873876518676211, 3.604272671140195],
        ]

        assert np.allclose(ans, config_cands)
        for i in range(opt._tpe_samplers[metric_name]._n_ei_candidates):
            eval_config = {hp: a[-1] for a, hp in zip(ans, self.hp_names_cat)}
            ord_v = opt._revert_eval_config(eval_config)[self.hp_names_cat[-1]]
            assert ord_v in self.cs_cat.get_hyperparameter(self.hp_names_cat[-1]).sequence

        cleanup()

    def test_compute_probability_improvement(self) -> None:
        max_evals = 5
        opt = TPEOptimizer(
            obj_func=sphere,
            config_space=self.cs,
            max_evals=max_evals,
            resultfile="test",
            seed=0,
            min_bandwidth_factor=1e-2,
            top=0.8,
        )
        opt.optimize(logger_name="test")
        metric_name = opt._metric_name

        config_cands = [np.array([0.0]) for _ in range(len(self.hp_names))]
        ll_ratio = opt._tpe_samplers[metric_name].compute_probability_improvement(config_cands)

        assert ll_ratio.size == 1
        self.assertAlmostEqual(ll_ratio[0], 0.333536394021221)


class TestTPEOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        dim = 10
        self.cs = CS.ConfigurationSpace()
        for d in range(dim):
            self.cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-5, upper=5))

        self.hp_names = list(self.cs._hyperparameters.keys())
        self.logger = get_logger(file_name="test", logger_name="test")

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
                    resultfile="test",
                    seed=i,
                    min_bandwidth_factor=1e-2,
                    top=0.8,
                )
                _, best_loss = opt.optimize(logger_name="test")
                losses[i] = best_loss
                assert opt._tpe_samplers[opt._metric_name].observations["x0"].size == max_evals

            # performance test
            assert losses.mean() < threshold

        cleanup()


if __name__ == "__main__":
    unittest.main()
