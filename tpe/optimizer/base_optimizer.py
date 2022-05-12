import time
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple

import ConfigSpace as CS

import numpy as np

from tpe.utils.constants import NumericType
from tpe.utils.utils import get_logger, get_random_sample, revert_eval_config, store_results


class ObjectiveFunc(Protocol):
    def __call__(self, eval_config: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
        """
        Objective func prototype.

        Args:
            eval_config (Dict[str, Any]):
                A configuration after the reversion.

        Returns:
            results (Dict[str, float]):
                Each metric obtained by the objective function.
            runtime (float):
                The runtime required to get all metrics
        """
        raise NotImplementedError


class BestUpdateFunc(Protocol):
    def __call__(self, results: Dict[str, float], loss: float, best_loss: float) -> bool:
        """
        Objective func prototype.

        Args:
            results (Dict[str, float]):
                Each metric obtained by the objective function.
            loss (float):
                The loss metric.
            best_loss (float):
                The best loss metric up to now.

        Returns:
            _ (bool):
                Whether the current results are the best.
        """
        raise NotImplementedError


def default_best_update(results: Dict[str, float], loss: float, best_loss: float) -> bool:
    return loss < best_loss


class BaseOptimizer(metaclass=ABCMeta):
    def __init__(
        self,
        obj_func: ObjectiveFunc,
        config_space: CS.ConfigurationSpace,
        resultfile: str,
        n_init: int,
        max_evals: int,
        seed: Optional[int],
        metric_name: str,
        runtime_name: str,
        only_requirements: bool,
        result_keys: List[str],
    ):
        """
        Attributes:
            rng (np.random.RandomState): random state to maintain the reproducibility
            resultfile (str): The name of the result file to output in the end
            n_init (int): The number of random sampling before using TPE
            obj_func (Callable): The objective function
            hp_names (List[str]): The list of hyperparameter names
            metric_name (str): The name of the metric (or objective function value)
            observations (Dict[str, Any]): The storage of the observations
            config_space (CS.ConfigurationSpace): The searching space of the task
            is_categoricals (Dict[str, bool]): Whether the given hyperparameter is categorical
            is_ordinals (Dict[str, bool]): Whether the given hyperparameter is ordinal
            only_requirements (bool): If True, we only save runtime and loss.
            result_keys (List[str]): Keys of results.
        """

        self._rng = np.random.RandomState(seed)
        self._n_init, self._max_evals = n_init, max_evals
        self.resultfile = resultfile
        self._obj_func = obj_func
        self._hp_names = list(config_space._hyperparameters.keys())
        self._metric_name = metric_name
        self._runtime_name = runtime_name
        self._result_keys = result_keys[:]
        self._requirements = [metric_name, self._runtime_name] if only_requirements else None

        self._config_space = config_space
        self._is_categoricals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == "CategoricalHyperparameter"
            for hp_name in self._hp_names
        }
        self._is_ordinals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == "OrdinalHyperparameter"
            for hp_name in self._hp_names
        }

    def optimize(
        self, logger_name: Optional[str] = None, best_update: BestUpdateFunc = default_best_update
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize obj_func using TPE Sampler and store the results in the end.

        Args:
            logger_name (Optional[str]):
                The name of logger to write the intermediate results

        Returns:
            best_config (Dict[str, Any]): The configuration that has the best loss
            best_loss (float): The best loss value during the optimization
        """
        use_logger = logger_name is not None
        logger_name = logger_name if use_logger else "temp"
        logger = get_logger(logger_name, logger_name, disable=use_logger)
        best_config, best_loss, t = {}, np.inf, 0

        while True:
            logger.info(f"\nIteration: {t + 1}")
            start = time.time()
            eval_config = self.initial_sample() if t < self._n_init else self.sample()
            time2sample = time.time() - start

            results, runtime = self._obj_func(eval_config)
            self.update(eval_config=eval_config, results=results, runtime=runtime + time2sample)
            loss = results[self._metric_name]

            if best_update(results=results, loss=loss, best_loss=best_loss):
                best_loss = loss
                best_config = eval_config

            logger.info(f"Cur. loss: {loss:.4e}, Cur. Config: {eval_config}")
            logger.info(f"Best loss: {best_loss:.4e}, Best Config: {best_config}")
            t += 1

            if t >= self._max_evals:
                break

        observations = self.fetch_observations()
        logger.info(f"Best loss: {best_loss:.4e}")
        store_results(
            best_config=best_config,
            logger=logger,
            observations=observations,
            file_name=self.resultfile,
            requirements=self._requirements,
        )

        return best_config, best_loss

    @abstractmethod
    def update(self, eval_config: Dict[str, Any], results: Dict[str, float], runtime: float) -> None:
        """
        Update of the child sampler.

        Args:
            eval_config (Dict[str, Any]): The configuration to be evaluated
            results (Dict[str, float]):
                Each metric obtained by the objective function.
            runtime (float):
                The runtime required to get all metrics
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_observations(self) -> Dict[str, np.ndarray]:
        """
        Fetch observations of this optimization.

        Returns:
            observations (Dict[str, np.ndarray]):
                observations of this optimization.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> Dict[str, Any]:
        """
        Sample a configuration using a child class instance sampler

        Returns:
            eval_config (Dict[str, Any]): A sampled configuration
        """
        raise NotImplementedError

    def initial_sample(self) -> Dict[str, Any]:
        """
        Sampling method up to n_init configurations

        Returns:
            samples (Dict[str, Any]):
                Typically randomly sampled configurations

        """
        eval_config = {hp_name: self._get_random_sample(hp_name=hp_name) for hp_name in self._hp_names}
        return self._revert_eval_config(eval_config=eval_config)

    def _get_random_sample(self, hp_name: str) -> NumericType:
        return get_random_sample(
            hp_name=hp_name,
            rng=self._rng,
            config_space=self._config_space,
            is_categorical=self._is_categoricals[hp_name],
            is_ordinal=self._is_ordinals[hp_name],
        )

    def _revert_eval_config(self, eval_config: Dict[str, NumericType]) -> Dict[str, Any]:
        return revert_eval_config(
            eval_config=eval_config,
            config_space=self._config_space,
            is_categoricals=self._is_categoricals,
            is_ordinals=self._is_ordinals,
            hp_names=self._hp_names,
        )
