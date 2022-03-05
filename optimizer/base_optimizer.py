from abc import abstractmethod, ABCMeta
from logging import Logger
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

import ConfigSpace as CS

from util.constants import NumericType
from util.utils import get_random_sample, revert_eval_config, store_results


class BaseOptimizer(metaclass=ABCMeta):
    def __init__(
        self,
        obj_func: Callable,
        config_space: CS.ConfigurationSpace,
        resultfile: str,
        n_init: int = 10,
        max_evals: int = 100,
        seed: Optional[int] = None,
        metric_name: str = "loss",
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
        """

        self._rng = np.random.RandomState(seed)
        self._n_init, self._max_evals = n_init, max_evals
        self.resultfile = resultfile
        self._obj_func = obj_func
        self._hp_names = list(config_space._hyperparameters.keys())
        self._metric_name = metric_name

        self._config_space = config_space
        self._is_categoricals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == "CategoricalHyperparameter"
            for hp_name in self._hp_names
        }
        self._is_ordinals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == "OrdinalHyperparameter"
            for hp_name in self._hp_names
        }

    def optimize(self, logger: Logger) -> Tuple[Dict[str, Any], float]:
        """
        Optimize obj_func using TPE Sampler and store the results in the end.

        Args:
            logger (Logger): The logging to write the intermediate results

        Returns:
            best_config (Dict[str, Any]): The configuration that has the best loss
            best_loss (float): The best loss value during the optimization
        """

        best_config, best_loss, t = {}, np.inf, 0

        while True:
            logger.info(f"\nIteration: {t + 1}")
            eval_config = self.initial_sample() if t < self._n_init else self.sample()

            loss = self._obj_func(eval_config)
            self.update(eval_config=eval_config, loss=loss)

            if best_loss > loss:
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
        )

        return best_config, best_loss

    @abstractmethod
    def update(self, eval_config: Dict[str, Any], loss: float) -> None:
        """
        Update of the child sampler.

        Args:
            eval_config (Dict[str, Any]): The configuration to be evaluated
            loss (float): The loss value of the eval_config
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
