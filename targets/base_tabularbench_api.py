from abc import abstractmethod, ABCMeta
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

import json

import ConfigSpace as CS

from util.utils import get_config_space, ParameterSettings


class BaseTabularBenchAPI(metaclass=ABCMeta):
    def __init__(self, hp_module_path: str, dataset_name: str, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed)
        self._oracle: Optional[float] = None
        js = open(f"{hp_module_path}/params.json")
        searching_space: Dict[str, ParameterSettings] = json.load(js)
        self._config_space = get_config_space(searching_space, hp_module_path=".".join(hp_module_path.split("/")))

    def find_oracle(self) -> Tuple[float, float]:
        """
        Find the oracle.

        Returns:
            best_oracle, worst_oracle (Tuple[float, float]):
                The best and worst possible loss value available in this benchmark.
                It considers each seed independently.
        """
        loss_vals = self.fetch_all_losses()
        return loss_vals.min(), loss_vals.max()

    @staticmethod
    def _validate_choice(
        choice: Union[str, Enum],
        choice_enum: Enum
    ) -> Enum:

        enum_name = choice_enum.__name__
        enum_keys = list(choice_enum.__members__.keys())
        if isinstance(choice, choice_enum):
            return choice
        elif isinstance(choice, str):
            for c in choice_enum:
                if c.name == choice:
                    return c
            else:
                raise ValueError(f"Expect the choice to be in {enum_keys}, but got `{choice}``")
        else:
            raise TypeError(f"dataset_choice must be str or {enum_name}, but got {type(choice)}")

    @abstractmethod
    def fetch_all_losses(self) -> np.ndarray:
        """
        Fetch the loss values in this instance.

        Returns:
            losses (np.ndarray):
                The loss values available in this benchmark.
        """
        raise NotImplementedError

    @abstractmethod
    def objective_func(self, config: Dict[str, Any], budget: Dict[str, Any] = {}) -> float:
        """
        Args:
            config (Dict[str, Any]):
                The dict of the configuration and the corresponding value
            budget (Dict[str, Any]):
                The budget information

        Returns:
            val_error (float):
                The validation error given a configuration and a budget.
        """
        raise NotImplementedError

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        """The config space of the child tabular benchmark"""
        return self._config_space

    @property
    def oracle(self) -> Optional[float]:
        """The global best performance given a constraint"""
        return self._oracle

    @property
    @abstractmethod
    def data(self) -> Any:
        """API for the target dataset"""
        raise NotImplementedError
