import os
from logging import basicConfig, getLogger, DEBUG, FileHandler, Formatter, Logger
from typing import Any, Dict, List, Literal

import json

import numpy as np

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from util.constants import NumericType, config2type


bool_json = Literal['True', 'False']


def get_logger(file_name: str, logger_name: str) -> Logger:
    os.makedirs('log/', exist_ok=True)

    file_path = f'log/{file_name}.log'
    fmt = "%(asctime)s [%(levelname)s/%(filename)s:%(lineno)d] %(message)s"
    basicConfig(filename=file_path, format=fmt, level=DEBUG)
    logger = getLogger(logger_name)
    file_handler = FileHandler(file_path, mode='a')
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(Formatter(fmt))
    logger.propagate = False
    logger.addHandler(file_handler)

    return logger


def get_random_sample(hp_name: str, is_categorical: bool, rng: np.random.RandomState,
                      config_space: CS.ConfigurationSpace) -> NumericType:
    """
    Random sample of a provided hyperparameter

    Args:
        config_space (CS.ConfigurationSpace): The searching space information
        is_categorical (bool): Whether this hyperparameter is categorical
        hp_name (str): The names of a hyperparameter
        rng (np.random.RandomState): The random state for numpy

    Returns:
        value (NumericType): A sampled value before conversion
            * categorical -> index of a symbol
            * log -> a value in a log-scale
            * q -> no quantization here
    """
    config = config_space.get_hyperparameter(hp_name)

    if is_categorical:
        choices = config.choices
        sample = rng.randint(len(choices))
    else:
        lb = np.log(config.lower) if config.log else config.lower
        ub = np.log(config.upper) if config.log else config.upper
        sample = rng.uniform() * (ub - lb) + lb

    return sample


def check_value_range(hp_name: str, config: CSH.Hyperparameter, val: NumericType) -> None:
    if val < config.lower or val > config.upper:
        raise ValueError('The sampled value for {} must be [{},{}], but got {}.'.format(
            hp_name, config.lower, config.upper, val
        ))


def revert_eval_config(eval_config: Dict[str, NumericType], config_space: CS.ConfigurationSpace,
                       is_categoricals: Dict[str, bool], hp_names: List[str]) -> Dict[str, Any]:
    """
    Revert the eval_config into the original value range.
    For example,
        * categorical: index of a symbol -> the symbol
        * log float: log(value) -> exp(log(value)) = value
        * quantized value: value -> steped by q, i.e. q, 2q, 3q, ...

    Args:
        eval_config (Dict[str, NumericType]): The configuration to evaluate and revert
        config_space (CS.ConfigurationSpace): The searching space information
        is_categoricals (Dict[str, bool]): Whether each hyperparameter is categorical
        hp_names (List[str]): The list of the names of hyperparameters

    Returns:
        reverted_eval_config (Dict[str, Any])
    """
    converted_eval_config: Dict[str, Any] = {}
    for hp_name in hp_names:
        is_categorical = is_categoricals[hp_name]
        config = config_space.get_hyperparameter(hp_name)
        val = eval_config[hp_name]

        if is_categorical:
            converted_eval_config[hp_name] = config.choices[val]
        else:
            dtype = config2type[config.__class__.__name__]
            q = config.q
            if config.log:
                val = np.exp(val)
            if q is not None or dtype is int:
                q = 1 if q is None and dtype is int else q
                val = np.round(val / q) * q

            check_value_range(hp_name=hp_name, config=config, val=val)

            converted_eval_config[hp_name] = dtype(val)

    return converted_eval_config


def save_observations(filename: str, observations: Dict[str, np.ndarray]) -> None:
    """
    Save the observations during the optimization procedure

    Args:
        filename (str): The name of the json file
        observations (Dict[str, np.ndarray]): The observations to save
    """
    dir_name = 'results/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    with open(f'{dir_name}{filename}.json', mode='w') as f:
        json.dump({k: v.tolist() for k, v in observations.items()}, f, indent=4)
