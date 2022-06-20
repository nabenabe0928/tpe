import json
import os
from logging import DEBUG, FileHandler, Formatter, Logger, basicConfig, getLogger
from typing import Any, Dict, List, Literal, Optional, Type, TypedDict, Union

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from tpe.utils.constants import NumericType, config2type


bool_json = Literal["True", "False"]


class MetaInformation(TypedDict):
    lower: Optional[NumericType]  # The lower bound of parameters
    upper: Optional[NumericType]  # The upper bound of parameters
    log: Optional[Union[bool_json, bool]]  # scale: If True, log, otherwise uniform
    q: Optional[NumericType]  # The quantization parameter"


class ParameterSettings(TypedDict):
    param_type: Union[Type[int], Type[float], Type[str], Type[bool]]  # parameter type
    ignore: bool_json  # Whether we ignore this parameter or not
    lower: Optional[NumericType]  # The lower bound of parameters
    upper: Optional[NumericType]  # The upper bound of parameters
    log: Optional[Union[bool_json, bool]]  # scale: If True, log, otherwise uniform
    q: Optional[NumericType]  # The quantization parameter"
    sequence: Optional[List[NumericType]]  # The choices for numerical parameters.
    choices: Optional[List[str]]  # The choices for categorical parameters. Must be str for config space
    dataclass: Optional[str]  # The choices for categorical parameters in dataclass
    default_value: Optional[Union[NumericType, str]]  # The default value for this parameter
    meta: Optional[MetaInformation]  # Any information that is useful


def get_logger(file_name: str, logger_name: str, disable: bool = False) -> Logger:
    subdirs = "/".join(file_name.split("/")[:-1])
    os.makedirs(f"log/{subdirs}", exist_ok=True)

    file_path = f"log/{file_name}.log"
    fmt = "%(asctime)s [%(levelname)s/%(filename)s:%(lineno)d] %(message)s"

    if not disable:
        basicConfig(filename=file_path, format=fmt, level=DEBUG)

    logger = getLogger(logger_name)
    file_handler = FileHandler(file_path, mode="a")
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(Formatter(fmt))
    logger.propagate = False
    logger.addHandler(file_handler)

    return logger


def get_random_sample(
    hp_name: str,
    is_categorical: bool,
    is_ordinal: bool,
    rng: np.random.RandomState,
    config_space: CS.ConfigurationSpace,
) -> NumericType:
    """
    Random sample of a provided hyperparameter

    Args:
        config_space (CS.ConfigurationSpace): The searching space information
        is_categorical (bool): Whether this hyperparameter is categorical
        is_ordinal (bool): Whether this hyperparameter is ordinal
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
    elif is_ordinal:
        if config.meta is None:
            raise ValueError("The meta information of the ordinal hyperparameter must be provided")

        log = config.meta.get("log", False)
        seq = config.sequence
        sample = seq[rng.randint(len(seq))]
        sample = np.log(sample) if log else sample
    else:
        lb = np.log(config.lower) if config.log else config.lower
        ub = np.log(config.upper) if config.log else config.upper
        sample = rng.uniform() * (ub - lb) + lb

    return sample


def check_value_range(hp_name: str, config: CSH.Hyperparameter, val: NumericType) -> None:
    if val < config.lower or val > config.upper:
        raise ValueError(f"The sampled value for {hp_name} must be [{config.lower},{config.upper}], but got {val}.")


def revert_eval_config(
    eval_config: Dict[str, NumericType],
    config_space: CS.ConfigurationSpace,
    is_categoricals: Dict[str, bool],
    is_ordinals: Dict[str, bool],
    hp_names: List[str],
) -> Dict[str, Any]:
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
        is_ordinals (Dict[str, bool]): Whether each hyperparameter is ordinal
        hp_names (List[str]): The list of the names of hyperparameters

    Returns:
        reverted_eval_config (Dict[str, Any])
    """
    converted_eval_config: Dict[str, Any] = {}
    for hp_name in hp_names:
        is_categorical, is_ordinal = is_categoricals[hp_name], is_ordinals[hp_name]
        config = config_space.get_hyperparameter(hp_name)
        val = eval_config[hp_name]

        if is_categorical:
            converted_eval_config[hp_name] = config.choices[val]
        elif is_ordinal:
            if config.meta is None:
                raise ValueError("The meta information of the ordinal hyperparameter must be provided")

            log = config.meta.get("log", False)
            vals = np.log(config.sequence) if log else np.array(config.sequence)
            diff = np.abs(vals - val)
            converted_eval_config[hp_name] = config.sequence[diff.argmin()]
        else:
            dtype = config2type[config.__class__.__name__]
            q = config.q
            if config.log:
                val = np.exp(val)
            if q is not None or dtype is int:
                lb = config.lower
                q = 1 if q is None and dtype is int else q
                val = np.round((val - lb) / q) * q + lb

            check_value_range(hp_name=hp_name, config=config, val=val)

            converted_eval_config[hp_name] = dtype(val)

    return converted_eval_config


def store_results(
    logger: Logger,
    observations: Dict[str, np.ndarray],
    file_name: str,
    requirements: Optional[List[str]] = None,
) -> None:

    logger.info(f"\nThe observations: {observations}")
    if requirements is None:
        save_observations(filename=file_name, observations=observations)
    else:
        required_observations = {k: v for k, v in observations.items() if k in requirements}
        save_observations(filename=file_name, observations=required_observations)


def save_observations(filename: str, observations: Dict[str, np.ndarray]) -> None:
    """
    Save the observations during the optimization procedure

    Args:
        filename (str): The name of the json file
        observations (Dict[str, np.ndarray]): The observations to save
    """
    subdirs = "/".join(filename.split("/")[:-1])
    os.makedirs(f"results/{subdirs}", exist_ok=True)

    with open(f"results/{filename}.json", mode="w") as f:
        json.dump({k: v.tolist() for k, v in observations.items()}, f, indent=4)
