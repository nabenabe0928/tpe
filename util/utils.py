import os
from logging import basicConfig, getLogger, DEBUG, FileHandler, Formatter, Logger
from typing import Any, Dict, List, Literal, Optional, Type, TypedDict, Union

import json

import numpy as np

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from util.constants import CategoricalHPType, NumericalHPType, NumericType, config2type, type2config


bool_json = Literal['True', 'False']


class ParameterSettings(TypedDict):
    param_type: Union[Type[int], Type[float], Type[str], Type[bool]]  # parameter type
    ignore: bool_json  # Whether we ignore this parameter or not
    lower: Optional[NumericType]  # The lower bound of parameters
    upper: Optional[NumericType]  # The upper bound of parameters
    log: Optional[Union[bool_json, bool]]  # scale: If True, log, otherwise uniform
    q: Optional[NumericType]  # The quantization parameter"
    choices: Optional[List[str]]  # The choices for categorical parameters. Must be str for config space
    dataclass: Optional[str]  # The choices for categorical parameters in dataclass
    default_value: Optional[Union[NumericType, str]]  # The default value for this parameter


def get_hyperparameter_module(hp_module_path: str) -> Any:
    return getattr(__import__(hp_module_path), 'hyperparameters')


def get_hyperparameter(
    param_name: str,
    config_type: Union[CategoricalHPType, NumericalHPType],
    settings: ParameterSettings,
    hp_module_path: str
) -> Union[CategoricalHPType, NumericalHPType]:
    """
    Create a hyperparameter class by CSH.<config_type>

    Args:
        param_name (str): The name of hyperparameter
        config_type (Union[CategoricalHPType, NumericalHPType]):
            The class of hyperparameter in the CSH package
        settings (ParameterSettings):
            Parameter settings from a json file

    Returns:
        hp (Union[CategoricalHPType, NumericalHPType]):
            The hyperparameter for ConfigurationSpace

    TODO: Add tests
    """
    config_dict = {key: val for key, val in settings.items() if key not in ['ignore', 'param_type', 'dataclass']}
    hp_module = get_hyperparameter_module(hp_module_path)

    if 'dataclass' in settings.keys():
        assert(isinstance(settings['dataclass'], str))
        choices = getattr(hp_module, settings['dataclass'])
        config_dict['choices'] = [c.name for c in choices]

    hyperparameter = getattr(CSH, config_type)(name=param_name, **config_dict)

    return hyperparameter


def get_config_space(searching_space: Dict[str, ParameterSettings],
                     hp_module_path: str) -> CS.ConfigurationSpace:
    """
    Create the config space by CS.ConfigurationSpace based on searching space dict

    Args:
        searching_space (Dict[str, ParameterSettings]):
            The dict of the pairs of a parameter name and the details
        hp_module_path (str): The path of the module describing the searching space

    Returns:
        cs (CS.ConfigurationSpace):
            The configuration space that includes the parameters specified in searching_space

    TODO: Add tests
    """

    settings_keys = set(ParameterSettings.__annotations__.keys())
    cs = CS.ConfigurationSpace()
    type_choices = ['int', 'float', 'bool', 'str']
    for param_name, settings in searching_space.items():
        if param_name.startswith('_') or settings['ignore'] == 'True':
            continue

        if settings['param_type'] not in type_choices:
            raise ValueError('`param_type` in params.json must have one of {}, '
                             'but got {}.'.format(type_choices, settings['param_type']))

        if len(set(settings.keys()) - settings_keys) >= 1:
            raise ValueError('params.json must not include the keys not specified in {}.'.format(
                settings_keys
            ))

        if 'log' in settings.keys():
            if settings['log'] not in bool_json.__args__:  # type: ignore
                raise ValueError('`log` in params.json must have either "True" or "False", '
                                 'but got {}.'.format(settings['log']))
            settings['log'] = eval(settings['log'])  # type: ignore

        config_type = type2config[eval(settings['param_type'])]  # type: ignore
        cs.add_hyperparameter(get_hyperparameter(param_name=param_name, config_type=config_type,
                                                 settings=settings, hp_module_path=hp_module_path))

    return cs


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


def extract_hyperparameter(eval_config: Dict[str, Any],
                           config_space: CS.ConfigurationSpace,
                           searching_space: Dict[str, ParameterSettings],
                           hp_module_path: str,
                           ) -> Dict[str, Any]:

    return_config = {}
    hp_module = get_hyperparameter_module(hp_module_path)
    for key, val in eval_config.items():
        if not isinstance(val, str):
            return_config[key] = val
            continue

        if 'dataclass' in searching_space[key].keys():
            choices_class_name = searching_space[key]['dataclass']
            assert(isinstance(choices_class_name, str))
            choices = getattr(hp_module, choices_class_name)
            return_config[key] = getattr(choices, val)
        elif hasattr(config_space.get_hyperparameter(key), 'choices'):
            return_config[key] = eval(val)

    return return_config


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


def store_results(best_config: Dict[str, np.ndarray], logger: Logger,
                  observations: Dict[str, np.ndarray], file_name: str) -> None:
    logger.info('\nThe observations: {}'.format(observations))
    save_observations(filename=file_name, observations=observations)

    os.makedirs('incumbents/', exist_ok=True)
    with open(f'incumbents/opt_{file_name}.json', mode='w') as f:
        json.dump(best_config, f, indent=4)


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
