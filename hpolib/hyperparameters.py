from dataclasses import dataclass
from typing import Literal


@dataclass
class BudgetConfig:
    epochs: int = 100


@dataclass
class Hyperparameters:
    # Architecture parameters
    n_units_1: int = 64
    n_units_2: int = 64
    dropout_1: float = 0.3
    dropout_2: float = 0.3
    activation_fn_1: Literal['relu', 'tanh'] = 'relu'
    activation_fn_2: Literal['relu', 'tanh'] = 'relu'
    # Training parameters
    init_lr: float = 1e-2
    lr_schedule: Literal["cosine", "const"] = 'cosine'
    batch_size: int = 32
