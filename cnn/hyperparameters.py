from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Optional

import torch


class LossChoices(Enum):
    cross_entropy = partial(torch.nn.CrossEntropyLoss)
    mse = partial(torch.nn.MSELoss)


class OptimizerChoices(Enum):
    adam = partial(torch.optim.Adam)
    adamw = partial(torch.optim.AdamW)
    adad = partial(torch.optim.Adadelta)
    sgd = partial(torch.optim.SGD)


@dataclass
class BudgetConfig:
    num_classes: int = 17
    epochs: int = 50
    input_size: int = 16


@dataclass
class Hyperparameters:
    # Architecture parameters
    n_conv_layers: int = 3
    kernel_size: int = 3
    n_fc_layers: int = 3
    dropout_rate: float = 0.25
    batch_normalization: bool = True
    global_avg_pooling: bool = False
    n_channels_conv0: Optional[int] = 950
    n_channels_conv1: Optional[int] = 60
    n_channels_conv2: Optional[int] = 900
    n_channels_fc0: Optional[int] = 850
    n_channels_fc1: Optional[int] = 50
    n_channels_fc2: Optional[int] = 1500
    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 415  # The total amount of training data is 80 x 17 = 1360
    optimizer: OptimizerChoices = OptimizerChoices.adam
    loss_fn: LossChoices = LossChoices.cross_entropy
