from dataclasses import dataclass
from typing import Literal


@dataclass
class BudgetConfig:
    epochs: int = 200  # in [1, 200]


@dataclass
class Hyperparameters:
    # The component to connect the i-th node and the j-th node (i < j)
    edge_0_1: Literal["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
    edge_0_2: Literal["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
    edge_0_3: Literal["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
    edge_1_2: Literal["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
    edge_1_3: Literal["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
    edge_2_3: Literal["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
