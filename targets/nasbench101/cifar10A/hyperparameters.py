from dataclasses import dataclass
from typing import Literal

import itertools
import numpy as np

from nasbench import api
from targets.nasbench101.api import MAX_VERTICES


@dataclass
class BudgetConfig:
    epochs: int = 108  # in {4, 12, 36, 108}


@dataclass
class Hyperparameters:
    op_node_0: Literal["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]
    op_node_1: Literal["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]
    op_node_2: Literal["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]
    op_node_3: Literal["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]
    op_node_4: Literal["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]
    # Whether to take each edge. i-th vertex to j-th vertex (i < j)
    edge_0_1: bool
    edge_0_2: bool
    edge_0_3: bool
    edge_0_4: bool
    edge_0_5: bool
    edge_0_6: bool
    edge_1_2: bool
    edge_1_3: bool
    edge_1_4: bool
    edge_1_5: bool
    edge_1_6: bool
    edge_2_3: bool
    edge_2_4: bool
    edge_2_5: bool
    edge_2_6: bool
    edge_3_4: bool
    edge_3_5: bool
    edge_3_6: bool
    edge_4_5: bool
    edge_4_6: bool
    edge_5_6: bool


def config2spec(config: Hyperparameters) -> api.ModelSpec:
    adjacent_matrix = np.zeros((MAX_VERTICES, MAX_VERTICES), np.int8)

    for (src, to) in itertools.combinations(range(MAX_VERTICES), 2):
        # map: {"True", "False"} -> {True, False} -> {1, 0}
        adjacent_matrix[src, to] = int(eval(getattr(config, f"edge_{src}_{to}")))

    labeling = [getattr(config, f"op_node_{i}") for i in range(5)]
    labeling = ["input"] + labeling + ["output"]
    return api.ModelSpec(adjacent_matrix, labeling)
