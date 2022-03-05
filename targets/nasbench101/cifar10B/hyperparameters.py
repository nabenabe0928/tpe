from dataclasses import dataclass
from typing import Literal

import numpy as np

from nasbench import api
from targets.nasbench101.api import MAX_VERTICES, MAX_EDGES


IDX2ROW, IDX2COL = np.triu_indices(MAX_VERTICES, k=1)

EdgeIndices = Literal[
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"
]


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
    # Which edges to take
    edge_0: EdgeIndices
    edge_1: EdgeIndices
    edge_2: EdgeIndices
    edge_3: EdgeIndices
    edge_4: EdgeIndices
    edge_5: EdgeIndices
    edge_6: EdgeIndices
    edge_7: EdgeIndices
    edge_8: EdgeIndices


def config2spec(config: Hyperparameters) -> api.ModelSpec:
    # descending order
    adjacent_matrix = np.zeros((MAX_VERTICES, MAX_VERTICES), np.int8)
    indices = [int(getattr(config, f"edge_{i}")) for i in range(MAX_EDGES)]

    for idx in indices:
        row, col = IDX2ROW[idx], IDX2COL[idx]
        adjacent_matrix[row, col] = 1

    labeling = [getattr(config, f"op_node_{i}") for i in range(5)]
    labeling = ["input"] + labeling + ["output"]
    return api.ModelSpec(adjacent_matrix, labeling)
