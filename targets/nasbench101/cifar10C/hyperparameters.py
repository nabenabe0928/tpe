from dataclasses import dataclass
from typing import Literal

import itertools
import numpy as np

from nasbench import api
from targets.nasbench101.api import MAX_VERTICES


IDX2ROW, IDX2COL = np.triu_indices(MAX_VERTICES, k=1)


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
    n_edges: int  # [0, 9]
    # The likelihood of each edge. i-th vertex to j-th vertex (i < j)
    edge_0_1: float
    edge_0_2: float
    edge_0_3: float
    edge_0_4: float
    edge_0_5: float
    edge_0_6: float
    edge_1_2: float
    edge_1_3: float
    edge_1_4: float
    edge_1_5: float
    edge_1_6: float
    edge_2_3: float
    edge_2_4: float
    edge_2_5: float
    edge_2_6: float
    edge_3_4: float
    edge_3_5: float
    edge_3_6: float
    edge_4_5: float
    edge_4_6: float
    edge_5_6: float


def config2spec(config: Hyperparameters) -> api.ModelSpec:
    edge_prob = np.array(
        [getattr(config, f"edge_{src}_{to}") for (src, to) in itertools.combinations(range(MAX_VERTICES), 2)]
    )
    # descending order
    indices = np.argsort(-edge_prob)[: config.n_edges]
    adjacent_matrix = np.zeros((MAX_VERTICES, MAX_VERTICES), np.int8)

    for idx in indices:
        row, col = IDX2ROW[idx], IDX2COL[idx]
        adjacent_matrix[row, col] = 1

    labeling = [getattr(config, f"op_node_{i}") for i in range(5)]
    labeling = ["input"] + labeling + ["output"]
    return api.ModelSpec(adjacent_matrix, labeling)
