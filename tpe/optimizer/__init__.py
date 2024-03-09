from __future__ import annotations

import warnings

from tpe.optimizer.random_search import RandomSearch
from tpe.optimizer.tpe_optimizer import TPEOptimizer


warnings.filterwarnings("ignore")

opts = {
    "tpe": TPEOptimizer,
    "random_search": RandomSearch,
}
