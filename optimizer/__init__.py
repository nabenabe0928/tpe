import logging
import warnings

from optimizer.random_search import RandomSearch
from optimizer.tpe import TPEOptimizer


logging.getLogger("ax.core.parameter").setLevel(logging.CRITICAL)
logging.getLogger("ax.core.experiment").setLevel(logging.CRITICAL)
logging.getLogger("ax.service.managed_loop").setLevel(logging.CRITICAL)
logging.getLogger("ax.modelbridge.dispatch_utils").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


opts = {
    "tpe": TPEOptimizer,
    "random_search": RandomSearch,
}
