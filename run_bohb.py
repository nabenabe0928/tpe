import time
from typing import Dict

import ConfigSpace as CS

from hpbandster.core import nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB


class ObjFunc(Worker):
    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleep_interval = sleep_interval

    def compute(self, config: Dict, budget: float, **kwargs):
        start = time.time()
        res = {
            "loss": config["x"] ** 2,
            "info": {"budget": budget, "cost": time.time() - start}
        }
        return res

    @staticmethod
    def get_configspace() -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x", -5, 5))
        return config_space


n_parallels = 4
run_id = "test-bohb"
ns_host = "127.0.0.1"
ns = hpns.NameServer(run_id=run_id, host=ns_host, port=None)
ns.start()

workers = []
for i in range(n_parallels):
    worker = ObjFunc(sleep_interval=0.5, nameserver=ns_host, run_id=run_id, id=i)
    worker.run(background=True)
    workers.append(worker)

bohb = BOHB(
    configspace=ObjFunc.get_configspace(),
    run_id=run_id,
    min_budget=1,
    max_budget=100,
)
bohb.run(n_iterations=50, min_n_workers=4)

bohb.shutdown(shutdown_workers=True)
ns.shutdown()
