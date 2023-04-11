import time
from typing import Dict

import neps


def run_pipeline(**eval_config) -> Dict[str, float]:
    start = time.time()
    return dict(loss=eval_config["x"]**2, cost=time.time() - start)


pipeline_space = dict(
    x=neps.FloatParameter(lower=-5.0, upper=5.0),
    epoch=neps.IntegerParameter(lower=1, upper=100, is_fidelity=True),
)

neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="neps-log",
    max_cost_total=1e-4,  # total seconds of obj funcs
)
