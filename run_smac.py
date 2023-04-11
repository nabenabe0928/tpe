import ConfigSpace as CS

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband


def obj_func(config: CS.Configuration, budget: int, seed: int = 0) -> float:
    return config["x"] ** 2


if __name__ == "__main__":
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x", -5, 5))
    scenario = Scenario(
        config_space,
        walltime_limit=10,
        n_trials=50,
        min_budget=1,
        max_budget=100,
        n_workers=4,
    )
    smac = MFFacade(
        scenario,
        obj_func,
        initial_design=MFFacade.get_initial_design(scenario, n_configs=5),
        intensifier=Hyperband(scenario, incumbent_selection="highest_budget"),
        overwrite=True,
    )
    incumbent = smac.optimize()
    print("Finish")
