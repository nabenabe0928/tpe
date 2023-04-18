import ConfigSpace as CS

from baseline_utils._smac import run_smac


def obj_func(eval_config: CS.Configuration, budget: int) -> float:
    return dict(loss=eval_config["x"] ** 2, runtime=budget / 10.0)


if __name__ == "__main__":
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x", -5, 5))
    run_smac(
        obj_func=obj_func,
        config_space=config_space,
        min_budget=1,
        max_budget=100,
        n_workers=4,
        subdir_name="smac",
        n_init=5,
    )
