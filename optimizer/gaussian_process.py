import numpy as np
import torch
from optimizer.base_optimizer import BaseOptimizer
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import joint_optimize


def optimize_EI(gp, best_f, n_dim):
    """
    Reference: https://botorch.org/api/optim.html

    bounds: 2d-ndarray (2, D)
        The values of lower and upper bound of each parameter.
    q: int
        The number of candidates to sample
    num_restarts: int
        The number of starting points for multistart optimization.
    raw_samples: int
        The number of initial points.

    Returns for joint_optimize is (num_restarts, q, D)
    """

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    ei = ExpectedImprovement(gp, best_f=best_f, maximize=False)
    bounds = torch.from_numpy(np.array([[0.] * n_dim, [1.] * n_dim]))
    x = joint_optimize(ei,
                       bounds=bounds,
                       q=1,
                       num_restarts=3,
                       raw_samples=15)

    return np.array(x[0])


class GPBO(BaseOptimizer):
    def __init__(self,
                 hp_utils,
                 n_parallels=1,
                 n_init=10,
                 max_evals=100,
                 n_experiments=0,
                 restart=True,
                 seed=None,
                 verbose=True,
                 print_freq=1
                 ):

        super().__init__(hp_utils,
                         n_parallels=n_parallels,
                         n_init=n_init,
                         max_evals=max_evals,
                         n_experiments=n_experiments,
                         restart=restart,
                         seed=seed,
                         verbose=verbose,
                         print_freq=print_freq
                         )
        self.opt = self.sample
        self.n_dim = len(hp_utils.config_space._hyperparameters)

    def sample(self):
        """
        Training Data: ndarray (N, D)
        Training Label: ndarray (N, )
        """

        X, Y = self.hp_utils.load_hps_conf(convert=True, do_sort=False)
        X, y = map(np.asarray, [X, Y[0]])
        X, y = torch.from_numpy(X), torch.from_numpy(y)

        gp = SingleTaskGP(X, y)
        x = optimize_EI(gp, y.min(), self.n_dim)

        return self.hp_utils.revert_hp_conf(x)
