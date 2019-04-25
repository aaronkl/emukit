import pickle
import torch
import numpy as np

from functools import partial
from typing import Tuple

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.examples.profet.meta_benchmarks.architecture import get_architecture, get_cost_architecture


def meta_svm(fname_objective: str, fname_cost: str, noise: bool=True) -> Tuple[UserFunctionWrapper, ParameterSpace]:
    """

    :param high_noise_std_deviation: Standard deviation of Gaussian observation noise on high fidelity observations.
                                     Defaults to zero.
    :param low_noise_std_deviation: Standard deviation of Gaussian observation noise on low fidelity observations.
                                     Defaults to zero.
    :return: Tuple of user function object and parameter space
    """
    parameter_space = ParameterSpace([
        ContinuousParameter('log_C', -10, 10),
        ContinuousParameter('log_gamma', -10, 10)])
    data = pickle.load(open(fname_objective, "rb"))
    objective = get_architecture(7).float()

    objective.load_state_dict(data["state_dict"])
    x_mean_objective = data["x_mean"]
    x_std_objective = data["x_std"]
    task_feature_objective = data["task_feature"]

    data = pickle.load(open(fname_cost, "rb"))
    cost = get_cost_architecture(7).float()

    cost.load_state_dict(data["state_dict"])
    x_mean_cost = data["x_mean"]
    x_std_cost = data["x_std"]
    task_feature_cost = data["task_feature"]

    def objective_function(config, with_noise=True):

        Ht = np.repeat(task_feature_objective[None, :], config.shape[0], axis=0)
        x = np.concatenate((config, Ht), axis=1)
        x_norm = torch.from_numpy((x - x_mean_objective) / x_std_objective).float()
        o = objective.forward(x_norm).data.numpy()
        m = o[:, 0]
        log_v = o[:, 1]
        if with_noise:
            feval = np.random.randn() * np.sqrt(np.exp(log_v)) + m
        else:
            feval = m

        Ht = np.repeat(task_feature_cost[None, :], config.shape[0], axis=0)
        x = np.concatenate((config, Ht), axis=1)
        x_norm = torch.from_numpy((x - x_mean_cost) / x_std_cost).float()
        o = cost.forward(x_norm).data.numpy()
        log_m = o[:, 0]
        log_log_v = o[:, 1]
        if with_noise:
            log_c = np.random.randn() * np.sqrt(np.exp(log_log_v)) + log_m
        else:
            log_c = log_m

        return feval[:, None], np.exp(log_c)[:, None]

    f = partial(objective_function, with_noise=noise)
    # user_function = UserFunctionWrapper(f)
    return f, parameter_space
