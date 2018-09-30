from typing import Tuple

import numpy as np
import GPy
from ..core.interfaces.models import IModel, IDifferentiable


class BOGP(IModel, IDifferentiable):

    def __init__(self, X_init, Y_init, noise=1e-8):
        """

        :param X_init:
        :param Y_init:
        :param noise:
        """

        super(BOGP, self).__init__()

        self.noise = noise

        # self.X_mean = np.mean(X_init)
        # self.X_std = np.std(X_init)
        # self._X = (X_init - self.X_mean) / self.X_std

        self._X = X_init
        # self.Y_mean = np.mean(Y_init)
        # self.Y_std = np.std(Y_init)
        # self._Y = (Y_init - self.Y_mean) / self.Y_std
        self._Y = Y_init
        kernel = GPy.kern.Matern52(input_dim=self.X.shape[1], active_dims=[i for i in range(self.X.shape[1])],
                                   variance=np.var(self.Y), ARD=True)

        self.gp = GPy.models.GPRegression(self.X, self.Y, kernel=kernel, noise_var=noise)
        # self.gp.likelihood.set_prior(GPy.priors.Exponential(1))
        self.gp.likelihood.constrain_positive()

    def optimize(self, num_restarts=3, verbose=False):
        self.gp.likelihood.constrain_fixed(self.noise)
        self.gp.optimize_restarts(messages=verbose, num_restarts=num_restarts, robust=True)
        self.gp.likelihood.constrain_positive()
        self.gp.optimize_restarts(messages=verbose, num_restarts=num_restarts, robust=True)

    def predict(self, X):
        # X_ = (X - self.X_mean) / self.X_std
        X_ = X
        m, v = self.gp.predict(X_)
        # return (m * self.Y_std + self.Y_mean), v * self.Y_std ** 2
        return m, v

    def update_data(self, X, Y):
        # self.X_mean = np.mean(X)
        # self.X_std = np.std(X)
        # self._X = (X - self.X_mean) / self.X_std

        # self.Y_mean = np.mean(Y)
        # self.Y_std = np.std(Y)
        # self._Y = (Y - self.Y_mean) / self.Y_std

        self._X = X
        self._Y = Y
        self.gp.set_XY(self.X, self.Y)

    def get_f_minimum(self):
        return np.min(self.Y)

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get gradient of the predictions
        :return: (mean gradient, variance gradient) n_points x n_dimensions arrays of the gradients of the predictive
                 distribution at each input location
        """
        # X_ = (X - self.X_mean) / self.X_std
        X_ = X
        d_mean_dx, d_variance_dx = self.gp.predictive_gradients(X_)
        return d_mean_dx[:, :, 0], d_variance_dx