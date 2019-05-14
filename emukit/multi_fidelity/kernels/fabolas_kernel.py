import GPy
import numpy as np


class FabolasKernel(GPy.kern.Kern):

    def __init__(self, input_dim, basis_func, a=1., b=1., active_dims=None):

        super(FabolasKernel, self).__init__(input_dim, active_dims, "fabolas_kernel")

        assert input_dim == 1

        self.basis_func = basis_func

        self.a = GPy.core.parameterization.Param("a", a)
        self.b = GPy.core.parameterization.Param("b", b)

        self.link_parameters(self.a, self.b)

    def K(self, X, X2):
        if X2 is None: X2 = X

        X_ = self.basis_func(X)
        X2_ = self.basis_func(X2)
        k = np.dot(X_ * self.b, X2_.T) + self.a

        return k

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X
        X_ = self.basis_func(X)
        X2_ = self.basis_func(X2)
        self.a.gradient = np.sum(dL_dK)
        self.b.gradient = np.sum(np.dot(np.dot(X_, X2_.T), dL_dK))

    def Kdiag(self, X):
        return np.diag(self.K(X, X))