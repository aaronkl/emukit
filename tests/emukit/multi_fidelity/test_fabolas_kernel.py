import numpy as np

from GPy.testing.kernel_tests import check_kernel_gradient_functions

from emukit.multi_fidelity.kernels import FabolasKernel
from emukit.multi_fidelity.models.fabolas_model import quad


def test_gradients():

    k = FabolasKernel(input_dim=1, basis_func=quad)
    k.randomize()

    X = np.array([[1]])
    X2 = np.array([[2]])
    assert check_kernel_gradient_functions(k, X=X, X2=X2, verbose=True)
