import numpy as np
import pytest

from GPy.models import GPRegression
from GPy.kern import RBF

from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import LogExpectedImprovement


@pytest.fixture
def acquisition():
    rng = np.random.RandomState(42)
    x_init = rng.rand(5, 2)
    y_init = rng.rand(5, 1)
    model = GPRegression(x_init, y_init, RBF(2))
    return LogExpectedImprovement(GPyModelWrapper(model))


def test_log_expected_improvement_shape(acquisition):

    rng = np.random.RandomState(43)

    x_test = rng.rand(10, 2)
    result = acquisition.evaluate(x_test)
    print(result)
    assert(result.shape == (10, 1))
