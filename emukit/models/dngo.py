import numpy as np

from ..core.interfaces.models import IModel

try:
    from pybnn import dngo
except ImportError:
    raise ImportError("""
        This module is missing required dependencies. Try running

        pip install git+https://github.com/automl/pybnn.git
    """)


class DNGO(IModel):

    def __init__(self, X_init, Y_init, **kwargs):
        """
        """
        super().__init__()

        self.model = dngo.DNGO()

        self._X = X_init
        self._Y = Y_init

        self.model.train(X_init, Y_init[:, 0], **kwargs)

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for given points

        :param X: points to run prediction for
        """
        m, v = self.model.predict(X)

        return m[:, None], v[:, None]

    def update_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Updates model with new data points.

        :param X: new points
        :param Y: function values at new points X
        """
        self._X = np.append(self._X, X, axis=0)
        self._Y = np.append(self._Y, Y, axis=0)

        self.model.train(self._X, self._Y[:, 0])

    def optimize(self) -> None:
        pass

    def get_f_minimum(self):
        return np.min(self._Y)
