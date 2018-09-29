from typing import Tuple, Union
from scipy.stats import norm
from GPyOpt.util.general import get_quantiles
import numpy as np

from ...core.interfaces import IModel, IDifferentiable
from ...core.acquisition import Acquisition


class LogExpectedImprovement(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], jitter: np.float64 = np.float64(0))-> None:
        """

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """

        self.model = model
        self.jitter = jitter

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """

        m, v = self.model.predict(x)

        eta = np.min(self.model.Y, axis=0)

        f_min = eta - self.jitter

        s = np.sqrt(v)

        z = (f_min - m) / s

        log_ei = np.zeros([m.size, 1])
        for i in range(0, m.size):
            mu, sigma = m[i], s[i]

        #    par_s = self.par * sigma

            # Degenerate case 1: first term vanishes
            if np.any(abs(f_min - mu) == 0):
                if sigma > 0:
                    log_ei[i] = np.log(sigma) + norm.logpdf(z[i])
                else:
                    log_ei[i] = -np.Infinity
            # Degenerate case 2: second term vanishes and first term
            # has a special form.
            elif sigma == 0:
                if np.any(mu < f_min):
                    log_ei[i] = np.log(f_min - mu)
                else:
                    log_ei[i] = -np.Infinity
            # Normal case
            else:
                b = np.log(sigma) + norm.logpdf(z[i])
                # log(y+z) is tricky, we distinguish two cases:
                if np.any(f_min > mu):
                    # When y>0, z>0, we define a=ln(y), b=ln(z).
                    # Then y+z = exp[ max(a,b) + ln(1 + exp(-|b-a|)) ],
                    # and thus log(y+z) = max(a,b) + ln(1 + exp(-|b-a|))
                    a = np.log(f_min - mu) + norm.logcdf(z[i])

                    log_ei[i] = max(a, b) + np.log(1 + np.exp(-abs(b - a)))
                else:
                    # When y<0, z>0, we define a=ln(-y), b=ln(z),
                    # and it has to be true that b >= a in
                    # order to satisfy y+z>=0.
                    # Then y+z = exp[ b + ln(exp(b-a) -1) ],
                    # and thus log(y+z) = a + ln(exp(b-a) -1)
                    a = np.log(mu - f_min) + norm.logcdf(z[i])
                    if a >= b:
                        # a>b can only happen due to numerical inaccuracies
                        # or approximation errors
                        log_ei[i] = -np.Infinity
                    else:
                        log_ei[i] = b + np.log(1 - np.exp(a - b))

        return log_ei

    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False
