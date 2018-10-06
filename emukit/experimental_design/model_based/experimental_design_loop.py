from ...core.acquisition import Acquisition
from ...core.interfaces.models import IModel
from ...core.loop import OuterLoop, Sequential, FixedIntervalUpdater
from ...core.loop.loop_state import create_loop_state
from ...core.optimization import AcquisitionOptimizer
from ...core.parameter_space import ParameterSpace

from .acquisitions import ModelVariance


class ExperimentalDesignLoop(OuterLoop):
    def __init__(self, space: ParameterSpace, model: IModel, acquisition: Acquisition = None,
                 update_interval: int = 1):
        """
        An outer loop class for use with Experimental design

        :param space: Definition of domain bounds to collect points within
        :param model: The model that approximates the underlying function
        :param acquisition: experimental design acquisition function object. Default: ModelVariance acquisition
        :param update_interval: How many iterations pass before next model optimization
        """

        if acquisition is None:
            acquisition = ModelVariance(model)

        # This AcquisitionOptimizer object deals with optimizing the acquisition to find the next point to collect
        acquisition_optimizer = AcquisitionOptimizer(space)

        # Construct emukit classes
        candidate_point_calculator = Sequential(acquisition, acquisition_optimizer)
        model_updater = FixedIntervalUpdater(model, update_interval)
        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, [model_updater], loop_state)
