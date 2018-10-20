# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Union, Callable

import numpy as np

from .loop_state import LoopState
from .user_function_result import UserFunctionResult
from .candidate_point_calculators import CandidatePointCalculator
from .model_updaters import ModelUpdater
from .user_function import UserFunction, UserFunctionWrapper
from .stopping_conditions import StoppingCondition, FixedIterationsStoppingCondition


import logging
_log = logging.getLogger(__name__)

class OuterLoop(object):
    """
    Generic outer loop that provides the framework for decision making parts of Emukit.

    The loop can be used in two modes:

    1. Emukit calculates the next point(s) to try and evaluates your function at these points until some stopping
       criterion is met.
    2. Emukit only calculates the next points(s) to try and you evaluate your function or perform the experiment.
    """
    def __init__(self, candidate_point_calculator: CandidatePointCalculator, model_updaters: List[ModelUpdater],
                 loop_state: LoopState = None) -> None:
        """
        :param candidate_point_calculator: Finds next points to evaluate by optimizing the acquisition function
        :param model_updaters: Updates the data in the model and the model hyper-parameters when we observe new data
        :param loop_state: Object that keeps track of the history of the loop. Default: None, resulting in empty initial state
        """
        self.candidate_point_calculator = candidate_point_calculator
        self.model_updaters = model_updaters
        self.loop_state = loop_state
        if self.loop_state is None:
            self.loop_state = LoopState([])

    def run_loop(self, user_function: Union[UserFunction, Callable], stopping_condition: Union[StoppingCondition, int]) -> None:
        """
        :param user_function: The function that we are emulating
        :param stopping_condition: If integer - a number of iterations to run,
                                   if object - a stopping condition object that decides
                                   whether we should stop collecting more points
        """
        if not (isinstance(stopping_condition, int) or isinstance(stopping_condition, StoppingCondition)):
            raise ValueError("Expected stopping_condition to be an int or a StoppingCondition instance, but received {}".format(type(stopping_condition)))

        if not isinstance(user_function, UserFunction):
            user_function = UserFunctionWrapper(user_function)

        if isinstance(stopping_condition, int):
            stopping_condition = FixedIterationsStoppingCondition(stopping_condition)

        _log.info("Starting outer loop")

        while not stopping_condition.should_stop(self.loop_state):
            _log.info("Iteration {}".format(self.loop_state.iteration))
            [mu.update(self.loop_state) for mu in self.model_updaters]
            new_x = self.candidate_point_calculator.compute_next_points(self.loop_state)
            results = user_function.evaluate(new_x)
            self.loop_state.update(results)
            self.custom_step()

        _log.info("Finished outer loop")

    def get_next_points(self, results: List[UserFunctionResult]) -> np.ndarray:
        """
        This method is used when the user doesn't want Emukit to evaluate the function of interest but rather just wants
        the input locations to evaluate the function at. This method calculates the new input locations.

        :param results: Function results since last loop step
        :return: Next batch of points to run
        """
        if results:
            self.loop_state.update(results)
            [mu.update(self.loop_state) for mu in self.model_updater]
        return self.candidate_point_calculator.compute_next_points(self.loop_state)

    def custom_step(self) -> None:
        """
        The user can insert custom code into the loop by overloading this method
        """
        pass
