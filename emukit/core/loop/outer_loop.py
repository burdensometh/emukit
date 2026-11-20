# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from ..event_handler import EventHandler
from .candidate_point_calculators import CandidatePointCalculator
from .external_evaluation import (
    PendingAwarePointCalculator,
    PendingObservationTracker,
    PendingStatus,
)
from .loop_state import LoopState
from .model_updaters import ModelUpdater
from .stopping_conditions import ConvergenceStoppingCondition, FixedIterationsStoppingCondition, StoppingCondition
from .user_function import UserFunction, UserFunctionWrapper
from .user_function_result import UserFunctionResult

_log = logging.getLogger(__name__)


class OuterLoop(object):
    """
    Generic outer loop that provides the framework for decision making parts of Emukit.

    The loop can be used in three modes:

    1. Emukit calculates the next point(s) to try and evaluates your function at these points until some stopping
       criterion is met.
    2. Emukit only calculates the next points(s) to try and you evaluate your function or perform the experiment.
    3. Asynchronous external evaluation: :meth:`ask` returns candidate points together with traceable
       observation ids and registers them as pending experiments. You run the experiments (which may finish
       at different times) and report each outcome out of order through :meth:`tell`, :meth:`tell_failure`
       or :meth:`tell_cancelled`. Only the first confirmed, successful observation for a claimed point is
       treated as a real observation and updates the model; repeated or conflicting reports are rejected.

    This object exposes the following events. See :py:mod:`emukit.core.event_handler` for details of how to subscribe:

    - ``loop_start_event`` called at the start of the run_loop method
    - ``iteration_end_event`` called at the end of each iteration
    """

    def __init__(
        self,
        candidate_point_calculator: CandidatePointCalculator,
        model_updaters: Union[ModelUpdater, List[ModelUpdater]],
        loop_state: LoopState = None,
    ) -> None:
        """
        :param candidate_point_calculator: Finds next points to evaluate by optimizing the acquisition function
        :param model_updaters: Updates the data in the model(s) and the model hyper-parameters when we observe new data
        :param loop_state: Object that keeps track of the history of the loop.
                           Default: None, resulting in empty initial state
        """
        self.candidate_point_calculator = candidate_point_calculator

        if isinstance(model_updaters, list):
            self.model_updaters = model_updaters
        else:
            self.model_updaters = [model_updaters]
        self.loop_state = loop_state
        if self.loop_state is None:
            self.loop_state = LoopState([])

        self.loop_start_event = EventHandler()
        self.iteration_end_event = EventHandler()

        # Registry backing the asynchronous external evaluation mode (ask/tell)
        self.pending_tracker = PendingObservationTracker()

        # A PendingAwarePointCalculator may be supplied directly; make sure it
        # tracks the same registry as this loop rather than a foreign one.
        if isinstance(self.candidate_point_calculator, PendingAwarePointCalculator):
            self.candidate_point_calculator.tracker = self.pending_tracker

    def run_loop(
        self,
        user_function: Union[UserFunction, Callable],
        stopping_condition: Union[StoppingCondition, int],
        context: dict = None,
    ) -> None:
        """
        :param user_function: The function that we are emulating
        :param stopping_condition: If integer - a number of iterations to run, or an object - a stopping
                        condition object that decides whether we should stop collecting more points.
                        Note that stopping conditions can be logically combined (&, |)
                        to represent complex stopping criteria.
        :param context: The context is used to force certain parameters of the inputs to the function of interest to
                        have a given value. It is a dictionary whose keys are the parameter names to fix and the values
                        are the values to fix the parameters to.
        """

        is_int = isinstance(stopping_condition, int)
        is_single_condition = isinstance(stopping_condition, StoppingCondition)

        if not (is_int or is_single_condition):
            raise ValueError(
                "Expected stopping_condition to be an int or a StoppingCondition instance,"
                "but received {}".format(type(stopping_condition))
            )

        if not isinstance(user_function, UserFunction):
            user_function = UserFunctionWrapper(user_function)

        if isinstance(stopping_condition, int):
            stopping_condition = FixedIterationsStoppingCondition(stopping_condition + self.loop_state.iteration)

        _log.info("Starting outer loop")

        self.loop_start_event(self, self.loop_state)

        while not stopping_condition.should_stop(self.loop_state):
            _log.info("Iteration {}".format(self.loop_state.iteration))

            self._update_models()
            new_x = self.candidate_point_calculator.compute_next_points(self.loop_state, context)
            _log.debug("Next suggested point(s): {}".format(new_x))
            results = user_function.evaluate(new_x)
            _log.debug("User function returned: {}".format(results))
            self.loop_state.update(results)
            self.iteration_end_event(self, self.loop_state)

        self._update_models()
        _log.info("Finished outer loop")

    def _update_models(self):
        for model_updater in self.model_updaters:
            model_updater.update(self.loop_state)

    def get_next_points(self, results: List[UserFunctionResult] = None, context: dict = {}) -> np.ndarray:
        """
        This method is used when the user doesn't want Emukit to evaluate the function of interest but rather just wants
        the input locations to evaluate the function at. This method calculates the new input locations.

        :param results: Function results since last loop step
        :param context: A dictionary of fixed parameters, identical to the context used in
                        self.run_loop()
        :return: Next batch of points to run
        """
        if results:
            self.loop_state.update(results)
            self._update_models()
        new_x = self.candidate_point_calculator.compute_next_points(self.loop_state, context)
        return new_x

    # --------------------------------------------------------------------------------------------------
    # Asynchronous external evaluation mode
    #
    # Experiments are claimed via ask() and confirmed later via tell()/tell_failure()/tell_cancelled().
    # A claim only creates a pending placeholder: it never enters loop_state and therefore never looks
    # like a real observation. The model is updated once per confirmed successful observation, never
    # when a point is merely asked for, so the number of model updates equals the number of confirmed
    # observations rather than the number of ask() calls.
    # --------------------------------------------------------------------------------------------------

    def enable_pending_point_handling(
        self,
        strategy: str = "exclude",
        max_attempts: int = 100,
        penalty_sigma: float = 1e-2,
        penalty_strength: float = 5.0,
    ) -> None:
        """
        Wraps the candidate point calculator so that points which are currently pending are excluded or
        penalized when new candidate points are requested through :meth:`ask`.

        This is called automatically on the first :meth:`ask` call but can also be called explicitly to
        choose the ``"penalize"`` strategy. Calling it more than once has no additional effect.

        :param strategy: ``"exclude"`` rejects candidate points coinciding with a pending point,
                         ``"penalize"`` multiplies the acquisition by a localized penalty around pending
                         points
        :param max_attempts: Maximum re-optimization attempts in exclude mode
        :param penalty_sigma: Length scale of the penalty when using the ``"penalize"`` strategy
        :param penalty_strength: Barrier strength when using the ``"penalize"`` strategy; the
                                 acquisition value directly on a pending point is reduced by a factor of
                                 :math:`e^{-\\text{penalty_strength}}`
        """
        if not isinstance(self.candidate_point_calculator, PendingAwarePointCalculator):
            self.candidate_point_calculator = PendingAwarePointCalculator(
                self.candidate_point_calculator,
                self.pending_tracker,
                strategy=strategy,
                max_attempts=max_attempts,
                penalty_sigma=penalty_sigma,
                penalty_strength=penalty_strength,
            )

    def ask(self, context: dict = None) -> Dict[str, np.ndarray]:
        """
        Suggests the next point to evaluate and claims it as a pending experiment.

        The point is registered in the pending observation tracker with a unique, traceable observation
        id but is **not** added to the loop state and does not trigger a model update. Any previously
        claimed, still pending points are excluded or penalized.

        :param context: A dictionary of fixed parameters, identical to the context used in
                        self.run_loop()
        :return: Dictionary with keys ``"observation_id"`` (the traceable id to use when reporting the
                 result) and ``"X"`` (the (1 x n_dims) array containing the point to evaluate)
        """
        self.enable_pending_point_handling()
        new_x = self.candidate_point_calculator.compute_next_points(self.loop_state, context)
        if new_x.shape[0] != 1:
            raise ValueError(
                "ask() expects the candidate point calculator to return exactly one point, "
                "got {} points".format(new_x.shape[0])
            )
        observation_id = self.pending_tracker.register(new_x[0])
        _log.debug("Claimed point {} with observation id {}".format(new_x[0], observation_id))
        return {"observation_id": observation_id, "X": new_x}

    def tell(
        self,
        results: Union[UserFunctionResult, List[UserFunctionResult]],
        observation_ids: Optional[Union[str, List[str]]] = None,
    ) -> List[str]:
        """
        Reports successful observations for previously claimed points.

        Reports may arrive out of order. Each result is paired with the traceable id of the point it was
        claimed with, either by passing the id alongside a single result or by passing equally long lists.
        Only the first confirmed success for a claimed point is accepted: once a point has reached a
        terminal status (succeeded, failed or cancelled) any further report for the same id is rejected
        with an error, and a result whose input does not match the claimed point is rejected as a
        conflict.

        Only confirmed successful observations are added to the loop state and advance the loop
        iteration / model updates. Failed or cancelled placeholders have to be released explicitly with
        :meth:`tell_failure` and :meth:`tell_cancelled`.

        :param results: A single :class:`UserFunctionResult` or a list of them
        :param observation_ids: The observation id returned by :meth:`ask` for a single result, or a list
                                of ids matching ``results``
        :return: The observation ids that were confirmed as successful, in report order
        """
        if isinstance(results, UserFunctionResult):
            results = [results]
        if not results:
            raise ValueError("Cannot tell an empty list of results")
        if isinstance(observation_ids, str):
            observation_ids = [observation_ids]
        if observation_ids is None or len(observation_ids) != len(results):
            raise ValueError(
                "Expected one observation id per result, got {} results and {} ids".format(
                    len(results), 0 if observation_ids is None else len(observation_ids)
                )
            )

        # Validate every report before mutating any state so a conflicting batch changes nothing.
        for result, observation_id in zip(results, observation_ids):
            observation = self.pending_tracker.get(observation_id)
            if observation.status != PendingStatus.PENDING:
                raise ValueError(
                    "Observation '{}' is already in terminal status '{}' and cannot be reported again".format(
                        observation_id, observation.status.value
                    )
                )
            if not np.allclose(observation.X, result.X):
                raise ValueError(
                    "Result X {} conflicts with the point X {} claimed for observation '{}'".format(
                        result.X, observation.X, observation_id
                    )
                )

        confirmed_ids = []
        for result, observation_id in zip(results, observation_ids):
            self.pending_tracker.mark_succeeded(observation_id, result)
            confirmed_ids.append(observation_id)

        self.loop_state.update(results)
        self._update_models()
        self.iteration_end_event(self, self.loop_state)
        _log.debug("Confirmed successful observations: {}".format(confirmed_ids))
        return confirmed_ids

    def tell_failure(self, observation_ids: Union[str, List[str]]) -> List[str]:
        """
        Reports that one or more claimed experiments failed.

        The pending placeholders are released so the points can be suggested again by future
        :meth:`ask` calls. No observation is added to the loop state and no model update is performed.

        :param observation_ids: A single observation id or a list of ids returned by :meth:`ask`
        :return: The observation ids marked as failed, in report order
        """
        return self._report_terminal(observation_ids, self.pending_tracker.mark_failed, PendingStatus.FAILED)

    def tell_cancelled(self, observation_ids: Union[str, List[str]]) -> List[str]:
        """
        Reports that one or more claimed experiments were cancelled before completion.

        The pending placeholders are released so the points can be suggested again by future
        :meth:`ask` calls. No observation is added to the loop state and no model update is performed.

        :param observation_ids: A single observation id or a list of ids returned by :meth:`ask`
        :return: The observation ids marked as cancelled, in report order
        """
        return self._report_terminal(observation_ids, self.pending_tracker.mark_cancelled, PendingStatus.CANCELLED)

    def _report_terminal(
        self, observation_ids: Union[str, List[str]], mark: Callable[[str], None], status: PendingStatus
    ) -> List[str]:
        if isinstance(observation_ids, str):
            observation_ids = [observation_ids]
        if not observation_ids:
            raise ValueError("Cannot report an empty list of observation ids")

        # Validate first so a batch containing an unknown or already terminal id changes nothing.
        for observation_id in observation_ids:
            if not self.pending_tracker.is_pending(observation_id):
                observation = self.pending_tracker.get(observation_id)
                raise ValueError(
                    "Observation '{}' is already in terminal status '{}' and cannot be reported again".format(
                        observation_id, observation.status.value
                    )
                )

        for observation_id in observation_ids:
            mark(observation_id)
        _log.debug("Marked observations as {}: {}".format(status.value, observation_ids))
        return list(observation_ids)
