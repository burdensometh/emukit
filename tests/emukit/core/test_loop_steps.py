# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import mock
import numpy as np
import pytest

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel
from emukit.core.loop import (
    FixedIntervalUpdater,
    FixedIterationsStoppingCondition,
    LoopState,
    RandomSampling,
    SequentialPointCalculator,
    UserFunctionResult,
    UserFunctionWrapper,
)
from emukit.core.optimization import GradientAcquisitionOptimizer


def test_every_iteration_model_updater():
    mock_model = mock.create_autospec(IModel)
    mock_model.optimize.return_value(None)
    updater = FixedIntervalUpdater(mock_model, 1)

    loop_state_mock = mock.create_autospec(LoopState)
    loop_state_mock.iteration = 1
    loop_state_mock.X.return_value(np.random.rand(5, 1))
    loop_state_mock.Y.return_value(np.random.rand(5, 1))
    updater.update(loop_state_mock)
    mock_model.optimize.assert_called_once()


def test_every_iteration_model_updater_with_cost():
    """
    Tests that the model updater can use a different attribute
    from loop_state as the training targets
    """

    class MockModel(IModel):
        def optimize(self):
            pass

        def set_data(self, X: np.ndarray, Y: np.ndarray):
            self._X = X
            self._Y = Y

        @property
        def X(self):
            return self._X

        @property
        def Y(self):
            return self._Y

    mock_model = MockModel()
    updater = FixedIntervalUpdater(mock_model, 1, lambda loop_state: loop_state.cost)

    loop_state_mock = mock.create_autospec(LoopState)
    loop_state_mock.iteration = 1
    loop_state_mock.X.return_value(np.random.rand(5, 1))

    loop_state_mock.cost = np.random.rand(5, 1)

    cost = np.random.rand(5, 1)
    loop_state_mock.cost = cost
    updater.update(loop_state_mock)
    assert np.array_equiv(mock_model.X, cost)


def test_sequential_evaluator():
    # SequentialPointCalculator should just return result of the acquisition optimizer
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition_optimizer = mock.create_autospec(GradientAcquisitionOptimizer)
    mock_acquisition_optimizer.optimize.return_value = (np.array([[0.0]]), None)
    loop_state_mock = mock.create_autospec(LoopState)
    seq = SequentialPointCalculator(mock_acquisition, mock_acquisition_optimizer)
    next_points = seq.compute_next_points(loop_state_mock)

    # "SequentialPointCalculator" should only ever return 1 value
    assert len(next_points) == 1
    # Value should be result of acquisition optimization
    assert np.equal(np.array([[0.0]]), next_points[0])


def test_sequential_with_context():
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition.has_gradients = False
    mock_acquisition.evaluate = lambda x: np.sum(x**2, axis=1)[:, None]
    space = ParameterSpace([ContinuousParameter("x", 0, 1), ContinuousParameter("y", 0, 1)])
    acquisition_optimizer = GradientAcquisitionOptimizer(space)

    loop_state_mock = mock.create_autospec(LoopState)
    seq = SequentialPointCalculator(mock_acquisition, acquisition_optimizer)
    next_points = seq.compute_next_points(loop_state_mock, context={"x": 0.25})

    # "SequentialPointCalculator" should only ever return 1 value
    assert len(next_points) == 1
    # Context value should be what we set
    assert np.isclose(next_points[0, 0], 0.25)


def test_sequential_with_all_parameters_fixed():
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition.has_gradients = False
    mock_acquisition.evaluate = lambda x: np.sum(x**2, axis=1)[:, None]
    space = ParameterSpace([ContinuousParameter("x", 0, 1), ContinuousParameter("y", 0, 1)])
    acquisition_optimizer = GradientAcquisitionOptimizer(space)

    loop_state_mock = mock.create_autospec(LoopState)
    seq = SequentialPointCalculator(mock_acquisition, acquisition_optimizer)
    next_points = seq.compute_next_points(loop_state_mock, context={"x": 0.25, "y": 0.25})
    assert np.array_equiv(next_points, np.array([0.25, 0.25]))


def test_random_sampling_without_context():
    space = ParameterSpace([ContinuousParameter("x", 0, 1), ContinuousParameter("y", 0, 1)])
    rs = RandomSampling(space)
    loop_state_mock = mock.create_autospec(LoopState)
    next_points = rs.compute_next_points(loop_state_mock)
    assert len(next_points) == 1


def test_random_sampling_with_context():
    space = ParameterSpace([ContinuousParameter("x", 0, 1), ContinuousParameter("y", 0, 1)])
    rs = RandomSampling(space)
    loop_state_mock = mock.create_autospec(LoopState)
    next_points = rs.compute_next_points(loop_state_mock, context={"x": 0.25})
    assert len(next_points) == 1
    # Context value should be what we set
    assert np.isclose(next_points[0, 0], 0.25)


def test_user_function_wrapper():
    def function_test(x):
        return x[:, 0:1] ** 2 + x[:, 1:2] ** 2

    user_function = UserFunctionWrapper(function_test)
    results = user_function.evaluate(np.random.rand(10, 2))

    assert len(results) == 10, "A different number of results were expected"
    for res in results:
        assert res.X.ndim == 1, "X results are expected to be 1 dimensional"
        assert res.Y.ndim == 1, "Y results are expected to be 1 dimensional"


class _RecordingModel(IModel):
    """Model that records the data it is updated with"""

    def __init__(self):
        self._X = None
        self._Y = None
        self.optimize_calls = 0

    def predict(self, x):
        pass

    def set_data(self, X: np.ndarray, Y: np.ndarray):
        self._X = X
        self._Y = Y

    def optimize(self):
        self.optimize_calls += 1

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y


def test_updater_trains_only_on_observations_of_its_output():
    results = [
        UserFunctionResult(np.array([1.0]), np.array([10.0]), cost=np.array([1.0])),
        UserFunctionResult(np.array([2.0]), np.array([20.0]), cost=None),
        UserFunctionResult(np.array([3.0]), None, cost=np.array([3.0])),
    ]
    loop_state = LoopState(results)

    model_objective = _RecordingModel()
    model_cost = _RecordingModel()

    FixedIntervalUpdater(model_objective, 1, output_name="Y").update(loop_state)
    FixedIntervalUpdater(model_cost, 1, output_name="cost").update(loop_state)

    # Both X and Y are filtered to the same observations; Y is not filtered against the full X
    assert np.array_equal(model_objective.X, np.array([[1.0], [2.0]]))
    assert np.array_equal(model_objective.Y, np.array([[10.0], [20.0]]))
    assert np.array_equal(model_cost.X, np.array([[1.0], [3.0]]))
    assert np.array_equal(model_cost.Y, np.array([[1.0], [3.0]]))


def test_updater_skips_when_output_not_observed_yet():
    loop_state = LoopState([UserFunctionResult(np.array([1.0]), np.array([10.0]))])

    model_constraint = _RecordingModel()
    FixedIntervalUpdater(model_constraint, 1, output_name="Y_constraint").update(loop_state)

    assert model_constraint.X is None
    assert model_constraint.optimize_calls == 0


def test_updater_does_not_train_on_nan_targets():
    results = [
        UserFunctionResult(np.array([1.0]), np.array([10.0]), cost=np.array([np.nan])),
        UserFunctionResult(np.array([2.0]), np.array([20.0]), cost=np.array([2.0])),
    ]
    loop_state = LoopState(results)

    model_cost = _RecordingModel()
    FixedIntervalUpdater(model_cost, 1, output_name="cost").update(loop_state)

    assert np.array_equal(model_cost.X, np.array([[2.0]]))
    assert np.array_equal(model_cost.Y, np.array([[2.0]]))


def test_updater_accepts_extractor_returning_matched_X_and_Y():
    results = [
        UserFunctionResult(np.array([1.0]), np.array([10.0]), cost=np.array([1.0])),
        UserFunctionResult(np.array([2.0]), None, cost=np.array([2.0])),
    ]
    loop_state = LoopState(results)

    extracted = loop_state.output_view("cost")
    model_cost = _RecordingModel()
    FixedIntervalUpdater(model_cost, 1, lambda state: (extracted.X, extracted.Y)).update(loop_state)

    assert np.array_equal(model_cost.X, np.array([[1.0], [2.0]]))
    assert np.array_equal(model_cost.Y, np.array([[1.0], [2.0]]))


def test_updater_rejects_targets_that_are_not_aligned_with_inputs():
    results = [
        UserFunctionResult(np.array([1.0]), np.array([10.0]), cost=np.array([1.0])),
        UserFunctionResult(np.array([2.0]), np.array([20.0]), cost=np.array([2.0])),
    ]
    loop_state = LoopState(results)

    model_cost = _RecordingModel()
    # Extracting the cost and silently pairing it with all inputs is exactly the failure mode that must be prevented
    updater = FixedIntervalUpdater(model_cost, 1, lambda state: state.cost[:1])

    with pytest.raises(ValueError):
        updater.update(loop_state)


def test_updater_rejects_nan_targets_from_custom_extractor():
    results = [
        UserFunctionResult(np.array([1.0]), np.array([10.0]), cost=np.array([1.0])),
        UserFunctionResult(np.array([2.0]), np.array([20.0]), cost=np.array([np.nan])),
    ]
    loop_state = LoopState(results)

    model_cost = _RecordingModel()
    updater = FixedIntervalUpdater(model_cost, 1, lambda state: state.cost)

    with pytest.raises(ValueError):
        updater.update(loop_state)
    assert model_cost.X is None


def test_updater_only_updates_the_model_whose_output_is_observed():
    loop_state = LoopState(
        [
            UserFunctionResult(np.array([1.0]), np.array([10.0]), Y_constraint=np.array([-1.0])),
            UserFunctionResult(np.array([2.0]), np.array([20.0]), Y_constraint=np.array([1.0])),
        ]
    )
    model_objective = _RecordingModel()
    model_constraint = _RecordingModel()

    # A late result supplies only the objective
    loop_state.update([UserFunctionResult(np.array([3.0]), np.array([30.0]))])

    FixedIntervalUpdater(model_objective, 1, output_name="Y").update(loop_state)
    FixedIntervalUpdater(model_constraint, 1, output_name="Y_constraint").update(loop_state)

    assert np.array_equal(model_objective.Y, np.array([[10.0], [20.0], [30.0]]))
    assert np.array_equal(model_constraint.Y, np.array([[-1.0], [1.0]]))
    assert model_constraint.optimize_calls == 1
