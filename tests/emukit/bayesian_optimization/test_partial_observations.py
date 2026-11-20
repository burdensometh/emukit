# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.bayesian_optimization.loops import UnknownConstraintBayesianOptimizationLoop
from emukit.bayesian_optimization.loops.cost_sensitive_bayesian_optimization_loop import (
    CostSensitiveBayesianOptimizationLoop,
)
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.interfaces import IModel
from emukit.core.loop import UserFunctionResult


class RecordingModel(IModel):
    def __init__(self, X, Y):
        self._X = np.asarray(X, dtype=float)
        self._Y = np.asarray(Y, dtype=float)
        self._updates = []
        self._optimize_calls = 0

    def predict(self, X):
        return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))

    def set_data(self, X, Y):
        self._updates.append((np.array(X), np.array(Y)))
        self._X = np.array(X)
        self._Y = np.array(Y)

    def optimize(self):
        self._optimize_calls += 1

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y


def test_constraint_loop_trains_each_model_on_its_own_observations():
    space = ParameterSpace([ContinuousParameter("x", 0, 1)])
    model_objective = RecordingModel([[0.1], [0.2], [0.3]], [[1.0], [2.0], [3.0]])
    model_constraint = RecordingModel([[0.2], [0.3], [0.4]], [[-1.0], [1.0], [-1.0]])

    loop = UnknownConstraintBayesianOptimizationLoop(space, model_objective, model_constraint)

    # The shared initial point must appear only once in the loop state
    assert loop.loop_state.X.shape[0] == 4
    assert np.allclose(model_objective.X, [[0.1], [0.2], [0.3]])
    assert np.allclose(model_constraint.X, [[0.2], [0.3], [0.4]])

    # A new evaluation observes the objective but not the constraint
    loop.loop_state.update([UserFunctionResult(np.array([0.5]), np.array([5.0]), Y_constraint=None)])
    for updater in loop.model_updaters:
        updater.update(loop.loop_state)

    assert np.allclose(model_objective.X, [[0.1], [0.2], [0.3], [0.5]])
    assert np.allclose(model_objective.Y, [[1.0], [2.0], [3.0], [5.0]])
    # The constraint model did not receive the point whose constraint was not observed
    assert np.allclose(model_constraint.X, [[0.2], [0.3], [0.4]])
    assert np.allclose(model_constraint.Y, [[-1.0], [1.0], [-1.0]])


def test_cost_sensitive_loop_supports_models_initialised_on_different_points():

    space = ParameterSpace([ContinuousParameter("x", 0, 1)])
    model_objective = RecordingModel([[0.1], [0.2]], [[1.0], [2.0]])
    model_cost = RecordingModel([[0.2], [0.9]], [[10.0], [90.0]])

    loop = CostSensitiveBayesianOptimizationLoop(space, model_objective, model_cost)

    assert np.allclose(loop.loop_state.X, [[0.1], [0.2], [0.9]])
    assert loop.loop_state.output_view("Y").n_observations == 2
    assert loop.loop_state.output_view("cost").n_observations == 2
    assert np.allclose(loop.loop_state.X_for("cost"), [[0.2], [0.9]])

    # An evaluation that reports a cost but no objective updates only the cost model
    loop.loop_state.update([UserFunctionResult(np.array([0.6]), None, cost=np.array([60.0]))])
    for updater in loop.model_updaters:
        updater.update(loop.loop_state)

    assert np.allclose(model_objective.Y, [[1.0], [2.0]])
    assert np.allclose(model_cost.X, [[0.2], [0.9], [0.6]])
    assert np.allclose(model_cost.Y, [[10.0], [90.0], [60.0]])
