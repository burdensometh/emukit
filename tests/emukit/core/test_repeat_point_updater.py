# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from numpy.testing import assert_array_equal

from emukit.core.interfaces import IModel, IModelWithPerPointNoise
from emukit.core.loop import (
    AggregatedPointObservations,
    ExactPointIdentifier,
    FunctionPointIdentifier,
    LoopState,
    PointIdentifier,
    RepeatPointUpdater,
    aggregate_repeat_measurements,
    aggregated_arrays,
)
from emukit.core.loop.user_function_result import UserFunctionResult
from emukit.model_wrappers import HeteroscedasticGaussianProcessModel

NOISE_PRIOR = 0.25


def make_loop_state():
    # Point x=0 has 4 repeats, x=1 has 2 repeats, x=2 has a single measurement.
    X = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [2.0]])
    Y = np.array([[1.0], [1.2], [0.8], [1.0], [3.0], [3.4], [5.0]])
    results = [UserFunctionResult(X[i], Y[i]) for i in range(X.shape[0])]
    return LoopState(results), X, Y


def test_aggregate_repeat_measurements_statistics():
    _, X, Y = make_loop_state()
    aggregated = aggregate_repeat_measurements(X, Y, NOISE_PRIOR, ExactPointIdentifier())

    assert len(aggregated) == 3
    assert [point.n_measurements for point in aggregated] == [4, 2, 1]
    assert [tuple(point.X) for point in aggregated] == [(0.0,), (1.0,), (2.0,)]

    # Point with 4 repeats: mean 1.0, unbiased sample variance 0.08/3, mean variance s^2/n.
    point_0 = aggregated[0]
    assert point_0.Y_mean == pytest.approx(np.array([1.0]))
    assert point_0.sample_variance == pytest.approx(np.array([0.08 / 3]))
    assert point_0.mean_variance == pytest.approx(np.array([0.08 / 3 / 4]))

    # Point with 2 repeats.
    point_1 = aggregated[1]
    assert point_1.Y_mean == pytest.approx(np.array([3.2]))
    assert point_1.sample_variance == pytest.approx(np.array([0.08]))
    assert point_1.mean_variance == pytest.approx(np.array([0.04]))

    # Singly measured point uses the declared noise prior.
    point_2 = aggregated[2]
    assert point_2.Y_mean == pytest.approx(np.array([5.0]))
    assert point_2.sample_variance == pytest.approx(np.array([0.0]))
    assert point_2.mean_variance == pytest.approx(np.array([NOISE_PRIOR]))


def test_aggregation_preserves_first_appearance_order():
    X = np.array([[2.0], [1.0], [2.0], [1.0], [0.0]])
    Y = np.arange(5).reshape(5, 1).astype(float)
    aggregated = aggregate_repeat_measurements(X, Y, NOISE_PRIOR, ExactPointIdentifier())

    assert [tuple(point.X) for point in aggregated] == [(2.0,), (1.0,), (0.0,)]
    assert [point.n_measurements for point in aggregated] == [2, 2, 1]


def test_aggregated_arrays_shapes():
    _, X, Y = make_loop_state()
    X_agg, Y_mean, mean_variance, n_measurements = aggregated_arrays(
        aggregate_repeat_measurements(X, Y, NOISE_PRIOR, ExactPointIdentifier())
    )

    assert X_agg.shape == (3, 1)
    assert Y_mean.shape == (3, 1)
    assert mean_variance.shape == (3, 1)
    assert n_measurements.shape == (3,)
    assert_array_equal(n_measurements, np.array([4, 2, 1]))


def test_aggregated_arrays_rejects_empty_input():
    with pytest.raises(ValueError):
        aggregated_arrays([])


def test_aggregate_validates_inputs():
    with pytest.raises(ValueError):
        aggregate_repeat_measurements(np.zeros((3, 1)), np.zeros((2, 1)), NOISE_PRIOR, ExactPointIdentifier())
    with pytest.raises(ValueError):
        aggregate_repeat_measurements(np.zeros((3, 1)), np.zeros((3, 1)), 0.0, ExactPointIdentifier())


def test_exact_point_identifier_with_rounding():
    identifier = ExactPointIdentifier(decimals=1)
    X = np.array([[0.001], [0.004], [1.0], [0.996]])
    identities = identifier.identify(X)

    assert identities[0] == identities[1]
    assert identities[2] == identities[3]
    assert identities[0] != identities[2]


def test_function_point_identifier():
    identifier = FunctionPointIdentifier(lambda X: np.round(X[:, 0], 0))
    identities = identifier.identify(np.array([[0.1], [0.4], [1.2]]))

    assert identities[0] == identities[1]
    assert identities[0] != identities[2]


def test_function_point_identifier_validates_output_shape():
    identifier = FunctionPointIdentifier(lambda X: np.zeros((X.shape[0], 2)))
    with pytest.raises(ValueError):
        identifier.identify(np.zeros((3, 1)))


def test_point_identifier_is_abstract():
    with pytest.raises(TypeError):
        PointIdentifier()


def test_repeat_point_updater_passes_per_point_noise_to_model():
    loop_state, X, Y = make_loop_state()
    _, Y_mean, mean_variance, _ = aggregated_arrays(
        aggregate_repeat_measurements(X, Y, NOISE_PRIOR, ExactPointIdentifier())
    )

    model = HeteroscedasticGaussianProcessModel(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)))
    updater = RepeatPointUpdater(model, noise_prior_variance=NOISE_PRIOR, interval=100)
    updater.update(loop_state)

    # Model holds one mean observation per distinct point, not one per raw measurement.
    assert model.X.shape == (3, 1)
    assert_array_equal(model.X, np.array([[0.0], [1.0], [2.0]]))
    assert model.Y == pytest.approx(Y_mean)
    assert model.noise_variance == pytest.approx(mean_variance)
    # The aggregated mean uncertainty is exactly what reaches the model.
    assert model.noise_variance.ravel() == pytest.approx(np.array([0.08 / 3 / 4, 0.04, NOISE_PRIOR]))

    # Updater exposes counts and uncertainties for inspection.
    assert_array_equal(updater.last_n_measurements, np.array([4, 2, 1]))
    assert updater.last_mean_variance == pytest.approx(mean_variance)
    assert all(isinstance(point, AggregatedPointObservations) for point in updater.last_aggregated)


def test_repeat_point_updater_keeps_raw_measurements_in_loop_state():
    loop_state, _, _ = make_loop_state()
    model = HeteroscedasticGaussianProcessModel(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)))
    updater = RepeatPointUpdater(model, noise_prior_variance=NOISE_PRIOR, interval=100)
    updater.update(loop_state)

    # The loop state still stores every individual noisy measurement.
    assert len(loop_state.results) == 7
    assert loop_state.X.shape == (7, 1)
    assert loop_state.Y.shape == (7, 1)
    # While the model was trained on the three aggregated points.
    assert model.X.shape[0] == 3


def test_repeat_point_updater_falls_back_to_plain_set_data():
    class PlainNoiseModel(IModel):
        def __init__(self):
            self.optimized = False
            self._X = None
            self._Y = None

        def set_data(self, X, Y):
            self._X = X
            self._Y = Y

        def optimize(self):
            self.optimized = True

        @property
        def X(self):
            return self._X

        @property
        def Y(self):
            return self._Y

    loop_state, X, Y = make_loop_state()
    loop_state.iteration = 1
    model = PlainNoiseModel()
    assert not isinstance(model, IModelWithPerPointNoise)

    updater = RepeatPointUpdater(model, noise_prior_variance=NOISE_PRIOR, interval=100)
    updater.update(loop_state)

    # Aggregated means are still used; per-point noise is retained on the updater.
    assert model.X.shape == (3, 1)
    np.testing.assert_array_equal(model.Y, np.array([[1.0], [3.2], [5.0]]))
    assert updater.last_mean_variance.ravel() == pytest.approx(np.array([0.08 / 3 / 4, 0.04, NOISE_PRIOR]))
    assert not model.optimized


def test_repeat_point_updater_optimizes_at_interval():
    loop_state, _, _ = make_loop_state()
    model = HeteroscedasticGaussianProcessModel(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)))
    updater = RepeatPointUpdater(model, noise_prior_variance=NOISE_PRIOR, interval=2)

    optimize_calls = []
    original_optimize = model.optimize

    def counting_optimize():
        optimize_calls.append(loop_state.iteration)
        original_optimize()

    model.optimize = counting_optimize

    loop_state.iteration = 0
    updater.update(loop_state)
    loop_state.iteration = 1
    updater.update(loop_state)
    loop_state.iteration = 2
    updater.update(loop_state)

    assert optimize_calls == [0, 2]


def test_repeat_point_updater_ignores_empty_loop_state():
    # OuterLoop updates models before the first evaluation, so an empty state must be tolerated.
    model = HeteroscedasticGaussianProcessModel(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)))
    updater = RepeatPointUpdater(model, noise_prior_variance=NOISE_PRIOR)
    empty_state = LoopState([])

    updater.update(empty_state)

    # Model and recorded aggregation outputs are left untouched.
    assert model.X.shape == (1, 1)
    assert updater.last_aggregated == []
    assert updater.last_mean_variance is None
    assert updater.last_n_measurements is None


def test_repeat_point_updater_uses_custom_identifier_and_targets():
    # Two numerically distinct rows that denote the same experimental point.
    X = np.array([[0.001], [0.004], [1.0]])
    Y = np.array([[2.0], [4.0], [10.0]])
    loop_state = LoopState([UserFunctionResult(X[i], Y[i]) for i in range(3)])

    model = HeteroscedasticGaussianProcessModel(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)))
    updater = RepeatPointUpdater(
        model,
        noise_prior_variance=NOISE_PRIOR,
        point_identifier=ExactPointIdentifier(decimals=1),
        interval=100,
    )
    updater.update(loop_state)

    assert model.X.shape == (2, 1)
    assert model.Y.ravel() == pytest.approx(np.array([3.0, 10.0]))
    assert model.noise_variance.ravel()[0] == pytest.approx(1.0)  # s^2/n = 2/2
