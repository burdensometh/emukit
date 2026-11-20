# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.loop import OuterLoop, RepeatPointUpdater, SequentialPointCalculator
from emukit.core.optimization import RandomSearchAcquisitionOptimizer
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.model_wrappers import HeteroscedasticGaussianProcessModel

NOISE_PRIOR = 0.25


def _noisy_user_function(x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    # Underlying function plus heteroscedastic measurement noise.
    return np.sin(3 * x) + rng.normal(scale=0.1, size=x.shape)


def test_loop_runs_with_aggregated_repeat_measurements():
    rng = np.random.RandomState(42)
    space = ParameterSpace([ContinuousParameter("x", 0.0, 1.0)])

    # Initial design contains explicit repeats at the same experimental points.
    x_init = np.array([[0.1], [0.1], [0.1], [0.5], [0.5], [0.9]])
    y_init = _noisy_user_function(x_init, rng)

    # Seed the model with one observation per distinct point and prior noise.
    from emukit.core.loop import ExactPointIdentifier, aggregate_repeat_measurements, aggregated_arrays

    aggregated = aggregate_repeat_measurements(x_init, y_init, NOISE_PRIOR, ExactPointIdentifier())
    x_agg, y_agg, noise_agg, n_agg = aggregated_arrays(aggregated)
    model = HeteroscedasticGaussianProcessModel(x_agg, y_agg, noise_agg)

    acquisition = ModelVariance(model)
    updater = RepeatPointUpdater(model, noise_prior_variance=NOISE_PRIOR)
    optimizer = RandomSearchAcquisitionOptimizer(space, num_eval_points=20)
    loop = OuterLoop(
        SequentialPointCalculator(acquisition, optimizer),
        updater,
    )

    # Seed the loop state with the raw, individual measurements.
    from emukit.core.loop.loop_state import create_loop_state

    loop.loop_state = create_loop_state(x_init, y_init)

    n_iterations = 4
    raw_count_before = loop.loop_state.X.shape[0]
    loop.run_loop(lambda x: _noisy_user_function(x, rng), n_iterations)

    # Raw measurements are all retained, one per evaluation.
    assert loop.loop_state.X.shape[0] == raw_count_before + n_iterations

    # The model is trained on the distinct aggregated points, never on raw repeats.
    n_distinct = np.unique(loop.loop_state.X, axis=0).shape[0]
    assert model.X.shape[0] == n_distinct
    assert model.X.shape[0] < loop.loop_state.X.shape[0]

    # Counts and mean uncertainties are consistent with the raw measurement history.
    assert updater.last_n_measurements.sum() == loop.loop_state.X.shape[0]
    assert np.all(updater.last_mean_variance > 0)
    np.testing.assert_array_equal(model.noise_variance, updater.last_mean_variance)

    # Acquisition predictions come from the aggregated posterior and remain finite.
    x_test = space.sample_uniform(10)
    mean, variance = model.predict(x_test)
    assert mean.shape == (10, 1)
    assert variance.shape == (10, 1)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(variance))
    assert np.all(variance > 0)


def test_repeated_points_become_more_precise_in_posterior():
    # A point measured many times with small noise must end up with a smaller posterior variance
    # than an equally spaced point measured only once.
    rng = np.random.RandomState(0)

    x_single = np.array([[0.0]])
    y_single = np.array([[0.0]])

    x_repeated = np.repeat([[1.0]], 20, axis=0)
    y_repeated = rng.normal(loc=1.0, scale=0.1, size=(20, 1))

    x_all = np.concatenate([x_single, x_repeated], axis=0)
    y_all = np.concatenate([y_single, y_repeated], axis=0)

    from emukit.core.loop import ExactPointIdentifier, aggregate_repeat_measurements, aggregated_arrays

    aggregated = aggregate_repeat_measurements(x_all, y_all, NOISE_PRIOR, ExactPointIdentifier())
    x_agg, y_agg, noise_agg, n_agg = aggregated_arrays(aggregated)

    np.testing.assert_array_equal(n_agg, np.array([1, 20]))
    # The replicated point's mean variance is estimated from the data and far below the prior.
    assert noise_agg[1, 0] < NOISE_PRIOR
    # The singly measured point falls back to the declared prior.
    assert noise_agg[0, 0] == NOISE_PRIOR

    model = HeteroscedasticGaussianProcessModel(x_agg, y_agg, noise_agg)
    _, variance = model.predict_noiseless(x_agg)

    assert variance[1, 0] < variance[0, 0]
