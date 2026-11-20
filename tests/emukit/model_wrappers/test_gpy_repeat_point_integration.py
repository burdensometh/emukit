# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

pytest.importorskip("GPy", reason="GPy not installed; install emukit[gpy]")
pytestmark = pytest.mark.gpy

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.loop import (
    ExactPointIdentifier,
    OuterLoop,
    RepeatPointUpdater,
    SequentialPointCalculator,
    aggregate_repeat_measurements,
    aggregated_arrays,
)
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization import RandomSearchAcquisitionOptimizer
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.model_wrappers import create_heteroscedastic_gpy_model

NOISE_PRIOR = 0.25


def test_gpy_loop_uses_aggregated_means_with_true_mean_uncertainty():
    rng = np.random.RandomState(7)
    space = ParameterSpace([ContinuousParameter("x", 0.0, 1.0)])

    # Three distinct points: one measured 4 times, one twice, one once.
    x_init = np.array([[0.1], [0.1], [0.1], [0.1], [0.5], [0.5], [0.9]])
    y_init = np.sin(3 * x_init) + rng.normal(scale=0.1, size=x_init.shape)

    aggregated = aggregate_repeat_measurements(x_init, y_init, NOISE_PRIOR, ExactPointIdentifier())
    x_agg, y_agg, noise_agg, n_agg = aggregated_arrays(aggregated)
    model = create_heteroscedastic_gpy_model(x_agg, y_agg, noise_agg)

    acquisition = ModelVariance(model)
    updater = RepeatPointUpdater(model, noise_prior_variance=NOISE_PRIOR)
    optimizer = RandomSearchAcquisitionOptimizer(space, num_eval_points=15)
    loop = OuterLoop(SequentialPointCalculator(acquisition, optimizer), updater)
    loop.loop_state = create_loop_state(x_init, y_init)

    n_iterations = 3
    loop.run_loop(lambda x: np.sin(3 * x) + rng.normal(scale=0.1, size=x.shape), n_iterations)

    # One aggregated training observation per distinct point, raw measurements retained in state.
    n_distinct = np.unique(loop.loop_state.X, axis=0).shape[0]
    assert model.X.shape[0] == n_distinct
    assert loop.loop_state.X.shape[0] == x_init.shape[0] + n_iterations

    # The GP genuinely holds the per-point mean variances from the aggregation.
    np.testing.assert_array_equal(model.noise_variance, updater.last_mean_variance)
    assert updater.last_n_measurements.sum() == loop.loop_state.X.shape[0]

    # The singly measured point keeps the declared prior; replicated points use s^2 / n.
    counts = updater.last_n_measurements
    variances = updater.last_mean_variance.ravel()
    single_variances = variances[counts == 1]
    np.testing.assert_allclose(single_variances, NOISE_PRIOR)
    assert np.all(variances[counts > 1] != NOISE_PRIOR)

    # Acquisition predictions come from the aggregated posterior.
    x_test = space.sample_uniform(10)
    mean, variance = model.predict(x_test)
    assert mean.shape == (10, 1)
    assert variance.shape == (10, 1)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(variance))


def test_gpy_replicated_point_is_treated_as_more_precise():
    rng = np.random.RandomState(11)

    x = np.concatenate([np.full((1, 1), 0.2), np.full((15, 1), 0.8)], axis=0)
    y = np.sin(3 * x) + rng.normal(scale=0.1, size=x.shape)

    aggregated = aggregate_repeat_measurements(x, y, NOISE_PRIOR, ExactPointIdentifier())
    x_agg, y_agg, noise_agg, n_agg = aggregated_arrays(aggregated)

    np.testing.assert_array_equal(n_agg, np.array([1, 15]))
    assert noise_agg[0, 0] == NOISE_PRIOR
    assert noise_agg[1, 0] < NOISE_PRIOR

    model = create_heteroscedastic_gpy_model(x_agg, y_agg, noise_agg)
    _, latent_variance = model.predict_noiseless(x_agg)
    _, predictive_variance = model.predict(x_agg)

    # The well-replicated point has both a tighter latent posterior and smaller observation noise.
    assert latent_variance[1, 0] < latent_variance[0, 0]
    np.testing.assert_allclose(predictive_variance - latent_variance, noise_agg, atol=1e-10)
