# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from emukit.core.interfaces import IModel, IModelWithNoise, IModelWithPerPointNoise
from emukit.model_wrappers import HeteroscedasticGaussianProcessModel


@pytest.fixture
def training_data():
    x = np.linspace(-3, 3, 10)[:, None]
    y = np.sin(x)
    # Distinct, strictly positive noise variances for every training point.
    noise_variance = np.linspace(0.01, 0.2, x.shape[0])[:, None]
    return x, y, noise_variance


@pytest.fixture
def hetero_gp(training_data):
    x, y, noise_variance = training_data
    return HeteroscedasticGaussianProcessModel(x, y, noise_variance)


def test_implements_per_point_noise_interface(hetero_gp):
    assert isinstance(hetero_gp, IModelWithPerPointNoise)
    assert isinstance(hetero_gp, IModelWithNoise)
    assert isinstance(hetero_gp, IModel)


def test_constructor_validates_noise(training_data):
    x, y, _ = training_data
    with pytest.raises(ValueError):
        HeteroscedasticGaussianProcessModel(x, y, np.zeros_like(y))
    with pytest.raises(ValueError):
        HeteroscedasticGaussianProcessModel(x, y, np.ones((x.shape[0] + 1, 1)))


def test_constructor_rejects_multiple_outputs(training_data):
    x, y, _ = training_data
    y_multi = np.repeat(y, 2, axis=1)
    with pytest.raises(ValueError):
        HeteroscedasticGaussianProcessModel(x, y_multi, np.ones_like(y_multi))


def test_set_data_updates_per_point_noise(hetero_gp, training_data):
    x, y, _ = training_data
    new_noise = np.full_like(y, 0.07)
    hetero_gp.set_data(x, 2 * y, new_noise)

    np.testing.assert_array_equal(hetero_gp.X, x)
    np.testing.assert_array_equal(hetero_gp.Y, 2 * y)
    np.testing.assert_array_equal(hetero_gp.noise_variance, new_noise)


def test_set_data_rejects_invalid_noise(hetero_gp, training_data):
    x, y, _ = training_data
    with pytest.raises(ValueError):
        hetero_gp.set_data(x, y, -np.ones_like(y))


def test_predict_shapes(hetero_gp, training_data):
    x, _, _ = training_data
    x_new = np.linspace(-3, 3, 25)[:, None]
    mean, variance = hetero_gp.predict(x_new)

    assert mean.shape == (25, 1)
    assert variance.shape == (25, 1)
    assert np.all(variance > 0)


def test_predict_includes_per_point_noise_at_training_locations(hetero_gp, training_data):
    x, _, noise_variance = training_data
    mean_noisy, variance_noisy = hetero_gp.predict(x)
    _, variance_latent = hetero_gp.predict_noiseless(x)

    # The only difference between noisy and noise-free predictions at training points is the
    # declared per-point observation noise.
    np.testing.assert_allclose(variance_noisy - variance_latent, noise_variance, atol=1e-10)
    np.testing.assert_allclose(mean_noisy, hetero_gp.predict_noiseless(x)[0], atol=1e-12)


def test_predict_noiseless_at_unknown_locations(hetero_gp):
    x_new = np.array([[10.0], [-10.0]])
    mean_noisy, variance_noisy = hetero_gp.predict(x_new)
    _, variance_latent = hetero_gp.predict_noiseless(x_new)

    # Locations far from any training point carry no observation noise.
    np.testing.assert_allclose(variance_noisy, variance_latent, atol=1e-12)


def test_per_point_noise_changes_posterior():
    # Two identical designs: a well-replicated point (small mean variance) must produce a tighter
    # posterior than a noisy one, when everything else is held fixed.
    x = np.array([[0.0], [1.0]])
    y = np.array([[0.0], [1.0]])

    small_noise = np.array([[1e-4], [1e-4]])
    large_noise = np.array([[1.0], [1.0]])

    tight_gp = HeteroscedasticGaussianProcessModel(x, y, small_noise, jitter=0.0)
    loose_gp = HeteroscedasticGaussianProcessModel(x, y, large_noise, jitter=0.0)

    _, tight_var = tight_gp.predict_noiseless(x)
    _, loose_var = loose_gp.predict_noiseless(x)

    assert np.all(tight_var < loose_var)


def test_optimize_keeps_per_point_noise_fixed(hetero_gp, training_data):
    _, _, noise_variance = training_data
    original_noise = noise_variance.copy()
    hetero_gp.optimize()

    np.testing.assert_array_equal(hetero_gp.noise_variance, original_noise)


def test_predictions_track_training_data_after_aggregation_style_update():
    # Simulates the data a RepeatPointUpdater would hand over: aggregated means and mean variances.
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([[1.0], [3.2], [5.0]])
    # A point with 4 repeats gets a much smaller noise than a singly measured point.
    noise_variance = np.array([[0.00666667], [0.04], [0.25]])

    gp = HeteroscedasticGaussianProcessModel(x, y, noise_variance)
    mean, variance = gp.predict(x)

    # Posterior means shrink toward the GP mean according to each point's noise but stay finite and
    # ordered with the aggregated targets.
    assert np.all(np.isfinite(mean))
    assert mean[0, 0] < mean[1, 0] < mean[2, 0]
    # Variances at training points are ordered with their observation noise.
    assert variance[0, 0] < variance[1, 0] < variance[2, 0]
