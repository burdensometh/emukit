# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

pytest.importorskip("GPy", reason="GPy not installed; install emukit[gpy]")
pytestmark = pytest.mark.gpy

from emukit.core.interfaces import IModel, IModelWithNoise, IModelWithPerPointNoise
from emukit.model_wrappers import GPyHeteroscedasticModelWrapper, create_heteroscedastic_gpy_model


@pytest.fixture
def data():
    np.random.seed(0)
    x = np.linspace(0.0, 1.0, 8)[:, None]
    y = np.sin(3 * x) + 0.05 * np.random.randn(8, 1)
    noise_variance = np.linspace(0.01, 0.2, 8)[:, None]
    return x, y, noise_variance


def test_implements_per_point_noise_interface(data):
    x, y, noise_variance = data
    model = create_heteroscedastic_gpy_model(x, y, noise_variance)

    assert isinstance(model, IModelWithPerPointNoise)
    assert isinstance(model, IModelWithNoise)
    assert isinstance(model, IModel)


def test_factory_validates_noise(data):
    x, y, noise_variance = data
    with pytest.raises(ValueError):
        create_heteroscedastic_gpy_model(x, y, np.zeros_like(y))
    with pytest.raises(ValueError):
        create_heteroscedastic_gpy_model(x, y, noise_variance[:-1])


def test_factory_rejects_multiple_outputs(data):
    x, y, _ = data
    y_multi = np.repeat(y, 2, axis=1)
    with pytest.raises(ValueError):
        create_heteroscedastic_gpy_model(x, y_multi, np.ones_like(y_multi))


def test_set_data_rejects_multiple_outputs(data):
    x, y, noise_variance = data
    model = create_heteroscedastic_gpy_model(x, y, noise_variance)
    y_multi = np.repeat(y, 2, axis=1)
    with pytest.raises(ValueError):
        model.set_data(x, y_multi, np.ones_like(y_multi))


def test_likelihood_variance_is_fixed(data):
    x, y, noise_variance = data
    model = create_heteroscedastic_gpy_model(x, y, noise_variance)

    assert model.model.likelihood.variance.is_fixed


def test_posterior_matches_manual_heteroscedastic_gp(data):
    x, y, noise_variance = data
    model = create_heteroscedastic_gpy_model(x, y, noise_variance)
    kernel = model.model.kern

    # Independently build the heteroscedastic GP posterior by Cholesky.
    K = kernel.K(x) + np.diag(noise_variance.ravel()) + 1e-8 * np.eye(x.shape[0])
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    x_test = np.linspace(0.0, 1.0, 15)[:, None]
    K_s = kernel.K(x, x_test)
    expected_mean = K_s.T @ alpha
    expected_variance = kernel.Kdiag(x_test) - np.sum(np.linalg.solve(L, K_s) ** 2, axis=0)

    mean, variance = model.predict_noiseless(x_test)
    np.testing.assert_allclose(mean, expected_mean, atol=1e-8)
    np.testing.assert_allclose(variance.ravel(), expected_variance, atol=1e-8)


def test_predict_adds_per_point_noise_at_training_locations(data):
    x, y, noise_variance = data
    model = create_heteroscedastic_gpy_model(x, y, noise_variance)

    _, variance_noisy = model.predict(x)
    _, variance_latent = model.predict_noiseless(x)

    np.testing.assert_allclose(variance_noisy - variance_latent, noise_variance, atol=1e-10)


def test_predict_at_unknown_locations_has_no_extra_noise(data):
    x, y, noise_variance = data
    model = create_heteroscedastic_gpy_model(x, y, noise_variance)

    x_new = np.array([[5.0], [-5.0]])
    _, variance_noisy = model.predict(x_new)
    _, variance_latent = model.predict_noiseless(x_new)

    np.testing.assert_allclose(variance_noisy, variance_latent, atol=1e-12)


def test_set_data_rebuilds_with_new_noise(data):
    x, y, noise_variance = data
    model = create_heteroscedastic_gpy_model(x, y, noise_variance)

    new_noise = 0.5 * noise_variance
    model.set_data(x, 2 * y, new_noise)

    np.testing.assert_array_equal(model.X, x)
    np.testing.assert_array_equal(model.Y, 2 * y)
    np.testing.assert_array_equal(model.noise_variance, new_noise)

    # The rebuilt posterior matches a manual GP using the new noise.
    kernel = model.model.kern
    K = kernel.K(x) + np.diag(new_noise.ravel()) + 1e-8 * np.eye(x.shape[0])
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, 2 * y))
    expected_mean = kernel.K(x, x).T @ alpha

    mean, _ = model.predict_noiseless(x)
    np.testing.assert_allclose(mean, expected_mean, atol=1e-8)


def test_set_data_rejects_invalid_noise(data):
    x, y, noise_variance = data
    model = create_heteroscedastic_gpy_model(x, y, noise_variance)

    with pytest.raises(ValueError):
        model.set_data(x, y, -np.ones_like(y))


def test_optimize_keeps_noise_fixed_and_updates_kernel(data):
    x, y, noise_variance = data
    model = create_heteroscedastic_gpy_model(x, y, noise_variance)
    original_noise = model.noise_variance.copy()

    model.optimize()

    np.testing.assert_array_equal(model.noise_variance, original_noise)
    assert model.model.likelihood.variance.is_fixed


def test_heteroscedastic_noise_changes_posterior(data):
    x, y, _ = data
    small_noise = np.full_like(y, 1e-4)
    large_noise = np.ones_like(y)

    tight = create_heteroscedastic_gpy_model(x, y, small_noise)
    loose = create_heteroscedastic_gpy_model(x, y, large_noise)

    _, tight_variance = tight.predict_noiseless(x)
    _, loose_variance = loose.predict_noiseless(x)

    assert np.all(tight_variance < loose_variance)


def test_wrapper_type(data):
    x, y, noise_variance = data
    model = create_heteroscedastic_gpy_model(x, y, noise_variance)
    assert isinstance(model, GPyHeteroscedasticModelWrapper)
