# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import numpy as np
import scipy.linalg
import scipy.optimize

from emukit.core.interfaces import IModelWithNoise, IModelWithPerPointNoise


class HeteroscedasticGaussianProcessModel(IModelWithPerPointNoise, IModelWithNoise):
    """
    A Gaussian process with an RBF kernel (without ARD) that supports a different observation noise
    variance for every training point.

    The model is a small, pure-numpy implementation intended for use with aggregated repeat
    measurements: each distinct experimental point contributes one mean observation together with the
    variance of that mean. It mirrors :class:`~emukit.model_wrappers.SimpleGaussianProcessModel` but
    replaces the single shared likelihood variance with a per-point noise vector, and additionally
    exposes noise-free predictions through :meth:`predict_noiseless`.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        noise_variance: np.ndarray,
        lengthscale: float = 1.0,
        kernel_variance: float = 1.0,
        jitter: float = 1e-6,
    ) -> None:
        """
        :param x: (n_points, n_dims) array containing training features
        :param y: (n_points, 1) array containing training targets
        :param noise_variance: (n_points, 1) array of strictly positive observation noise variances,
                               one per training point
        :param lengthscale: Initial RBF kernel lengthscale
        :param kernel_variance: Initial RBF kernel output variance
        :param jitter: Numerical jitter added to the diagonal of the training covariance matrix
        """
        self._validate_noise(x, y, noise_variance)
        self.x = x
        self.y = y
        self.noise_variance = noise_variance.astype(float)
        self.lengthscale = lengthscale
        self.kernel_variance = kernel_variance
        self.jitter = jitter

    @staticmethod
    def _validate_noise(x: np.ndarray, y: np.ndarray, noise_variance: np.ndarray) -> None:
        noise_variance = np.asarray(noise_variance)
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError(
                "This Gaussian process supports a single output only, so Y must have shape "
                "(n_points, 1), got {}".format(y.shape)
            )
        if noise_variance.shape != y.shape:
            raise ValueError(
                "noise_variance must have the same shape as Y, got {} and {}".format(noise_variance.shape, y.shape)
            )
        if np.any(noise_variance <= 0):
            raise ValueError("All observation noise variances must be strictly positive")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "X and Y must contain the same number of points, got {} and {}".format(x.shape[0], y.shape[0])
            )

    def __repr__(self):
        """
        Prints the kernel hyper-parameters
        """
        return "Lengthscale: {:.4f} \n Kernel variance: {:.4f} \n Per-point noise variance: {}".format(
            self.lengthscale, self.kernel_variance, self.noise_variance.ravel()
        )

    def set_data(self, X: np.ndarray, Y: np.ndarray, noise_variance: np.ndarray) -> None:
        """
        Set training data and per-point observation noise to new values

        :param X: (n_points, n_dims) array containing training features
        :param Y: (n_points, 1) array containing training targets
        :param noise_variance: (n_points, 1) array of strictly positive observation noise variances
        """
        self._validate_noise(X, Y, noise_variance)
        self.x = X
        self.y = Y
        self.noise_variance = noise_variance.astype(float)

    def optimize(self) -> None:
        """
        Optimize the kernel lengthscale and kernel variance. The per-point noise variances are kept
        fixed because they are known estimates derived from the repeat measurements.
        """

        def optimize_fcn(log_hyper_parameters):
            # take exponential to ensure positive values
            hyper_parameters = np.exp(log_hyper_parameters)
            self.lengthscale = hyper_parameters[0]
            self.kernel_variance = hyper_parameters[1]
            return self._negative_marginal_log_likelihood()

        lower_bound = np.log(1e-6)
        upper_bound = np.log(1e8)

        bounds = [(lower_bound, upper_bound), (lower_bound, upper_bound)]
        scipy.optimize.minimize(
            optimize_fcn,
            np.log(np.array([self.lengthscale, self.kernel_variance])),
            bounds=bounds,
        )

    def _training_covariance(self) -> np.ndarray:
        """
        :return: Training covariance matrix K(X, X) with jitter and per-point noise on the diagonal
        """
        K = self._calc_kernel(self.x)
        K += np.identity(self.x.shape[0]) * self.jitter
        K += np.diag(self.noise_variance.ravel())
        return K

    def _posterior_quantities(self, x_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the Cholesky factor of the training covariance and the triangular solves shared by
        prediction and likelihood calculations.

        :param x_new: (n_points_new, n_dims) array of prediction locations
        :return: Tuple of the lower Cholesky factor, the solved kernel cross-covariance and the
                 solved training targets
        """
        K = self._training_covariance()
        L = np.linalg.cholesky(K)

        K_xs = self._calc_kernel(self.x, x_new)

        tmp = scipy.linalg.solve_triangular(L, K_xs, lower=True)
        tmp2 = scipy.linalg.solve_triangular(L, self.y, lower=True)
        return L, tmp, tmp2

    def predict(self, x_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict from the model including the observation noise.

        Per-point noise only applies at the training locations, so the noise added to the predictive
        variance at a new location is the noise declared for that exact training point when
        ``x_new`` coincides with one, and zero otherwise.

        :param x_new: (n_points, n_dims) array containing points at which the predictive
                      distributions will be computed
        :return: Tuple containing two (n_points, 1) arrays representing the mean and variance of the
                 predictive distribution at the specified input locations
        """
        mean, latent_variance = self.predict_noiseless(x_new)
        return mean, latent_variance + self._noise_at(x_new)

    def predict_noiseless(self, x_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the latent function mean and variance without any observation noise.

        :param x_new: (n_points, n_dims) array containing points at which the predictive
                      distributions will be computed
        :return: Tuple containing two (n_points, 1) arrays representing the mean and latent variance
                 of the predictive distribution at the specified input locations
        """
        _, tmp, tmp2 = self._posterior_quantities(x_new)

        mean = np.dot(tmp.T, tmp2)
        variance = (self.kernel_variance - np.sum(np.square(tmp), axis=0))[:, None]
        return mean, variance

    def _noise_at(self, x_new: np.ndarray) -> np.ndarray:
        """
        Returns the observation noise variance associated with each prediction location. Training
        points carry their declared per-point noise; any other location has no observation noise.

        :param x_new: (n_points, n_dims) array of prediction locations
        :return: (n_points, 1) array of observation noise variances
        """
        noise = np.zeros((x_new.shape[0], 1))
        for i, candidate in enumerate(x_new):
            matches = np.where(np.all(self.x == candidate, axis=1))[0]
            if matches.size > 0:
                noise[i, 0] = self.noise_variance[matches[0], 0]
        return noise

    def _calc_kernel(self, X: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Implements an RBF kernel with no ARD

        :param X: array of shape (n_points_1, n_dims) containing input points of first argument to kernel function
        :param X2: array of shape (n_points_2, n_dims) containing input points of second argument to kernel function.
                   If not supplied K(X, X) is computed.
        :return: Kernel matrix K(X, X2) or K(X, X) if X2 not supplied.
        """
        if X2 is None:
            X2 = X

        X1sq = np.sum(np.square(X), 1)
        X2sq = np.sum(np.square(X2), 1)
        r2 = -2.0 * np.dot(X, X2.T) + (X1sq[:, None] + X2sq[None, :])
        r2 = np.clip(r2, 0, np.inf)
        return self.kernel_variance * np.exp(-0.5 * r2 / self.lengthscale**2)

    def _negative_marginal_log_likelihood(self) -> float:
        """
        :return: Negative marginal log likelihood of model with current hyper-parameters
        """
        K = self._training_covariance()

        # cholesky decomposition of covariance matrix
        L = np.linalg.cholesky(K)

        # Log determinant of the covariance matrix
        log_det = 2.0 * np.sum(np.log(np.diag(L)))

        # calculate y^T K^{-1} y using the cholesky of K
        tmp = scipy.linalg.solve_triangular(L, self.y, lower=True)
        alpha = scipy.linalg.solve_triangular(L.T, tmp, lower=False)
        log_2_pi = np.log(2 * np.pi)
        return -0.5 * (-self.y.size * log_2_pi - self.y.shape[1] * log_det - np.sum(alpha * self.y))

    @property
    def X(self) -> np.ndarray:
        return self.x

    @property
    def Y(self) -> np.ndarray:
        return self.y
