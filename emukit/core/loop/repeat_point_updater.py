# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import abc
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..interfaces import IModel, IModelWithPerPointNoise
from . import LoopState
from .model_updaters import ModelUpdater


class PointIdentifier(abc.ABC):
    """
    Maps each individual noisy measurement to the identity of the experimental point it was taken at.

    All measurements that share the same point identity are treated as repeats of the same underlying
    function value and are aggregated into a single training observation.
    """

    @abc.abstractmethod
    def identify(self, X: np.ndarray) -> np.ndarray:
        """
        Assigns a point identity to every row of ``X``.

        :param X: Function inputs of shape (n_measurements x n_input_dims)
        :return: One-dimensional array of length n_measurements containing hashable point identities.
                 Rows with identical identities are aggregated together as repeat measurements.
        """
        pass


class ExactPointIdentifier(PointIdentifier):
    """Identifies points by exact equality of their (optionally rounded) input vectors."""

    def __init__(self, decimals: Optional[int] = None) -> None:
        """
        :param decimals: Number of decimals to round each input to before comparison. ``None`` compares
                         the raw input values exactly.
        """
        self.decimals = decimals

    def identify(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: Function inputs of shape (n_measurements x n_input_dims)
        :return: One-dimensional object array of tuple identities, one per row of ``X``
        """
        if self.decimals is not None:
            X = np.round(X, decimals=self.decimals)
        identities = np.empty(X.shape[0], dtype=object)
        identities[:] = [tuple(row) for row in X]
        return identities


class FunctionPointIdentifier(PointIdentifier):
    """Identifies points using a user-supplied function applied to the input array."""

    def __init__(self, identity_fcn: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        :param identity_fcn: Function that takes the (n_measurements x n_input_dims) input array and
                             returns a one-dimensional array of hashable point identities.
        """
        self.identity_fcn = identity_fcn

    def identify(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: Function inputs of shape (n_measurements x n_input_dims)
        :return: One-dimensional array of point identities, one per row of ``X``
        """
        identities = np.asarray(self.identity_fcn(X))
        if identities.ndim != 1 or identities.shape[0] != X.shape[0]:
            raise ValueError(
                "identity_fcn must return a 1-dimensional array with one identity per measurement, "
                "got shape {}".format(identities.shape)
            )
        return identities


class AggregatedPointObservations:
    """
    Aggregated statistics of all noisy measurements taken at a single experimental point.

    :param X: Input location of the point, shape (n_input_dims,)
    :param Y_mean: Mean of the measured outputs, shape (n_output_dims,)
    :param sample_variance: Unbiased sample variance of the measured outputs (divisor ``n - 1``),
                            shape (n_output_dims,). Set to zero for a single measurement.
    :param mean_variance: Variance of the sample mean (the uncertainty of ``Y_mean``),
                          shape (n_output_dims,). Equals the declared noise prior variance for
                          points with fewer than two measurements.
    :param n_measurements: Number of individual measurements aggregated at this point
    """

    def __init__(
        self,
        X: np.ndarray,
        Y_mean: np.ndarray,
        sample_variance: np.ndarray,
        mean_variance: np.ndarray,
        n_measurements: int,
    ) -> None:
        self.X = X
        self.Y_mean = Y_mean
        self.sample_variance = sample_variance
        self.mean_variance = mean_variance
        self.n_measurements = n_measurements


def aggregate_repeat_measurements(
    X: np.ndarray,
    Y: np.ndarray,
    noise_prior_variance: float,
    point_identifier: PointIdentifier,
) -> List[AggregatedPointObservations]:
    """
    Groups individual noisy measurements by point identity and computes the mean, sample variance and
    effective observation noise of the mean for every distinct experimental point.

    For a point with ``n`` repeat measurements the sample mean is

    .. math:: \\bar{y} = \\frac{1}{n}\\sum_{i=1}^{n} y_i

    with unbiased sample variance

    .. math:: s^2 = \\frac{1}{n-1}\\sum_{i=1}^{n} (y_i - \\bar{y})^2

    and mean observation noise (the variance to feed to the Gaussian process)

    .. math:: \\sigma^2_{\\bar{y}} = s^2 / n.

    Points with fewer than two measurements cannot estimate a sample variance, so the declared
    ``noise_prior_variance`` is used as the variance of their (single) observation.

    :param X: All individual measurement inputs, shape (n_measurements x n_input_dims)
    :param Y: All individual measurement outputs, shape (n_measurements x n_output_dims)
    :param noise_prior_variance: Declared observation noise variance used when a point has fewer
                                 than two measurements
    :param point_identifier: Object that assigns a point identity to each row of ``X``
    :return: One :class:`AggregatedPointObservations` per distinct point, ordered by first appearance
             in ``X``
    """
    if X.ndim != 2:
        raise ValueError("X is expected to be 2-dimensional, actual dimensionality is {}".format(X.ndim))
    if Y.ndim != 2:
        raise ValueError("Y is expected to be 2-dimensional, actual dimensionality is {}".format(Y.ndim))
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y must contain the same number of measurements, got {} and {}".format(X.shape[0], Y.shape[0])
        )
    if noise_prior_variance <= 0:
        raise ValueError("noise_prior_variance must be strictly positive, got {}".format(noise_prior_variance))

    identities = point_identifier.identify(X)

    # Preserve the order in which distinct points first appear, so aggregation results are stable
    # across loop iterations.
    group_order: List[object] = []
    groups: dict = {}
    for index, identity in enumerate(identities):
        identity = identity.item() if isinstance(identity, np.generic) else identity
        if identity not in groups:
            groups[identity] = []
            group_order.append(identity)
        groups[identity].append(index)

    aggregated = []
    for identity in group_order:
        indices = np.asarray(groups[identity], dtype=int)
        point_X = X[indices[0]]
        point_Y = Y[indices]
        n_measurements = indices.size
        y_mean = np.mean(point_Y, axis=0)

        if n_measurements < 2:
            # No repeats: the sample variance cannot be estimated, fall back to the declared prior.
            sample_variance = np.zeros(point_Y.shape[1])
            mean_variance = np.full(point_Y.shape[1], noise_prior_variance)
        else:
            # ddof=1 gives the unbiased sample variance (divisor n - 1).
            sample_variance = np.var(point_Y, axis=0, ddof=1)
            mean_variance = sample_variance / n_measurements

        aggregated.append(
            AggregatedPointObservations(
                X=point_X,
                Y_mean=y_mean,
                sample_variance=sample_variance,
                mean_variance=mean_variance,
                n_measurements=n_measurements,
            )
        )

    return aggregated


def aggregated_arrays(
    aggregated: List[AggregatedPointObservations],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stacks a list of aggregated point observations into arrays suitable for model training.

    :param aggregated: Aggregated observations, e.g. the output of
                       :func:`aggregate_repeat_measurements`
    :return: Tuple ``(X, Y_mean, mean_variance, n_measurements)`` where ``X`` has shape
             (n_points x n_input_dims), ``Y_mean`` and ``mean_variance`` have shape
             (n_points x n_output_dims) and ``n_measurements`` is a one-dimensional integer array
             of length n_points
    """
    if not aggregated:
        raise ValueError("Cannot stack an empty list of aggregated point observations")

    X = np.stack([point.X for point in aggregated])
    Y_mean = np.stack([point.Y_mean for point in aggregated])
    mean_variance = np.stack([point.mean_variance for point in aggregated])
    n_measurements = np.asarray([point.n_measurements for point in aggregated], dtype=int)
    return X, Y_mean, mean_variance, n_measurements


class RepeatPointUpdater(ModelUpdater):
    """
    Model updater that aggregates repeated noisy measurements of the same experimental point before
    updating the model.

    The raw, individual measurements remain stored in :attr:`LoopState.results`; only the data handed
    to the model is aggregated. Each distinct point becomes one training observation whose target is
    the sample mean and whose observation noise is the variance of that mean. Models implementing
    :class:`~emukit.core.interfaces.IModelWithPerPointNoise` receive the per-point mean variances, so
    well-replicated points are correctly treated as more precise than singly measured points. Models
    without that interface are updated through the standard ``set_data`` method using the aggregated
    means, and the mean uncertainties are exposed via :attr:`last_mean_variance`.
    """

    def __init__(
        self,
        model: IModel,
        noise_prior_variance: float,
        point_identifier: PointIdentifier = None,
        interval: int = 1,
        targets_extractor_fcn: Callable = None,
    ) -> None:
        """
        :param model: Emukit emulator model. If it implements
                      :class:`~emukit.core.interfaces.IModelWithPerPointNoise` the aggregated mean
                      variances are passed as per-point observation noise.
        :param noise_prior_variance: Declared observation noise variance assigned to points with
                                     fewer than two measurements
        :param point_identifier: Object that decides which measurements belong to the same
                                 experimental point. Defaults to :class:`ExactPointIdentifier`.
        :param interval: Number of loop iterations between model hyper-parameter optimizations
        :param targets_extractor_fcn: Function that takes the loop state and returns the training
                                      targets. Defaults to a function returning ``loop_state.Y``.
        """
        self.model = model
        self.noise_prior_variance = noise_prior_variance
        self.point_identifier = point_identifier if point_identifier is not None else ExactPointIdentifier()
        self.interval = interval

        if targets_extractor_fcn is None:
            self.targets_extractor_fcn = lambda loop_state: loop_state.Y
        else:
            self.targets_extractor_fcn = targets_extractor_fcn

        # Outputs of the most recent aggregation, kept available for inspection and testing.
        self.last_aggregated: List[AggregatedPointObservations] = []
        self.last_mean_variance: Optional[np.ndarray] = None
        self.last_n_measurements: Optional[np.ndarray] = None

    def update(self, loop_state: LoopState) -> None:
        """
        Aggregates all raw measurements in the loop state by point identity and updates the model
        with one mean observation and one mean variance per distinct point.

        An empty loop state leaves the model untouched, mirroring
        :class:`FixedIntervalUpdater` so the updater can be used before the first evaluation.

        :param loop_state: Object containing the full history of individual measurements
        """
        if not loop_state.results:
            return

        targets = self.targets_extractor_fcn(loop_state)
        aggregated = aggregate_repeat_measurements(
            loop_state.X, targets, self.noise_prior_variance, self.point_identifier
        )
        X, Y_mean, mean_variance, n_measurements = aggregated_arrays(aggregated)

        self.last_aggregated = aggregated
        self.last_mean_variance = mean_variance
        self.last_n_measurements = n_measurements

        if isinstance(self.model, IModelWithPerPointNoise):
            self.model.set_data(X, Y_mean, mean_variance)
        else:
            self.model.set_data(X, Y_mean)

        if (loop_state.iteration % self.interval) == 0:
            self.model.optimize()
