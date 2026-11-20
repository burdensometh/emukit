# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Support for the asynchronous, external evaluation mode of the outer loop.

In this mode the user first *claims* candidate points (see :meth:`OuterLoop.ask`)
and later reports the outcomes of the corresponding experiments out of order
(see :meth:`OuterLoop.tell`). Experiments can finish at different times, so this
module keeps track of every claimed point until its outcome is confirmed:

- points that have been claimed but not yet confirmed are *pending*,
- a pending point whose real observation has been reported is *succeeded*,
- a pending point whose experiment failed is *failed*,
- a pending point whose experiment was cancelled before completion is *cancelled*.

Pending points never enter :class:`~emukit.core.loop.loop_state.LoopState`, i.e.
they are never treated as real observations of the user function. They do,
however, participate in the selection of new candidate points: the
:class:`PendingAwarePointCalculator` either excludes them from the candidate set
or down-weights the acquisition function around them so that the same point is
not suggested twice while an experiment is in flight.
"""

import abc
import enum
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..acquisition import Acquisition
from .candidate_point_calculators import CandidatePointCalculator
from .user_function_result import UserFunctionResult

_log = logging.getLogger(__name__)


class PendingStatus(enum.Enum):
    """Lifecycle status of a point claimed for external evaluation."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PendingObservation:
    """
    A single point that has been claimed for external evaluation.

    :param observation_id: Unique, traceable identifier returned by ``ask``
    :param X: The claimed function input. Shape: (function input dimension,)
    :param status: Current lifecycle status
    :param result: The confirmed :class:`UserFunctionResult`, only set once the
                   observation has succeeded
    """

    def __init__(
        self,
        observation_id: str,
        X: np.ndarray,
        status: PendingStatus = PendingStatus.PENDING,
        result: Optional[UserFunctionResult] = None,
    ) -> None:
        self.observation_id = observation_id
        self.X = np.asarray(X)
        self.status = status
        self.result = result

    def __repr__(self) -> str:
        return "PendingObservation(id: {}, X: {}, status: {})".format(self.observation_id, self.X, self.status.value)


class PendingObservationTracker:
    """
    Registry of all points that have been claimed for external evaluation.

    The tracker is the single source of truth for the lifecycle of claimed
    points. Each claimed point is stored exactly once, keyed by the traceable
    identifier returned when it was claimed, and moves through the statuses in
    :class:`PendingStatus` at most once: terminal states (succeeded, failed,
    cancelled) cannot be overwritten, so conflicting or repeated reports are
    rejected.
    """

    def __init__(self) -> None:
        self._observations: Dict[str, PendingObservation] = {}
        self._next_id = 0

    def register(self, X: np.ndarray) -> str:
        """
        Registers a newly claimed point with status :py:attr:`PendingStatus.PENDING`.

        :param X: The claimed function input. Shape: (function input dimension,)
        :return: The unique, traceable identifier assigned to this observation
        """
        observation_id = self._new_id()
        self._observations[observation_id] = PendingObservation(observation_id, np.asarray(X))
        return observation_id

    def register_batch(self, X: np.ndarray) -> List[str]:
        """
        Registers a batch of newly claimed points.

        :param X: The claimed function inputs. Shape: (n_points x n_dims)
        :return: The unique identifiers assigned to the observations, in the
                 same order as the rows of ``X``
        """
        return [self.register(row) for row in np.atleast_2d(X)]

    def mark_succeeded(self, observation_id: str, result: UserFunctionResult) -> None:
        """
        Confirms the first real, successful observation for a pending point.

        :param observation_id: Identifier returned when the point was claimed
        :param result: The observed :class:`UserFunctionResult`
        """
        observation = self._require_pending(observation_id)
        if not np.allclose(observation.X, result.X):
            raise ValueError(
                "Result X {} does not match the point X {} claimed for observation '{}'".format(
                    result.X, observation.X, observation_id
                )
            )
        observation.status = PendingStatus.SUCCEEDED
        observation.result = result

    def mark_failed(self, observation_id: str) -> None:
        """
        Marks a pending point as failed. The placeholder is released and no
        observation is ever added to the loop state.

        :param observation_id: Identifier returned when the point was claimed
        """
        observation = self._require_pending(observation_id)
        observation.status = PendingStatus.FAILED

    def mark_cancelled(self, observation_id: str) -> None:
        """
        Marks a pending point as cancelled. The placeholder is released and no
        observation is ever added to the loop state.

        :param observation_id: Identifier returned when the point was claimed
        """
        observation = self._require_pending(observation_id)
        observation.status = PendingStatus.CANCELLED

    def get(self, observation_id: str) -> PendingObservation:
        """
        :param observation_id: Identifier returned when the point was claimed
        :return: The registered observation
        """
        try:
            return self._observations[observation_id]
        except KeyError:
            raise KeyError("Unknown observation id '{}'".format(observation_id))

    def observations(self, status: Optional[PendingStatus] = None) -> List[PendingObservation]:
        """
        :param status: If given, only observations in this status are returned
        :return: All matching observations, ordered by the time they were claimed
        """
        if status is None:
            return list(self._observations.values())
        return [observation for observation in self._observations.values() if observation.status == status]

    @property
    def pending(self) -> List[PendingObservation]:
        """:return: All observations that are still awaiting confirmation"""
        return self.observations(PendingStatus.PENDING)

    @property
    def pending_X(self) -> Optional[np.ndarray]:
        """
        :return: (n_pending x n_dims) array of the inputs of all pending
                 observations, or ``None`` if no point is currently pending
        """
        pending = self.pending
        if not pending:
            return None
        return np.array([observation.X for observation in pending])

    def is_pending(self, observation_id: str) -> bool:
        """
        :param observation_id: Identifier returned when the point was claimed
        :return: Whether the observation is still pending
        """
        return (
            observation_id in self._observations and self._observations[observation_id].status == PendingStatus.PENDING
        )

    def _new_id(self) -> str:
        observation_id = "obs-{}".format(self._next_id)
        self._next_id += 1
        return observation_id

    def _require_pending(self, observation_id: str) -> PendingObservation:
        if observation_id not in self._observations:
            raise KeyError("Unknown observation id '{}'".format(observation_id))
        observation = self._observations[observation_id]
        if observation.status != PendingStatus.PENDING:
            raise ValueError(
                "Observation '{}' is already in terminal status '{}' and cannot be reported again".format(
                    observation_id, observation.status.value
                )
            )
        return observation


class PendingPointStrategy(abc.ABC):
    """Strategy used by :class:`PendingAwarePointCalculator` to handle pending points."""

    @abc.abstractmethod
    def is_taken(self, X: np.ndarray, pending_X: np.ndarray) -> bool:
        """
        :param X: Candidate point. Shape: (n_dims,)
        :param pending_X: All currently pending points. Shape: (n_pending x n_dims)
        :return: Whether the candidate point must be rejected as a duplicate
        """
        pass


class ExcludePendingPoints(PendingPointStrategy):
    """Rejects candidate points that coincide with a point that is still pending.

    By default points are compared with zero tolerances, i.e. they only coincide
    when they are exactly equal (as happens in discrete or categorical spaces,
    or when the acquisition optimizer deterministically returns the same
    maximizer). Pass larger tolerances to also exclude points that are merely
    very close.
    """

    def __init__(self, rtol: float = 0.0, atol: float = 0.0) -> None:
        """
        :param rtol: Relative tolerance used when comparing points
        :param atol: Absolute tolerance used when comparing points
        """
        self.rtol = rtol
        self.atol = atol

    def is_taken(self, X: np.ndarray, pending_X: np.ndarray) -> bool:
        # np.atleast_2d makes the function work for both 1-D single points and
        # (1 x n_dims) candidate rows returned by point calculators.
        X = np.atleast_2d(X)
        return bool(np.any(np.all(np.isclose(pending_X, X, rtol=self.rtol, atol=self.atol), axis=1)))


class PendingPenalty(Acquisition):
    """
    Acquisition wrapper that down-weights an acquisition around pending points.

    The wrapper adds a log-space barrier of configurable strength that is zero
    far away from every pending point and grows (negatively) as a candidate
    approaches one:

    .. math::
        \\log a_{\\text{penalized}}(x) = \\log a(x) - \\kappa
        \\sum_j \\exp\\!\\left(-\\frac{\\|x - x_j\\|^2}{2\\sigma^2}\\right)

    where :math:`x_j` are the currently pending points and :math:`\\kappa` is the
    barrier strength. Exponentiating gives a strictly positive multiplicative
    penalty, so every point remains selectable but points close to an in-flight
    experiment are strongly discouraged (the value directly on a pending point is
    reduced by a factor of :math:`e^{-\\kappa}`).

    This implementation does not provide gradients; the acquisition optimizer
    automatically falls back to its gradient-free routine when
    :attr:`has_gradients` is ``False``.
    """

    def __init__(self, acquisition: Acquisition, sigma: float = 1e-2, strength: float = 5.0) -> None:
        """
        :param acquisition: The base acquisition function to penalize
        :param sigma: Length scale of the penalty. Smaller values make the
                      penalty more localised around the pending points
        :param strength: Barrier strength :math:`\\kappa`. The acquisition value
                         directly on a pending point is multiplied by
                         :math:`e^{-\\kappa}`. Must be positive
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive, got {}".format(sigma))
        if strength <= 0:
            raise ValueError("strength must be positive, got {}".format(strength))
        self.acquisition = acquisition
        self.sigma = sigma
        self.strength = strength
        self.pending_X: Optional[np.ndarray] = None

    def update_pending(self, pending_X: Optional[np.ndarray]) -> None:
        """
        :param pending_X: (n_pending x n_dims) array of pending points or
                          ``None`` if no point is currently pending
        """
        self.pending_X = None if pending_X is None else np.atleast_2d(pending_X)

    @property
    def has_gradients(self) -> bool:
        """:return: Always False, gradients of the penalty are not provided"""
        return False

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: (n_points x n_dims) array of points at which to evaluate
        :return: (n_points x 1) array of penalized acquisition values
        """
        value = self.acquisition.evaluate(x)
        if self.pending_X is None or self.pending_X.shape[0] == 0:
            return value
        squared_distance = np.sum((x[:, None, :] - self.pending_X[None, :, :]) ** 2, axis=2)
        barrier = np.exp(-squared_distance / (2 * self.sigma**2)).sum(axis=1, keepdims=True)
        return value * np.exp(-self.strength * barrier)

    def update_parameters(self) -> None:
        self.acquisition.update_parameters()


class PendingAwarePointCalculator(CandidatePointCalculator):
    """
    Candidate point calculator that keeps pending points out of new suggestions.

    Wraps any :class:`CandidatePointCalculator`. Before every ``ask`` the pending
    points registered in the :class:`PendingObservationTracker` are made visible
    to the wrapped calculator in one of two ways:

    - ``"exclude"`` (default): candidate points coinciding with a pending point
      are rejected and the acquisition is re-optimized until a free point is
      found. This is exact for discrete and categorical spaces.
    - ``"penalize"``: the acquisition is multiplied by a :class:`PendingPenalty`
      centered on the pending points, which smoothly discourages re-suggesting
      in-flight points in continuous spaces without failing when an exact
      duplicate cannot be avoided.

    Pending points are the only extra information used: no fake observations are
    added to the model or to the loop state.
    """

    def __init__(
        self,
        candidate_point_calculator: CandidatePointCalculator,
        tracker: PendingObservationTracker,
        strategy: str = "exclude",
        max_attempts: int = 100,
        penalty_sigma: float = 1e-2,
        penalty_strength: float = 5.0,
    ) -> None:
        """
        :param candidate_point_calculator: The calculator used to optimize the
                                           underlying acquisition function
        :param tracker: Registry of claimed points and their statuses
        :param strategy: Either ``"exclude"`` or ``"penalize"``
        :param max_attempts: Maximum number of re-optimization attempts in
                             exclude mode before the best available point is
                             returned
        :param penalty_sigma: Length scale of the penalty in penalize mode
        :param penalty_strength: Barrier strength of the penalty in penalize
                                 mode. The acquisition value directly on a
                                 pending point is reduced by a factor of
                                 :math:`e^{-\\text{penalty_strength}}`
        """
        if strategy not in ("exclude", "penalize"):
            raise ValueError("strategy must be 'exclude' or 'penalize', got '{}'".format(strategy))
        if max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer, got {}".format(max_attempts))
        self.candidate_point_calculator = candidate_point_calculator
        self.tracker = tracker
        self.strategy = strategy
        self.max_attempts = max_attempts
        self.exclusion = ExcludePendingPoints()
        self.penalty: Optional[PendingPenalty] = None
        if strategy == "penalize":
            self.penalty = PendingPenalty(
                _AcquisitionAdapter(candidate_point_calculator), penalty_sigma, penalty_strength
            )

    def compute_next_points(self, loop_state, context: dict = None) -> np.ndarray:
        """
        Computes the next point to evaluate while accounting for pending points.

        :param loop_state: Object that contains current state of the loop
        :param context: Contains variables to fix through optimization of the
                        acquisition function
        :return: (1 x n_dims) array of the next input to evaluate
        """
        pending_X = self.tracker.pending_X
        if self.strategy == "penalize":
            return self._compute_penalized(loop_state, context, pending_X)
        return self._compute_excluded(loop_state, context, pending_X)

    def _compute_excluded(self, loop_state, context: Optional[dict], pending_X: Optional[np.ndarray]) -> np.ndarray:
        if pending_X is None:
            return self.candidate_point_calculator.compute_next_points(loop_state, context)

        last_suggestion = None
        for _ in range(self.max_attempts):
            new_x = self.candidate_point_calculator.compute_next_points(loop_state, context)
            last_suggestion = new_x
            if not self.exclusion.is_taken(new_x[:1], pending_X):
                return new_x

        # The space may be exhausted (e.g. every discrete point is pending):
        # return the last suggestion rather than failing. The caller can still
        # detect and cancel the duplicate claim once the conflict is observed.
        _log.warning(
            "Could not find a point distinct from all pending points after %d attempt(s); "
            "returning the last suggestion",
            self.max_attempts,
        )
        return last_suggestion

    def _compute_penalized(self, loop_state, context: Optional[dict], pending_X: Optional[np.ndarray]) -> np.ndarray:
        assert self.penalty is not None
        self.penalty.update_pending(pending_X)
        return _PenalizedCalculator(self.candidate_point_calculator, self.penalty).compute_next_points(
            loop_state, context
        )


class _AcquisitionAdapter(Acquisition):
    """Exposes the acquisition used inside an arbitrary point calculator for penalization."""

    def __init__(self, candidate_point_calculator: CandidatePointCalculator) -> None:
        acquisition = getattr(candidate_point_calculator, "acquisition", None)
        if not isinstance(acquisition, Acquisition):
            raise ValueError(
                "The 'penalize' strategy requires a candidate point calculator exposing an "
                "'acquisition' attribute (e.g. SequentialPointCalculator)"
            )
        self.acquisition = acquisition

    @property
    def has_gradients(self) -> bool:
        return self.acquisition.has_gradients

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.acquisition.evaluate(x)

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.acquisition.evaluate_with_gradients(x)

    def update_parameters(self) -> None:
        self.acquisition.update_parameters()


class _PenalizedCalculator(CandidatePointCalculator):
    """Temporarily swaps the wrapped calculator's acquisition for the penalized one."""

    def __init__(self, candidate_point_calculator: CandidatePointCalculator, penalty: PendingPenalty) -> None:
        self.candidate_point_calculator = candidate_point_calculator
        self.penalty = penalty

    def compute_next_points(self, loop_state, context: dict = None) -> np.ndarray:
        calculator = self.candidate_point_calculator
        original_acquisition = calculator.acquisition
        calculator.acquisition = self.penalty
        try:
            return calculator.compute_next_points(loop_state, context)
        finally:
            calculator.acquisition = original_acquisition
