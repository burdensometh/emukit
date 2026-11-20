# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import mock
import numpy as np
import pytest

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.loop import (
    CandidatePointCalculator,
    FixedIntervalUpdater,
    OuterLoop,
    PendingAwarePointCalculator,
    PendingObservationTracker,
    PendingPenalty,
    PendingStatus,
    SequentialPointCalculator,
    UserFunctionResult,
)
from emukit.core.loop.external_evaluation import ExcludePendingPoints
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.loop.model_updaters import ModelUpdater
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.model_wrappers import SimpleGaussianProcessModel


class CountingUpdater(ModelUpdater):
    def __init__(self):
        self.n_updates = 0

    def update(self, loop_state):
        self.n_updates += 1


class ScriptedPointCalculator(CandidatePointCalculator):
    """Returns points from a fixed script, cycling over the last entry once exhausted."""

    def __init__(self, points):
        self.points = [np.atleast_1d(np.asarray(p, dtype=float)) for p in points]
        self.n_calls = 0

    def compute_next_points(self, loop_state, context=None):
        index = min(self.n_calls, len(self.points) - 1)
        self.n_calls += 1
        return self.points[index][None, :]


class FixedAcquisition(Acquisition):
    """Acquisition whose value at a grid point is given by a lookup table."""

    has_gradients = False

    def __init__(self, grid, values):
        self.grid = grid
        self.values = values

    def evaluate(self, x):
        out = np.zeros((x.shape[0], 1))
        for i, row in enumerate(x):
            out[i, 0] = self.values[np.argmax(np.all(np.isclose(self.grid, row), axis=1))]
        return out


class GridPointCalculator(CandidatePointCalculator):
    """Picks the grid point with the highest (possibly penalized) acquisition value."""

    def __init__(self, acquisition, grid):
        self.acquisition = acquisition
        self.grid = grid

    def compute_next_points(self, loop_state, context=None):
        self.acquisition.update_parameters()
        values = self.acquisition.evaluate(self.grid)
        return self.grid[np.argmax(values)][None, :]


def make_result(x, y=0.0):
    return UserFunctionResult(np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float)))


# ---------------------------------------------------------------------------
# PendingObservationTracker
# ---------------------------------------------------------------------------


def test_tracker_assigns_unique_ids_and_starts_pending():
    tracker = PendingObservationTracker()
    id_0 = tracker.register(np.array([0.0]))
    id_1 = tracker.register(np.array([1.0]))

    assert id_0 == "obs-0"
    assert id_1 == "obs-1"
    assert id_0 != id_1
    assert tracker.is_pending(id_0) and tracker.is_pending(id_1)
    assert [obs.observation_id for obs in tracker.pending] == [id_0, id_1]
    np.testing.assert_array_equal(tracker.pending_X, np.array([[0.0], [1.0]]))


def test_tracker_register_batch_preserves_order():
    tracker = PendingObservationTracker()
    ids = tracker.register_batch(np.array([[0.0], [1.0], [2.0]]))
    assert ids == ["obs-0", "obs-1", "obs-2"]
    assert tracker.pending_X.shape == (3, 1)


def test_tracker_terminal_transitions_release_pending_placeholders():
    tracker = PendingObservationTracker()
    id_0 = tracker.register(np.array([0.0]))
    id_1 = tracker.register(np.array([1.0]))
    id_2 = tracker.register(np.array([2.0]))

    result = make_result(0.0, 0.5)
    tracker.mark_succeeded(id_0, result)
    tracker.mark_failed(id_1)
    tracker.mark_cancelled(id_2)

    assert tracker.get(id_0).status == PendingStatus.SUCCEEDED
    assert tracker.get(id_0).result is result
    assert tracker.get(id_1).status == PendingStatus.FAILED
    assert tracker.get(id_2).status == PendingStatus.CANCELLED
    assert tracker.pending == []
    assert tracker.pending_X is None
    status_counts = {
        status: len(tracker.observations(status))
        for status in [PendingStatus.PENDING, PendingStatus.SUCCEEDED, PendingStatus.FAILED, PendingStatus.CANCELLED]
    }
    assert status_counts == {
        PendingStatus.PENDING: 0,
        PendingStatus.SUCCEEDED: 1,
        PendingStatus.FAILED: 1,
        PendingStatus.CANCELLED: 1,
    }


def test_tracker_rejects_repeated_terminal_reports():
    tracker = PendingObservationTracker()
    observation_id = tracker.register(np.array([0.0]))
    tracker.mark_succeeded(observation_id, make_result(0.0))

    with pytest.raises(ValueError):
        tracker.mark_succeeded(observation_id, make_result(0.0))
    with pytest.raises(ValueError):
        tracker.mark_failed(observation_id)
    with pytest.raises(ValueError):
        tracker.mark_cancelled(observation_id)


def test_tracker_unknown_id_raises_keyerror():
    tracker = PendingObservationTracker()
    with pytest.raises(KeyError):
        tracker.get("does-not-exist")
    with pytest.raises(KeyError):
        tracker.mark_failed("does-not-exist")
    assert not tracker.is_pending("does-not-exist")


def test_tracker_mark_succeeded_rejects_x_conflict():
    tracker = PendingObservationTracker()
    observation_id = tracker.register(np.array([0.0]))
    with pytest.raises(ValueError):
        tracker.mark_succeeded(observation_id, make_result(0.99))
    # rejected report must not have changed the status
    assert tracker.is_pending(observation_id)


# ---------------------------------------------------------------------------
# ExcludePendingPoints / PendingPenalty
# ---------------------------------------------------------------------------


def test_exclude_pending_points_matches_exact_points_only_by_default():
    exclusion = ExcludePendingPoints()
    pending_X = np.array([[0.0], [1.0]])

    assert exclusion.is_taken(np.array([0.0]), pending_X)
    assert exclusion.is_taken(np.array([[1.0]]), pending_X)
    assert not exclusion.is_taken(np.array([0.5]), pending_X)
    # a distinct but very close point is not treated as a duplicate by default
    assert not exclusion.is_taken(np.array([1e-6]), pending_X)


def test_exclude_pending_points_custom_tolerance():
    exclusion = ExcludePendingPoints(atol=1e-4)
    pending_X = np.array([[0.0]])
    assert exclusion.is_taken(np.array([1e-6]), pending_X)
    assert not exclusion.is_taken(np.array([1e-3]), pending_X)


def test_pending_penalty_downweights_pending_locations_only():
    grid = np.array([[0.0], [1.0]])
    base = FixedAcquisition(grid, np.array([0.1, 1.0]))
    penalty = PendingPenalty(base, sigma=0.1, strength=5.0)

    # without pending points the base acquisition is returned unchanged
    np.testing.assert_array_equal(penalty.evaluate(grid), base.evaluate(grid))

    penalty.update_pending(np.array([[1.0]]))
    values = penalty.evaluate(grid).flatten()
    # point 1.0 is the base maximizer but is now strongly penalized
    assert values[0] > values[1]
    # far-away point is essentially untouched
    assert values[0] == pytest.approx(0.1)
    # pending point value is reduced by exp(-strength)
    assert values[1] == pytest.approx(1.0 * np.exp(-5.0))
    assert penalty.has_gradients is False


def test_pending_penalty_validates_arguments():
    base = FixedAcquisition(np.array([[0.0]]), np.array([1.0]))
    with pytest.raises(ValueError):
        PendingPenalty(base, sigma=0.0)
    with pytest.raises(ValueError):
        PendingPenalty(base, strength=-1.0)


# ---------------------------------------------------------------------------
# PendingAwarePointCalculator
# ---------------------------------------------------------------------------


def test_pending_aware_calculator_excludes_duplicate_discrete_points():
    tracker = PendingObservationTracker()
    calculator = ScriptedPointCalculator([0.0, 0.0, 1.0])
    wrapped = PendingAwarePointCalculator(calculator, tracker, strategy="exclude")

    first = wrapped.compute_next_points(None)
    tracker.register(first[0])
    second = wrapped.compute_next_points(None)

    np.testing.assert_array_equal(second, np.array([[1.0]]))
    assert calculator.n_calls == 3  # duplicate suggestion was retried


def test_pending_aware_calculator_without_pending_points_delegates_directly():
    tracker = PendingObservationTracker()
    calculator = ScriptedPointCalculator([0.0])
    wrapped = PendingAwarePointCalculator(calculator, tracker)

    out = wrapped.compute_next_points(None)
    np.testing.assert_array_equal(out, np.array([[0.0]]))
    assert calculator.n_calls == 1


def test_pending_aware_calculator_returns_last_suggestion_when_space_exhausted(caplog):
    tracker = PendingObservationTracker()
    tracker.register(np.array([0.0]))
    calculator = ScriptedPointCalculator([0.0])
    wrapped = PendingAwarePointCalculator(calculator, tracker, strategy="exclude", max_attempts=3)

    out = wrapped.compute_next_points(None)
    np.testing.assert_array_equal(out, np.array([[0.0]]))
    assert calculator.n_calls == 3
    assert "Could not find a point distinct" in caplog.text


def test_pending_aware_calculator_penalize_picks_unpenalized_point():
    tracker = PendingObservationTracker()
    grid = np.array([[0.0], [1.0]])
    base_calculator = GridPointCalculator(FixedAcquisition(grid, np.array([0.1, 1.0])), grid)
    wrapped = PendingAwarePointCalculator(base_calculator, tracker, strategy="penalize", penalty_sigma=0.1)

    first = wrapped.compute_next_points(None)
    np.testing.assert_array_equal(first, np.array([[1.0]]))
    tracker.register(first[0])

    second = wrapped.compute_next_points(None)
    np.testing.assert_array_equal(second, np.array([[0.0]]))
    # the base calculator keeps its original acquisition after the call
    assert isinstance(base_calculator.acquisition, FixedAcquisition)


def test_pending_aware_calculator_invalid_strategy():
    tracker = PendingObservationTracker()
    with pytest.raises(ValueError):
        PendingAwarePointCalculator(ScriptedPointCalculator([0.0]), tracker, strategy="bogus")
    with pytest.raises(ValueError):
        PendingAwarePointCalculator(ScriptedPointCalculator([0.0]), tracker, max_attempts=0)


def test_pending_aware_calculator_penalize_requires_acquisition_attribute():
    tracker = PendingObservationTracker()

    class BareCalculator(CandidatePointCalculator):
        def compute_next_points(self, loop_state, context=None):
            return np.array([[0.0]])

    with pytest.raises(ValueError):
        PendingAwarePointCalculator(BareCalculator(), tracker, strategy="penalize")


# ---------------------------------------------------------------------------
# OuterLoop.ask / tell
# ---------------------------------------------------------------------------


@pytest.fixture
def loop():
    calculator = ScriptedPointCalculator([0.0, 1.0, 2.0])
    return OuterLoop(calculator, CountingUpdater())


def test_ask_returns_traceable_id_and_registers_pending_point(loop):
    claim = loop.ask()

    assert claim["observation_id"] == "obs-0"
    np.testing.assert_array_equal(claim["X"], np.array([[0.0]]))
    observation = loop.pending_tracker.get("obs-0")
    assert observation.status == PendingStatus.PENDING
    np.testing.assert_array_equal(observation.X, np.array([0.0]))


def test_ask_does_not_update_model_or_loop_state(loop):
    loop.ask()
    loop.ask()

    assert loop.loop_state.iteration == 0
    assert loop.loop_state.results == []
    assert loop.model_updaters[0].n_updates == 0
    assert len(loop.pending_tracker.pending) == 2


def test_tell_confirms_out_of_order_and_updates_once_per_observation(loop):
    claim_0 = loop.ask()
    claim_1 = loop.ask()

    # second experiment finishes first
    confirmed = loop.tell(make_result(claim_1["X"][0], 0.9), claim_1["observation_id"])
    assert confirmed == [claim_1["observation_id"]]
    assert loop.loop_state.iteration == 1
    assert loop.model_updaters[0].n_updates == 1
    assert loop.loop_state.X.shape[0] == 1
    np.testing.assert_array_equal(loop.loop_state.Y, np.array([[0.9]]))
    assert not loop.pending_tracker.is_pending(claim_1["observation_id"])

    loop.tell(make_result(claim_0["X"][0], 0.1), claim_0["observation_id"])
    assert loop.loop_state.iteration == 2
    assert loop.model_updaters[0].n_updates == 2
    assert loop.loop_state.X.shape[0] == 2
    # only real, confirmed observations are present - no pending placeholders
    np.testing.assert_array_equal(np.sort(loop.loop_state.X.flatten()), np.array([0.0, 1.0]))


def test_tell_accepts_batches_and_single_result(loop):
    claim_0 = loop.ask()
    claim_1 = loop.ask()

    confirmed = loop.tell(
        [make_result(claim_0["X"][0]), make_result(claim_1["X"][0])],
        [claim_0["observation_id"], claim_1["observation_id"]],
    )
    assert confirmed == [claim_0["observation_id"], claim_1["observation_id"]]
    assert loop.loop_state.iteration == 1
    assert loop.model_updaters[0].n_updates == 1
    assert loop.loop_state.X.shape[0] == 2


def test_tell_rejects_conflicting_re_report(loop):
    claim = loop.ask()
    loop.tell(make_result(claim["X"][0], 0.5), claim["observation_id"])

    # the same id cannot be reported again, even with a different Y
    with pytest.raises(ValueError):
        loop.tell(make_result(claim["X"][0], 0.2), claim["observation_id"])
    # the first confirmed observation is untouched
    np.testing.assert_array_equal(loop.loop_state.Y, np.array([[0.5]]))
    assert loop.loop_state.iteration == 1


def test_tell_rejects_x_conflict_without_mutating_state(loop):
    claim = loop.ask()
    with pytest.raises(ValueError):
        loop.tell(make_result(0.42), claim["observation_id"])

    assert loop.pending_tracker.is_pending(claim["observation_id"])
    assert loop.loop_state.iteration == 0
    assert loop.model_updaters[0].n_updates == 0


def test_tell_rejects_unknown_id(loop):
    loop.ask()
    with pytest.raises(KeyError):
        loop.tell(make_result(0.0), "unknown-id")
    assert loop.loop_state.iteration == 0


def test_tell_requires_matching_id_and_result_counts(loop):
    claim = loop.ask()
    with pytest.raises(ValueError):
        loop.tell(make_result(claim["X"][0]))
    with pytest.raises(ValueError):
        loop.tell([make_result(claim["X"][0])], ["obs-0", "obs-1"])
    with pytest.raises(ValueError):
        loop.tell([])


def test_tell_batch_is_atomic_when_one_report_is_invalid(loop):
    claim_0 = loop.ask()
    claim_1 = loop.ask()
    with pytest.raises(KeyError):
        loop.tell(
            [make_result(claim_0["X"][0]), make_result(claim_1["X"][0])],
            [claim_0["observation_id"], "unknown-id"],
        )
    # nothing was confirmed and no model update happened
    assert loop.pending_tracker.is_pending(claim_0["observation_id"])
    assert loop.pending_tracker.is_pending(claim_1["observation_id"])
    assert loop.loop_state.iteration == 0
    assert loop.model_updaters[0].n_updates == 0


def test_tell_failure_and_cancel_release_placeholders_without_updates(loop):
    claim_0 = loop.ask()
    claim_1 = loop.ask()

    failed_ids = loop.tell_failure(claim_0["observation_id"])
    assert failed_ids == [claim_0["observation_id"]]
    assert loop.pending_tracker.get(claim_0["observation_id"]).status == PendingStatus.FAILED
    assert loop.loop_state.iteration == 0
    assert loop.model_updaters[0].n_updates == 0

    cancelled_ids = loop.tell_cancelled([claim_1["observation_id"]])
    assert cancelled_ids == [claim_1["observation_id"]]
    assert loop.pending_tracker.get(claim_1["observation_id"]).status == PendingStatus.CANCELLED
    assert loop.loop_state.iteration == 0
    assert loop.model_updaters[0].n_updates == 0
    assert loop.loop_state.results == []


def test_failed_point_can_be_claimed_again(loop):
    claim = loop.ask()
    loop.tell_failure(claim["observation_id"])

    new_claim = loop.ask()
    assert new_claim["observation_id"] == "obs-1"
    assert loop.pending_tracker.is_pending(new_claim["observation_id"])


def test_terminal_ids_cannot_be_re_reported(loop):
    claim = loop.ask()
    loop.tell_failure(claim["observation_id"])
    with pytest.raises(ValueError):
        loop.tell_failure(claim["observation_id"])
    with pytest.raises(ValueError):
        loop.tell_cancelled(claim["observation_id"])
    with pytest.raises(ValueError):
        loop.tell(make_result(claim["X"][0]), claim["observation_id"])


def test_tell_failure_validates_before_mutating_batch(loop):
    claim_0 = loop.ask()
    claim_1 = loop.ask()
    with pytest.raises(KeyError):
        loop.tell_failure([claim_0["observation_id"], claim_1["observation_id"], "unknown-id"])
    assert loop.pending_tracker.is_pending(claim_0["observation_id"])
    assert loop.pending_tracker.is_pending(claim_1["observation_id"])
    with pytest.raises(ValueError):
        loop.tell_cancelled([])


def test_iteration_end_event_fires_on_confirmed_observations_only(loop):
    iterations = []
    loop.iteration_end_event.append(lambda outer_loop, state: iterations.append(state.iteration))

    claim_0 = loop.ask()
    claim_1 = loop.ask()
    assert iterations == []

    loop.tell(make_result(claim_0["X"][0]), claim_0["observation_id"])
    loop.tell_failure(claim_1["observation_id"])
    assert iterations == [1]


def test_ask_wraps_existing_calculator_and_rebinds_supplied_tracker():
    tracker = PendingObservationTracker()
    calculator = PendingAwarePointCalculator(ScriptedPointCalculator([0.0]), tracker)
    loop = OuterLoop(calculator, CountingUpdater())

    # a pre-wrapped calculator must use the loop's own registry
    assert loop.candidate_point_calculator.tracker is loop.pending_tracker
    claim = loop.ask()
    assert loop.pending_tracker.is_pending(claim["observation_id"])


def test_enable_pending_point_handling_is_idempotent(loop):
    loop.enable_pending_point_handling()
    wrapped_once = loop.candidate_point_calculator
    loop.enable_pending_point_handling()
    assert loop.candidate_point_calculator is wrapped_once


def test_ask_rejects_calculator_returning_multiple_points():
    calculator = mock.create_autospec(CandidatePointCalculator)
    calculator.compute_next_points.return_value = np.array([[0.0], [1.0]])
    loop = OuterLoop(calculator, CountingUpdater())

    with pytest.raises(ValueError):
        loop.ask()
    # nothing registered on failure
    assert loop.pending_tracker.observations() == []


def test_pending_points_participate_in_candidate_exclusion(loop):
    # claim point 0; the next ask must not suggest 0 again while it is pending
    first = loop.ask()
    np.testing.assert_array_equal(first["X"], np.array([[0.0]]))
    second = loop.ask()
    np.testing.assert_array_equal(second["X"], np.array([[1.0]]))
    third = loop.ask()
    np.testing.assert_array_equal(third["X"], np.array([[2.0]]))
    assert len(loop.pending_tracker.pending) == 3


# ---------------------------------------------------------------------------
# End-to-end behaviour with a real model
# ---------------------------------------------------------------------------


def _make_gp_loop(strategy):
    space = ParameterSpace([ContinuousParameter("x", 0, 1)])
    x_init = np.linspace(0, 1, 5)[:, None]
    y_init = x_init**2
    model = SimpleGaussianProcessModel(x_init, y_init)
    calculator = SequentialPointCalculator(ModelVariance(model), GradientAcquisitionOptimizer(space))
    loop_state = create_loop_state(x_init, y_init)
    loop = OuterLoop(calculator, FixedIntervalUpdater(model), loop_state)
    loop.enable_pending_point_handling(strategy=strategy, penalty_sigma=0.05)
    return loop, model


def test_async_workflow_with_gp_model_out_of_order_confirmation():
    # The penalize strategy gives a smooth barrier that always keeps successive
    # claims distinct on a continuous space, so it is used for the full workflow.
    loop, model = _make_gp_loop("penalize")
    n_initial = model.X.shape[0]
    events = []
    loop.iteration_end_event.append(lambda outer_loop, state: events.append(state.iteration))

    # three experiments are claimed before any result comes back
    claims = [loop.ask() for _ in range(3)]
    ids = [claim["observation_id"] for claim in claims]
    assert len({claim["X"].tobytes() for claim in claims}) == 3

    # asking never creates real observations or model updates
    assert model.X.shape[0] == n_initial
    assert loop.loop_state.iteration == 0
    assert loop.loop_state.X.shape[0] == n_initial
    assert events == []

    # the last claimed experiment finishes first (out-of-order confirmation)
    loop.tell(make_result(claims[2]["X"][0], 0.31), ids[2])
    assert model.X.shape[0] == n_initial + 1
    assert loop.loop_state.iteration == 1 and events == [1]

    # a failed experiment only releases its placeholder
    loop.tell_failure(ids[1])
    assert model.X.shape[0] == n_initial + 1
    assert loop.loop_state.iteration == 1 and events == [1]
    assert loop.pending_tracker.get(ids[1]).status == PendingStatus.FAILED

    replacement = loop.ask()
    assert len(loop.pending_tracker.pending) == 2

    # the first claimed experiment now succeeds
    loop.tell(make_result(claims[0]["X"][0], 0.07), ids[0])
    assert model.X.shape[0] == n_initial + 2
    assert loop.loop_state.iteration == 2 and events == [1, 2]

    loop.tell_cancelled(replacement["observation_id"])
    assert model.X.shape[0] == n_initial + 2
    assert loop.loop_state.iteration == 2

    # exactly the two new, successful observations were added to the initial data
    assert loop.loop_state.X.shape[0] == n_initial + 2
    statuses = {obs.observation_id: obs.status for obs in loop.pending_tracker.observations()}
    assert list(statuses.values()) == [
        PendingStatus.SUCCEEDED,
        PendingStatus.FAILED,
        PendingStatus.SUCCEEDED,
        PendingStatus.CANCELLED,
    ]


def test_async_workflow_lifecycle_is_strategy_independent():
    # Exact exclusion is only guaranteed for discrete/categorical points; on a
    # continuous space a deterministic optimizer may exhaust itself, so the
    # exclude-mode E2E test asserts the lifecycle rather than point distinctness.
    loop, model = _make_gp_loop("exclude")
    n_initial = model.X.shape[0]

    claims = [loop.ask() for _ in range(3)]
    ids = [claim["observation_id"] for claim in claims]
    # every claim is registered exactly once regardless of point distinctness
    assert len(loop.pending_tracker.pending) == 3
    assert loop.loop_state.iteration == 0
    assert model.X.shape[0] == n_initial

    loop.tell(make_result(claims[2]["X"][0], 0.31), ids[2])
    loop.tell_failure(ids[1])
    loop.tell(make_result(claims[0]["X"][0], 0.07), ids[0])

    # two confirmed successes drive exactly two iterations/model updates; the
    # failure in between neither adds data nor advances the iteration counter
    assert loop.loop_state.iteration == 2
    assert loop.loop_state.X.shape[0] == n_initial + 2
    assert [obs.status for obs in loop.pending_tracker.observations()] == [
        PendingStatus.SUCCEEDED,
        PendingStatus.FAILED,
        PendingStatus.SUCCEEDED,
    ]


def test_sync_run_loop_keeps_working_alongside_pending_machinery():
    loop, model = _make_gp_loop("exclude")
    n_before = model.X.shape[0]
    loop.run_loop(lambda x: x, 2)
    assert model.X.shape[0] == n_before + 2
    assert loop.loop_state.iteration == 2
