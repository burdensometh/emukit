# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from numpy.testing import assert_array_equal

from emukit.core.loop.loop_state import (
    LoopState,
    create_loop_state,
    create_loop_state_from_output_data,
)
from emukit.core.loop.user_function_result import UserFunctionResult


def test_create_loop_state():
    x_init = np.array([[1], [2], [3]])
    y_init = np.array([[4], [5], [6]])

    loop_state = create_loop_state(x_init, y_init)

    assert_array_equal(loop_state.X, x_init)
    assert_array_equal(loop_state.Y, y_init)
    assert loop_state.iteration == 0


def test_create_loop_state_with_cost():
    x_init = np.array([[1], [2], [3]])
    y_init = np.array([[4], [5], [6]])
    cost = np.array([[5], [2], [0]])

    loop_state = create_loop_state(x_init, y_init, cost=cost)

    assert_array_equal(loop_state.X, x_init)
    assert_array_equal(loop_state.Y, y_init)
    assert_array_equal(loop_state.cost, cost)
    assert loop_state.iteration == 0


def test_create_loop_error():
    x_init = np.array([[1], [2], [3]])
    y_init = np.array([[4], [5]])

    with pytest.raises(ValueError):
        create_loop_state(x_init, y_init)


def test_loop_state_update():
    x = np.array([[1], [2], [3], [4]])
    y = np.array([[4], [5], [6], [7]])

    loop_state = create_loop_state(x[:3, :], y[:3, :])
    step_result = UserFunctionResult(x[3, :], y[3, :])
    loop_state.update([step_result])

    assert_array_equal(loop_state.X, x)
    assert_array_equal(loop_state.Y, y)
    assert loop_state.iteration == 1


def test_loop_state_update_error():
    x = np.array([[1], [2], [3], [4]])
    y = np.array([[4], [5], [6], [7]])

    loop_state = create_loop_state(x[:3, :], y[:3, :])
    with pytest.raises(ValueError):
        loop_state.update(None)

    with pytest.raises(ValueError):
        loop_state.update([])


def test_create_loop_state_wrong_size_input_fails():
    x = np.array([[1], [2], [3], [4]])
    y = np.array([[4], [5], [6], [7]])
    c = np.array([[4], [5], [6], [7], [8]])

    with pytest.raises(ValueError):
        create_loop_state(x, y, cost=c)


def test_output_view_aligns_inputs_and_targets_per_output():
    x_init = np.array([[1], [2], [3]])
    y_init = np.array([[10], [20], [30]])
    cost = np.array([[1], [np.nan], [3]])

    loop_state = create_loop_state(x_init, y_init, cost=cost)

    cost_view = loop_state.output_view("cost")
    assert_array_equal(cost_view.X, np.array([[1], [3]]))
    assert_array_equal(cost_view.Y, np.array([[1], [3]]))
    assert_array_equal(cost_view.mask, np.array([True, False, True]))
    assert_array_equal(cost_view.result_indices, np.array([0, 2]))
    assert cost_view.n_observations == 2
    assert cost_view.n_missing == 1
    assert cost_view.output_dim == 1

    y_view = loop_state.output_view("Y")
    assert_array_equal(y_view.X, x_init)
    assert_array_equal(y_view.Y, y_init)
    assert np.all(y_view.mask)


def test_output_view_excludes_none_values():
    results = [
        UserFunctionResult(np.array([1.0]), np.array([10.0]), cost=np.array([1.0])),
        UserFunctionResult(np.array([2.0]), None, cost=np.array([2.0])),
        UserFunctionResult(np.array([3.0]), np.array([30.0])),
    ]
    loop_state = LoopState(results)

    assert_array_equal(loop_state.output_view("Y").X, np.array([[1.0], [3.0]]))
    assert_array_equal(loop_state.output_view("cost").X, np.array([[1.0], [2.0]]))


def test_loop_state_Y_raises_when_objective_missing():
    results = [
        UserFunctionResult(np.array([1.0]), np.array([10.0])),
        UserFunctionResult(np.array([2.0]), None),
    ]
    loop_state = LoopState(results)

    with pytest.raises(ValueError):
        loop_state.Y

    assert_array_equal(loop_state.Y_for("Y"), np.array([[10.0]]))


def test_loop_state_attribute_raises_when_extra_output_missing():
    results = [
        UserFunctionResult(np.array([1.0]), np.array([10.0]), cost=np.array([1.0])),
        UserFunctionResult(np.array([2.0]), np.array([20.0]), cost=None),
    ]
    loop_state = LoopState(results)

    with pytest.raises(ValueError):
        loop_state.cost

    assert_array_equal(loop_state.Y_for("cost"), np.array([[1.0]]))


def test_output_view_missing_output_raises():
    loop_state = create_loop_state(np.array([[1], [2]]), np.array([[1], [2]]))

    with pytest.raises(ValueError):
        loop_state.output_view("cost")


def test_output_dim_is_reported_for_multi_dimensional_output():
    x_init = np.array([[1], [2], [3]])
    y_init = np.array([[1, 2], [3, 4], [5, 6]])

    loop_state = create_loop_state(x_init, y_init)

    assert loop_state.output_view("Y").output_dim == 2


def test_create_loop_state_with_partial_initial_outputs():
    x_init = np.array([[1], [2], [3]])
    y_init = np.array([[10], [np.nan], [30]])
    cost = np.array([[1], [2], [np.nan]])

    loop_state = create_loop_state(x_init, y_init, cost=cost)

    assert_array_equal(loop_state.X, x_init)
    assert_array_equal(loop_state.Y_for("Y"), np.array([[10], [30]]))
    assert_array_equal(loop_state.X_for("Y"), np.array([[1], [3]]))
    assert_array_equal(loop_state.Y_for("cost"), np.array([[1], [2]]))
    assert_array_equal(loop_state.X_for("cost"), np.array([[1], [2]]))


def test_create_loop_state_with_different_initial_points_per_output():
    x_objective = np.array([[1.0], [2.0]])
    y_objective = np.array([[10.0], [20.0]])
    x_constraint = np.array([[2.0], [3.0]])
    y_constraint = np.array([[-1.0], [1.0]])

    loop_state = create_loop_state_from_output_data(
        {"Y": (x_objective, y_objective), "Y_constraint": (x_constraint, y_constraint)}
    )

    # Point 2 is shared between both outputs, points 1 and 3 are observed only by one output
    assert_array_equal(loop_state.X, np.array([[1.0], [2.0], [3.0]]))
    assert_array_equal(loop_state.Y_for("Y"), np.array([[10.0], [20.0]]))
    assert_array_equal(loop_state.X_for("Y"), x_objective)
    assert_array_equal(loop_state.Y_for("Y_constraint"), np.array([[-1.0], [1.0]]))
    assert_array_equal(loop_state.X_for("Y_constraint"), x_constraint)
    assert_array_equal(loop_state.output_view("Y").mask, np.array([True, True, False]))
    assert_array_equal(loop_state.output_view("Y_constraint").mask, np.array([False, True, True]))


def test_create_loop_state_from_output_data_keeps_repeated_points_separate():
    x = np.array([[1.0], [1.0]])
    y = np.array([[10.0], [11.0]])
    cost = np.array([[1.0], [2.0]])

    loop_state = create_loop_state_from_output_data({"Y": (x, y), "cost": (x, cost)})

    assert len(loop_state.results) == 2
    assert_array_equal(loop_state.Y, y)
    assert_array_equal(loop_state.cost, cost)


def test_create_loop_state_from_output_data_requires_objective():
    with pytest.raises(ValueError):
        create_loop_state_from_output_data({"cost": (np.array([[1.0]]), np.array([[1.0]]))})


def test_loop_state_update_with_partial_outputs():
    x_init = np.array([[1.0], [2.0]])
    y_init = np.array([[10.0], [20.0]])
    loop_state = create_loop_state(x_init, y_init, cost=np.array([[1.0], [2.0]]))

    # The next evaluation returns only a cost, the objective is not observed
    loop_state.update([UserFunctionResult(np.array([3.0]), None, cost=np.array([3.0]))])

    assert_array_equal(loop_state.output_view("Y").X, x_init)
    assert_array_equal(loop_state.output_view("cost").X, np.array([[1.0], [2.0], [3.0]]))


def test_create_loop_state_with_unsorted_different_initial_points_per_output():
    # Objective points are not a prefix of the merged order; alignment must follow the input values
    loop_state = create_loop_state_from_output_data(
        {"Y": (np.array([[3.0], [1.0]]), np.array([[30.0], [10.0]])), "cost": (np.array([[2.0]]), np.array([[2.0]]))}
    )

    assert_array_equal(loop_state.X, np.array([[3.0], [1.0], [2.0]]))
    assert_array_equal(loop_state.X_for("Y"), np.array([[3.0], [1.0]]))
    assert_array_equal(loop_state.Y_for("Y"), np.array([[30.0], [10.0]]))
    assert_array_equal(loop_state.X_for("cost"), np.array([[2.0]]))
    assert_array_equal(loop_state.output_view("Y").mask, np.array([True, True, False]))


def test_loop_state_attribute_access_requires_output_in_all_results():
    # Legacy attribute access returns an array only if the output is present everywhere; use output_view otherwise
    loop_state = create_loop_state_from_output_data(
        {"Y": (np.array([[1.0], [2.0]]), np.array([[10.0], [20.0]])), "cost": (np.array([[2.0]]), np.array([[5.0]]))}
    )

    with pytest.raises(ValueError):
        loop_state.cost

    assert_array_equal(loop_state.Y_for("cost"), np.array([[5.0]]))
