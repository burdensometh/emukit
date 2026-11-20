# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from time import time

import numpy as np

from ...core.loop import LoopState, OuterLoop


class Metric:
    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> None:
        pass

    def reset(self) -> None:
        pass


class MeanSquaredErrorMetric(Metric):
    """
    Mean squared error metric stored in loop state metric dictionary with key "mean_squared_error".
    """

    def __init__(self, x_test: np.ndarray, y_test: np.ndarray, name: str = "mean_squared_error"):
        """
        :param x_test: Input locations of test data
        :param y_test: Test targets
        """
        self.x_test = x_test
        self.y_test = y_test
        self.name = name

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        """
        Calculate and store mean squared error

        :param loop: Outer loop
        :param loop_state: Object containing history of the loop that we add results to
        """
        # Calculate mean squared error
        predictions = loop.model_updaters[0].model.predict(self.x_test)[0]
        mse = np.mean(np.square(self.y_test - predictions), axis=0)
        return mse


class MinimumObservedValueMetric(Metric):
    """
    The result is stored in the "metrics" dictionary in the loop state with the key "minimum_observed_value"
    """

    def __init__(self, name: str = "minimum_observed_value"):
        self.name = name

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        """
        Evaluates minimum observed value

        :param loop: Outer loop
        :param loop_state: Object containing history of the loop that we add results to
        """
        view = loop_state.output_view("Y", allow_missing=True)
        if view.n_observations == 0:
            return np.full((1,), np.nan)
        # Missing objective values are excluded rather than treated as zero
        y_min = np.min(view.Y, axis=0)
        return y_min


class TimeMetric(Metric):
    """
    Time taken between each iteration of the loop
    """

    def __init__(self, name: str = "time"):
        """
        :param name: Name of the metric. Defaults to "time"
        """
        self.start_time = None
        self.name = name

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        """
        Returns difference between time now and when the reset method was last called
        """
        time_since_start = time() - self.start_time
        # Add to metrics dictionary in loop state
        return np.array([time_since_start])

    def reset(self) -> None:
        """
        Resets the start time
        """
        self.start_time = time()


class CumulativeCostMetric(Metric):
    """

    Accumulates the cost of each function evaluation.  The result is stored in the "metrics"
    dictionary in the loop state with the key "cumulative_costs"
    """

    def __init__(self, name: str = "cumulative_costs"):
        self.name = name
        self.cumulative_costs = np.array([0.0])
        self.last_observed_iter = 0

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        """
        Computes the cumulative cost of all function evaluations after the last observed iteration

        :param loop: Outer loop
        :param loop_state: Object containing history of the loop that we add results to

        :return cumulative cost
        """
        # Only observed costs are accumulated; evaluations whose cost is missing contribute nothing
        view = loop_state.output_view("cost", allow_missing=True)
        is_new = view.result_indices >= self.last_observed_iter
        if np.any(is_new):
            self.cumulative_costs += np.cumsum(view.Y[is_new], axis=0)[-1]
        self.last_observed_iter = len(loop_state.results)

        return self.cumulative_costs

    def reset(self) -> None:
        """
        Resets the cumulative cost and the internal counter back to 0
        """
        self.last_observed_iter = 0
        self.cumulative_costs = np.array([0.0])
