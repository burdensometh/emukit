# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional

import numpy as np


class UserFunctionResult(object):
    """
    A class that records the inputs, outputs and meta-data of an evaluation of the user function.

    An output can be missing from a result. This happens when a measurement provides only a subset of the outputs
    of interest, for example when the objective value is observed but the cost of the evaluation is not. A missing
    output is represented either by passing ``None`` or by omitting the corresponding key word argument. Values that
    are not finite (``NaN`` or infinity) are also interpreted as missing by
    :class:`~emukit.core.loop.loop_state.LoopState`.
    """

    def __init__(self, X: np.ndarray, Y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        :param X: Function input. Shape: (function input dimension,)
        :param Y: Function output(s). Shape: (function output dimension,), or None if the objective was not observed
        :param kwargs: Extra outputs of the UserFunction to store. Shape: (extra output dimension,).
                       A value of None indicates that this output was not observed in this evaluation.
        """
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError("x is expected to be 1-dimensional, actual dimensionality is {}".format(X.ndim))

        if Y is not None:
            Y = np.asarray(Y)
            if Y.ndim != 1:
                raise ValueError("y is expected to be 1-dimensional, actual dimensionality is {}".format(Y.ndim))

        extra_outputs = dict()
        for key, val in kwargs.items():
            if val is None:
                extra_outputs[key] = None
                continue
            val = np.asarray(val)
            if val.ndim != 1:
                raise ValueError("Key word arguments must be 1-dimensional but {} is {}d".format(key, val.ndim))
            extra_outputs[key] = val

        self.extra_outputs = extra_outputs
        self.X = X
        self.Y = Y

    @property
    def available_outputs(self) -> List[str]:
        """
        :return: Names of the outputs that were observed in this evaluation, in the order "Y" followed by the
                 names of the extra outputs. Outputs that are None or non-finite are excluded.
        """
        names = []
        if is_observed_output(self.Y):
            names.append("Y")
        for key, val in self.extra_outputs.items():
            if is_observed_output(val):
                names.append(key)
        return names

    def has_output(self, name: str) -> bool:
        """
        :param name: Name of the output to check. "Y" refers to the objective, other names to extra outputs.
        :return: Whether a finite value for the given output was observed in this evaluation
        """
        return name in self.available_outputs

    def __eq__(self, other):
        if not isinstance(other, UserFunctionResult):
            return False
        if not np.array_equal(self.X, other.X):
            return False
        if not _values_equal(self.Y, other.Y):
            return False
        if set(self.extra_outputs) != set(other.extra_outputs):
            return False
        return all(_values_equal(self.extra_outputs[key], other.extra_outputs[key]) for key in self.extra_outputs)

    def __getattr__(self, item):
        """
        Allow extra output values to be accessed as an attribute

        :param item: The name of the extra output to be accessed
        :return: The value of the extra output
        """
        extra_outputs = self.__dict__.get("extra_outputs", {})
        if item in extra_outputs:
            return extra_outputs[item]
        raise AttributeError("{} has no attribute {}".format(type(self).__name__, item))

    def __repr__(self):
        return "UserFunctionResult(X: {}, Y: {}, extra_outputs: {})".format(self.X, self.Y, self.extra_outputs)


def is_observed_output(value: Optional[np.ndarray]) -> bool:
    """
    :param value: Value of one output in one result, or None
    :return: Whether the value is present and finite
    """
    if value is None:
        return False
    value = np.asarray(value)
    if value.size == 0:
        return False
    return bool(np.all(np.isfinite(value)))


def _values_equal(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return np.array_equal(np.asarray(left), np.asarray(right))
