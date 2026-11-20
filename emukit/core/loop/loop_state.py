# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from .user_function_result import UserFunctionResult, is_observed_output

OBJECTIVE_OUTPUT_NAME = "Y"


class OutputView(object):
    """
    Pairing of inputs and targets for a single output, aligned by observation identity.

    The loop state stores one :class:`~emukit.core.loop.user_function_result.UserFunctionResult` per observation.
    Not every observation necessarily contains every output: a measurement can provide the objective but not the
    cost, or the cost but not a constraint value. An ``OutputView`` selects the observations in which a given output
    was actually observed and exposes exactly the inputs and targets of those observations.

    ``X`` and ``Y`` always have the same number of rows, so they can be passed directly to a model's ``set_data``.
    ``mask`` refers to the full observation history: ``mask[i]`` is True if ``results[i]`` is part of this view.
    """

    def __init__(self, name: str, X: np.ndarray, Y: np.ndarray, mask: np.ndarray, result_indices: np.ndarray) -> None:
        """
        :param name: Name of the output this view belongs to
        :param X: Inputs for observations in which the output was observed. Shape: (n_observations x n_input_dims)
        :param Y: Targets for observations in which the output was observed. Shape: (n_observations x output_dim)
        :param mask: Boolean array of length (n_results,) that is True for results included in this view
        :param result_indices: Indices into the loop state results for observations included in this view
        """
        self.name = name
        self.X = X
        self.Y = Y
        self.mask = mask
        self.result_indices = result_indices

    @property
    def n_observations(self) -> int:
        """
        :return: Number of observations in which this output is available
        """
        return int(self.result_indices.shape[0])

    @property
    def n_missing(self) -> int:
        """
        :return: Number of observations in which this output is missing
        """
        return int(self.mask.shape[0] - self.n_observations)

    @property
    def output_dim(self) -> int:
        """
        :return: Number of target dimensions of this output. Zero if no observation is available.
        """
        return int(self.Y.shape[1]) if self.Y.ndim == 2 else 0

    def __len__(self) -> int:
        return self.n_observations

    def __repr__(self) -> str:
        return "OutputView(name={}, n_observations={}, output_dim={}, n_missing={})".format(
            self.name, self.n_observations, self.output_dim, self.n_missing
        )


class LoopState(object):
    """
    Contains the state of the loop, which includes a history of all function evaluations.

    Each function evaluation is stored as one :class:`UserFunctionResult` and identified by its position in
    ``results``. Outputs may be missing from individual results. Use :meth:`output_view` to obtain the inputs and
    targets actually observed for a given output, aligned by observation identity.
    """

    def __init__(self, initial_results: List[UserFunctionResult]) -> None:
        """
        :param initial_results: The function results from previous function evaluations
        """
        self.results = initial_results
        self.iteration = 0

    def update(self, results: List[UserFunctionResult]) -> None:
        """
        :param results: The latest function results since last update
        """
        if not results:
            raise ValueError("Cannot update state with empty result list.")

        self.iteration += 1
        self.results += results

    @property
    def X(self) -> np.ndarray:
        """
        :return: Function inputs for all function evaluations in a 2d array: number of points by input dimensions.
        """
        return np.array([result.X for result in self.results])

    @property
    def Y(self) -> np.ndarray:
        """
        :return: Function outputs for all function evaluations in a 2d array: number of points by output dimensions.
        :raises ValueError: if the objective is missing from some evaluation, use :meth:`output_view` instead.
        """
        if not self.results:
            return np.array([])
        view = self.output_view(OBJECTIVE_OUTPUT_NAME, allow_missing=True)
        if view.n_observations != len(self.results):
            raise ValueError(
                "The objective is missing from some function evaluations. Use output_view('Y') to obtain the "
                "inputs and targets of the evaluations in which it was observed."
            )
        return view.Y

    @property
    def output_names(self) -> List[str]:
        """
        :return: Names of all outputs that appear in the stored results. The objective "Y" comes first if present,
                 followed by extra outputs in order of first appearance.
        """
        names = []
        if any(result.Y is not None for result in self.results):
            names.append(OBJECTIVE_OUTPUT_NAME)
        for result in self.results:
            for name in result.extra_outputs:
                if name not in names:
                    names.append(name)
        return names

    def has_output(self, name: str) -> bool:
        """
        :param name: Name of an output. "Y" refers to the objective, other names to extra outputs.
        :return: Whether the output appears in the stored results
        """
        return name in self.output_names

    def output_view(self, name: str = OBJECTIVE_OUTPUT_NAME, allow_missing: bool = False) -> OutputView:
        """
        Creates a view of the inputs and targets for one output, aligned by observation identity. Observations in
        which the output is missing (``None`` or not finite) are excluded from the view and marked in its mask.

        :param name: Name of the output. "Y" refers to the objective, other names to extra outputs.
        :param allow_missing: If True, an output that no stored result mentions at all yields an empty view instead of
                              raising. This allows a model to be registered before its output has been observed.
        :return: View holding the paired inputs and targets of the observations in which the output was observed
        :raises ValueError: if no stored result contains the requested output and ``allow_missing`` is False
        """
        if not self.results:
            empty = np.empty((0, 0))
            return OutputView(name, empty, empty, np.zeros(0, dtype=bool), np.zeros(0, dtype=int))

        if name not in self.output_names and not allow_missing:
            raise ValueError("{} not found in results object. Available outputs: {}".format(name, self.output_names))

        X_rows = []
        Y_rows = []
        result_indices = []
        for i, result in enumerate(self.results):
            if not result.has_output(name):
                continue
            value = result.Y if name == OBJECTIVE_OUTPUT_NAME else result.extra_outputs[name]
            X_rows.append(result.X)
            Y_rows.append(np.atleast_1d(value))
            result_indices.append(i)

        mask = np.zeros(len(self.results), dtype=bool)
        mask[result_indices] = True

        if not X_rows:
            n_input_dims = self.X.shape[1] if self.X.ndim == 2 else 0
            return OutputView(
                name,
                np.empty((0, n_input_dims)),
                np.empty((0, 0)),
                mask,
                np.zeros(0, dtype=int),
            )

        output_dims = {row.shape[0] for row in Y_rows}
        if len(output_dims) != 1:
            raise ValueError(
                "Output {} has inconsistent dimensionality across evaluations: {}".format(name, sorted(output_dims))
            )

        return OutputView(
            name,
            np.array(X_rows),
            np.array(Y_rows),
            mask,
            np.array(result_indices, dtype=int),
        )

    def output_views(self) -> Dict[str, OutputView]:
        """
        :return: Mapping from the name of every output that appears in the stored results to its aligned view
        """
        return {name: self.output_view(name) for name in self.output_names}

    def X_for(self, name: str = OBJECTIVE_OUTPUT_NAME) -> np.ndarray:
        """
        :param name: Name of an output. "Y" refers to the objective, other names to extra outputs.
        :return: Inputs of the observations in which the output was observed. Shape: (n_observations x n_input_dims)
        """
        return self.output_view(name).X

    def Y_for(self, name: str = OBJECTIVE_OUTPUT_NAME) -> np.ndarray:
        """
        :param name: Name of an output. "Y" refers to the objective, other names to extra outputs.
        :return: Targets of the observations in which the output was observed. Shape: (n_observations x output_dim)
        """
        return self.output_view(name).Y

    def __getattr__(self, item) -> np.array:
        """
        Overriding this method allows us to customise behaviour for accessing attributes. We use this to allow arbitrary
        fields in the loop state. These are usually extra outputs from the user function such as cost, constraint values
        etc. These fields are stored in each individual "UserFunctionResult" object in the "extra_outputs" dictionary

        :param item: The name of the item to acquire. Must match the key value in the "extra_outputs" dictionary in the
                     stored "UserFunctionResults" objects
        :return: The specified output for all function evaluations in a 2d array of size (n_points x n_dimensions)
        """
        results = self.__dict__.get("results")
        if results is None:
            raise AttributeError(item)
        if item.startswith("__") and item.endswith("__"):
            # Protocol lookups such as __deepcopy__ or __setstate__ must raise AttributeError, not ValueError
            raise AttributeError(item)

        # check key appears in all results objects
        is_valid = all([item in res.extra_outputs for res in results])
        if not is_valid:
            raise ValueError("{} not found in results object".format(item))

        values = [result.extra_outputs[item] for result in results]
        if not all(is_observed_output(value) for value in values):
            raise ValueError(
                "{} is missing from some function evaluations. Use output_view('{}') to obtain the inputs and "
                "targets of the evaluations in which it was observed.".format(item, item)
            )
        return np.array(values)


def create_loop_state(x_init: Optional[np.ndarray], y_init: Optional[np.ndarray] = None, **kwargs) -> LoopState:
    """
    Creates a loop state object using the provided data.

    Outputs may be missing from individual evaluations. A missing entry is encoded either as ``None``, which may
    be used in keyword arguments, or as a non-finite value such as ``NaN``. Missing entries are not zero-filled and
    are excluded from the per-output views of the loop state.

    To describe an output that was observed at inputs different from ``x_init``, pass a tuple
    ``(x_output, y_output)`` for that keyword argument instead of an array aligned with ``x_init``.

    :param x_init: x values for initial function evaluations. Shape: (n_initial_points x n_input_dims), or None if
                   every output is provided together with its own inputs
    :param y_init: y values for initial function evaluations. Shape: (n_initial_points x n_output_dims), or None if
                   the objective was not observed
    :param kwargs: extra outputs observed from a function evaluation. Shape: (n_initial_points x n_dims) for an
                   output aligned with ``x_init``, or a tuple ``(x_output, y_output)`` with the inputs of that output
    """
    if x_init is not None:
        x_init = np.asarray(x_init)
        if x_init.ndim != 2:
            raise ValueError("x_init is expected to be 2-dimensional, actual dimensionality is {}".format(x_init.ndim))

    output_data = dict()
    if y_init is not None:
        if x_init is None:
            raise ValueError("x_init must be provided if y_init is given.")
        y_init = _as_2d_output(y_init, "y_init")
        _validate_aligned_length(x_init, y_init, "y_init")
        output_data[OBJECTIVE_OUTPUT_NAME] = (x_init, y_init)
    elif x_init is not None:
        # Inputs are known but the objective was not observed there. Keep the evaluation with a missing objective.
        output_data[OBJECTIVE_OUTPUT_NAME] = (x_init, np.full((x_init.shape[0], 0), np.nan))

    for key, value in kwargs.items():
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(
                    "Keyword argument {} must be an array or a (X, Y) tuple but received a tuple of length {}".format(
                        key, len(value)
                    )
                )
            x_output = np.asarray(value[0])
            y_output = _as_2d_output(value[1], key)
            if x_output.ndim != 2:
                raise ValueError(
                    "Expected keyword argument {} to have 2-dimensional inputs but its dimensionality is {}".format(
                        key, x_output.ndim
                    )
                )
            _validate_aligned_length(x_output, y_output, key)
            output_data[key] = (x_output, y_output)
        else:
            if x_init is None:
                raise ValueError("Keyword argument {} must be a (X, Y) tuple if x_init is not provided.".format(key))
            if value is None:
                n_rows = x_init.shape[0]
                y_output = np.full((n_rows, 0), np.nan)
            else:
                y_output = _as_2d_output(value, key)
            _validate_aligned_length(x_init, y_output, key)
            output_data[key] = (x_init, y_output)

    return create_loop_state_from_output_data(output_data)


def create_loop_state_from_output_data(output_data: Mapping[str, Tuple[np.ndarray, np.ndarray]]) -> LoopState:
    """
    Creates a loop state from inputs and targets given separately for each output.

    Each output may have been observed at its own set of inputs. Observations at identical inputs are merged into a
    single :class:`UserFunctionResult`, so an output missing at a given input remains missing rather than being
    zero-filled. Non-finite target entries are treated as missing.

    :param output_data: Mapping from output name ("Y" for the objective, other names for extra outputs) to a tuple
                        ``(X, Y)`` of shape (n_points x n_input_dims) and (n_points x output_dim). The same input may
                        appear more than once within one output; repeated occurrences are matched in order.
    :return: Loop state containing the merged observations
    """
    if OBJECTIVE_OUTPUT_NAME not in output_data:
        raise ValueError("Output data must contain the objective output '{}'.".format(OBJECTIVE_OUTPUT_NAME))

    slots: List[Dict[str, object]] = []
    lookup: Dict[Tuple[bytes, int], int] = dict()

    for name, (X, Y) in output_data.items():
        X = np.asarray(X)
        Y = _as_2d_output(Y, name)
        if X.ndim != 2:
            raise ValueError("Inputs for output {} are expected to be 2-dimensional.".format(name))
        _validate_aligned_length(X, Y, name)

        # Occurrences are counted per output, so the k-th observation of an input within one output is merged with
        # the k-th observation of the same input within every other output.
        occurrences: Dict[bytes, int] = dict()
        for x_row, y_row in zip(X, Y):
            x_bytes = np.ascontiguousarray(x_row).tobytes()
            occurrence = occurrences.get(x_bytes, 0)
            occurrences[x_bytes] = occurrence + 1
            key = (x_bytes, occurrence)

            slot_index = lookup.get(key)
            if slot_index is None:
                slot_index = len(slots)
                lookup[key] = slot_index
                slots.append({"X": np.array(x_row), "outputs": dict()})

            value = np.array(y_row) if is_observed_output(y_row) else None
            slots[slot_index]["outputs"][name] = value

    results = []
    for slot in slots:
        outputs = slot["outputs"]
        y = outputs.pop(OBJECTIVE_OUTPUT_NAME, None)
        results.append(UserFunctionResult(slot["X"], y, **outputs))

    return LoopState(results)


def _as_2d_output(value: np.ndarray, name: str) -> np.ndarray:
    """
    :param value: Array of output values
    :param name: Name of the output, used in error messages
    :return: The value as a 2d array
    """
    value = np.asarray(value)
    if value.ndim != 2:
        raise ValueError(
            "Expected output {} to have shape (n_points x n_dims) but actual dimensionality is {}".format(
                name, value.ndim
            )
        )
    return value


def _validate_aligned_length(X: np.ndarray, Y: np.ndarray, name: str) -> None:
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "Expected keyword argument {} to have length {} but actual length is {}".format(
                name, X.shape[0], Y.shape[0]
            )
        )
