# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import abc
import logging
from typing import Callable, Optional, Tuple

import numpy as np

from ..interfaces import IModel
from . import LoopState
from .loop_state import OBJECTIVE_OUTPUT_NAME, OutputView

_log = logging.getLogger(__name__)


class ModelUpdater(abc.ABC):
    """
    Updates the data of a model with the observations relevant to that model.

    A loop state can contain evaluations in which only some outputs were observed. A model trained on one output
    must only receive the evaluations in which that output was observed, paired with their own inputs. Use
    :attr:`output_name` to declare which output a model learns from and :meth:`get_output_view` to obtain the
    matched inputs and targets.
    """

    output_name: Optional[str] = None

    def get_output_view(self, loop_state: LoopState) -> OutputView:
        """
        Returns the observations in which this updater's output was observed, aligned by observation identity.

        :param loop_state: Object that contains current state of the loop
        :return: View with matched inputs and targets, a mask over the full result history and the output dimension.
                 The view is empty if the output has not been observed yet.
        """
        name = self.output_name if self.output_name is not None else OBJECTIVE_OUTPUT_NAME
        return loop_state.output_view(name, allow_missing=True)

    @abc.abstractmethod
    def update(self, loop_state: LoopState) -> None:
        """
        Updates the training data of the model. Chooses whether to update hyper-parameters and how to do it

        :param loop_state: Object that contains current state of the loop
        """
        pass


class NoopModelUpdater(ModelUpdater):
    def update(self, loop_state: LoopState) -> None:
        """
        Dummy model updater that does nothing. It can be used for example for random search

        :param loop_state: Object that contains current state of the loop
        """
        pass


class FixedIntervalUpdater(ModelUpdater):
    """Updates hyper-parameters every nth iteration, where n is defined by the user"""

    def __init__(
        self,
        model: IModel,
        interval: int = 1,
        targets_extractor_fcn: Optional[Callable] = None,
        output_name: Optional[str] = None,
    ) -> None:
        """
        :param model: Emukit emulator model
        :param interval: Number of function evaluations between optimizing model hyper-parameters
        :param targets_extractor_fcn: A function that takes in loop state and returns the training targets, or a
                                      tuple ``(X, Y)`` of matched inputs and targets. If it returns only targets they
                                      must be aligned with ``loop_state.X``. Defaults to using the output selected by
                                      ``output_name``, including only the evaluations in which it was observed.
        :param output_name: Name of the output the model is trained on. "Y" refers to the objective, other names to
                            extra outputs such as a constraint or cost. Defaults to "Y". Ignored if
                            ``targets_extractor_fcn`` is given.
        """
        self.model = model
        self.interval = interval
        self.targets_extractor_fcn = targets_extractor_fcn
        self.output_name = output_name

    def _get_training_data(self, loop_state: LoopState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collects the inputs and targets the model should be trained on. Both parts are aligned to the same
        observations, so no evaluation whose output is missing enters the training set.

        :param loop_state: Object that contains current state of the loop
        :return: Tuple of inputs and targets with matching number of rows
        """
        if self.targets_extractor_fcn is None:
            view = self.get_output_view(loop_state)
            return view.X, view.Y

        extracted = self.targets_extractor_fcn(loop_state)
        if isinstance(extracted, OutputView):
            return extracted.X, extracted.Y
        if isinstance(extracted, tuple):
            if len(extracted) != 2:
                raise ValueError(
                    "Targets extractor must return a targets array, an (X, Y) tuple or an OutputView but returned a "
                    "tuple of length {}".format(len(extracted))
                )
            X, targets = extracted
            X = np.asarray(X)
            targets = np.asarray(targets)
            if targets.shape[0] != X.shape[0]:
                raise ValueError(
                    "Targets extractor returned {} targets for {} inputs. Targets must be paired with the inputs "
                    "they belong to.".format(targets.shape[0], X.shape[0])
                )
            return X, targets

        targets = np.asarray(extracted)
        X = loop_state.X
        if isinstance(X, np.ndarray) and targets.ndim >= 1 and targets.shape[0] != X.shape[0]:
            raise ValueError(
                "Targets extractor returned {} targets for {} inputs. Targets must be aligned with the inputs they "
                "belong to; use output_name to train the model on an output that is missing from some evaluations, "
                "or return an (X, Y) tuple.".format(targets.shape[0], X.shape[0])
            )
        return X, targets

    def update(self, loop_state: LoopState) -> None:
        """
        :param loop_state: Object that contains current state of the loop
        """
        view_name = self.output_name if self.output_name is not None else OBJECTIVE_OUTPUT_NAME
        if self.targets_extractor_fcn is None:
            view = self.get_output_view(loop_state)
            if view.n_observations == 0:
                _log.info("No observations of output %r yet, skipping model update", view.name)
                return

        X, targets = self._get_training_data(loop_state)
        _validate_finite_targets(targets, view_name)
        self.model.set_data(X, targets)
        if (loop_state.iteration % self.interval) == 0:
            _log.info("Updating parameters of the model")
            self.model.optimize()


def _validate_finite_targets(targets: np.ndarray, output_name: Optional[str]) -> None:
    """
    Guards against training on missing values that were encoded as NaN. Missing values must be excluded from the
    training data rather than passed to the model.

    :param targets: Training targets about to be passed to a model
    :param output_name: Name of the output the targets belong to, used in the error message
    """
    if isinstance(targets, np.ndarray) and np.issubdtype(targets.dtype, np.number) and not np.all(np.isfinite(targets)):
        raise ValueError(
            "Training targets for output {!r} contain non-finite values. Missing outputs must be excluded from the "
            "training data, for example by selecting the observations with output_name, rather than trained on as "
            "NaN.".format(output_name if output_name is not None else OBJECTIVE_OUTPUT_NAME)
        )
