# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from ...bayesian_optimization.acquisitions import ExpectedImprovement
from ...core.acquisition import Acquisition, acquisition_per_expected_cost
from ...core.interfaces import IModel
from ...core.loop import FixedIntervalUpdater, OuterLoop, SequentialPointCalculator
from ...core.loop.loop_state import create_loop_state_from_output_data
from ...core.optimization import AcquisitionOptimizerBase, GradientAcquisitionOptimizer
from ...core.parameter_space import ParameterSpace


class CostSensitiveBayesianOptimizationLoop(OuterLoop):
    def __init__(
        self,
        space: ParameterSpace,
        model_objective: IModel,
        model_cost: IModel,
        acquisition: Acquisition = None,
        update_interval: int = 1,
        acquisition_optimizer: AcquisitionOptimizerBase = None,
    ):
        """
        Emukit class that implements a loop for building modular cost sensitive Bayesian optimization.

        :param space: Input space where the optimization is carried out.
        :param model_objective: The model that approximates the underlying objective function
        :param model_cost: The model that approximates the cost of evaluating the objective function
        :param acquisition: The acquisition function that will be used to collect new points (default, EI).
        :param update_interval:  Number of iterations between optimization of model hyper-parameters. Defaults to 1.
        :param acquisition_optimizer: Optimizer selecting next evaluation points
                                      by maximizing acquisition.
                                      Gradient based optimizer is used if None.
                                      Defaults to None.
        """

        if acquisition is None:
            expected_improvement = ExpectedImprovement(model_objective)
            acquisition = acquisition_per_expected_cost(expected_improvement, model_cost)

        model_updater_objective = FixedIntervalUpdater(model_objective, update_interval, output_name="Y")
        model_updater_cost = FixedIntervalUpdater(model_cost, update_interval, output_name="cost")

        if acquisition_optimizer is None:
            acquisition_optimizer = GradientAcquisitionOptimizer(space)
        candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)

        # The two models may have been initialized at different inputs. The loop state merges the observations by
        # input and keeps the objective and cost of each observation separate, so each model is only trained on the
        # evaluations in which its own output was observed.
        loop_state = create_loop_state_from_output_data(
            {"Y": (model_objective.X, model_objective.Y), "cost": (model_cost.X, model_cost.Y)}
        )

        super(CostSensitiveBayesianOptimizationLoop, self).__init__(
            candidate_point_calculator, [model_updater_objective, model_updater_cost], loop_state
        )
