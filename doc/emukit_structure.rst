Emukit Structure
================

.. contents::
    :local:

We have structured Emukit in a way that reflects our thinking about the different decision making processes it performs.
Bayesian optimization, experimental design and Bayesian quadrature are all decision making processes that follow a 
similar pattern. 
We think of all these decision making processes as implementations of a common abstract loop::

    while stopping condition is not met:
        optimize acquisition function
        evaluate user function
        update model with new observation

We have built Emukit in a modular way such that each fundamental component of this loop can be swapped out. 
If you are a machine learning researcher interested implemented in your method in Emukit - this is where to find out
which parts you need to implement!

Loop
____
The ``emukit.core.loop.OuterLoop`` class is the abstract loop where the different components come together.
There are more specific loops for Bayesian optimization and experimental design that construct some of the component 
parts for you.


Model
_____
All ``Emukit`` loops need a probabilistic model of the underlying system.
Emukit does not provide functionality to build models as there are already many good modelling frameworks available in python.
Instead, we provide a way of interfacing third part modelling libraries with Emukit. 
We already provide a wrapper for using a model created with ``GPy``.
For instructions on how to include your own model please see `this notebook <notebooks/Emukit-tutorial-custom-model.ipynb>`_.

Different models and modelling frameworks will provide different functionality. 
For instance a Gaussian process will usually have derivatives of the predictions available but random forests will not. 
These different functionalities are represented by a set of interfaces which a model implements. 
The basic interface that all models must implement is ``IModel``, which implements functionality to make predictions and
update the model but a model may implement any number of other interfaces such as ``IDifferentiable`` which indicates a
model has prediction derivatives available.

Candidate Point Calculator
__________________________
This class decides which point to evaluate next.
The simplest implementation, ``SequentialPointCalculator``, collects one point at a time by finding where the acquisition is a maximum
by applying the acquisition optimizer to the acquisition function.
More complex implementations will enable batches of points to be collected so that the user function can be evaluated
in parallel.

Acquisition
___________
The acquisition is a heuristic quantification of how valuable collecting a future point might be.
It is used by the candidate point calculator to decide which point(s) to collect next.

Acquisition Optimizer
_____________________
The ``AcquisitionOptimizer`` optimizes the acquisition function to find the point at which the acquisition is a maximum.
This will use the acquisition function gradients if they are available. 
If gradients of the acquisition function are not available it will either estimate them numerically or use a gradient 
free optimizer.

User Function
_____________
This is the function that we are trying to reason about. 
It can be either evaluated by the user or it can be passed into the loop and evaluated by Emukit.

Model Updater
_____________
The ``ModelUpdater`` class updates the model with new training data after a new point is observed and optimizes any
hyper-parameters of the model.
It can decide whether hyper-parameters need updating based on some internal logic.

An output of interest, such as the objective, a constraint or the cost of an evaluation, is not necessarily observed
in every evaluation. The ``LoopState`` stores one ``UserFunctionResult`` per evaluation and keeps missing outputs
missing: an output may be passed as ``None`` or as ``NaN``, and is never zero-filled. ``LoopState.output_view(name)``
returns an ``OutputView`` with the inputs and targets of the evaluations in which that output was observed, aligned
by evaluation, together with a mask over the full history and the output dimensionality. ``X`` and ``Y`` of an
``OutputView`` always have the same number of rows and can be passed directly to a model's ``set_data``.

``FixedIntervalUpdater`` takes an ``output_name`` and trains its model only on the observations of that output. This
lets the objective model, the constraint model and the cost model each receive their own paired data, even when they
were initialized at different inputs or when an output is first observed after the loop has started. The
``create_loop_state_from_output_data`` function builds a loop state from per-output ``(X, Y)`` pairs and merges
evaluations at identical inputs.


Stopping Condition
__________________
The ``StoppingCondition`` class chooses when we should stop collecting points.
The most commonly used example is to stop when a set number of iterations have been reached.



