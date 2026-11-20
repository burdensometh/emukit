# Changelog
All notable changes to Emukit will be documented in this file.

## [Unreleased]
- Add `RepeatPointUpdater` and configurable point identifiers (`ExactPointIdentifier`, `FunctionPointIdentifier`) that aggregate repeated noisy measurements at the same experimental point into sample means, unbiased sample variances and per-point mean variances (`s^2 / n`), retaining the raw measurements in `LoopState`. Points with fewer than two measurements use a declared noise prior.
- Add `IModelWithPerPointNoise` interface for models accepting per-point observation noise variances.
- Add `HeteroscedasticGaussianProcessModel` (pure numpy) and `GPyHeteroscedasticModelWrapper` / `create_heteroscedastic_gpy_model` (GPy) models that train on aggregated means with their true mean uncertainty as per-point noise.

## [0.5.1]
- Use scipy for model free designs (#483)
- Fix division by 0 warning (#481)
- Document release process (#479)

## [0.5.0]
- Upgrade codebase to allow `numpy` versions greater than 2.0, while maintaining backwards compatibility. `GPy` remains pinned to `numpy<2.0` due to upstream constraints, but core Emukit functionality is now independent of it. Users can now install and use Emukit core API with `numpy>=2.0`.
- Packaging: Adopt PEP 621 metadata in `pyproject.toml`; dynamic version from `emukit.__version__`.
- Packaging: Introduced setuptools extras (`gpy`, `bnn`, `sklearn`, `docs`, `examples`, `tests`, `dev`).
- CI: Workflows now install extras via `pip install -e .[tests]` (and `[tests,gpy]`) instead of requirements files.
- Docs: Updated installation guide, README, CONTRIBUTING to prefer extras over legacy `requirements/` files.
- Tests: Documented pytest marker and optional dependency usage.
- Docs build: Use `docs` extra (includes GPy).
- Maintenance: Legacy requirements files retained temporarily for reference.

## [0.4.11]
- Various bugfixes, including installation on Windows
- Updated copyright info

## [0.4.10]
- Wrapper for SKlearn Guassian process
- Black and isort formatting
- Brownian motion quadrature kernel and product embedding
- ProductMatern52 quadrature kernel embedding 
- Multiple improvements to quadrature integration measures
- QuadratureProductKernel base class
- Doc improvements
- Bug fixes, including scipy compatibility fixes

## [0.4.9]
- Update to newest version of GPy, which shall fix installation issues
- Mean Plug-in Expected Improvement
- Square root warping for BQ and WSABI
- Improved validation of categorical variables
- Updates and fixes of Local Penalization acquisition function
- bug fixes
- doc fixes

## [0.4.8]
- Added sobol initial design
- BanditParameter
- Boolean operations for stopping conditions
- Preferential Bayesian optimization example
- MUMBO acquisition function
- Revised dependecies' versions requirements
- Bug fixes
- Doc fixes

## [0.4.7]
- Added simple GP model for examples
- Bayesian optimization with unknown constraints
- Removed dependency on libomp
- Max value entropy search acquisition function
- Multi point expected improvement acquisition function
- Moved model free designs to core
- Profet implementation
- Added citation info
- QRBF for uniform and Gaussian measures
- uncertainty sampling acquisition for bq
- Bayesian Monte Carlo
- Bugfixes
- Doc fixes

## [0.4.6]
- Added support for inequality constraints
- Fabolas as an example
- Bugfixes

## [0.4.5]
- Confirmed support for Python 3.7
- Removed dependency on GPyOpt
- Implemented generic IntegratedHyperParameterAcquisition
- Added notebooks validation automation
- Random baseline for benchmarking
- Implemented a range of discrete optimizers
- Uniform measure and mutual information acquisition for BQ
- Added sample_uniform method to parameter space and individual parameter types
- Improved unit test coverage
- Various fixes in code, comments and notebooks
