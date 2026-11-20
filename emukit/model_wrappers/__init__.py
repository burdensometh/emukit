# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Importing emukit.model_wrappers should succeed without GPy. Accessing GPy wrappers
# should raise an informative ImportError
# pointing users to install the optional extra: `pip install emukit[gpy]`.
# By doing it this way, we avoid breaking minimal installs of Emukit.

from importlib import util as _importlib_util

# Always expose HeteroscedasticGaussianProcessModel (pure numpy implementation)
from .heteroscedastic_gp_model import HeteroscedasticGaussianProcessModel  # noqa: F401

# Always expose SimpleGaussianProcessModel (pure numpy implementation)
from .simple_gp_model import SimpleGaussianProcessModel  # noqa: F401

if _importlib_util.find_spec("GPy") is not None:  # GPy available
    from .gpy_model_wrappers import (  # noqa: F401
        GPyHeteroscedasticModelWrapper,
        GPyModelWrapper,
        GPyMultiOutputWrapper,
        create_heteroscedastic_gpy_model,
    )
else:

    class _GPyMissingBase:
        _error_msg = (
            "GPy is not installed. Install optional dependency with 'pip install emukit[gpy]' " "to use {name}."
        )

        def __init__(self, *args, **kwargs):  # pragma: no cover - exercised in minimal installs
            raise ImportError(self._error_msg.format(name=self.__class__.__name__))

    class GPyModelWrapper(_GPyMissingBase):  # pragma: no cover - minimal installs
        """Placeholder for GPyModelWrapper. Requires `emukit[gpy]` extra."""

    class GPyMultiOutputWrapper(_GPyMissingBase):  # pragma: no cover - minimal installs
        """Placeholder for GPyMultiOutputWrapper. Requires `emukit[gpy]` extra."""

    class GPyHeteroscedasticModelWrapper(_GPyMissingBase):  # pragma: no cover - minimal installs
        """Placeholder for GPyHeteroscedasticModelWrapper. Requires `emukit[gpy]` extra."""

    def create_heteroscedastic_gpy_model(*args, **kwargs):  # pragma: no cover - minimal installs
        raise ImportError(
            "GPy is not installed. Install optional dependency with 'pip install emukit[gpy]' "
            "to use create_heteroscedastic_gpy_model."
        )


__all__ = [
    "GPyModelWrapper",
    "GPyMultiOutputWrapper",
    "GPyHeteroscedasticModelWrapper",
    "create_heteroscedastic_gpy_model",
    "SimpleGaussianProcessModel",
    "HeteroscedasticGaussianProcessModel",
]
