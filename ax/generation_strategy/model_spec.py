#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ax.adapter.base import Adapter
from ax.adapter.cross_validation import (
    compute_diagnostics,
    cross_validate,
    CVDiagnostics,
    CVResult,
    get_fit_and_std_quality_and_generalization_dict,
)
from ax.adapter.registry import ModelRegistryBase
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.exceptions.core import UserInputError
from ax.utils.common.base import SortableBase
from ax.utils.common.kwargs import consolidate_kwargs, get_function_argument_names
from ax.utils.common.serialization import SerializationMixin
from pyre_extensions import none_throws


TModelFactory = Callable[..., Adapter]


class GeneratorSpecJSONEncoder(json.JSONEncoder):
    """Generic encoder to avoid JSON errors in GeneratorSpec.__repr__"""

    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def default(self, o: Any) -> str:
        return repr(o)


@dataclass
class GeneratorSpec(SortableBase, SerializationMixin):
    model_enum: ModelRegistryBase
    # Kwargs to pass into the `Model` + `Adapter` constructors in
    # `ModelRegistryBase.__call__`.
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    # Kwargs to pass to `Adapter.gen`.
    model_gen_kwargs: dict[str, Any] = field(default_factory=dict)
    # Kwargs to pass to `cross_validate`.
    model_cv_kwargs: dict[str, Any] = field(default_factory=dict)
    # An optional override for the model key. Each `GeneratorSpec` in a
    # `GenerationNode` must have a unique key to ensure identifiability.
    model_key_override: str | None = None

    # Fitted model, constructed using specified `model_kwargs` and `Data`
    # on `GeneratorSpec.fit`
    _fitted_model: Adapter | None = None

    # Stored cross validation results set in cross validate.
    _cv_results: list[CVResult] | None = None

    # Stored cross validation diagnostics set in cross validate.
    _diagnostics: CVDiagnostics | None = None

    # Stored to check if the CV result & diagnostic cache is safe to reuse.
    _last_cv_kwargs: dict[str, Any] | None = None

    # Stored to check if the model can be safely updated in fit.
    _last_fit_arg_ids: dict[str, int] | None = None

    def __post_init__(self) -> None:
        self.model_kwargs = self.model_kwargs or {}
        self.model_gen_kwargs = self.model_gen_kwargs or {}
        self.model_cv_kwargs = self.model_cv_kwargs or {}

    @property
    def fitted_model(self) -> Adapter:
        """Returns the fitted Ax model, asserting fit() was called"""
        self._assert_fitted()
        return none_throws(self._fitted_model)

    @property
    def fixed_features(self) -> ObservationFeatures | None:
        """
        Fixed generation features to pass into the Model's `.gen` function.
        """
        return self.model_gen_kwargs.get("fixed_features", None)

    @fixed_features.setter
    def fixed_features(self, value: ObservationFeatures | None) -> None:
        """
        Fixed generation features to pass into the Model's `.gen` function.
        """
        self.model_gen_kwargs["fixed_features"] = value

    @property
    def model_key(self) -> str:
        """Key string to identify the model used by this ``GeneratorSpec``."""
        if self.model_key_override is not None:
            return self.model_key_override
        else:
            return self.model_enum.value

    def fit(
        self,
        experiment: Experiment,
        data: Data | None = None,
        **model_kwargs: Any,
    ) -> None:
        """Fits the specified model on the given experiment + data using the
        model kwargs set on the model spec, alongside any passed down as
        kwargs to this function (local kwargs take precedent)
        """
        # unset any cross validation cache
        self._cv_results, self._diagnostics = None, None
        # NOTE: It's important to copy `self.model_kwargs` here to avoid actually
        # adding contents of `model_kwargs` passed to this method, to
        # `self.model_kwargs`.
        combined_model_kwargs = {**self.model_kwargs, **model_kwargs}
        if self._fitted_model is not None and self._safe_to_update(
            experiment=experiment, combined_model_kwargs=combined_model_kwargs
        ):
            # Update the data on the adapter and call `_fit`.
            # This will skip model fitting if the data has not changed.
            observations, search_space = self.fitted_model._process_and_transform_data(
                experiment=experiment, data=data
            )
            self.fitted_model._fit_if_implemented(
                search_space=search_space, observations=observations, time_so_far=0.0
            )

        else:
            # Fit from scratch.
            self._fitted_model = self.model_enum(
                experiment=experiment,
                data=data,
                **combined_model_kwargs,
            )
            self._last_fit_arg_ids = self._get_fit_arg_ids(
                experiment=experiment, combined_model_kwargs=combined_model_kwargs
            )

    def cross_validate(
        self,
        model_cv_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[CVResult] | None, CVDiagnostics | None]:
        """
        Call cross_validate, compute_diagnostics and cache the results.
        If the model cannot be cross validated, warn and return None.

        NOTE: If there are cached results, and the cache was computed using
        the same kwargs, this will return the cached results.

        Args:
            model_cv_kwargs: Optional kwargs to pass into `cross_validate` call.
                These are combined with `self.model_cv_kwargs`, with the
                `model_cv_kwargs` taking precedence over `self.model_cv_kwargs`.

        Returns:
            A tuple of CV results (observed vs predicted values) and the
            corresponding diagnostics.
        """
        cv_kwargs = {**self.model_cv_kwargs, **(model_cv_kwargs or {})}
        if (
            self._cv_results is not None
            and self._diagnostics is not None
            and cv_kwargs == self._last_cv_kwargs
        ):
            return self._cv_results, self._diagnostics

        self._assert_fitted()
        try:
            self._cv_results = cross_validate(model=self.fitted_model, **cv_kwargs)
        except NotImplementedError:
            warnings.warn(
                f"{self.model_enum.value} cannot be cross validated", stacklevel=2
            )
            return None, None

        self._diagnostics = compute_diagnostics(self._cv_results)
        self._last_cv_kwargs = cv_kwargs
        return self._cv_results, self._diagnostics

    @property
    def cv_results(self) -> list[CVResult] | None:
        """
        Cached CV results from `self.cross_validate()`
        if it has been successfully called
        """
        return self._cv_results

    @property
    def diagnostics(self) -> CVDiagnostics | None:
        """
        Cached CV diagnostics from `self.cross_validate()`
        if it has been successfully called
        """
        return self._diagnostics

    def gen(self, **model_gen_kwargs: Any) -> GeneratorRun:
        """Generates candidates from the fitted model, using the model gen
        kwargs set on the model spec, alongside any passed as kwargs
        to this function (local kwargs take precedent)

        NOTE: Model must have been fit prior to calling gen()

        Args:
            n: Integer representing how many arms should be in the generator run
                produced by this method. NOTE: Some underlying models may ignore
                the ``n`` and produce a model-determined number of arms. In that
                case this method will also output a generator run with number of
                arms that can differ from ``n``.
            pending_observations: A map from metric name to pending
                observations for that metric, used by some models to avoid
                resuggesting points that are currently being evaluated.
        """
        fitted_model = self.fitted_model
        model_gen_kwargs = consolidate_kwargs(
            kwargs_iterable=[self.model_gen_kwargs, model_gen_kwargs],
            keywords=get_function_argument_names(fitted_model.gen),
        )
        # copy to ensure there is no in-place modification
        model_gen_kwargs = deepcopy(model_gen_kwargs)
        generator_run = fitted_model.gen(**model_gen_kwargs)
        fit_and_std_quality_and_generalization_dict = (
            get_fit_and_std_quality_and_generalization_dict(
                fitted_adapter=self.fitted_model,
            )
        )
        generator_run._gen_metadata = (
            {} if generator_run.gen_metadata is None else generator_run.gen_metadata
        )
        generator_run._gen_metadata.update(
            **fit_and_std_quality_and_generalization_dict
        )
        return generator_run

    def copy(self) -> GeneratorSpec:
        """`GeneratorSpec` is both a spec and an object that performs actions.
        Copying is useful to avoid changes to a singleton model spec.
        """
        return self.__class__(
            model_enum=self.model_enum,
            model_kwargs=deepcopy(self.model_kwargs),
            model_gen_kwargs=deepcopy(self.model_gen_kwargs),
            model_cv_kwargs=deepcopy(self.model_cv_kwargs),
            model_key_override=self.model_key_override,
        )

    def _safe_to_update(
        self,
        experiment: Experiment,
        combined_model_kwargs: dict[str, Any],
    ) -> bool:
        """Checks if the object id of any of the non-data fit arguments has changed.

        This is a cheap way of checking that we're attempting to re-fit the same
        model for the same experiment, which is a very reasonable expectation
        since this all happens on the same `GeneratorSpec` instance.
        """
        if self.model_key == "TRBO":
            # Temporary hack to unblock TRBO.
            # TODO[T167756515] Remove when TRBO revamp diff lands.
            return True
        return self._last_fit_arg_ids == self._get_fit_arg_ids(
            experiment=experiment, combined_model_kwargs=combined_model_kwargs
        )

    def _get_fit_arg_ids(
        self,
        experiment: Experiment,
        combined_model_kwargs: dict[str, Any],
    ) -> dict[str, int]:
        """Construct a dictionary mapping arg name to object id."""
        return {
            "experiment": id(experiment),
            **{k: id(v) for k, v in combined_model_kwargs.items()},
        }

    def _assert_fitted(self) -> None:
        """Helper that verifies a model was fitted, raising an error if not"""
        if self._fitted_model is None:
            raise UserInputError("No fitted model found. Call fit() to generate one")

    def _brief_repr(self) -> str:
        """Returns a brief string representation of this model spec.
        Includes just name and override, but not the various kwargs"""
        return (
            "GeneratorSpec("
            f"\tmodel_enum={self.model_enum.value}, "
            f"\tmodel_key_override={self.model_key_override}"
            ")"
        )

    def __repr__(self) -> str:
        model_kwargs = json.dumps(
            self.model_kwargs, sort_keys=True, cls=GeneratorSpecJSONEncoder
        )
        model_gen_kwargs = json.dumps(
            self.model_gen_kwargs, sort_keys=True, cls=GeneratorSpecJSONEncoder
        )
        model_cv_kwargs = json.dumps(
            self.model_cv_kwargs, sort_keys=True, cls=GeneratorSpecJSONEncoder
        )
        return (
            "GeneratorSpec("
            f"\tmodel_enum={self.model_enum.value}, "
            f"\tmodel_kwargs={model_kwargs}, "
            f"\tmodel_gen_kwargs={model_gen_kwargs}, "
            f"\tmodel_cv_kwargs={model_cv_kwargs}, "
            f"\tmodel_key_override={self.model_key_override}"
            ")"
        )

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: GeneratorSpec) -> bool:
        return repr(self) == repr(other)

    @property
    def _unique_id(self) -> str:
        """Returns the unique ID of this model spec"""
        # TODO @mgarrard verify that this is unique enough
        return str(hash(self))
