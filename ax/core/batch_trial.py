#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings

from collections import defaultdict, OrderedDict
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from logging import Logger
from typing import TYPE_CHECKING

import numpy as np
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.generator_run import ArmWeight, GeneratorRun, GeneratorRunType
from ax.core.trial import immutable_once_run
from ax.core.types import (
    TCandidateMetadata,
    TEvaluationOutcome,
    validate_evaluation_outcome,
)
from ax.exceptions.core import AxError, UnsupportedError, UserInputError
from ax.utils.common.base import SortableBase
from ax.utils.common.docutils import copy_doc
from ax.utils.common.equality import datetime_equals, equality_typechecker
from ax.utils.common.logger import _round_floats_for_logging, get_logger
from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401

BATCH_TRIAL_RAW_DATA_FORMAT_ERROR_MESSAGE = (
    "Raw data must be a dict for batched trials."
)


class LifecycleStage(int, Enum):
    EXPLORATION = 0
    ITERATION = 1
    BAKEOFF = 2
    OFFLINE_OPTIMIZED = 3
    EXPLORATION_CONCURRENT = 4


@dataclass
class AbandonedArm(SortableBase):
    """Class storing metadata of arm that has been abandoned within
    a BatchTrial.
    """

    name: str
    time: datetime
    reason: str | None = None

    @equality_typechecker
    def __eq__(self, other: AbandonedArm) -> bool:
        return (
            self.name == other.name
            and self.reason == other.reason
            and datetime_equals(self.time, other.time)
        )

    @property
    def _unique_id(self) -> str:
        return self.name


@dataclass
class GeneratorRunStruct(SortableBase):
    """Stores GeneratorRun object as well as the weight with which it was added."""

    generator_run: GeneratorRun
    weight: float

    @property
    def _unique_id(self) -> str:
        return self.generator_run._unique_id + ":" + str(self.weight)


class BatchTrial(BaseTrial):
    """Batched trial that has multiple attached arms, meant to be
    *deployed and evaluated together*, and possibly arm weights, which are
    a measure of how much of the total resources allocated to evaluating
    a batch should go towards evaluating the specific arm. For instance,
    for field experiments the weights could describe the fraction of the
    total experiment population assigned to the different treatment arms.
    Interpretation of the weights is defined in Runner.

    NOTE: A `BatchTrial` is not just a trial with many arms; it is a trial,
    for which it is important that the arms are evaluated simultaneously, e.g.
    in an A/B test where the evaluation results are subject to nonstationarity.
    For cases where multiple arms are evaluated separately and independently of
    each other, use multiple `Trial` objects with a single arm each.

    Args:
        experiment: Experiment, to which this trial is attached
        generator_run: GeneratorRun, associated with this trial. This can a
            also be set later through `add_arm` or `add_generator_run`, but a
            trial's associated generator run is immutable once set.
        generator_runs: GeneratorRuns, associated with this trial. This can a
            also be set later through `add_arm` or `add_generator_run`, but a
            trial's associated generator run is immutable once set.  This cannot
            be combined with the `generator_run` argument.
        trial_type: Type of this trial, if used in MultiTypeExperiment.
        add_status_quo_arm: If True, adds the status quo arm to the trial with a
            weight of 1.0. If False, the _status_quo is still set on the trial for
            tracking purposes, but without a weight it will not be an Arm present on
            the trial
        ttl_seconds: If specified, trials will be considered failed after
            this many seconds since the time the trial was ran, unless the
            trial is completed before then. Meant to be used to detect
            'dead' trials, for which the evaluation process might have
            crashed etc., and which should be considered failed after
            their 'time to live' has passed.
        index: If specified, the trial's index will be set accordingly.
            This should generally not be specified, as in the index will be
            automatically determined based on the number of existing trials.
            This is only used for the purpose of loading from storage.
        lifecycle_stage: The stage of the experiment lifecycle that this
            trial represents
    """

    def __init__(
        self,
        experiment: core.experiment.Experiment,
        generator_run: GeneratorRun | None = None,
        generator_runs: list[GeneratorRun] | None = None,
        trial_type: str | None = None,
        add_status_quo_arm: bool | None = False,
        ttl_seconds: int | None = None,
        index: int | None = None,
        lifecycle_stage: LifecycleStage | None = None,
    ) -> None:
        super().__init__(
            experiment=experiment,
            trial_type=trial_type,
            ttl_seconds=ttl_seconds,
            index=index,
        )
        self._arms_by_name: dict[str, Arm] = {}
        self._generator_run_structs: list[GeneratorRunStruct] = []
        self._abandoned_arms_metadata: dict[str, AbandonedArm] = {}
        self._status_quo: Arm | None = None
        self._status_quo_weight_override: float | None = None
        if generator_run is not None:
            if generator_runs is not None:
                raise UnsupportedError(
                    "Cannot specify both `generator_run` and `generator_runs`."
                )
            self.add_generator_run(generator_run=generator_run)
        elif generator_runs is not None:
            for gr in generator_runs:
                self.add_generator_run(generator_run=gr)

        self.add_status_quo_arm = add_status_quo_arm
        status_quo = experiment.status_quo
        if add_status_quo_arm:
            if status_quo is None:
                raise ValueError(
                    "Experiment does not have a status quo arm so "
                    "no weight can be set for it."
                )
            else:
                self.set_status_quo_with_weight(status_quo=status_quo, weight=1.0)
        else:
            # Set the status quo for tracking purposes
            # It will not be included in arm_weights
            self._status_quo = status_quo

        # Trial status quos are stored in the DB as a generator run
        # with one arm; thus we need to store two `db_id` values
        # for this object instead of one
        self._status_quo_generator_run_db_id: int | None = None
        self._status_quo_arm_db_id: int | None = None
        self._lifecycle_stage = lifecycle_stage

    @property
    def experiment(self) -> core.experiment.Experiment:
        """The experiment this batch belongs to."""
        return self._experiment

    @property
    def index(self) -> int:
        """The index of this batch within the experiment's batch list."""
        return self._index

    @property
    def generator_run_structs(self) -> list[GeneratorRunStruct]:
        """List of generator run structs attached to this trial.

        Struct holds generator_run object and the weight with which it was added.
        """
        return self._generator_run_structs

    @property
    def arm_weights(self) -> MutableMapping[Arm, float]:
        """The set of arms and associated weights for the trial.

        These are constructed by merging the arms and weights from
        each generator run that is attached to the trial.
        """
        arm_weights = OrderedDict()
        if len(self._generator_run_structs) == 0 and self.status_quo is None:
            return arm_weights
        for struct in self._generator_run_structs:
            multiplier = struct.weight
            for arm, weight in struct.generator_run.arm_weights.items():
                scaled_weight = weight * multiplier
                if arm in arm_weights:
                    arm_weights[arm] += scaled_weight
                else:
                    arm_weights[arm] = scaled_weight
        if self.status_quo is not None and self._status_quo_weight_override is not None:
            # If override is specified, this is the weight the status quo gets,
            # regardless of whether it appeared in any generator runs.
            # If no override is specified, status quo does not appear in arm_weights.
            arm_weights[self.status_quo] = self._status_quo_weight_override
        return arm_weights

    @property
    def lifecycle_stage(self) -> LifecycleStage | None:
        return self._lifecycle_stage

    @arm_weights.setter
    def arm_weights(self, arm_weights: MutableMapping[Arm, float]) -> None:
        raise NotImplementedError("Use `trial.add_arms_and_weights`")

    @immutable_once_run
    def add_arm(self, arm: Arm, weight: float = 1.0) -> BatchTrial:
        """Add a arm to the trial.

        Args:
            arm: The arm to be added.
            weight: The weight with which this arm should be added.

        Returns:
            The trial instance.
        """
        return self.add_arms_and_weights(arms=[arm], weights=[weight])

    @immutable_once_run
    def add_arms_and_weights(
        self,
        arms: list[Arm],
        weights: list[float] | None = None,
        multiplier: float = 1.0,
    ) -> BatchTrial:
        """Add arms and weights to the trial.

        Args:
            arms: The arms to be added.
            weights: The weights associated with the arms.
            multiplier: The multiplier applied to input weights before merging with
                the current set of arms and weights.

        Returns:
            The trial instance.
        """
        return self.add_generator_run(
            generator_run=GeneratorRun(
                arms=arms, weights=weights, type=GeneratorRunType.MANUAL.name
            ),
            multiplier=multiplier,
        )

    @immutable_once_run
    def add_generator_run(
        self, generator_run: GeneratorRun, multiplier: float = 1.0
    ) -> BatchTrial:
        """Add a generator run to the trial.

        The arms and weights from the generator run will be merged with
        the existing arms and weights on the trial, and the generator run
        object will be linked to the trial for tracking.

        Args:
            generator_run: The generator run to be added.
            multiplier: The multiplier applied to input weights before merging with
                the current set of arms and weights.

        Returns:
            The trial instance.
        """
        # First validate generator run arms
        for arm in generator_run.arms:
            self.experiment.search_space.check_types(arm.parameters, raise_error=True)

        # Clone arms to avoid mutating existing state
        generator_run._arm_weight_table = OrderedDict(
            {
                arm_sig: ArmWeight(arm_weight.arm.clone(), arm_weight.weight)
                for arm_sig, arm_weight in generator_run._arm_weight_table.items()
            }
        )

        # Add names to arms
        # For those not yet added to this experiment, create a new name
        # Else, use the name of the existing arm
        for arm in generator_run.arms:
            self._check_existing_and_name_arm(arm)

        self._generator_run_structs.append(
            GeneratorRunStruct(generator_run=generator_run, weight=multiplier)
        )
        generator_run.index = len(self._generator_run_structs) - 1

        if self.status_quo is not None and self.add_status_quo_arm:
            self.set_status_quo_with_weight(
                status_quo=none_throws(self.status_quo), weight=1.0
            )

        if generator_run._generation_step_index is not None:
            self._set_generation_step_index(
                generation_step_index=generator_run._generation_step_index
            )
        self._refresh_arms_by_name()
        return self

    @property
    def status_quo(self) -> Arm | None:
        """The control arm for this batch."""
        return self._status_quo

    @status_quo.setter
    def status_quo(self, status_quo: Arm | None) -> None:
        raise NotImplementedError(
            "Use `set_status_quo_with_weight` to set the status quo arm."
        )

    def unset_status_quo(self) -> None:
        """Set the status quo to None."""
        self._status_quo = None
        self._status_quo_weight_override = None
        self._refresh_arms_by_name()

    @immutable_once_run
    def set_status_quo_with_weight(
        self, status_quo: Arm, weight: float | None
    ) -> BatchTrial:
        """Sets status quo arm with given weight. This weight *overrides* any
        weight the status quo has from generator runs attached to this batch.
        Thus, this function is not the same as using add_arm, which will
        result in the weight being additive over all generator runs.
        """
        # Assign a name to this arm if none exists
        if weight is not None:
            if weight <= 0.0:
                raise ValueError("Status quo weight must be positive.")
            if status_quo is None:
                raise ValueError("Cannot set weight because status quo is not defined.")

        if status_quo is not None:
            self.experiment.search_space.check_types(
                status_quo.parameters, raise_error=True
            )
            self.experiment._name_and_store_arm_if_not_exists(
                arm=status_quo,
                proposed_name="status_quo_" + str(self.index),
                replace=True,
            )
        self._status_quo = status_quo.clone() if status_quo is not None else None
        self._status_quo_weight_override = weight
        self._refresh_arms_by_name()
        return self

    @property
    def arms(self) -> list[Arm]:
        """All arms contained in the trial."""
        arm_weights = self.arm_weights
        return [] if arm_weights is None else list(arm_weights.keys())

    @property
    def weights(self) -> list[float]:
        """Weights corresponding to arms contained in the trial."""
        arm_weights = self.arm_weights
        return [] if arm_weights is None else list(arm_weights.values())

    @property
    def arms_by_name(self) -> dict[str, Arm]:
        """Map from arm name to object for all arms in trial."""
        return self._arms_by_name

    def _refresh_arms_by_name(self) -> None:
        self._arms_by_name = {}
        for arm in self.arms:
            if not arm.has_name:
                raise ValueError("Arms attached to a trial must have a name.")
            self._arms_by_name[arm.name] = arm

    @property
    def abandoned_arms(self) -> list[Arm]:
        """List of arms that have been abandoned within this trial."""
        return [
            self.arms_by_name[arm.name]
            for arm in self._abandoned_arms_metadata.values()
        ]

    @property
    def abandoned_arm_names(self) -> set[str]:
        """Set of names of arms that have been abandoned within this trial."""
        return set(self._abandoned_arms_metadata.keys())

    @property
    def in_design_arms(self) -> list[Arm]:
        return [
            arm
            for arm in self.arms
            if self.experiment.search_space.check_membership(arm.parameters)
        ]

    # pyre-ignore[6]: T77111662.
    @copy_doc(BaseTrial.generator_runs)
    @property
    def generator_runs(self) -> list[GeneratorRun]:
        return [grs.generator_run for grs in self.generator_run_structs]

    @property
    def abandoned_arms_metadata(self) -> list[AbandonedArm]:
        return list(self._abandoned_arms_metadata.values())

    @property
    def is_factorial(self) -> bool:
        """Return true if the trial's arms are a factorial design with
        no linked factors.
        """
        # To match the model behavior, this should probably actually be pulled
        # from exp.parameters. However, that seems rather ugly when this function
        # intuitively should just depend on the arms.
        sufficient_factors = all(len(arm.parameters or []) >= 2 for arm in self.arms)
        if not sufficient_factors:
            return False
        param_levels: defaultdict[str, dict[str | float, int]] = defaultdict(dict)
        for arm in self.arms:
            for param_name, param_value in arm.parameters.items():
                param_levels[param_name][none_throws(param_value)] = 1
        param_cardinality = 1
        for param_values in param_levels.values():
            param_cardinality *= len(param_values)
        return len(self.arms) == param_cardinality

    def run(self) -> BatchTrial:
        return assert_is_instance(
            super().run(),
            BatchTrial,
        )

    def normalized_arm_weights(
        self, total: float = 1, trunc_digits: int | None = None
    ) -> MutableMapping[Arm, float]:
        """Returns arms with a new set of weights normalized
        to the given total.

        This method is useful for many runners where we need to normalize weights
        to a certain total without mutating the weights attached to a trial.

        Args:
            total: The total weight to which to normalize.
                Default is 1, in which case arm weights
                can be interpreted as probabilities.
            trunc_digits: The number of digits to keep. If the
                resulting total weight is not equal to `total`, re-allocate
                weight in such a way to maintain relative weights as best as
                possible.

        Returns:
            Mapping from arms to the new set of weights.

        """
        weights = np.array(self.weights)
        if trunc_digits is not None:
            atomic_weight = 10**-trunc_digits
            int_weights = (
                (total / atomic_weight) * (weights / np.sum(weights))
            ).astype(int)
            n_leftover = int(total / atomic_weight) - np.sum(int_weights)
            int_weights[:n_leftover] += 1
            weights = int_weights * atomic_weight
        else:
            weights = weights * (total / np.sum(weights))
        return OrderedDict(zip(self.arms, weights))

    def mark_arm_abandoned(
        self, arm_name: str, reason: str | None = None
    ) -> BatchTrial:
        """Mark a arm abandoned.

        Usually done after deployment when one arm causes issues but
        user wants to continue running other arms in the batch.

        NOTE: Abandoned arms are considered to be 'pending points' in
        experiment after their abandonment to avoid Ax models suggesting
        the same arm again as a new candidate. Abandoned arms are also
        excluded from model training data unless ``fit_abandoned``
        option is specified to adapter.

        Args:
            arm_name: The name of the arm to abandon.
            reason: The reason for abandoning the arm.

        Returns:
            The batch instance.
        """
        if arm_name not in self.arms_by_name:
            raise ValueError("Arm must be contained in batch.")

        abandoned_arm = AbandonedArm(name=arm_name, time=datetime.now(), reason=reason)
        self._abandoned_arms_metadata[arm_name] = abandoned_arm
        return self

    def clone(self) -> BatchTrial:
        """Clone the trial and attach it to the current experiment."""
        warnings.warn(
            "clone() method is getting deprecated. Please use clone_to() instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return self.clone_to(include_sq=False)

    def clone_to(
        self,
        experiment: core.experiment.Experiment | None = None,
        include_sq: bool = True,
        clear_trial_type: bool = False,
    ) -> BatchTrial:
        """Clone the trial and attach it to a specified experiment.
        If None provided, attach it to the current experiment.

        Args:
            experiment: The experiment to which the cloned trial will belong.
                If unspecified, uses the current experiment.
            include_sq: Whether to include status quo in the cloned trial.
            clear_trial_type: Whether to clear the trial type of the cloned trial.

        Returns:
            A new instance of the trial.
        """
        use_old_experiment = experiment is None
        experiment = self._experiment if experiment is None else experiment
        new_trial = experiment.new_batch_trial(
            trial_type=None if clear_trial_type else self._trial_type,
            ttl_seconds=self._ttl_seconds,
        )
        for struct in self._generator_run_structs:
            if use_old_experiment:
                # don't clone gen run in case we are attaching cloned trial to
                # the same experiment
                new_trial.add_generator_run(struct.generator_run, struct.weight)
            else:
                new_trial.add_generator_run(struct.generator_run.clone(), struct.weight)

        if (self._status_quo is not None) and include_sq:
            sq_weight = self._status_quo_weight_override
            new_trial.set_status_quo_with_weight(
                self._status_quo.clone(),
                weight=sq_weight,
            )
        self._update_trial_attrs_on_clone(new_trial=new_trial)
        return new_trial

    def attach_batch_trial_data(
        self,
        raw_data: dict[str, TEvaluationOutcome],
        sample_sizes: dict[str, int] | None = None,
        metadata: dict[str, str | int] | None = None,
    ) -> None:
        """Attaches data to the trial

        Args:
            raw_data: Map from arm name to metric outcomes.
            sample_sizes: Dict from arm name to sample size.
            metadata: Additional metadata to track about this run.
                importantly the start_date and end_date
            complete_trial: Whether to mark trial as complete after
                attaching data. Defaults to False.
        """
        # Validate type of raw_data
        if not isinstance(raw_data, dict):
            raise ValueError(BATCH_TRIAL_RAW_DATA_FORMAT_ERROR_MESSAGE)

        for key, value in raw_data.items():
            if not isinstance(key, str):
                raise ValueError(BATCH_TRIAL_RAW_DATA_FORMAT_ERROR_MESSAGE)

            try:
                validate_evaluation_outcome(outcome=value)
            except TypeError:
                raise ValueError(BATCH_TRIAL_RAW_DATA_FORMAT_ERROR_MESSAGE)

        # Format the data to save.
        not_trial_arm_names = set(raw_data.keys()) - set(self.arms_by_name.keys())
        if not_trial_arm_names:
            raise UserInputError(
                f"Arms {not_trial_arm_names} are not part of trial #{self.index}."
            )

        evaluations, data = self._make_evaluations_and_data(
            raw_data=raw_data, metadata=metadata, sample_sizes=sample_sizes
        )
        self._validate_batch_trial_data(data=data)

        self._run_metadata = self._run_metadata if metadata is None else metadata
        self.experiment.attach_data(data)

        data_for_logging = _round_floats_for_logging(item=evaluations)

        logger.debug(
            f"Updated trial {self.index} with data: "
            f"{_round_floats_for_logging(item=data_for_logging)}."
        )

    def __repr__(self) -> str:
        return (
            "BatchTrial("
            f"experiment_name='{self._experiment._name}', "
            f"index={self._index}, "
            f"status={self._status})"
        )

    def _get_candidate_metadata_from_all_generator_runs(
        self,
    ) -> dict[str, TCandidateMetadata]:
        """Retrieves combined candidate metadata from all generator runs on this
        batch trial in the form of { arm name -> candidate metadata} mapping.

        NOTE: this does not handle the case of the same arm appearing in multiple
        generator runs in the same trial: metadata from only one of the generator
        runs containing the arm will be retrieved.
        """
        cand_metadata = {}
        for gr_struct in self._generator_run_structs:
            gr = gr_struct.generator_run
            if gr.candidate_metadata_by_arm_signature:
                gr_cand_metadata = gr.candidate_metadata_by_arm_signature
                warn = False
                for arm in gr.arms:
                    if arm.name in cand_metadata:
                        warn = True
                    if gr_cand_metadata:
                        # Reformat the mapping to be by arm name, since arm signature
                        # is not stored in Ax data.
                        cand_metadata[arm.name] = gr_cand_metadata.get(arm.signature)
                if warn:
                    logger.debug(
                        "The same arm appears in multiple generator runs in batch "
                        f"{self.index}. Candidate metadata will only contain metadata "
                        "for one of those generator runs, and the candidate metadata "
                        "for the arm from another generator run will not be propagated."
                    )
        return cand_metadata

    def _get_candidate_metadata(self, arm_name: str) -> TCandidateMetadata:
        """Retrieves candidate metadata for a specific arm."""
        try:
            arm = self.arms_by_name[arm_name]
        except KeyError:
            raise ValueError(
                f"Arm by name {arm_name} is not part of trial #{self.index}."
            )
        for gr_struct in self._generator_run_structs:
            gr = gr_struct.generator_run
            if gr and gr.candidate_metadata_by_arm_signature and arm in gr.arms:
                return none_throws(gr.candidate_metadata_by_arm_signature).get(
                    arm.signature
                )
        return None

    def _validate_batch_trial_data(self, data: Data) -> None:
        """Utility function to validate batch data before further processing."""
        if (
            self.status_quo
            and none_throws(self.status_quo).name in self.arms_by_name
            and none_throws(self.status_quo).name not in data.df["arm_name"].values
        ):
            raise AxError(
                f"Trial #{self.index} was completed with data that did "
                "not contain status quo observations, but the trial has "
                "status quo set and therefore data for it is required."
            )

        for metric_name in data.df["metric_name"].values:
            if metric_name not in self.experiment.metrics:
                logger.debug(
                    f"Data was logged for metric {metric_name} that was not yet "
                    "tracked on the experiment. Please specify `tracking_metric_"
                    "names` argument in AxClient.create_experiment to add tracking "
                    "metrics to the experiment. Without those, all data users "
                    "specify is still attached to the experiment, but will not be "
                    "fetched in `experiment.fetch_data()`, but you can still use "
                    "`experiment.lookup_data_for_trial` to get all attached data."
                )
