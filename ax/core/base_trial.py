#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.formatting_utils import data_and_evaluations_from_raw_data
from ax.core.generator_run import GeneratorRun, GeneratorRunType
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric, MetricFetchResult
from ax.core.runner import Runner
from ax.core.trial_status import TrialStatus
from ax.core.types import TCandidateMetadata, TEvaluationOutcome
from ax.exceptions.core import UnsupportedError
from ax.utils.common.base import SortableBase
from ax.utils.common.constants import Keys
from pyre_extensions import none_throws


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401

MANUAL_GENERATION_METHOD_STR = "Manual"
UNKNOWN_GENERATION_METHOD_STR = "Unknown"
STATUS_QUO_GENERATION_METHOD_STR = "Status Quo"


def immutable_once_run(func: Callable) -> Callable:
    """Decorator for methods that should throw Error when
    trial is running or has ever run and immutable.
    """

    # no type annotation for now; breaks sphinx-autodoc-typehints
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _immutable_once_run(self, *args, **kwargs):
        if self._status != TrialStatus.CANDIDATE:
            raise ValueError(
                "Cannot modify a trial that is running or has ever run.",
                "Create a new trial using `experiment.new_trial()` "
                "or clone an existing trial using `trial.clone()`.",
            )
        return func(self, *args, **kwargs)

    return _immutable_once_run


class BaseTrial(ABC, SortableBase):
    """Base class for representing trials.

    Trials are containers for arms that are deployed together. There are
    two kinds of trials: regular Trial, which only contains a single arm,
    and BatchTrial, which contains an arbitrary number of arms.

    Args:
        experiment: Experiment, of which this trial is a part
        trial_type: Type of this trial, if used in MultiTypeExperiment.
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
    """

    def __init__(
        self,
        experiment: core.experiment.Experiment,
        trial_type: str | None = None,
        ttl_seconds: int | None = None,
        index: int | None = None,
    ) -> None:
        """Initialize trial.

        Args:
            experiment: The experiment this trial belongs to.
        """
        self._experiment = experiment
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("TTL must be a positive integer (or None).")
        self._ttl_seconds: int | None = ttl_seconds
        self._index: int = self._experiment._attach_trial(self, index=index)

        if trial_type is not None:
            if not self._experiment.supports_trial_type(trial_type):
                raise ValueError(
                    f"Experiment does not support trial_type {trial_type}."
                )
        else:
            trial_type = self._experiment.default_trial_type
        self._trial_type: str | None = trial_type

        self.__status: TrialStatus | None = None
        # Uses `_status` setter, which updates trial statuses to trial indices
        # mapping on the experiment, with which this trial is associated.
        self._status = TrialStatus.CANDIDATE
        self._time_created: datetime = datetime.now()

        # Initialize fields to be used later in lifecycle
        self._time_completed: datetime | None = None
        self._time_staged: datetime | None = None
        self._time_run_started: datetime | None = None

        self._abandoned_reason: str | None = None
        self._failed_reason: str | None = None
        self._run_metadata: dict[str, Any] = {}
        self._stop_metadata: dict[str, Any] = {}

        self._runner: Runner | None = None

        # Counter to maintain how many arms have been named by this BatchTrial
        self._num_arms_created = 0

        # If generator run(s) in this trial were generated from a generation
        # strategy, this property will be set to the generation step that produced
        # the generator run(s).
        self._generation_step_index: int | None = None
        # Please do not store any data related to trial deployment or data-
        # fetching in properties. It is intended to only store properties related
        # to core Ax functionality and not to any third-system that the trials
        # might be getting deployed to.
        # pyre-fixme[4]: Attribute must be annotated.
        self._properties = {}

    @property
    def experiment(self) -> core.experiment.Experiment:
        """The experiment this trial belongs to."""
        return self._experiment

    @property
    def index(self) -> int:
        """The index of this trial within the experiment's trial list."""
        return self._index

    @property
    def status(self) -> TrialStatus:
        """The status of the trial in the experimentation lifecycle."""
        self._mark_failed_if_past_TTL()
        return none_throws(self._status)

    @status.setter
    def status(self, status: TrialStatus) -> None:
        raise NotImplementedError("Use `trial.mark_*` methods to set trial status.")

    @property
    def ttl_seconds(self) -> int | None:
        """This trial's time-to-live once ran, in seconds. If not set, trial
        will never be automatically considered failed (i.e. infinite TTL).
        Reflects after how many seconds since the time the trial was run it
        will be considered failed unless completed.
        """
        return self._ttl_seconds

    @ttl_seconds.setter
    def ttl_seconds(self, ttl_seconds: int | None) -> None:
        """Sets this trial's time-to-live once ran, in seconds. If None, trial
        will never be automatically considered failed (i.e. infinite TTL).
        Reflects after how many seconds since the time the trial was run it
        will be considered failed unless completed.
        """
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("TTL must be a positive integer (or None).")
        self._ttl_seconds = ttl_seconds

    @property
    def completed_successfully(self) -> bool:
        """Checks if trial status is `COMPLETED`."""
        return self.status == TrialStatus.COMPLETED

    @property
    def did_not_complete(self) -> bool:
        """Checks if trial status is terminal, but not `COMPLETED`."""
        return self.status.is_terminal and not self.completed_successfully

    @property
    def runner(self) -> Runner | None:
        """The runner object defining how to deploy the trial."""
        return self._runner

    @runner.setter
    @immutable_once_run
    def runner(self, runner: Runner | None) -> None:
        self._runner = runner

    @property
    def deployed_name(self) -> str | None:
        """Name of the experiment created in external framework.

        This property is derived from the name field in run_metadata.
        """
        return self._run_metadata.get("name") if self._run_metadata else None

    @property
    def run_metadata(self) -> dict[str, Any]:
        """Dict containing metadata from the deployment process.

        This is set implicitly during `trial.run()`.
        """
        return self._run_metadata

    @property
    def stop_metadata(self) -> dict[str, Any]:
        """Dict containing metadata from the stopping process.

        This is set implicitly during `trial.stop()`.
        """
        return self._stop_metadata

    @property
    def trial_type(self) -> str | None:
        """The type of the trial.

        Relevant for experiments containing different kinds of trials
        (e.g. different deployment types).
        """
        return self._trial_type

    @trial_type.setter
    @immutable_once_run
    def trial_type(self, trial_type: str | None) -> None:
        """Identifier used to distinguish trial types in experiments
        with multiple trial types.
        """
        if self._experiment is not None:
            if not self._experiment.supports_trial_type(trial_type):
                raise ValueError(f"{trial_type} is not supported by the experiment.")

        self._trial_type = trial_type

    def assign_runner(self) -> BaseTrial:
        """Assigns default experiment runner if trial doesn't already have one."""
        runner = self.experiment.runner_for_trial(self)
        if runner is not None:
            self._runner = runner.clone()
        return self

    def update_run_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Updates the run metadata dict stored on this trial and returns the
        updated dict."""
        self._run_metadata.update(metadata)
        return self._run_metadata

    def update_stop_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Updates the stop metadata dict stored on this trial and returns the
        updated dict."""
        self._stop_metadata.update(metadata)
        return self._stop_metadata

    def run(self) -> BaseTrial:
        """Deploys the trial according to the behavior on the runner.

        The runner returns a `run_metadata` dict containining metadata
        of the deployment process. It also returns a `deployed_name` of the trial
        within the system to which it was deployed. Both these fields are set on
        the trial.

        Returns:
            The trial instance.
        """
        if self.status != TrialStatus.CANDIDATE:
            raise ValueError("Can only run a candidate trial.")

        # Default to experiment runner if trial doesn't have one
        self.assign_runner()

        if self._runner is None:
            raise ValueError("No runner set on trial or experiment.")

        self.update_run_metadata(none_throws(self._runner).run(self))

        if none_throws(self._runner).staging_required:
            self.mark_staged()
        else:
            self.mark_running()
        return self

    def stop(self, new_status: TrialStatus, reason: str | None = None) -> BaseTrial:
        """Stops the trial according to the behavior on the runner.

        The runner returns a `stop_metadata` dict containining metadata
        of the stopping process.

        Args:
            new_status: The new TrialStatus. Must be one of {TrialStatus.COMPLETED,
                TrialStatus.ABANDONED, TrialStatus.EARLY_STOPPED}
            reason: A message containing information why the trial is to be stopped.

        Returns:
            The trial instance.
        """
        if self.status not in {TrialStatus.STAGED, TrialStatus.RUNNING}:
            raise ValueError("Can only stop STAGED or RUNNING trials.")

        if new_status not in {
            TrialStatus.COMPLETED,
            TrialStatus.ABANDONED,
            TrialStatus.EARLY_STOPPED,
        }:
            raise ValueError(
                "New status of a stopped trial must either be "
                "COMPLETED, ABANDONED or EARLY_STOPPED."
            )

        # Default to experiment runner if trial doesn't have one
        self.assign_runner()
        if self._runner is None:
            raise ValueError("No runner set on trial or experiment.")
        runner = none_throws(self._runner)

        self._stop_metadata = runner.stop(self, reason=reason)
        self.mark_as(new_status)
        return self

    def complete(self, reason: str | None = None) -> BaseTrial:
        """Stops the trial if functionality is defined on runner
            and marks trial completed.

        Args:
            reason: A message containing information why the trial is to be
                completed.

        Returns:
            The trial instance.
        """
        if self.status != TrialStatus.RUNNING:
            raise ValueError("Can only stop a running trial.")
        try:
            self.stop(new_status=TrialStatus.COMPLETED, reason=reason)
        except NotImplementedError:
            self.mark_completed()
        return self

    def fetch_data_results(
        self, metrics: list[Metric] | None = None, **kwargs: Any
    ) -> dict[str, MetricFetchResult]:
        """Fetch data results for this trial for all metrics on experiment.

        Args:
            trial_index: The index of the trial to fetch data for.
            metrics: If provided, fetch data for these metrics instead of the ones
                defined on the experiment.
            kwargs: keyword args to pass to underlying metrics' fetch data functions.

        Returns:
            MetricFetchResults for this trial.
        """

        return self.experiment._fetch_trial_data(
            trial_index=self.index, metrics=metrics, **kwargs
        )

    def fetch_data(self, metrics: list[Metric] | None = None, **kwargs: Any) -> Data:
        """Fetch data for this trial for all metrics on experiment.

        # NOTE: This can be lossy (ex. a MapData could get implicitly cast to a Data and
        # lose rows)if some if Experiment.default_data_type is misconfigured!

        Args:
            trial_index: The index of the trial to fetch data for.
            metrics: If provided, fetch data for these metrics instead of the ones
                defined on the experiment.
            kwargs: keyword args to pass to underlying metrics' fetch data functions.

        Returns:
            Data for this trial.
        """
        base_metric_cls = (
            MapMetric if self.experiment.default_data_constructor == MapData else Metric
        )

        return base_metric_cls._unwrap_trial_data_multi(
            results=self.fetch_data_results(metrics=metrics, **kwargs)
        )

    def lookup_data(self) -> Data:
        """Lookup cached data on experiment for this trial.

        Returns:
            If not merging across timestamps, the latest ``Data`` object
            associated with the trial. If merging, all data for trial, merged.

        """
        return self.experiment.lookup_data_for_trial(trial_index=self.index)[0]

    def _check_existing_and_name_arm(self, arm: Arm) -> None:
        """Sets name for given arm; if this arm is already in the
        experiment, uses the existing arm name.
        """
        proposed_name = self._get_default_name()

        # Arm could already be in experiment, replacement is okay.
        self.experiment._name_and_store_arm_if_not_exists(
            arm=arm, proposed_name=proposed_name, replace=True
        )
        # If arm was named using given name, incremement the count
        if arm.name == proposed_name:
            self._num_arms_created += 1

    def _get_default_name(self, arm_index: int | None = None) -> str:
        if arm_index is None:
            arm_index = self._num_arms_created
        return f"{self.index}_{arm_index}"

    def _set_generation_step_index(self, generation_step_index: int | None) -> None:
        """Sets the `generation_step_index` property of the trial, to reflect which
        generation step of a given generation strategy (if any) produced the generator
        run(s) attached to this trial.
        """
        if (
            self._generation_step_index is not None
            and generation_step_index is not None
            and self._generation_step_index != generation_step_index
        ):
            raise ValueError(
                "Cannot add generator runs from different generation steps to a "
                "single trial."
            )
        self._generation_step_index = generation_step_index

    @abstractproperty
    def arms(self) -> list[Arm]:
        pass

    @abstractproperty
    def arms_by_name(self) -> dict[str, Arm]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractproperty
    def abandoned_arms(self) -> list[Arm]:
        """All abandoned arms, associated with this trial."""
        pass

    @property
    def active_arms(self) -> list[Arm]:
        """All non abandoned arms associated with this trial."""
        return [arm for arm in self.arms if arm not in self.abandoned_arms]

    @abstractproperty
    def generator_runs(self) -> list[GeneratorRun]:
        """All generator runs associated with this trial."""
        pass

    @abstractmethod
    def _get_candidate_metadata_from_all_generator_runs(
        self,
    ) -> dict[str, TCandidateMetadata]:
        """Retrieves combined candidate metadata from all generator runs associated
        with this trial.
        """
        ...

    @abstractmethod
    def _get_candidate_metadata(self, arm_name: str) -> TCandidateMetadata:
        """Retrieves candidate metadata for a specific arm."""
        ...

    # --- Trial lifecycle management functions ---

    @property
    def time_created(self) -> datetime:
        """Creation time of the trial."""
        return self._time_created

    @property
    def time_completed(self) -> datetime | None:
        """Completion time of the trial."""
        return self._time_completed

    @property
    def time_staged(self) -> datetime | None:
        """Staged time of the trial."""
        return self._time_staged

    @property
    def time_run_started(self) -> datetime | None:
        """Time the trial was started running (i.e. collecting data)."""
        return self._time_run_started

    @property
    def is_abandoned(self) -> bool:
        """Whether this trial is abandoned."""
        return self._status == TrialStatus.ABANDONED

    @property
    def abandoned_reason(self) -> str | None:
        return self._abandoned_reason

    @property
    def failed_reason(self) -> str | None:
        return self._failed_reason

    def mark_staged(self, unsafe: bool = False) -> BaseTrial:
        """Mark the trial as being staged for running.

        Args:
            unsafe: Ignore sanity checks on state transitions.
        Returns:
            The trial instance.
        """
        if not unsafe and self._status != TrialStatus.CANDIDATE:
            raise ValueError(
                f"Can only stage a candidate trial.  This trial is {self._status}"
            )
        self._status = TrialStatus.STAGED
        self._time_staged = datetime.now()
        return self

    def mark_running(
        self, no_runner_required: bool = False, unsafe: bool = False
    ) -> BaseTrial:
        """Mark trial has started running.

        Args:
            no_runner_required: Whether to skip the check for presence of a
                ``Runner`` on the experiment.
            unsafe: Ignore sanity checks on state transitions.

        Returns:
            The trial instance.
        """
        if self._runner is None and not no_runner_required:
            raise ValueError("Cannot mark trial running without setting runner.")

        prev_step = (
            TrialStatus.STAGED
            if self._runner is not None and self._runner.staging_required
            else TrialStatus.CANDIDATE
        )
        prev_step_str = "staged" if prev_step == TrialStatus.STAGED else "candidate"
        if not unsafe and self._status != prev_step:
            raise ValueError(
                f"Can only mark this trial as running when {prev_step_str}."
            )
        self._status = TrialStatus.RUNNING
        self._time_run_started = datetime.now()
        return self

    def mark_completed(self, unsafe: bool = False) -> BaseTrial:
        """Mark trial as completed.

        Args:
            unsafe: Ignore sanity checks on state transitions.
        Returns:
            The trial instance.
        """
        if not unsafe and self._status != TrialStatus.RUNNING:
            raise ValueError("Can only complete trial that is currently running.")
        self._status = TrialStatus.COMPLETED
        self._time_completed = datetime.now()
        return self

    def mark_abandoned(
        self, reason: str | None = None, unsafe: bool = False
    ) -> BaseTrial:
        """Mark trial as abandoned.

        NOTE: Arms in abandoned trials are considered to be 'pending points'
        in experiment after their abandonment to avoid Ax models suggesting
        the same arm again as a new candidate. Arms in abandoned trials are
        also excluded from model training data unless ``fit_abandoned`` option
        is specified to adapter.

        Args:
            abandoned_reason: The reason the trial was abandoned.
            unsafe: Ignore sanity checks on state transitions.

        Returns:
            The trial instance.
        """
        if not unsafe and none_throws(self._status).is_terminal:
            raise ValueError("Cannot abandon a trial in a terminal state.")

        self._abandoned_reason = reason
        self._status = TrialStatus.ABANDONED
        self._time_completed = datetime.now()
        return self

    def mark_failed(self, reason: str | None = None, unsafe: bool = False) -> BaseTrial:
        """Mark trial as failed.

        Args:
            unsafe: Ignore sanity checks on state transitions.
        Returns:
            The trial instance.
        """
        if not unsafe and self._status != TrialStatus.RUNNING:
            raise ValueError("Can only mark failed a trial that is currently running.")

        self._failed_reason = reason
        self._status = TrialStatus.FAILED
        self._time_completed = datetime.now()
        return self

    def mark_early_stopped(self, unsafe: bool = False) -> BaseTrial:
        """Mark trial as early stopped.

        Args:
            unsafe: Ignore sanity checks on state transitions.
        Returns:
            The trial instance.
        """
        if not unsafe:
            if self._status != TrialStatus.RUNNING:
                raise ValueError("Can only early stop trial that is currently running.")

            if self.lookup_data().df.empty:
                raise UnsupportedError(
                    "Cannot mark trial early stopped without data. Please mark trial "
                    "abandoned instead."
                )

        self._status = TrialStatus.EARLY_STOPPED
        self._time_completed = datetime.now()
        return self

    def mark_as(
        self, status: TrialStatus, unsafe: bool = False, **kwargs: Any
    ) -> BaseTrial:
        """Mark trial with a new TrialStatus.

        Args:
            status: The new status of the trial.
            unsafe: Ignore sanity checks on state transitions.
            kwargs: Additional keyword args, as can be ued in the respective `mark_`
                methods associated with the trial status.

        Returns:
            The trial instance.
        """
        if status == TrialStatus.STAGED:
            self.mark_staged(unsafe=unsafe)
        elif status == TrialStatus.RUNNING:
            no_runner_required = kwargs.get("no_runner_required", False)
            self.mark_running(no_runner_required=no_runner_required, unsafe=unsafe)
        elif status == TrialStatus.ABANDONED:
            self.mark_abandoned(reason=kwargs.get("reason"), unsafe=unsafe)
        elif status == TrialStatus.FAILED:
            self.mark_failed(reason=kwargs.get("reason"), unsafe=unsafe)
        elif status == TrialStatus.COMPLETED:
            self.mark_completed(unsafe=unsafe)
        elif status == TrialStatus.EARLY_STOPPED:
            self.mark_early_stopped(unsafe=unsafe)
        else:
            raise ValueError(f"Cannot mark trial as {status}.")
        return self

    def mark_arm_abandoned(self, arm_name: str, reason: str | None = None) -> BaseTrial:
        raise NotImplementedError(
            "Abandoning arms is only supported for `BatchTrial`. "
            "Use `trial.mark_abandoned` if applicable."
        )

    @property
    def generation_method_str(self) -> str:
        """Returns the generation method(s) used to generate this trial's arms,
        as a human-readable string (e.g. 'Sobol', 'BoTorch', 'Manual', etc.).
        Returns a comma-delimited string if multiple generation methods were used.
        """
        # Use model key provided during warm-starting if present, since the
        # generator run may not be present on warm-started trials.
        if (
            warm_start_model_key := self._properties.get(Keys.WARMSTART_TRIAL_MODEL_KEY)
        ) is not None:
            return warm_start_model_key

        generation_methods = {
            none_throws(generator_run._model_key)
            for generator_run in self.generator_runs
            if generator_run._model_key is not None
        }

        # Add generator-run-type strings for non-Adapter generator runs.
        gr_type_name_to_str = {
            GeneratorRunType.MANUAL.name: MANUAL_GENERATION_METHOD_STR,
            GeneratorRunType.STATUS_QUO.name: STATUS_QUO_GENERATION_METHOD_STR,
        }
        generation_methods |= {
            gr_type_name_to_str[generator_run.generator_run_type]
            for generator_run in self.generator_runs
            if generator_run.generator_run_type in gr_type_name_to_str
        }

        return (
            # Sort for deterministic output
            ", ".join(sorted(generation_methods))
            if generation_methods
            else UNKNOWN_GENERATION_METHOD_STR
        )

    def _mark_failed_if_past_TTL(self) -> None:
        """If trial has TTL set and is running, check if the TTL has elapsed
        and mark the trial failed if so.
        """
        if self.ttl_seconds is None or not none_throws(self._status).is_running:
            return
        time_run_started = self._time_run_started
        assert time_run_started is not None
        dt = datetime.now() - time_run_started
        if dt > timedelta(seconds=none_throws(self.ttl_seconds)):
            self.mark_failed()

    @property
    def _status(self) -> TrialStatus | None:
        """The status of the trial in the experimentation lifecycle. This private
        property exists to allow for a corresponding setter, since its important
        that the trial statuses mapping on the experiment is updated always when
        a trial status is updated. In addition, the private property can be None
        whereas the public `status` errors out if self._status is None.
        """
        return self.__status

    @_status.setter
    def _status(self, trial_status: TrialStatus) -> None:
        """Setter for the `_status` attribute that also updates the experiment's
        `_trial_indices_by_status mapping according to the newly set trial status.
        """
        status = self._status
        if status is not None:
            assert self.index in self._experiment._trial_indices_by_status[status]
            self._experiment._trial_indices_by_status[status].remove(self.index)
        self._experiment._trial_indices_by_status[trial_status].add(self.index)
        self.__status = trial_status

    @property
    def _unique_id(self) -> str:
        return str(self.index)

    def _make_evaluations_and_data(
        self,
        raw_data: dict[str, TEvaluationOutcome],
        metadata: dict[str, str | int] | None,
        sample_sizes: dict[str, int] | None = None,
    ) -> tuple[dict[str, TEvaluationOutcome], Data]:
        """Formats given raw data as Ax evaluations and `Data`.

        Args:
            raw_data: Map from arm name to
                metric outcomes.
            metadata: Additional metadata to track about this run.
            sample_size: Integer sample size for 1-arm trials, dict from arm
                name to sample size for batched trials. Optional.
        """

        metadata = metadata if metadata is not None else {}

        evaluations, data = data_and_evaluations_from_raw_data(
            raw_data=raw_data,
            metric_names=list(set(self.experiment.metrics)),
            trial_index=self.index,
            sample_sizes=sample_sizes or {},
            data_type=self.experiment.default_data_type,
            start_time=metadata.get("start_time"),
            end_time=metadata.get("end_time"),
        )
        return evaluations, data

    def _raise_cant_attach_if_completed(self) -> None:
        """
        Helper method used by `validate_can_attach_data` to raise an error if
        the user tries to attach data to a completed trial. Subclasses such as
        `Trial` override this by suggesting a remediation.
        """
        raise UnsupportedError(
            f"Trial {self.index} already has status 'COMPLETED', so data cannot "
            "be attached."
        )

    def _validate_can_attach_data(self) -> None:
        """Determines whether a trial is in a state that can be attached data."""
        if self.status.is_completed:
            self._raise_cant_attach_if_completed()
        if self.status.is_abandoned or self.status.is_failed:
            raise UnsupportedError(
                f"Trial {self.index} has been marked {self.status.name}, so it "
                "no longer expects data."
            )

    def _update_trial_attrs_on_clone(
        self,
        new_trial: BaseTrial,
    ) -> None:
        """Updates attributes of the trial that are not copied over when cloning
        a trial.

        Args:
            new_trial: The cloned trial.
            new_experiment: The experiment that the cloned trial belongs to.
            new_status: The new status of the cloned trial.
        """
        new_trial._run_metadata = deepcopy(self._run_metadata)
        new_trial._stop_metadata = deepcopy(self._stop_metadata)
        new_trial._num_arms_created = self._num_arms_created
        new_trial.runner = self._runner.clone() if self._runner else None

        # Set status and reason accordingly.
        if self.status == TrialStatus.CANDIDATE:
            return
        if self.status == TrialStatus.STAGED:
            new_trial.mark_staged()
            return
        # Other statuses require the state first be set to `RUNNING`.
        new_trial.mark_running(no_runner_required=True, unsafe=True)
        if self.status == TrialStatus.RUNNING:
            return
        if self.status == TrialStatus.ABANDONED:
            new_trial.mark_abandoned(reason=self.abandoned_reason)
            return
        if self.status == TrialStatus.FAILED:
            new_trial.mark_failed(reason=self.failed_reason)
            return
        new_trial.mark_as(self.status, unsafe=True)
