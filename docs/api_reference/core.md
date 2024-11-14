# ax.core

## Core Classes

### Arm

### *class* ax.core.arm.Arm(parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`SortableBase`](utils.md#ax.utils.common.base.SortableBase)

Base class for defining arms.

Randomization in experiments assigns units to a given arm. Thus, the arm
encapsulates the parametrization needed by the unit.

#### clone(clear_name: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [Arm](#ax.core.arm.Arm)

Create a copy of this arm.

* **Parameters:**
  **clear_name** – whether this cloned copy should set its
  name to None instead of the name of the arm being cloned.
  Defaults to False.

#### *property* has_name *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Return true if arm’s name is not None.

#### *static* md5hash(parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]) → [str](https://docs.python.org/3/library/stdtypes.html#str)

Return unique identifier for arm’s parameters.

* **Parameters:**
  **parameters** – Parameterization; mapping of param name
  to value.
* **Returns:**
  Hash of arm’s parameters.

#### *property* name *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

Get arm name. Throws if name is None.

#### *property* name_or_short_signature *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

Returns arm name if exists; else last 8 characters of the hash.

Used for presentation of candidates (e.g. plotting and tables),
where the candidates do not yet have names (since names are
automatically set upon addition to a trial).

#### *property* parameters *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]*

Get mapping from parameter names to values.

#### *property* signature *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

Get unique representation of a arm.

### BaseTrial

### *class* ax.core.base_trial.BaseTrial(experiment: [core.experiment.Experiment](#ax.core.experiment.Experiment), trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, ttl_seconds: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, index: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC), [`SortableBase`](utils.md#ax.utils.common.base.SortableBase)

Base class for representing trials.

Trials are containers for arms that are deployed together. There are
two kinds of trials: regular Trial, which only contains a single arm,
and BatchTrial, which contains an arbitrary number of arms.

* **Parameters:**
  * **experiment** – Experiment, of which this trial is a part
  * **trial_type** – Type of this trial, if used in MultiTypeExperiment.
  * **ttl_seconds** – If specified, trials will be considered failed after
    this many seconds since the time the trial was ran, unless the
    trial is completed before then. Meant to be used to detect
    ‘dead’ trials, for which the evaluation process might have
    crashed etc., and which should be considered failed after
    their ‘time to live’ has passed.
  * **index** – If specified, the trial’s index will be set accordingly.
    This should generally not be specified, as in the index will be
    automatically determined based on the number of existing trials.
    This is only used for the purpose of loading from storage.

#### *abstract property* abandoned_arms *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Arm](#ax.core.arm.Arm)]*

All abandoned arms, associated with this trial.

#### *property* abandoned_reason *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### *abstract property* arms *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Arm](#ax.core.arm.Arm)]*

#### *abstract property* arms_by_name *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Arm](#ax.core.arm.Arm)]*

#### assign_runner() → [BaseTrial](#ax.core.base_trial.BaseTrial)

Assigns default experiment runner if trial doesn’t already have one.

#### complete(reason: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [BaseTrial](#ax.core.base_trial.BaseTrial)

Stops the trial if functionality is defined on runner
: and marks trial completed.

* **Parameters:**
  **reason** – A message containing information why the trial is to be
  completed.
* **Returns:**
  The trial instance.

#### *property* completed_successfully *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Checks if trial status is COMPLETED.

#### *property* deployed_name *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

Name of the experiment created in external framework.

This property is derived from the name field in run_metadata.

#### *property* did_not_complete *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Checks if trial status is terminal, but not COMPLETED.

#### *property* experiment *: [core.experiment.Experiment](#ax.core.experiment.Experiment)*

The experiment this trial belongs to.

#### *property* failed_reason *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### fetch_data(metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → Data

Fetch data for this trial for all metrics on experiment.

# NOTE: This can be lossy (ex. a MapData could get implicitly cast to a Data and
# lose rows)if some if Experiment.default_data_type is misconfigured!

* **Parameters:**
  * **trial_index** – The index of the trial to fetch data for.
  * **metrics** – If provided, fetch data for these metrics instead of the ones
    defined on the experiment.
  * **kwargs** – keyword args to pass to underlying metrics’ fetch data functions.
* **Returns:**
  Data for this trial.

#### fetch_data_results(metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Result](utils.md#ax.utils.common.result.Result)[Data, MetricFetchE]]

Fetch data results for this trial for all metrics on experiment.

* **Parameters:**
  * **trial_index** – The index of the trial to fetch data for.
  * **metrics** – If provided, fetch data for these metrics instead of the ones
    defined on the experiment.
  * **kwargs** – keyword args to pass to underlying metrics’ fetch data functions.
* **Returns:**
  MetricFetchResults for this trial.

#### *abstract property* generator_runs *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[GeneratorRun](#ax.core.generator_run.GeneratorRun)]*

All generator runs associated with this trial.

#### *property* index *: [int](https://docs.python.org/3/library/functions.html#int)*

The index of this trial within the experiment’s trial list.

#### *property* is_abandoned *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Whether this trial is abandoned.

#### lookup_data() → Data

Lookup cached data on experiment for this trial.

* **Returns:**
  If not merging across timestamps, the latest `Data` object
  associated with the trial. If merging, all data for trial, merged.

#### mark_abandoned(reason: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, unsafe: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [BaseTrial](#ax.core.base_trial.BaseTrial)

Mark trial as abandoned.

NOTE: Arms in abandoned trials are considered to be ‘pending points’
in experiment after their abandonment to avoid Ax models suggesting
the same arm again as a new candidate. Arms in abandoned trials are
also excluded from model training data unless `fit_abandoned` option
is specified to model bridge.

* **Parameters:**
  * **abandoned_reason** – The reason the trial was abandoned.
  * **unsafe** – Ignore sanity checks on state transitions.
* **Returns:**
  The trial instance.

#### mark_arm_abandoned(arm_name: [str](https://docs.python.org/3/library/stdtypes.html#str), reason: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [BaseTrial](#ax.core.base_trial.BaseTrial)

#### mark_as(status: [TrialStatus](#ax.core.base_trial.TrialStatus), unsafe: [bool](https://docs.python.org/3/library/functions.html#bool) = False, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [BaseTrial](#ax.core.base_trial.BaseTrial)

Mark trial with a new TrialStatus.

* **Parameters:**
  * **status** – The new status of the trial.
  * **unsafe** – Ignore sanity checks on state transitions.
  * **kwargs** – Additional keyword args, as can be ued in the respective mark_
    methods associated with the trial status.
* **Returns:**
  The trial instance.

#### mark_completed(unsafe: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [BaseTrial](#ax.core.base_trial.BaseTrial)

Mark trial as completed.

* **Parameters:**
  **unsafe** – Ignore sanity checks on state transitions.
* **Returns:**
  The trial instance.

#### mark_early_stopped(unsafe: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [BaseTrial](#ax.core.base_trial.BaseTrial)

Mark trial as early stopped.

* **Parameters:**
  **unsafe** – Ignore sanity checks on state transitions.
* **Returns:**
  The trial instance.

#### mark_failed(reason: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, unsafe: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [BaseTrial](#ax.core.base_trial.BaseTrial)

Mark trial as failed.

* **Parameters:**
  **unsafe** – Ignore sanity checks on state transitions.
* **Returns:**
  The trial instance.

#### mark_running(no_runner_required: [bool](https://docs.python.org/3/library/functions.html#bool) = False, unsafe: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [BaseTrial](#ax.core.base_trial.BaseTrial)

Mark trial has started running.

* **Parameters:**
  * **no_runner_required** – Whether to skip the check for presence of a
    `Runner` on the experiment.
  * **unsafe** – Ignore sanity checks on state transitions.
* **Returns:**
  The trial instance.

#### mark_staged(unsafe: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [BaseTrial](#ax.core.base_trial.BaseTrial)

Mark the trial as being staged for running.

* **Parameters:**
  **unsafe** – Ignore sanity checks on state transitions.
* **Returns:**
  The trial instance.

#### run() → [BaseTrial](#ax.core.base_trial.BaseTrial)

Deploys the trial according to the behavior on the runner.

The runner returns a run_metadata dict containining metadata
of the deployment process. It also returns a deployed_name of the trial
within the system to which it was deployed. Both these fields are set on
the trial.

* **Returns:**
  The trial instance.

#### *property* run_metadata *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*

Dict containing metadata from the deployment process.

This is set implicitly during trial.run().

#### *property* runner *: [Runner](#ax.core.runner.Runner) | [None](https://docs.python.org/3/library/constants.html#None)*

The runner object defining how to deploy the trial.

#### *property* status *: [TrialStatus](#ax.core.base_trial.TrialStatus)*

The status of the trial in the experimentation lifecycle.

#### stop(new_status: [TrialStatus](#ax.core.base_trial.TrialStatus), reason: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [BaseTrial](#ax.core.base_trial.BaseTrial)

Stops the trial according to the behavior on the runner.

The runner returns a stop_metadata dict containining metadata
of the stopping process.

* **Parameters:**
  * **new_status** – The new TrialStatus. Must be one of {TrialStatus.COMPLETED,
    TrialStatus.ABANDONED, TrialStatus.EARLY_STOPPED}
  * **reason** – A message containing information why the trial is to be stopped.
* **Returns:**
  The trial instance.

#### *property* stop_metadata *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*

Dict containing metadata from the stopping process.

This is set implicitly during trial.stop().

#### *property* time_completed *: [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime) | [None](https://docs.python.org/3/library/constants.html#None)*

Completion time of the trial.

#### *property* time_created *: [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime)*

Creation time of the trial.

#### *property* time_run_started *: [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime) | [None](https://docs.python.org/3/library/constants.html#None)*

Time the trial was started running (i.e. collecting data).

#### *property* time_staged *: [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime) | [None](https://docs.python.org/3/library/constants.html#None)*

Staged time of the trial.

#### *property* trial_type *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

The type of the trial.

Relevant for experiments containing different kinds of trials
(e.g. different deployment types).

#### *property* ttl_seconds *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

This trial’s time-to-live once ran, in seconds. If not set, trial
will never be automatically considered failed (i.e. infinite TTL).
Reflects after how many seconds since the time the trial was run it
will be considered failed unless completed.

#### update_run_metadata(metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Updates the run metadata dict stored on this trial and returns the
updated dict.

#### update_stop_metadata(metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Updates the stop metadata dict stored on this trial and returns the
updated dict.

### *class* ax.core.base_trial.TrialStatus(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: [`int`](https://docs.python.org/3/library/functions.html#int), [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Enum of trial status.

General lifecycle of a trial is::

```default
CANDIDATE --> STAGED --> RUNNING --> COMPLETED
          ------------->         --> FAILED (retryable)
                                 --> EARLY_STOPPED (deemed unpromising)
          -------------------------> ABANDONED (non-retryable)
```

Trial is marked as a `CANDIDATE` immediately upon its creation.

Trials may be abandoned at any time prior to completion or failure.
The difference between abandonment and failure is that the `FAILED` state
is meant to express a possibly transient or retryable error, so trials in
that state may be re-run and arm(s) in them may be resuggested by Ax models
to be added to new trials.

`ABANDONED` trials on the other end, indicate
that the trial (and arms(s) in it) should not be rerun or added to new
trials. A trial might be marked `ABANDONED` as a result of human-initiated
action (if some trial in experiment is poorly-performing, deterministically
failing etc., and should not be run again in the experiment). It might also
be marked `ABANDONED` in an automated way if the trial’s execution
encounters an error that indicates that the arm(s) in the trial should bot
be evaluated in the experiment again (e.g. the parameterization in a given
arm deterministically causes trial evaluation to fail). Note that it’s also
possible to abandon a single arm in a BatchTrial via
`batch.mark_arm_abandoned`.

Early-stopped refers to trials that were deemed
unpromising by an early-stopping strategy and therefore terminated.

Additionally, when trials are deployed, they may be in an intermediate
staged state (e.g. scheduled but waiting for resources) or immediately
transition to running. Note that `STAGED` trial status is not always
applicable and depends on the `Runner` trials are deployed with
(and whether a `Runner` is present at all; for example, in Ax Service
API, trials are marked as `RUNNING` immediately when generated from
`get_next_trial`, skipping the `STAGED` status).

NOTE: Data for abandoned trials (or abandoned arms in batch trials) is
not passed to the model as part of training data, unless `fit_abandoned`
option is specified to model bridge. Additionally, data from MapMetrics is
typically excluded unless the corresponding trial is completed.

#### ABANDONED *= 5*

#### CANDIDATE *= 0*

#### COMPLETED *= 3*

#### DISPATCHED *= 6*

#### EARLY_STOPPED *= 7*

#### FAILED *= 2*

#### RUNNING *= 4*

#### STAGED *= 1*

#### *property* expecting_data *: [bool](https://docs.python.org/3/library/functions.html#bool)*

True if trial is expecting data.

#### *property* is_abandoned *: [bool](https://docs.python.org/3/library/functions.html#bool)*

True if this trial is an abandoned one.

#### *property* is_candidate *: [bool](https://docs.python.org/3/library/functions.html#bool)*

True if this trial is a candidate.

#### *property* is_completed *: [bool](https://docs.python.org/3/library/functions.html#bool)*

True if this trial is a successfully completed one.

#### *property* is_deployed *: [bool](https://docs.python.org/3/library/functions.html#bool)*

True if trial has been deployed but not completed.

#### *property* is_early_stopped *: [bool](https://docs.python.org/3/library/functions.html#bool)*

True if this trial is an early stopped one.

#### *property* is_failed *: [bool](https://docs.python.org/3/library/functions.html#bool)*

True if this trial is a failed one.

#### *property* is_running *: [bool](https://docs.python.org/3/library/functions.html#bool)*

True if this trial is a running one.

#### *property* is_terminal *: [bool](https://docs.python.org/3/library/functions.html#bool)*

True if trial is completed.

### ax.core.base_trial.immutable_once_run(func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Decorator for methods that should throw Error when
trial is running or has ever run and immutable.

### BatchTrial

### *class* ax.core.batch_trial.AbandonedArm(name: [str](https://docs.python.org/3/library/stdtypes.html#str), time: [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime), reason: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`SortableBase`](utils.md#ax.utils.common.base.SortableBase)

Class storing metadata of arm that has been abandoned within
a BatchTrial.

#### name *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### reason *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### time *: [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime)*

### *class* ax.core.batch_trial.BatchTrial(experiment: [core.experiment.Experiment](#ax.core.experiment.Experiment), generator_run: [GeneratorRun](#ax.core.generator_run.GeneratorRun) | [None](https://docs.python.org/3/library/constants.html#None) = None, generator_runs: [list](https://docs.python.org/3/library/stdtypes.html#list)[[GeneratorRun](#ax.core.generator_run.GeneratorRun)] | [None](https://docs.python.org/3/library/constants.html#None) = None, trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, optimize_for_power: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = False, ttl_seconds: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, index: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, lifecycle_stage: [LifecycleStage](#ax.core.batch_trial.LifecycleStage) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`BaseTrial`](#ax.core.base_trial.BaseTrial)

Batched trial that has multiple attached arms, meant to be
*deployed and evaluated together*, and possibly arm weights, which are
a measure of how much of the total resources allocated to evaluating
a batch should go towards evaluating the specific arm. For instance,
for field experiments the weights could describe the fraction of the
total experiment population assigned to the different treatment arms.
Interpretation of the weights is defined in Runner.

NOTE: A BatchTrial is not just a trial with many arms; it is a trial,
for which it is important that the arms are evaluated simultaneously, e.g.
in an A/B test where the evaluation results are subject to nonstationarity.
For cases where multiple arms are evaluated separately and independently of
each other, use multiple Trial objects with a single arm each.

* **Parameters:**
  * **experiment** – Experiment, to which this trial is attached
  * **generator_run** – GeneratorRun, associated with this trial. This can a
    also be set later through add_arm or add_generator_run, but a
    trial’s associated generator run is immutable once set.
  * **generator_runs** – GeneratorRuns, associated with this trial. This can a
    also be set later through add_arm or add_generator_run, but a
    trial’s associated generator run is immutable once set.  This cannot
    be combined with the generator_run argument.
  * **trial_type** – Type of this trial, if used in MultiTypeExperiment.
  * **optimize_for_power** – Whether to optimize the weights of arms in this
    trial such that the experiment’s power to detect effects of
    certain size is as high as possible. Refer to documentation of
    BatchTrial.set_status_quo_and_optimize_power for more detail.
  * **ttl_seconds** – If specified, trials will be considered failed after
    this many seconds since the time the trial was ran, unless the
    trial is completed before then. Meant to be used to detect
    ‘dead’ trials, for which the evaluation process might have
    crashed etc., and which should be considered failed after
    their ‘time to live’ has passed.
  * **index** – If specified, the trial’s index will be set accordingly.
    This should generally not be specified, as in the index will be
    automatically determined based on the number of existing trials.
    This is only used for the purpose of loading from storage.
  * **lifecycle_stage** – The stage of the experiment lifecycle that this
    trial represents

#### *property* abandoned_arm_names *: [set](https://docs.python.org/3/library/stdtypes.html#set)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

Set of names of arms that have been abandoned within this trial.

#### *property* abandoned_arms *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Arm](#ax.core.arm.Arm)]*

List of arms that have been abandoned within this trial.

#### *property* abandoned_arms_metadata *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[AbandonedArm](#ax.core.batch_trial.AbandonedArm)]*

#### add_arm(\*args, \*\*kwargs)

#### add_arms_and_weights(\*args, \*\*kwargs)

#### add_generator_run(\*args, \*\*kwargs)

#### *property* arm_weights *: [MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[Arm](#ax.core.arm.Arm), [float](https://docs.python.org/3/library/functions.html#float)]*

The set of arms and associated weights for the trial.

These are constructed by merging the arms and weights from
each generator run that is attached to the trial.

#### *property* arms *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Arm](#ax.core.arm.Arm)]*

All arms contained in the trial.

#### *property* arms_by_name *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Arm](#ax.core.arm.Arm)]*

Map from arm name to object for all arms in trial.

#### attach_batch_trial_data(raw_data: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]] | [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Hashable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]]], sample_sizes: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str) | [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)

Attaches data to the trial

* **Parameters:**
  * **raw_data** – Map from arm name to metric outcomes.
  * **sample_sizes** – Dict from arm name to sample size.
  * **metadata** – Additional metadata to track about this run.
    importantly the start_date and end_date
  * **complete_trial** – Whether to mark trial as complete after
    attaching data. Defaults to False.

#### clone() → [BatchTrial](#ax.core.batch_trial.BatchTrial)

Clone the trial and attach it to the current experiment.

#### clone_to(experiment: [core.experiment.Experiment](#ax.core.experiment.Experiment) | [None](https://docs.python.org/3/library/constants.html#None) = None, include_sq: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [BatchTrial](#ax.core.batch_trial.BatchTrial)

Clone the trial and attach it to a specified experiment.
If None provided, attach it to the current experiment.

* **Parameters:**
  * **experiment** – The experiment to which the cloned trial will belong.
    If unspecified, uses the current experiment.
  * **include_sq** – Whether to include status quo in the cloned trial.
* **Returns:**
  A new instance of the trial.

#### *property* experiment *: [core.experiment.Experiment](#ax.core.experiment.Experiment)*

The experiment this batch belongs to.

#### *property* generator_run_structs *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[GeneratorRunStruct](#ax.core.batch_trial.GeneratorRunStruct)]*

List of generator run structs attached to this trial.

Struct holds generator_run object and the weight with which it was added.

#### *property* generator_runs *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[GeneratorRun](#ax.core.generator_run.GeneratorRun)]*

All generator runs associated with this trial.

#### *property* in_design_arms *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Arm](#ax.core.arm.Arm)]*

#### *property* index *: [int](https://docs.python.org/3/library/functions.html#int)*

The index of this batch within the experiment’s batch list.

#### *property* is_factorial *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Return true if the trial’s arms are a factorial design with
no linked factors.

#### *property* lifecycle_stage *: [LifecycleStage](#ax.core.batch_trial.LifecycleStage) | [None](https://docs.python.org/3/library/constants.html#None)*

#### mark_arm_abandoned(arm_name: [str](https://docs.python.org/3/library/stdtypes.html#str), reason: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [BatchTrial](#ax.core.batch_trial.BatchTrial)

Mark a arm abandoned.

Usually done after deployment when one arm causes issues but
user wants to continue running other arms in the batch.

NOTE: Abandoned arms are considered to be ‘pending points’ in
experiment after their abandonment to avoid Ax models suggesting
the same arm again as a new candidate. Abandoned arms are also
excluded from model training data unless `fit_abandoned`
option is specified to model bridge.

* **Parameters:**
  * **arm_name** – The name of the arm to abandon.
  * **reason** – The reason for abandoning the arm.
* **Returns:**
  The batch instance.

#### normalized_arm_weights(total: [float](https://docs.python.org/3/library/functions.html#float) = 1, trunc_digits: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[Arm](#ax.core.arm.Arm), [float](https://docs.python.org/3/library/functions.html#float)]

Returns arms with a new set of weights normalized
to the given total.

This method is useful for many runners where we need to normalize weights
to a certain total without mutating the weights attached to a trial.

* **Parameters:**
  * **total** – The total weight to which to normalize.
    Default is 1, in which case arm weights
    can be interpreted as probabilities.
  * **trunc_digits** – The number of digits to keep. If the
    resulting total weight is not equal to total, re-allocate
    weight in such a way to maintain relative weights as best as
    possible.
* **Returns:**
  Mapping from arms to the new set of weights.

#### run() → [BatchTrial](#ax.core.batch_trial.BatchTrial)

Deploys the trial according to the behavior on the runner.

The runner returns a run_metadata dict containining metadata
of the deployment process. It also returns a deployed_name of the trial
within the system to which it was deployed. Both these fields are set on
the trial.

* **Returns:**
  The trial instance.

#### set_status_quo_and_optimize_power(\*args, \*\*kwargs)

#### set_status_quo_with_weight(\*args, \*\*kwargs)

#### *property* status_quo *: [Arm](#ax.core.arm.Arm) | [None](https://docs.python.org/3/library/constants.html#None)*

The control arm for this batch.

#### unset_status_quo() → [None](https://docs.python.org/3/library/constants.html#None)

Set the status quo to None.

#### *property* weights *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]*

Weights corresponding to arms contained in the trial.

### *class* ax.core.batch_trial.GeneratorRunStruct(generator_run: [GeneratorRun](#ax.core.generator_run.GeneratorRun), weight: [float](https://docs.python.org/3/library/functions.html#float))

Bases: [`SortableBase`](utils.md#ax.utils.common.base.SortableBase)

Stores GeneratorRun object as well as the weight with which it was added.

#### generator_run *: [GeneratorRun](#ax.core.generator_run.GeneratorRun)*

#### weight *: [float](https://docs.python.org/3/library/functions.html#float)*

### *class* ax.core.batch_trial.LifecycleStage(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: [`int`](https://docs.python.org/3/library/functions.html#int), [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

#### BAKEOFF *= 2*

#### EXPLORATION *= 0*

#### EXPLORATION_CONCURRENT *= 4*

#### ITERATION *= 1*

#### OFFLINE_OPTIMIZED *= 3*

### Data

### *class* ax.core.data.BaseData(df: pd.DataFrame | [None](https://docs.python.org/3/library/constants.html#None) = None, description: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Base`](utils.md#ax.utils.common.base.Base), [`SerializationMixin`](utils.md#ax.utils.common.serialization.SerializationMixin)

Class storing data for an experiment.

The dataframe is retrieved via the df property. The data can be stored
to an external store for future use by attaching it to an experiment using
experiment.attach_data() (this requires a description to be set.)

#### df

DataFrame with underlying data, and required columns. For BaseData, the
one required column is “arm_name”.

#### description

Human-readable description of data.

#### COLUMN_DATA_TYPES *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]* *= {'arm_name': <class 'str'>, 'end_time': pandas.Timestamp, 'fidelities': <class 'str'>, 'frac_nonnull': <class 'numpy.float64'>, 'mean': <class 'numpy.float64'>, 'metric_name': <class 'str'>, 'n': <class 'int'>, 'random_split': <class 'int'>, 'sem': <class 'numpy.float64'>, 'start_time': pandas.Timestamp, 'trial_index': <class 'int'>}*

#### REQUIRED_COLUMNS *= {'arm_name'}*

#### *classmethod* column_data_types(extra_column_types: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [type](https://docs.python.org/3/library/functions.html#type)] | [None](https://docs.python.org/3/library/constants.html#None) = None, excluded_columns: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [type](https://docs.python.org/3/library/functions.html#type)]

Type specification for all supported columns.

#### copy_structure_with_df(df: pandas.DataFrame) → TBaseData

Serialize the structural properties needed to initialize this class.
Used for storage and to help construct new similar objects. All kwargs
other than `df` and `description` are considered structural.

#### *classmethod* deserialize_init_args(args: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], decoder_registry: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [type](https://docs.python.org/3/library/functions.html#type)[T] | [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], T]] | [None](https://docs.python.org/3/library/constants.html#None) = None, class_decoder_registry: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]], [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Given a dictionary, extract the properties needed to initialize the object.
Used for storage.

#### *property* df *: pandas.DataFrame*

#### *property* df_hash *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

Compute hash of pandas DataFrame.

This first serializes the DataFrame and computes the md5 hash on the
resulting string. Note that this may cause performance issue for very large
DataFrames.

* **Parameters:**
  **df** – The DataFrame for which to compute the hash.

Returns
: str: The hash of the DataFrame.

#### *classmethod* from_evaluations(evaluations: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]], trial_index: [int](https://docs.python.org/3/library/functions.html#int), sample_sizes: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, start_time: [int](https://docs.python.org/3/library/functions.html#int) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, end_time: [int](https://docs.python.org/3/library/functions.html#int) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → TBaseData

Convert dict of evaluations to Ax data object.

* **Parameters:**
  * **evaluations** – Map from arm name to outcomes, which itself is a mapping of
    outcome names to values, means, or tuples of mean and SEM. If SEM is
    not specified, it will be set to None and inferred from data.
  * **trial_index** – Trial index to which this data belongs.
  * **sample_sizes** – Number of samples collected for each arm.
  * **start_time** – Optional start time of run of the trial that produced this
    data, in milliseconds or iso format.  Milliseconds will be automatically
    converted to iso format because iso format automatically works with the
    pandas column type Timestamp.
  * **end_time** – Optional end time of run of the trial that produced this
    data, in milliseconds or iso format.  Milliseconds will be automatically
    converted to iso format because iso format automatically works with the
    pandas column type Timestamp.
* **Returns:**
  Ax object of the enclosing class.

#### *classmethod* from_fidelity_evaluations(evaluations: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]]], trial_index: [int](https://docs.python.org/3/library/functions.html#int), sample_sizes: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, start_time: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, end_time: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None) → TBaseData

Convert dict of fidelity evaluations to Ax data object.

* **Parameters:**
  * **evaluations** – Map from arm name to list of (fidelity, outcomes)
    where outcomes is itself a mapping of outcome names to values, means,
    or tuples of mean and SEM. If SEM is not specified, it will be set
    to None and inferred from data.
  * **trial_index** – Trial index to which this data belongs.
  * **sample_sizes** – Number of samples collected for each arm.
  * **start_time** – Optional start time of run of the trial that produced this
    data, in milliseconds.
  * **end_time** – Optional end time of run of the trial that produced this
    data, in milliseconds.
* **Returns:**
  Ax object of type `cls`.

#### *classmethod* from_multiple(data: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[TBaseData]) → TBaseData

Combines multiple objects into one (with the concatenated
underlying dataframe).

* **Parameters:**
  **data** – Iterable of Ax objects of this class to combine.

#### get_filtered_results(\*\*filters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → pandas.DataFrame

Return filtered subset of data.

* **Parameters:**
  **filter** – Column names and values they must match.

Returns
: df: The filtered DataFrame.

#### *classmethod* required_columns() → [set](https://docs.python.org/3/library/stdtypes.html#set)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

Names of columns that must be present in the underlying `DataFrame`.

#### *classmethod* serialize_init_args(obj: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Serialize the class-dependent properties needed to initialize this Data.
Used for storage and to help construct new similar Data.

#### *classmethod* supported_columns(extra_column_names: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [set](https://docs.python.org/3/library/stdtypes.html#set)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

Names of columns supported (but not necessarily required) by this class.

#### *property* true_df *: pandas.DataFrame*

Return the DataFrame being used as the source of truth (avoid using
except for caching).

### *class* ax.core.data.Data(df: pd.DataFrame | [None](https://docs.python.org/3/library/constants.html#None) = None, description: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: `BaseData`

Class storing numerical data for an experiment.

The dataframe is retrieved via the df property. The data can be stored
to an external store for future use by attaching it to an experiment using
experiment.attach_data() (this requires a description to be set.)

#### df

DataFrame with underlying data, and required columns. For BaseData, the
required columns are “arm_name”, “metric_name”, “mean”, and “sem”, the
latter two of which must be numeric.

#### description

Human-readable description of data.

#### REQUIRED_COLUMNS *: [set](https://docs.python.org/3/library/stdtypes.html#set)[[str](https://docs.python.org/3/library/stdtypes.html#str)]* *= {'arm_name', 'mean', 'metric_name', 'sem'}*

#### clone() → Data

Returns a new Data object with the same underlying dataframe.

#### filter(trial_indices: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, metric_names: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → Data

Construct a new object with the subset of rows corresponding to the
provided trial indices AND metric names. If either trial_indices or
metric_names are not provided, that dimension will not be filtered.

#### *static* from_multiple_data(data: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[Data], subset_metrics: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → Data

Combines multiple objects into one (with the concatenated
underlying dataframe).

* **Parameters:**
  * **data** – Iterable of Ax objects of this class to combine.
  * **subset_metrics** – If specified, combined object will only contain
    metrics, names of which appear in this iterable,
    in the underlying dataframe.

#### *property* metric_names *: [set](https://docs.python.org/3/library/stdtypes.html#set)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

Set of metric names that appear in the underlying dataframe of
this object.

### ax.core.data.clone_without_metrics(data: Data, excluded_metric_names: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → Data

Returns a new data object where rows containing the outcomes specified by
metric_names are filtered out. Used to sanitize data before using it as
training data for a model that requires data rectangularity.

* **Parameters:**
  * **data** – Original data to clone.
  * **excluded_metric_names** – Metrics to avoid copying
* **Returns:**
  new version of the original data without specified metrics.

### ax.core.data.custom_data_class(column_data_types: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [type](https://docs.python.org/3/library/functions.html#type)] | [None](https://docs.python.org/3/library/constants.html#None) = None, required_columns: [set](https://docs.python.org/3/library/stdtypes.html#set)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, time_columns: [set](https://docs.python.org/3/library/stdtypes.html#set)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [type](https://docs.python.org/3/library/functions.html#type)[Data]

Creates a custom data class with additional columns.

All columns and their designations on the base data class are preserved,
the inputs here are appended to the definitions on the base class.

* **Parameters:**
  * **column_data_types** – Dict from column name to column type.
  * **required_columns** – Set of additional columns required for this data object.
  * **time_columns** – Set of additional columns to cast to timestamp.
* **Returns:**
  New data subclass with amended column definitions.

### ax.core.data.set_single_trial(data: Data) → Data

Returns a new Data object where we set all rows to have the same
trial index (i.e. 0). This is meant to be used with our IVW transform,
which will combine multiple observations of the same outcome.

### Experiment

### *class* ax.core.experiment.Experiment(search_space: SearchSpace, name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [OptimizationConfig](#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, tracking_metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric] | [None](https://docs.python.org/3/library/constants.html#None) = None, runner: [Runner](#ax.core.runner.Runner) | [None](https://docs.python.org/3/library/constants.html#None) = None, status_quo: [Arm](#ax.core.arm.Arm) | [None](https://docs.python.org/3/library/constants.html#None) = None, description: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, is_test: [bool](https://docs.python.org/3/library/functions.html#bool) = False, experiment_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, properties: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, default_data_type: [DataType](#ax.core.formatting_utils.DataType) | [None](https://docs.python.org/3/library/constants.html#None) = None, auxiliary_experiments_by_purpose: [None](https://docs.python.org/3/library/constants.html#None) | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[AuxiliaryExperimentPurpose](#ax.core.auxiliary.AuxiliaryExperimentPurpose), [list](https://docs.python.org/3/library/stdtypes.html#list)[[AuxiliaryExperiment](#ax.core.auxiliary.AuxiliaryExperiment)]] = None)

Bases: [`Base`](utils.md#ax.utils.common.base.Base)

Base class for defining an experiment.

#### add_tracking_metric(metric: Metric) → [Experiment](#ax.core.experiment.Experiment)

Add a new metric to the experiment.

* **Parameters:**
  **metric** – Metric to be added.

#### add_tracking_metrics(metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric]) → [Experiment](#ax.core.experiment.Experiment)

Add a list of new metrics to the experiment.

If any of the metrics are already defined on the experiment,
we raise an error and don’t add any of them to the experiment

* **Parameters:**
  **metrics** – Metrics to be added.

#### *property* arms_by_name *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Arm](#ax.core.arm.Arm)]*

The arms belonging to this experiment, by their name.

#### *property* arms_by_signature *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Arm](#ax.core.arm.Arm)]*

The arms belonging to this experiment, by their signature.

#### *property* arms_by_signature_for_deduplication *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Arm](#ax.core.arm.Arm)]*

The arms belonging to this experiment that should be used for deduplication
in `GenerationStrategy`, by their signature.

In its current form, this includes all arms except for those that are
associated with a `FAILED` trial.
- The `CANDIDATE`, `STAGED`, `RUNNING`, and `ABANDONED` arms are
included as pending points during generation, so they should be less likely

> to get suggested by the model again.
- The `EARLY_STOPPED` and `COMPLETED` trials were already evaluated, so

the model will have data for these and is unlikely to suggest them again.

#### attach_data(data: Data, combine_with_last_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False, overwrite_existing_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [int](https://docs.python.org/3/library/functions.html#int)

Attach data to experiment. Stores data in experiment._data_by_trial,
to be looked up via experiment.lookup_data_for_trial.

* **Parameters:**
  * **data** – Data object to store.
  * **combine_with_last_data** – 

    By default, when attaching data, it’s identified
    by its timestamp, and experiment.lookup_data_for_trial returns
    data by most recent timestamp. Sometimes, however, we want to combine
    the data from multiple calls to attach_data into one dataframe.
    This might be because:
    > - We attached data for some metrics at one point and data for

    > the rest of the metrics later on.
    > - We attached data for some fidelity at one point and data for
    > another fidelity later one.

    To achieve that goal, set combine_with_last_data to True.
    In this case, we will take the most recent previously attached
    data, append the newly attached data to it, attach a new
    Data object with the merged result, and delete the old one.
    Afterwards, calls to lookup_data_for_trial will return this
    new combined data object. This operation will also validate that the
    newly added data does not contain observations for metrics that
    already have observations at the same fidelity in the most recent data.
  * **overwrite_existing_data** – By default, we keep around all data that has
    ever been attached to the experiment. However, if we know that
    the incoming data contains all the information we need for a given
    trial, we can replace the existing data for that trial, thereby
    reducing the amount we need to store in the database.
* **Returns:**
  Timestamp of storage in millis.

#### attach_fetch_results(results: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[int](https://docs.python.org/3/library/functions.html#int), [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Result](utils.md#ax.utils.common.result.Result)[Data, MetricFetchE]]], combine_with_last_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False, overwrite_existing_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)

UNSAFE: Prefer to use attach_data directly instead.

Attach fetched data results to the Experiment so they will not have to be
fetched again. Returns the timestamp from attachment, which is used as a
dict key for \_data_by_trial.

NOTE: Any Errs in the results passed in will silently be dropped! This will
cause the Experiment to fail to find them in the \_data_by_trial cache and
attempt to refetch at fetch time. If this is not your intended behavior you
MUST resolve your results first and use attach_data directly instead.

#### attach_trial(parameterizations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]], arm_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, ttl_seconds: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, run_metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, optimize_for_power: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]], [int](https://docs.python.org/3/library/functions.html#int)]

Attach a new trial with the given parameterization to the experiment.

* **Parameters:**
  * **parameterizations** – List of parameterization for the new trial. If
    only one is provided a single-arm Trial is created. If multiple
    arms are provided a BatchTrial is created.
  * **arm_names** – Names of arm(s) in the new trial.
  * **ttl_seconds** – If specified, will consider the trial failed after this
    many seconds. Used to detect dead trials that were not marked
    failed properly.
  * **run_metadata** – Metadata to attach to the trial.
  * **optimize_for_power** – For BatchTrial only.
    Whether to optimize the weights of arms in this
    trial such that the experiment’s power to detect effects of
    certain size is as high as possible. Refer to documentation of
    BatchTrial.set_status_quo_and_optimize_power for more detail.
* **Returns:**
  Tuple of arm name to parameterization dict, and trial index from
  newly created trial.

#### clone_with(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [OptimizationConfig](#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, tracking_metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric] | [None](https://docs.python.org/3/library/constants.html#None) = None, runner: [Runner](#ax.core.runner.Runner) | [None](https://docs.python.org/3/library/constants.html#None) = None, status_quo: [Arm](#ax.core.arm.Arm) | [None](https://docs.python.org/3/library/constants.html#None) = None, description: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, is_test: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = None, properties: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, trial_indices: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, data: Data | [None](https://docs.python.org/3/library/constants.html#None) = None) → [Experiment](#ax.core.experiment.Experiment)

Return a copy of this experiment with some attributes replaced.

NOTE: This method only retains the latest data attached to the experiment.
This is the same data that would be accessed using common APIs such as
`Experiment.lookup_data()`.

* **Parameters:**
  * **search_space** – New search space. If None, it uses the cloned search space
    of the original experiment.
  * **name** – New experiment name. If None, it adds 

    ```
    cloned_experiment_
    ```

      prefix
    to the original experiment name.
  * **optimization_config** – New optimization config. If None, it clones the same
    optimization_config from the orignal experiment.
  * **tracking_metrics** – New list of metrics to track. If None, it clones the
    tracking metrics already attached to the main experiment.
  * **runner** – New runner. If None, it clones the existing runner.
  * **status_quo** – New status quo arm. If None, it clones the existing status quo.
  * **description** – New description. If None, it uses the same description.
  * **is_test** – Whether the cloned experiment should be considered a test. If None,
    it uses the same value.
  * **properties** – New properties dictionary. If None, it uses a copy of the
    same properties.
  * **trial_indices** – If specified, only clones the specified trials. If None,
    clones all trials.
  * **data** – If specified, attach this data to the cloned experiment. If None,
    clones the latest data attached to the original experiment if
    the experiment has any data.

#### *property* completed_trials *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[BaseTrial](#ax.core.base_trial.BaseTrial)]*

the list of all trials for which data has arrived
or is expected to arrive.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[BaseTrial](#ax.core.base_trial.BaseTrial)]

#### *property* data_by_trial *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [OrderedDict](https://docs.python.org/3/library/collections.html#collections.OrderedDict)[[int](https://docs.python.org/3/library/functions.html#int), Data]]*

Data stored on the experiment, indexed by trial index and storage time.

First key is trial index and second key is storage time in milliseconds.
For a given trial, data is ordered by storage time, so first added data
will appear first in the list.

#### *property* default_data_constructor *: [type](https://docs.python.org/3/library/functions.html#type)*

#### *property* default_data_type *: [DataType](#ax.core.formatting_utils.DataType)*

#### *property* default_trial_type *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

Default trial type assigned to trials in this experiment.

In the base experiment class this is always None. For experiments
with multiple trial types, use the MultiTypeExperiment class.

#### *property* experiment_type *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

The type of the experiment.

#### fetch_data(metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric] | [None](https://docs.python.org/3/library/constants.html#None) = None, combine_with_last_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False, overwrite_existing_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → Data

Fetches data for all trials on this experiment and for either the
specified metrics or all metrics currently on the experiment, if metrics
argument is not specified.

NOTE: For metrics that are not available while trial is running, the data
may be retrieved from cache on the experiment. Data is cached on the experiment
via calls to experiment.attach_data and whether a given metric class is
available while trial is running is determined by the boolean returned from its
is_available_while_running class method.

NOTE: This can be lossy (ex. a MapData could get implicitly cast to a Data and
lose rows) if Experiment.default_data_type is misconfigured!

* **Parameters:**
  * **metrics** – If provided, fetch data for these metrics instead of the ones
    defined on the experiment.
  * **kwargs** – keyword args to pass to underlying metrics’ fetch data functions.
* **Returns:**
  Data for the experiment.

#### fetch_data_results(metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric] | [None](https://docs.python.org/3/library/constants.html#None) = None, combine_with_last_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False, overwrite_existing_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Result](utils.md#ax.utils.common.result.Result)[Data, MetricFetchE]]]

Fetches data for all trials on this experiment and for either the
specified metrics or all metrics currently on the experiment, if metrics
argument is not specified.

If a metric fetch fails, the Exception will be captured in the
MetricFetchResult along with a message.

NOTE: For metrics that are not available while trial is running, the data
may be retrieved from cache on the experiment. Data is cached on the experiment
via calls to experiment.attach_data and whether a given metric class is
available while trial is running is determined by the boolean returned from its
is_available_while_running class method.

* **Parameters:**
  * **metrics** – If provided, fetch data for these metrics instead of the ones
    defined on the experiment.
  * **kwargs** – keyword args to pass to underlying metrics’ fetch data functions.
* **Returns:**
  A nested Dictionary from trial_index => metric_name => result

#### fetch_trials_data(trial_indices: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[int](https://docs.python.org/3/library/functions.html#int)], metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric] | [None](https://docs.python.org/3/library/constants.html#None) = None, combine_with_last_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False, overwrite_existing_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → Data

Fetches data for specific trials on the experiment.

NOTE: For metrics that are not available while trial is running, the data
may be retrieved from cache on the experiment. Data is cached on the experiment
via calls to experiment.attach_data and whetner a given metric class is
available while trial is running is determined by the boolean returned from its
is_available_while_running class method.

NOTE: This can be lossy (ex. a MapData could get implicitly cast to a Data and
lose rows) if Experiment.default_data_type is misconfigured!

* **Parameters:**
  * **trial_indices** – Indices of trials, for which to fetch data.
  * **metrics** – If provided, fetch data for these metrics instead of the ones
    defined on the experiment.
  * **kwargs** – Keyword args to pass to underlying metrics’ fetch data functions.
* **Returns:**
  Data for the specific trials on the experiment.

#### fetch_trials_data_results(trial_indices: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[int](https://docs.python.org/3/library/functions.html#int)], metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric] | [None](https://docs.python.org/3/library/constants.html#None) = None, combine_with_last_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False, overwrite_existing_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Result](utils.md#ax.utils.common.result.Result)[Data, MetricFetchE]]]

Fetches data for specific trials on the experiment.

If a metric fetch fails, the Exception will be captured in the
MetricFetchResult along with a message.

NOTE: For metrics that are not available while trial is running, the data
may be retrieved from cache on the experiment. Data is cached on the experiment
via calls to experiment.attach_data and whether a given metric class is
available while trial is running is determined by the boolean returned from its
is_available_while_running class method.

* **Parameters:**
  * **trial_indices** – Indices of trials, for which to fetch data.
  * **metrics** – If provided, fetch data for these metrics instead of the ones
    defined on the experiment.
  * **kwargs** – keyword args to pass to underlying metrics’ fetch data functions.
* **Returns:**
  A nested Dictionary from trial_index => metric_name => result

#### get_trials_by_indices(trial_indices: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[int](https://docs.python.org/3/library/functions.html#int)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[BaseTrial](#ax.core.base_trial.BaseTrial)]

Grabs trials on this experiment by their indices.

#### *property* has_name *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Return true if experiment’s name is not None.

#### *property* immutable_search_space_and_opt_config *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Boolean representing whether search space and metrics on this experiment
are mutable (by default they are).

NOTE: For experiments with immutable search spaces and metrics, generator
runs will not store copies of search space and metrics, which improves
storage layer performance. Not keeping copies of those on generator runs
also disables keeping track of changes to search space and metrics,
thereby necessitating that those attributes be immutable on experiment.

#### *property* is_moo_problem *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Whether the experiment’s optimization config contains multiple objectives.

#### *property* is_test *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Get whether the experiment is a test.

#### lookup_data(trial_indices: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → Data

Lookup stored data for trials on this experiment.

For each trial, returns latest data object present for this trial.
Returns empty data if no data is present. In particular, this method
will not fetch data from metrics - to do that, use fetch_data() instead.

* **Parameters:**
  **trial_indices** – Indices of trials for which to fetch data. If omitted,
  lookup data for all trials on the experiment.
* **Returns:**
  Data for the trials on the experiment.

#### lookup_data_for_trial(trial_index: [int](https://docs.python.org/3/library/functions.html#int)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Data, [int](https://docs.python.org/3/library/functions.html#int)]

Lookup stored data for a specific trial.

Returns latest data object and its storage timestamp present for this trial.
Returns empty data and -1 if no data is present. In particular, this method
will not fetch data from metrics - to do that, use fetch_data() instead.

* **Parameters:**
  **trial_index** – The index of the trial to lookup data for.
* **Returns:**
  The requested data object, and its storage timestamp in milliseconds.

#### lookup_data_for_ts(timestamp: [int](https://docs.python.org/3/library/functions.html#int)) → Data

Collect data for all trials stored at this timestamp.

Useful when many trials’ data was fetched and stored simultaneously
and user wants to retrieve same collection of data later.

Can also be used to lookup specific data for a single trial
when storage time is known.

* **Parameters:**
  **timestamp** – Timestamp in millis at which data was stored.
* **Returns:**
  Data object with all data stored at the timestamp.

#### *property* metric_config_summary_df *: pandas.DataFrame*

Creates a dataframe with information about each metric in the
experiment. The resulting dataframe has one row per metric, and the
following columns:

> - Name: the name of the metric.
> - Type: the metric subclass (e.g., Metric, BraninMetric).
> - Goal: the goal for this for this metric, based on the optimization
>   config (minimize, maximize, constraint or track).
> - Bound: the bound of this metric (e.g., “<=10.0”) if it is being used
>   as part of an ObjectiveThreshold or OutcomeConstraint.
> - Lower is Better: whether the user prefers this metric to be lower,
>   if provided.

#### *property* metrics *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Metric]*

The metrics attached to the experiment.

#### *property* name *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

Get experiment name. Throws if name is None.

#### new_batch_trial(generator_run: [GeneratorRun](#ax.core.generator_run.GeneratorRun) | [None](https://docs.python.org/3/library/constants.html#None) = None, generator_runs: [list](https://docs.python.org/3/library/stdtypes.html#list)[[GeneratorRun](#ax.core.generator_run.GeneratorRun)] | [None](https://docs.python.org/3/library/constants.html#None) = None, trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, optimize_for_power: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = False, ttl_seconds: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, lifecycle_stage: [LifecycleStage](#ax.core.batch_trial.LifecycleStage) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [BatchTrial](#ax.core.batch_trial.BatchTrial)

Create a new batch trial associated with this experiment.

* **Parameters:**
  * **generator_run** – GeneratorRun, associated with this trial. This can a
    also be set later through add_arm or add_generator_run, but a
    trial’s associated generator run is immutable once set.
  * **generator_runs** – GeneratorRuns, associated with this trial. This can a
    also be set later through add_arm or add_generator_run, but a
    trial’s associated generator run is immutable once set.  This cannot
    be combined with the generator_run argument.
  * **trial_type** – Type of this trial, if used in MultiTypeExperiment.
  * **optimize_for_power** – Whether to optimize the weights of arms in this
    trial such that the experiment’s power to detect effects of
    certain size is as high as possible. Refer to documentation of
    BatchTrial.set_status_quo_and_optimize_power for more detail.
  * **ttl_seconds** – If specified, trials will be considered failed after
    this many seconds since the time the trial was ran, unless the
    trial is completed before then. Meant to be used to detect
    ‘dead’ trials, for which the evaluation process might have
    crashed etc., and which should be considered failed after
    their ‘time to live’ has passed.
  * **lifecycle_stage** – The stage of the experiment lifecycle that this
    trial represents

#### new_trial(generator_run: [GeneratorRun](#ax.core.generator_run.GeneratorRun) | [None](https://docs.python.org/3/library/constants.html#None) = None, trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, ttl_seconds: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [Trial](#ax.core.trial.Trial)

Create a new trial associated with this experiment.

* **Parameters:**
  * **generator_run** – GeneratorRun, associated with this trial.
    Trial has only one arm attached to it and this generator_run
    must therefore contain one arm. This arm can also be set later
    through add_arm or add_generator_run, but a trial’s
    associated generator run is immutable once set.
  * **trial_type** – Type of this trial, if used in MultiTypeExperiment.
  * **ttl_seconds** – If specified, trials will be considered failed after
    this many seconds since the time the trial was ran, unless the
    trial is completed before then. Meant to be used to detect
    ‘dead’ trials, for which the evaluation process might have
    crashed etc., and which should be considered failed after
    their ‘time to live’ has passed.

#### *property* num_abandoned_arms *: [int](https://docs.python.org/3/library/functions.html#int)*

How many arms attached to this experiment are abandoned.

#### *property* num_trials *: [int](https://docs.python.org/3/library/functions.html#int)*

How many trials are associated with this experiment.

#### *property* optimization_config *: [OptimizationConfig](#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)*

The experiment’s optimization config.

#### *property* parameters *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Parameter](#ax.core.parameter.Parameter)]*

The parameters in the experiment’s search space.

#### remove_tracking_metric(metric_name: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [Experiment](#ax.core.experiment.Experiment)

Remove a metric that already exists on the experiment.

* **Parameters:**
  **metric_name** – Unique name of metric to remove.

#### reset_runners(runner: [Runner](#ax.core.runner.Runner)) → [None](https://docs.python.org/3/library/constants.html#None)

Replace all candidate trials runners.

* **Parameters:**
  **runner** – New runner to replace with.

#### runner_for_trial(trial: [BaseTrial](#ax.core.base_trial.BaseTrial)) → [Runner](#ax.core.runner.Runner) | [None](https://docs.python.org/3/library/constants.html#None)

The default runner to use for a given trial.

In the base experiment class, this is always the default experiment runner.
For experiments with multiple trial types, use the MultiTypeExperiment class.

#### *property* running_trial_indices *: [set](https://docs.python.org/3/library/stdtypes.html#set)[[int](https://docs.python.org/3/library/functions.html#int)]*

Indices of running trials, associated with the experiment.

#### *property* search_space *: SearchSpace*

The search space for this experiment.

When setting a new search space, all parameter names and types
must be preserved. However, if no trials have been created, all
modifications are allowed.

#### *property* status_quo *: [Arm](#ax.core.arm.Arm) | [None](https://docs.python.org/3/library/constants.html#None)*

The existing arm that new arms will be compared against.

#### *property* sum_trial_sizes *: [int](https://docs.python.org/3/library/functions.html#int)*

Sum of numbers of arms attached to each trial in this experiment.

#### supports_trial_type(trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)) → [bool](https://docs.python.org/3/library/functions.html#bool)

Whether this experiment allows trials of the given type.

The base experiment class only supports None. For experiments
with multiple trial types, use the MultiTypeExperiment class.

#### *property* time_created *: [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime)*

Creation time of the experiment.

#### *property* tracking_metrics *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric]*

#### *property* trial_indices_by_status *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[TrialStatus](#ax.core.base_trial.TrialStatus), [set](https://docs.python.org/3/library/stdtypes.html#set)[[int](https://docs.python.org/3/library/functions.html#int)]]*

Indices of trials associated with the experiment, grouped by trial
status.

#### *property* trials *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [BaseTrial](#ax.core.base_trial.BaseTrial)]*

The trials associated with the experiment.

NOTE: If some trials on this experiment specify their TTL, RUNNING trials
will be checked for whether their TTL elapsed during this call. Found past-
TTL trials will be marked as FAILED.

#### *property* trials_by_status *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[TrialStatus](#ax.core.base_trial.TrialStatus), [list](https://docs.python.org/3/library/stdtypes.html#list)[[BaseTrial](#ax.core.base_trial.BaseTrial)]]*

Trials associated with the experiment, grouped by trial status.

#### *property* trials_expecting_data *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[BaseTrial](#ax.core.base_trial.BaseTrial)]*

the list of all trials for which data has arrived
or is expected to arrive.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[BaseTrial](#ax.core.base_trial.BaseTrial)]

#### update_tracking_metric(metric: Metric) → [Experiment](#ax.core.experiment.Experiment)

Redefine a metric that already exists on the experiment.

* **Parameters:**
  **metric** – New metric definition.

#### validate_trials(trials: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[BaseTrial](#ax.core.base_trial.BaseTrial)]) → [None](https://docs.python.org/3/library/constants.html#None)

Raise ValueError if any of the trials in the input are not from
this experiment.

#### warm_start_from_old_experiment(old_experiment: [Experiment](#ax.core.experiment.Experiment), copy_run_metadata_keys: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, trial_statuses_to_copy: [list](https://docs.python.org/3/library/stdtypes.html#list)[[TrialStatus](#ax.core.base_trial.TrialStatus)] | [None](https://docs.python.org/3/library/constants.html#None) = None, search_space_check_membership_raise_error: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Trial](#ax.core.trial.Trial)]

Copy all completed trials with data from an old Ax expeirment to this one.
This function checks that the parameters of each trial are members of the
current experiment’s search_space.

NOTE: Currently only handles experiments with 1-arm `Trial`-s, not
`BatchTrial`-s as there has not yet been need for support of the latter.

* **Parameters:**
  * **old_experiment** – The experiment from which to transfer trials and data
  * **copy_run_metadata_keys** – A list of keys denoting which items to copy over
    from each trial’s run_metadata. Defaults to
    `old_experiment.runner.run_metadata_report_keys`.
  * **trial_statuses_to_copy** – All trials with a status in this list will be
    copied. By default, copies all `RUNNING`, `COMPLETED`,
    `ABANDONED`, and `EARLY_STOPPED` trials.
  * **search_space_check_membership_raise_error** – Whether to raise an exception
    if the warm started trials being imported fall outside of the
    defined search space.
* **Returns:**
  List of trials successfully copied from old_experiment to this one

### ax.core.experiment.add_arm_and_prevent_naming_collision(new_trial: [Trial](#ax.core.trial.Trial), old_trial: [Trial](#ax.core.trial.Trial), old_experiment_name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)

### AuxiliaryExperiment

<a id="module-ax.core.auxiliary"></a>

### *class* ax.core.auxiliary.AuxiliaryExperiment(experiment: [core.experiment.Experiment](#ax.core.experiment.Experiment), data: Data | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`SortableBase`](utils.md#ax.utils.common.base.SortableBase)

Class for defining an auxiliary experiment.

### *class* ax.core.auxiliary.AuxiliaryExperimentPurpose(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

### GenerationStrategyInterface

### *class* ax.core.generation_strategy_interface.GenerationStrategyInterface(name: [str](https://docs.python.org/3/library/stdtypes.html#str))

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC), [`Base`](utils.md#ax.utils.common.base.Base)

Interface for all generation strategies: standard Ax
`GenerationStrategy`, as well as non-standard (e.g. remote, external)
generation strategies.

NOTE: Currently in Beta; please do not use without discussion with the Ax
developers.

#### DEFAULT_N *: [int](https://docs.python.org/3/library/functions.html#int)* *= 1*

#### *abstract* clone_reset() → [GenerationStrategyInterface](#ax.core.generation_strategy_interface.GenerationStrategyInterface)

Returns a clone of this generation strategy with all state reset.

#### *property* experiment *: [Experiment](#ax.core.experiment.Experiment)*

Experiment, currently set on this generation strategy.

#### *abstract* gen_for_multiple_trials_with_multiple_models(experiment: [Experiment](#ax.core.experiment.Experiment), data: Data | [None](https://docs.python.org/3/library/constants.html#None) = None, num_generator_runs: [int](https://docs.python.org/3/library/functions.html#int) = 1, n: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[GeneratorRun](#ax.core.generator_run.GeneratorRun)]]

Produce `GeneratorRun`-s for multiple trials at once with the possibility
of joining `GeneratorRun`-s from multiple models into one `BatchTrial`.

* **Parameters:**
  * **experiment** – `Experiment`, for which the generation strategy is producing
    a new generator run in the course of `gen`, and to which that
    generator run will be added as trial(s). Information stored on the
    experiment (e.g., trial statuses) is used to determine which model
    will be used to produce the generator run returned from this method.
  * **data** – Optional data to be passed to the underlying model’s `gen`, which
    is called within this method and actually produces the resulting
    generator run. By default, data is all data on the `experiment`.
  * **n** – Integer representing how many trials should be in the generator run
    produced by this method. NOTE: Some underlying models may ignore
    the `n` and produce a model-determined number of arms. In that
    case this method will also output a generator run with number of
    arms that can differ from `n`.
  * **pending_observations** – A map from metric name to pending
    observations for that metric, used by some models to avoid
    resuggesting points that are currently being evaluated.
* **Returns:**
  A list of lists of `GeneratorRun`-s. Each outer list item represents
  a `(Batch)Trial` being suggested, with a list of `GeneratorRun`-s for
  that trial.

#### *property* name *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

Name of this generation strategy.

### GeneratorRun

### *class* ax.core.generator_run.ArmWeight(arm: [Arm](#ax.core.arm.Arm), weight: [float](https://docs.python.org/3/library/functions.html#float))

Bases: [`Base`](utils.md#ax.utils.common.base.Base)

NamedTuple for tying together arms and weights.

#### arm *: [Arm](#ax.core.arm.Arm)*

#### weight *: [float](https://docs.python.org/3/library/functions.html#float)*

### *class* ax.core.generator_run.GeneratorRun(arms: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Arm](#ax.core.arm.Arm)], weights: [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [OptimizationConfig](#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, model_predictions: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]]]] | [None](https://docs.python.org/3/library/constants.html#None) = None, best_arm_predictions: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[Arm](#ax.core.arm.Arm), [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]] | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None, type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, fit_time: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, gen_time: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, model_key: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, model_kwargs: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, bridge_kwargs: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, gen_metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, model_state_after_gen: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, generation_step_index: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, candidate_metadata_by_arm_signature: [None](https://docs.python.org/3/library/constants.html#None) | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)] = None, generation_node_name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`SortableBase`](utils.md#ax.utils.common.base.SortableBase)

An object that represents a single run of a generator.

This object is created each time the `gen` method of a generator is
called. It stores the arms and (optionally) weights that were
generated by the run. When we add a generator run to a trial, its
arms and weights will be merged with those from previous generator
runs that were already attached to the trial.

#### add_arm(arm: [Arm](#ax.core.arm.Arm), weight: [float](https://docs.python.org/3/library/functions.html#float) = 1.0) → [None](https://docs.python.org/3/library/constants.html#None)

Adds an arm to this generator run.  This should not be used to
mutate generator runs that are attached to trials.

* **Parameters:**
  * **arm** – The arm to add.
  * **weight** – The weight to associate with the arm.

#### *property* arm_signatures *: [set](https://docs.python.org/3/library/stdtypes.html#set)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

Returns signatures of arms generated by this run.

#### *property* arm_weights *: [MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[Arm](#ax.core.arm.Arm), [float](https://docs.python.org/3/library/functions.html#float)]*

Mapping from arms to weights (order matches order in
arms property).

#### *property* arms *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Arm](#ax.core.arm.Arm)]*

Returns arms generated by this run.

#### *property* best_arm_predictions *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[Arm](#ax.core.arm.Arm), [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]] | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None)*

Best arm in this run (according to the optimization config) and its
optional respective model predictions.

#### *property* candidate_metadata_by_arm_signature *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None)*

Retrieves model-produced candidate metadata as a mapping from arm name (for
the arm the candidate became when added to experiment) to the metadata dict.

#### clone() → [GeneratorRun](#ax.core.generator_run.GeneratorRun)

Return a deep copy of a GeneratorRun.

#### *property* fit_time *: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Time taken to fit the model in seconds.

#### *property* gen_metadata *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)*

Returns metadata generated by this run.

#### *property* gen_time *: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Time taken to generate in seconds.

#### *property* generator_run_type *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

The type of the generator run.

#### *property* index *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

The index of this generator run within a trial’s list of generator run
structs. This field is set when the generator run is added to a trial.

#### *property* model_predictions *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]]]] | [None](https://docs.python.org/3/library/constants.html#None)*

Means and covariances for the arms in this run recorded at
the time the run was executed.

#### *property* model_predictions_by_arm *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]] | [None](https://docs.python.org/3/library/constants.html#None)]] | [None](https://docs.python.org/3/library/constants.html#None)*

Model predictions for each arm in this run, at the time the run was
executed.

#### *property* optimization_config *: [OptimizationConfig](#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)*

The optimization config used during generation of this run.

#### *property* param_df *: pandas.DataFrame*

Constructs a Pandas dataframe with the parameter values for each arm.

Useful for inspecting the contents of a generator run.

* **Returns:**
  a dataframe with the generator run’s arms.
* **Return type:**
  pd.DataFrame

#### *property* search_space *: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None)*

The search used during generation of this run.

#### *property* time_created *: [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime)*

Creation time of the batch.

#### *property* weights *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]*

Returns weights associated with arms generated by this run.

### *class* ax.core.generator_run.GeneratorRunType(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Class for enumerating generator run types.

#### MANUAL *= 1*

#### STATUS_QUO *= 0*

### ax.core.generator_run.extract_arm_predictions(model_predictions: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]]]], arm_idx: [int](https://docs.python.org/3/library/functions.html#int)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]] | [None](https://docs.python.org/3/library/constants.html#None)]

Extract a particular arm from model_predictions.

* **Parameters:**
  * **model_predictions** – Mean and Cov for all arms.
  * **arm_idx** – Index of arm in prediction list.
* **Returns:**
  (mean, cov) for specified arm.

### MapData

### *class* ax.core.map_data.MapData(df: pd.DataFrame | [None](https://docs.python.org/3/library/constants.html#None) = None, map_key_infos: Iterable[MapKeyInfo] | [None](https://docs.python.org/3/library/constants.html#None) = None, description: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: `Data`

Class storing mapping-like results for an experiment.

Data is stored in a dataframe, and auxiliary information ((key name,
default value) pairs) are stored in a collection of MapKeyInfo objects.

Mapping-like results occur whenever a metric is reported as a collection
of results, each element corresponding to a tuple of values.

The simplest case is a sequence. For instance a time series is
a mapping from the 1-tuple (timestamp) to (mean, sem) results.

Another example: MultiFidelity results. This is a mapping from
(fidelity_feature_1, …, fidelity_feature_n) to (mean, sem) results.

The dataframe is retrieved via the map_df property. The data can be stored
to an external store for future use by attaching it to an experiment using
experiment.attach_data() (this requires a description to be set.)

#### DEDUPLICATE_BY_COLUMNS *= ['arm_name', 'metric_name']*

#### clone() → MapData

Returns a new `MapData` object with the same underlying dataframe
and map key infos.

#### *classmethod* deserialize_init_args(args: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], decoder_registry: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [type](https://docs.python.org/3/library/functions.html#type)[T] | [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], T]] | [None](https://docs.python.org/3/library/constants.html#None) = None, class_decoder_registry: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]], [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Given a dictionary, extract the properties needed to initialize the metric.
Used for storage.

#### *property* df *: pandas.DataFrame*

Returns a Data shaped DataFrame

#### filter(trial_indices: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, metric_names: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → MapData

Construct a new object with the subset of rows corresponding to the
provided trial indices AND metric names. If either trial_indices or
metric_names are not provided, that dimension will not be filtered.

#### *static* from_map_evaluations(evaluations: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Hashable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]]], trial_index: [int](https://docs.python.org/3/library/functions.html#int), map_key_infos: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[MapKeyInfo] | [None](https://docs.python.org/3/library/constants.html#None) = None) → MapData

#### *static* from_multiple_data(data: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[Data], subset_metrics: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → MapData

Downcast instances of Data into instances of MapData with empty
map_key_infos if necessary then combine as usual (filling in empty cells with
default values).

#### *static* from_multiple_map_data(data: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[MapData], subset_metrics: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → MapData

#### *property* map_df *: pandas.DataFrame*

#### *property* map_key_infos *: [list](https://docs.python.org/3/library/stdtypes.html#list)[MapKeyInfo]*

#### *property* map_key_to_type *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [type](https://docs.python.org/3/library/functions.html#type)]*

#### *property* map_keys *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### *classmethod* serialize_init_args(obj: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Serialize the class-dependent properties needed to initialize this Data.
Used for storage and to help construct new similar Data.

#### subsample(map_key: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, keep_every: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, limit_rows_per_group: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, limit_rows_per_metric: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, include_first_last: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → MapData

Subsample the map_key column in an equally-spaced manner (if there is
a self.map_keys is length one, then map_key can be set to None). The
values of the map_key column are not taken into account, so this function
is most reasonable when those values are equally-spaced. There are three
ways that this can be done:

> 1. If keep_every = k is set, then every kth row of the DataFrame in the
>    : map_key column is kept after grouping by DEDUPLICATE_BY_COLUMNS.
>      In other words, every kth step of each (arm, metric) will be kept.
> 2. If limit_rows_per_group = n, the method will find the (arm, metric)
>    : pair with the largest number of rows in the map_key column and select
>      an approprioate keep_every such that each (arm, metric) has at most
>      n rows in the map_key column.
> 3. If limit_rows_per_metric = n, the method will select an
>    : appropriate keep_every such that the total number of rows per
>      metric is less than n.

If multiple of keep_every, limit_rows_per_group, limit_rows_per_metric,
then the priority is in the order above: 1. keep_every,
2. limit_rows_per_group, and 3. limit_rows_per_metric.

Note that we want all curves to be subsampled with nearly the same spacing.
Internally, the method converts limit_rows_per_group and
limit_rows_per_metric to a keep_every quantity that will satisfy the
original request.

When include_first_last is True, then the method will use the keep_every
as a guideline and for each group, produce (nearly) evenly spaced points that
include the first and last points.

#### *property* true_df *: pandas.DataFrame*

Return the DataFrame being used as the source of truth (avoid using
except for caching).

### *class* ax.core.map_data.MapKeyInfo(key: [str](https://docs.python.org/3/library/stdtypes.html#str), default_value: T)

Bases: [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`T`], [`SortableBase`](utils.md#ax.utils.common.base.SortableBase)

Helper class storing map keys and auxilary info for use in MapData

#### clone() → MapKeyInfo[T]

Return a copy of this MapKeyInfo.

#### *property* default_value *: T*

#### *property* key *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### *property* value_type *: [type](https://docs.python.org/3/library/functions.html#type)*

### MapMetric

### *class* ax.core.map_metric.MapMetric(name: [str](https://docs.python.org/3/library/stdtypes.html#str), lower_is_better: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = None, properties: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: `Metric`

Base class for representing metrics that return MapData.

The fetch_trial_data method is the essential method to override when
subclassing, which specifies how to retrieve a Metric, for a given trial.

A MapMetric must return a MapData object, which requires (at minimum) the following:
: [https://ax.dev/api/_modules/ax/core/data.html#Data.required_columns](https://ax.dev/api/_modules/ax/core/data.html#Data.required_columns)

#### lower_is_better

Flag for metrics which should be minimized.

#### properties

Properties specific to a particular metric.

#### data_constructor

alias of `MapData`

#### map_key_info *: MapKeyInfo[[float](https://docs.python.org/3/library/functions.html#float)]* *= <ax.core.map_data.MapKeyInfo object>*

### Metric

### *class* ax.core.metric.Metric(name: [str](https://docs.python.org/3/library/stdtypes.html#str), lower_is_better: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = None, properties: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`SortableBase`](utils.md#ax.utils.common.base.SortableBase), [`SerializationMixin`](utils.md#ax.utils.common.serialization.SerializationMixin)

Base class for representing metrics.

The fetch_trial_data method is the essential method to override when
subclassing, which specifies how to retrieve a Metric, for a given trial.

A Metric must return a Data object, which requires (at minimum) the following:
: [https://ax.dev/api/_modules/ax/core/data.html#Data.required_columns](https://ax.dev/api/_modules/ax/core/data.html#Data.required_columns)

#### lower_is_better

Flag for metrics which should be minimized.

#### properties

Properties specific to a particular metric.

#### bulk_fetch_experiment_data(experiment: [core.experiment.Experiment](#ax.core.experiment.Experiment), metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric], trials: [list](https://docs.python.org/3/library/stdtypes.html#list)[[core.base_trial.BaseTrial](#ax.core.base_trial.BaseTrial)] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: Any) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), MetricFetchResult]]

Fetch multiple metrics data for multiple trials on an experiment, using
instance attributes of the metrics.

Returns Dict of metric_name => Result
Default behavior calls fetch_trial_data for each metric.
Subclasses should override this to trial data computation for multiple metrics.

#### bulk_fetch_trial_data(trial: [core.base_trial.BaseTrial](#ax.core.base_trial.BaseTrial), metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric], \*\*kwargs: Any) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), MetricFetchResult]

Fetch multiple metrics data for one trial, using instance attributes
of the metrics.

Returns Dict of metric_name => Result
Default behavior calls fetch_trial_data for each metric. Subclasses should
override this to perform trial data computation for multiple metrics.

#### clone() → Metric

Create a copy of this Metric.

#### data_constructor

alias of `Data`

#### fetch_data_prefer_lookup(experiment: [core.experiment.Experiment](#ax.core.experiment.Experiment), metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric], trials: [list](https://docs.python.org/3/library/stdtypes.html#list)[[core.base_trial.BaseTrial](#ax.core.base_trial.BaseTrial)] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: Any) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), MetricFetchResult]], [bool](https://docs.python.org/3/library/functions.html#bool)]

Fetch or lookup (with fallback to fetching) data for given metrics,
depending on whether they are available while running. Return a tuple
containing the data, along with a boolean that will be True if new
data was fetched, and False if all data was looked up from cache.

If metric is available while running, its data can change (and therefore
we should always re-fetch it). If metric is available only upon trial
completion, its data does not change, so we can look up that data on
the experiment and only fetch the data that is not already attached to
the experiment.

NOTE: If fetching data for a metrics class that is only available upon
trial completion, data fetched in this function (data that was not yet
available on experiment) will be attached to experiment.

#### *classmethod* fetch_experiment_data_multi(experiment: [core.experiment.Experiment](#ax.core.experiment.Experiment), metrics: Iterable[Metric], trials: Iterable[[core.base_trial.BaseTrial](#ax.core.base_trial.BaseTrial)] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: Any) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), MetricFetchResult]]

Fetch multiple metrics data for an experiment.

Returns Dict of trial_index => (metric_name => Result)
Default behavior calls fetch_trial_data_multi for each trial.
Subclasses should override to batch data computation across trials + metrics.

#### *property* fetch_multi_group_by_metric *: [type](https://docs.python.org/3/library/functions.html#type)[Metric]*

Metric class, with which to group this metric in
Experiment._metrics_by_class, which is used to combine metrics on experiment
into groups and then fetch their data via Metric.fetch_trial_data_multi for
each group.

NOTE: By default, this property will just return the class on which it is
defined; however, in some cases it is useful to group metrics by their
superclass, in which case this property should return that superclass.

#### fetch_trial_data(trial: [core.base_trial.BaseTrial](#ax.core.base_trial.BaseTrial), \*\*kwargs: Any) → MetricFetchResult

Fetch data for one trial.

#### *classmethod* fetch_trial_data_multi(trial: [core.base_trial.BaseTrial](#ax.core.base_trial.BaseTrial), metrics: Iterable[Metric], \*\*kwargs: Any) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), MetricFetchResult]

Fetch multiple metrics data for one trial.

Returns Dict of metric_name => Result
Default behavior calls fetch_trial_data for each metric.
Subclasses should override this to trial data computation for multiple metrics.

#### *classmethod* is_available_while_running() → [bool](https://docs.python.org/3/library/functions.html#bool)

Whether metrics of this class are available while the trial is running.
Metrics that are not available while the trial is running are assumed to be
available only upon trial completion. For such metrics, data is assumed to
never change once the trial is completed.

NOTE: If this method returns False, data-fetching via experiment.fetch_data
will return the data cached on the experiment (for the metrics of the given
class) whenever its available. Data is cached on experiment when attached
via experiment.attach_data.

#### maybe_raise_deprecation_warning_on_class_methods() → [None](https://docs.python.org/3/library/constants.html#None)

#### *property* name *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

Get name of metric.

#### *classmethod* period_of_new_data_after_trial_completion() → [timedelta](https://docs.python.org/3/library/datetime.html#datetime.timedelta)

Period of time metrics of this class are still expecting new data to arrive
after trial completion.  This is useful for metrics whose results are processed
by some sort of data pipeline, where the pipeline will continue to land
additional data even after the trial is completed.

If the metric is not available after trial completion, this method will
return timedelta(0). Otherwise, it should return the maximum amount of time
that the metric may have new data arrive after the trial is completed.

NOTE: This property will not prevent new data from attempting to be refetched
for completed trials when calling experiment.fetch_data().  Its purpose is to
prevent experiment.fetch_data() from being called in Scheduler and anywhere
else it is checked.

#### *property* summary_dict *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*

Returns a dictionary containing the metric’s name and properties.

### *class* ax.core.metric.MetricFetchE(message: 'str', exception: 'Exception | None')

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

#### exception *: [Exception](https://docs.python.org/3/library/exceptions.html#Exception) | [None](https://docs.python.org/3/library/constants.html#None)*

#### message *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### tb_str() → [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)

### MultiTypeExperiment

### *class* ax.core.multi_type_experiment.MultiTypeExperiment(name: [str](https://docs.python.org/3/library/stdtypes.html#str), search_space: SearchSpace, default_trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str), default_runner: [Runner](#ax.core.runner.Runner), optimization_config: [OptimizationConfig](#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, status_quo: [Arm](#ax.core.arm.Arm) | [None](https://docs.python.org/3/library/constants.html#None) = None, description: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, is_test: [bool](https://docs.python.org/3/library/functions.html#bool) = False, experiment_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, properties: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, default_data_type: [DataType](#ax.core.formatting_utils.DataType) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Experiment`](#ax.core.experiment.Experiment)

Class for experiment with multiple trial types.

A canonical use case for this is tuning a large production system
with limited evaluation budget and a simulator which approximates
evaluations on the main system. Trial deployment and data fetching
is separate for the two systems, but the final data is combined and
fed into multi-task models.

See the Multi-Task Modeling tutorial for more details.

#### name

Name of the experiment.

#### description

Description of the experiment.

#### add_tracking_metric(metric: Metric, trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str), canonical_name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [MultiTypeExperiment](#ax.core.multi_type_experiment.MultiTypeExperiment)

Add a new metric to the experiment.

* **Parameters:**
  * **metric** – The metric to add.
  * **trial_type** – The trial type for which this metric is used.
  * **canonical_name** – The default metric for which this metric is a proxy.

#### add_trial_type(trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str), runner: [Runner](#ax.core.runner.Runner)) → [MultiTypeExperiment](#ax.core.multi_type_experiment.MultiTypeExperiment)

Add a new trial_type to be supported by this experiment.

* **Parameters:**
  * **trial_type** – The new trial_type to be added.
  * **runner** – The default runner for trials of this type.

#### *property* default_trial_type *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

Default trial type assigned to trials in this experiment.

#### *property* default_trials *: [set](https://docs.python.org/3/library/stdtypes.html#set)[[int](https://docs.python.org/3/library/functions.html#int)]*

Return the indicies for trials of the default type.

#### fetch_data(metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric] | [None](https://docs.python.org/3/library/constants.html#None) = None, combine_with_last_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False, overwrite_existing_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → Data

Fetches data for all trials on this experiment and for either the
specified metrics or all metrics currently on the experiment, if metrics
argument is not specified.

NOTE: For metrics that are not available while trial is running, the data
may be retrieved from cache on the experiment. Data is cached on the experiment
via calls to experiment.attach_data and whether a given metric class is
available while trial is running is determined by the boolean returned from its
is_available_while_running class method.

NOTE: This can be lossy (ex. a MapData could get implicitly cast to a Data and
lose rows) if Experiment.default_data_type is misconfigured!

* **Parameters:**
  * **metrics** – If provided, fetch data for these metrics instead of the ones
    defined on the experiment.
  * **kwargs** – keyword args to pass to underlying metrics’ fetch data functions.
* **Returns:**
  Data for the experiment.

#### *property* metric_to_trial_type *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]*

Map metrics to trial types.

Adds in default trial type for OC metrics to custom defined trial types..

#### *property* optimization_config *: [OptimizationConfig](#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)*

The experiment’s optimization config.

#### remove_tracking_metric(metric_name: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [MultiTypeExperiment](#ax.core.multi_type_experiment.MultiTypeExperiment)

Remove a metric that already exists on the experiment.

* **Parameters:**
  **metric_name** – Unique name of metric to remove.

#### reset_runners(runner: [Runner](#ax.core.runner.Runner)) → [None](https://docs.python.org/3/library/constants.html#None)

Replace all candidate trials runners.

* **Parameters:**
  **runner** – New runner to replace with.

#### runner_for_trial(trial: [BaseTrial](#ax.core.base_trial.BaseTrial)) → [Runner](#ax.core.runner.Runner) | [None](https://docs.python.org/3/library/constants.html#None)

The default runner to use for a given trial.

Looks up the appropriate runner for this trial type in the trial_type_to_runner.

#### runner_for_trial_type(trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [Runner](#ax.core.runner.Runner) | [None](https://docs.python.org/3/library/constants.html#None)

The default runner to use for a given trial type.

Looks up the appropriate runner for this trial type in the trial_type_to_runner.

#### supports_trial_type(trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)) → [bool](https://docs.python.org/3/library/functions.html#bool)

Whether this experiment allows trials of the given type.

Only trial types defined in the trial_type_to_runner are allowed.

#### update_runner(trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str), runner: [Runner](#ax.core.runner.Runner)) → [MultiTypeExperiment](#ax.core.multi_type_experiment.MultiTypeExperiment)

Update the default runner for an existing trial_type.

* **Parameters:**
  * **trial_type** – The new trial_type to be added.
  * **runner** – The new runner for trials of this type.

#### update_tracking_metric(metric: Metric, trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str), canonical_name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [MultiTypeExperiment](#ax.core.multi_type_experiment.MultiTypeExperiment)

Update an existing metric on the experiment.

* **Parameters:**
  * **metric** – The metric to add.
  * **trial_type** – The trial type for which this metric is used.
  * **canonical_name** – The default metric for which this metric is a proxy.

### ax.core.multi_type_experiment.filter_trials_by_type(trials: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[BaseTrial](#ax.core.base_trial.BaseTrial)], trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[BaseTrial](#ax.core.base_trial.BaseTrial)]

Filter trials by trial type if provided.

This filters trials by trial type if the experiment is a
MultiTypeExperiment.

* **Parameters:**
  **trials** – Trials to filter.
* **Returns:**
  Filtered trials.

### ax.core.multi_type_experiment.get_trial_indices_for_statuses(experiment: [Experiment](#ax.core.experiment.Experiment), statuses: [set](https://docs.python.org/3/library/stdtypes.html#set)[[TrialStatus](#ax.core.base_trial.TrialStatus)], trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [set](https://docs.python.org/3/library/stdtypes.html#set)[[int](https://docs.python.org/3/library/functions.html#int)]

Get trial indices for a set of statuses.

* **Parameters:**
  **statuses** – Set of statuses to get trial indices for.
* **Returns:**
  Set of trial indices for the given statuses.

### Objective

### *class* ax.core.objective.MultiObjective(objectives: [list](https://docs.python.org/3/library/stdtypes.html#list)[Objective] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*extra_kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

Bases: `Objective`

Class for an objective composed of a multiple component objectives.

The Acquisition function determines how the objectives are weighted.

#### objectives

List of objectives.

#### clone() → MultiObjective

Create a copy of the objective.

#### *property* metric *: Metric*

Override base method to error.

#### *property* metrics *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric]*

Get the objective metrics.

#### *property* objective_weights *: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Objective, [float](https://docs.python.org/3/library/functions.html#float)]]*

Get the objectives and weights.

#### *property* objectives *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Objective]*

Get the objectives.

#### weights *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]*

### *class* ax.core.objective.Objective(metric: Metric, minimize: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`SortableBase`](utils.md#ax.utils.common.base.SortableBase)

Base class for representing an objective.

#### minimize

If True, minimize metric.

#### clone() → Objective

Create a copy of the objective.

#### get_unconstrainable_metrics() → [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric]

Return a list of metrics that are incompatible with OutcomeConstraints.

#### *property* metric *: Metric*

Get the objective metric.

#### *property* metric_names *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

Get a list of objective metric names.

#### *property* metrics *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric]*

Get a list of objective metrics.

### *class* ax.core.objective.ScalarizedObjective(metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric], weights: [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, minimize: [bool](https://docs.python.org/3/library/functions.html#bool) = False)

Bases: `Objective`

Class for an objective composed of a linear scalarization of metrics.

#### metrics

List of metrics.

#### weights

Weights for scalarization; default to 1.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]

#### clone() → ScalarizedObjective

Create a copy of the objective.

#### *property* metric *: Metric*

Override base method to error.

#### *property* metric_weights *: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Metric, [float](https://docs.python.org/3/library/functions.html#float)]]*

Get the metrics and weights.

#### *property* metrics *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric]*

Get the metrics.

#### weights *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]*

### Observation

### *class* ax.core.observation.Observation(features: [ObservationFeatures](#ax.core.observation.ObservationFeatures), data: [ObservationData](#ax.core.observation.ObservationData), arm_name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Base`](utils.md#ax.utils.common.base.Base)

Represents an observation.

A set of features (ObservationFeatures) and corresponding measurements
(ObservationData). Optionally, an arm name associated with the features.

#### features

* **Type:**
  [ObservationFeatures](#ax.core.observation.ObservationFeatures)

#### data

* **Type:**
  [ObservationData](#ax.core.observation.ObservationData)

#### arm_name

* **Type:**
  Optional[[str](https://docs.python.org/3/library/stdtypes.html#str)]

### *class* ax.core.observation.ObservationData(metric_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], means: ndarray, covariance: ndarray)

Bases: [`Base`](utils.md#ax.utils.common.base.Base)

Outcomes observed at a point.

The “point” corresponding to this ObservationData would be an
ObservationFeatures object.

#### metric_names

A list of k metric names that were observed

#### means

a k-array of observed means

#### covariance

a (k x k) array of observed covariances

#### *property* covariance_matrix *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]]*

Extract covariance matric from this observation data as mapping from
metric name (m1) to mapping of another metric name (m2) to the covariance
of the two metrics (m1 and m2).

#### *property* means_dict *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]*

Extract means from this observation data as mapping from metric name to
mean.

### *class* ax.core.observation.ObservationFeatures(parameters: TParameterization, trial_index: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, start_time: pd.Timestamp | [None](https://docs.python.org/3/library/constants.html#None) = None, end_time: pd.Timestamp | [None](https://docs.python.org/3/library/constants.html#None) = None, random_split: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, metadata: TCandidateMetadata = None)

Bases: [`Base`](utils.md#ax.utils.common.base.Base)

The features of an observation.

These include both the arm parameters and the features of the
observation found in the Data object: trial index, times,
and random split. This object is meant to contain everything needed to
represent this observation in a model feature space. It is essentially a
row of Data joined with the arm parameters.

An ObservationFeatures object would typically have a corresponding
ObservationData object that provides the observed outcomes.

#### parameters

arm parameters

#### trial_index

trial index

#### start_time

batch start time

#### end_time

batch end time

#### random_split

random split

#### clone(replace_parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [ObservationFeatures](#ax.core.observation.ObservationFeatures)

Make a copy of these `ObservationFeatures`.

* **Parameters:**
  **replace_parameters** – An optimal parameterization, to which to set the
  parameters of the cloned `ObservationFeatures`. Useful when
  transforming observation features in a way that requires a
  change to parameterization –– for example, while casting it to
  a hierarchical search space.

#### *static* from_arm(arm: [Arm](#ax.core.arm.Arm), trial_index: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, start_time: pd.Timestamp | [None](https://docs.python.org/3/library/constants.html#None) = None, end_time: pd.Timestamp | [None](https://docs.python.org/3/library/constants.html#None) = None, random_split: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, metadata: TCandidateMetadata = None) → [ObservationFeatures](#ax.core.observation.ObservationFeatures)

Convert a Arm to an ObservationFeatures, including additional
data as specified.

#### update_features(new_features: [ObservationFeatures](#ax.core.observation.ObservationFeatures)) → [ObservationFeatures](#ax.core.observation.ObservationFeatures)

Updates the existing ObservationFeatures with the fields of the the input.

Adds all of the new parameters to the existing parameters and overwrites
any other fields that are not None on the new input features.

### ax.core.observation.get_feature_cols(data: Data, is_map_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

### ax.core.observation.observations_from_data(experiment: [Experiment](#ax.core.experiment.Experiment), data: Data, statuses_to_include: [set](https://docs.python.org/3/library/stdtypes.html#set)[[TrialStatus](#ax.core.base_trial.TrialStatus)] | [None](https://docs.python.org/3/library/constants.html#None) = None, statuses_to_include_map_metric: [set](https://docs.python.org/3/library/stdtypes.html#set)[[TrialStatus](#ax.core.base_trial.TrialStatus)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](#ax.core.observation.Observation)]

Convert Data to observations.

Converts a Data object to a list of Observation objects. Pulls arm parameters from
from experiment. Overrides fidelity parameters in the arm with those found in the
Data object.

Uses a diagonal covariance matrix across metric_names.

* **Parameters:**
  * **experiment** – Experiment with arm parameters.
  * **data** – Data of observations.
  * **statuses_to_include** – data from non-MapMetrics will only be included for trials
    with statuses in this set. Defaults to all statuses except abandoned.
  * **statuses_to_include_map_metric** – data from MapMetrics will only be included for
    trials with statuses in this set. Defaults to completed status only.
* **Returns:**
  List of Observation objects.

### ax.core.observation.observations_from_map_data(experiment: [Experiment](#ax.core.experiment.Experiment), map_data: MapData, statuses_to_include: [set](https://docs.python.org/3/library/stdtypes.html#set)[[TrialStatus](#ax.core.base_trial.TrialStatus)] | [None](https://docs.python.org/3/library/constants.html#None) = None, statuses_to_include_map_metric: [set](https://docs.python.org/3/library/stdtypes.html#set)[[TrialStatus](#ax.core.base_trial.TrialStatus)] | [None](https://docs.python.org/3/library/constants.html#None) = None, map_keys_as_parameters: [bool](https://docs.python.org/3/library/functions.html#bool) = False, limit_rows_per_metric: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, limit_rows_per_group: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](#ax.core.observation.Observation)]

Convert MapData to observations.

Converts a MapData object to a list of Observation objects. Pulls arm parameters
from experiment. Overrides fidelity parameters in the arm with those found in the
Data object.

Uses a diagonal covariance matrix across metric_names.

* **Parameters:**
  * **experiment** – Experiment with arm parameters.
  * **map_data** – MapData of observations.
  * **statuses_to_include** – data from non-MapMetrics will only be included for trials
    with statuses in this set. Defaults to all statuses except abandoned.
  * **statuses_to_include_map_metric** – data from MapMetrics will only be included for
    trials with statuses in this set. Defaults to all statuses except abandoned.
  * **map_keys_as_parameters** – Whether map_keys should be returned as part of
    the parameters of the Observation objects.
  * **limit_rows_per_metric** – If specified, uses MapData.subsample() with
    limit_rows_per_metric equal to the specified value on the first
    map_key (map_data.map_keys[0]) to subsample the MapData. This is
    useful in, e.g., cases where learning curves are frequently
    updated, leading to an intractable number of Observation objects
    created.
  * **limit_rows_per_group** – If specified, uses MapData.subsample() with
    limit_rows_per_group equal to the specified value on the first
    map_key (map_data.map_keys[0]) to subsample the MapData.
* **Returns:**
  List of Observation objects.

### ax.core.observation.recombine_observations(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](#ax.core.observation.ObservationFeatures)], observation_data: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationData](#ax.core.observation.ObservationData)], arm_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](#ax.core.observation.Observation)]

Construct a list of 

```
`
```

Observation\`s from the given arguments.

In the returned list of Observation\`s, element \`i has features from
observation_features[i], data from observation_data[i], and, if
applicable, arm_name from arm_names[i].

### ax.core.observation.separate_observations(observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](#ax.core.observation.Observation)], copy: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](#ax.core.observation.ObservationFeatures)], [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationData](#ax.core.observation.ObservationData)]]

Split out observations into features+data.

* **Parameters:**
  **observations** – input observations
* **Returns:**
  ObservationFeatures
  observation_data: ObservationData
* **Return type:**
  observation_features

### OptimizationConfig

### *class* ax.core.optimization_config.MultiObjectiveOptimizationConfig(objective: MultiObjective | ScalarizedObjective, outcome_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint] | [None](https://docs.python.org/3/library/constants.html#None) = None, objective_thresholds: [list](https://docs.python.org/3/library/stdtypes.html#list)[ObjectiveThreshold] | [None](https://docs.python.org/3/library/constants.html#None) = None, risk_measure: [RiskMeasure](#ax.core.risk_measures.RiskMeasure) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`OptimizationConfig`](#ax.core.optimization_config.OptimizationConfig)

An optimization configuration for multi-objective optimization,
which comprises multiple objective, outcome constraints, objective
thresholds, and an optional risk measure.

There is no minimum or maximum number of outcome constraints, but an
individual metric can have at most two constraints–which is how we
represent metrics with both upper and lower bounds.

ObjectiveThresholds should be present for every objective. A good
rule of thumb is to set them 10% below the minimum acceptable value
for each metric.

#### *property* all_constraints *: [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint]*

Get all constraints and thresholds.

#### clone_with_args(objective: ~ax.core.objective.MultiObjective | ~ax.core.objective.ScalarizedObjective | None = None, outcome_constraints: None | list[~ax.core.outcome_constraint.OutcomeConstraint] = [OutcomeConstraint( >= 0%)], objective_thresholds: None | list[~ax.core.outcome_constraint.ObjectiveThreshold] = [ObjectiveThreshold( <= 0%)], risk_measure: ~ax.core.risk_measures.RiskMeasure | None = RiskMeasure(risk_measure=, options={})) → [MultiObjectiveOptimizationConfig](#ax.core.optimization_config.MultiObjectiveOptimizationConfig)

Make a copy of this optimization config.

#### *property* metrics *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Metric]*

#### *property* objective *: MultiObjective | ScalarizedObjective*

Get objective.

#### *property* objective_thresholds *: [list](https://docs.python.org/3/library/stdtypes.html#list)[ObjectiveThreshold]*

Get objective thresholds.

#### *property* objective_thresholds_dict *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), ObjectiveThreshold]*

Get a mapping from objective metric name to the corresponding
threshold.

### *class* ax.core.optimization_config.OptimizationConfig(objective: Objective, outcome_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint] | [None](https://docs.python.org/3/library/constants.html#None) = None, risk_measure: [RiskMeasure](#ax.core.risk_measures.RiskMeasure) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Base`](utils.md#ax.utils.common.base.Base)

An optimization configuration, which comprises an objective,
outcome constraints and an optional risk measure.

There is no minimum or maximum number of outcome constraints, but an
individual metric can have at most two constraints–which is how we
represent metrics with both upper and lower bounds.

#### *property* all_constraints *: [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint]*

Get outcome constraints.

#### clone() → [OptimizationConfig](#ax.core.optimization_config.OptimizationConfig)

Make a copy of this optimization config.

#### clone_with_args(objective: ~ax.core.objective.Objective | None = None, outcome_constraints: None | list[~ax.core.outcome_constraint.OutcomeConstraint] = [OutcomeConstraint( >= 0%)], risk_measure: ~ax.core.risk_measures.RiskMeasure | None = RiskMeasure(risk_measure=, options={})) → [OptimizationConfig](#ax.core.optimization_config.OptimizationConfig)

Make a copy of this optimization config.

#### *property* is_moo_problem *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### *property* metrics *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Metric]*

#### *property* objective *: Objective*

Get objective.

#### *property* outcome_constraints *: [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint]*

Get outcome constraints.

### ax.core.optimization_config.check_objective_thresholds_match_objectives(objectives_by_name: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Objective], objective_thresholds: [list](https://docs.python.org/3/library/stdtypes.html#list)[ObjectiveThreshold]) → [None](https://docs.python.org/3/library/constants.html#None)

Error if thresholds on objective_metrics bound from the wrong direction or
if there is a mismatch between objective thresholds and objectives.

### OutcomeConstraint

### *class* ax.core.outcome_constraint.ObjectiveThreshold(metric: Metric, bound: [float](https://docs.python.org/3/library/functions.html#float), relative: [bool](https://docs.python.org/3/library/functions.html#bool) = True, op: [ComparisonOp](#ax.core.types.ComparisonOp) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: `OutcomeConstraint`

Class for representing Objective Thresholds.

An objective threshold represents the threshold for an objective metric
to contribute to hypervolume calculations. A list containing the objective
threshold for each metric collectively form a reference point.

Objective thresholds may bound the metric from above or from below.
The bound can be expressed as an absolute measurement or relative
to the status quo (if applicable).

The direction of the bound is inferred from the Metric’s lower_is_better attribute.

#### metric

Metric to constrain.

#### bound

The bound in the constraint.

#### relative

Whether you want to bound on an absolute or relative
scale. If relative, bound is the acceptable percent change.

#### op

automatically inferred, but manually overwritable.
specifies whether metric should be greater or equal to, or less
than or equal to, some bound.

#### clone() → ObjectiveThreshold

Create a copy of this ObjectiveThreshold.

### *class* ax.core.outcome_constraint.OutcomeConstraint(metric: Metric, op: [ComparisonOp](#ax.core.types.ComparisonOp), bound: [float](https://docs.python.org/3/library/functions.html#float), relative: [bool](https://docs.python.org/3/library/functions.html#bool) = True)

Bases: [`SortableBase`](utils.md#ax.utils.common.base.SortableBase)

Base class for representing outcome constraints.

Outcome constraints may of the form metric >= bound or metric <= bound,
where the bound can be expressed as an absolute measurement or relative
to the status quo (if applicable).

#### metric

Metric to constrain.

#### op

Specifies whether metric should be greater or equal
to, or less than or equal to, some bound.

#### bound

The bound in the constraint.

#### relative

[default `True`] Whether the provided bound value is relative to
some status-quo arm’s metric value. If False, `bound` is interpreted as an
absolute number, else `bound` specifies percent-difference from the
observed metric value on the status-quo arm. That is, the bound’s absolute
value will be `(1 + bound/100.0) * status_quo_metric_value`. This requires
specification of a status-quo arm in `Experiment`.

#### clone() → OutcomeConstraint

Create a copy of this OutcomeConstraint.

#### *property* metric *: Metric*

#### *property* op *: [ComparisonOp](#ax.core.types.ComparisonOp)*

### *class* ax.core.outcome_constraint.ScalarizedOutcomeConstraint(metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric], op: [ComparisonOp](#ax.core.types.ComparisonOp), bound: [float](https://docs.python.org/3/library/functions.html#float), relative: [bool](https://docs.python.org/3/library/functions.html#bool) = True, weights: [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: `OutcomeConstraint`

Class for presenting outcome constraints composed of a linear
scalarization of metrics.

#### metrics

List of metrics.

#### weights

Weights for scalarization; default to 1.0 / len(metrics).

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]

#### op

Specifies whether metric should be greater or equal
to, or less than or equal to, some bound.

#### bound

The bound in the constraint.

#### relative

[default `True`] Whether the provided bound value is relative to
some status-quo arm’s metric value. If False, `bound` is interpreted as an
absolute number, else `bound` specifies percent-difference from the
observed metric value on the status-quo arm. That is, the bound’s absolute
value will be `(1 + bound/100.0) * status_quo_metric_value`. This requires
specification of a status-quo arm in `Experiment`.

#### clone() → ScalarizedOutcomeConstraint

Create a copy of this ScalarizedOutcomeConstraint.

#### *property* metric *: Metric*

Override base method to error.

#### *property* metric_weights *: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Metric, [float](https://docs.python.org/3/library/functions.html#float)]]*

Get the objective metrics and weights.

#### *property* metrics *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Metric]*

#### *property* op *: [ComparisonOp](#ax.core.types.ComparisonOp)*

#### weights *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]*

### Parameter

### *class* ax.core.parameter.ChoiceParameter(name: [str](https://docs.python.org/3/library/stdtypes.html#str), parameter_type: [ParameterType](#ax.core.parameter.ParameterType), values: [list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], is_ordered: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = None, is_task: [bool](https://docs.python.org/3/library/functions.html#bool) = False, is_fidelity: [bool](https://docs.python.org/3/library/functions.html#bool) = False, target_value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int) = None, sort_values: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = None, dependents: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int), [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Parameter`](#ax.core.parameter.Parameter)

Parameter object that specifies a discrete set of values.

* **Parameters:**
  * **name** – Name of the parameter.
  * **parameter_type** – Enum indicating the type of parameter
    value (e.g. string, int).
  * **values** – List of allowed values for the parameter.
  * **is_ordered** – If False, the parameter is a categorical variable.
    Defaults to False if parameter_type is STRING and `values`
    is longer than 2, else True.
  * **is_task** – Treat the parameter as a task parameter for modeling.
  * **is_fidelity** – Whether this parameter is a fidelity parameter.
  * **target_value** – Target value of this parameter if it’s a fidelity or
    task parameter.
  * **sort_values** – Whether to sort `values` before encoding.
    Defaults to False if `parameter_type` is STRING, else
    True.
  * **dependents** – Optional mapping for parameters in hierarchical search
    spaces; format is { value -> list of dependent parameter names }.

#### add_values(values: [list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]) → [ChoiceParameter](#ax.core.parameter.ChoiceParameter)

Add input list to the set of allowed values for parameter.

Cast all input values to the parameter type.

* **Parameters:**
  **values** – Values being added to the allowed list.

#### *property* available_flags *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

List of boolean attributes that can be set on this parameter.

#### cardinality() → [float](https://docs.python.org/3/library/functions.html#float)

#### clone() → [ChoiceParameter](#ax.core.parameter.ChoiceParameter)

#### *property* dependents *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int), [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]]*

#### *property* domain_repr *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

Returns a string representation of the domain.

#### *property* is_ordered *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### *property* is_task *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### set_values(values: [list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]) → [ChoiceParameter](#ax.core.parameter.ChoiceParameter)

Set the list of allowed values for parameter.

Cast all input values to the parameter type.

* **Parameters:**
  **values** – New list of allowed values.

#### *property* sort_values *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### validate(value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)) → [bool](https://docs.python.org/3/library/functions.html#bool)

Checks that the input is in the list of allowed values.

* **Parameters:**
  **value** – Value being checked.
* **Returns:**
  True if valid, False otherwise.

#### *property* values *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]*

### *class* ax.core.parameter.FixedParameter(name: [str](https://docs.python.org/3/library/stdtypes.html#str), parameter_type: [ParameterType](#ax.core.parameter.ParameterType), value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int), is_fidelity: [bool](https://docs.python.org/3/library/functions.html#bool) = False, target_value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int) = None, dependents: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int), [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Parameter`](#ax.core.parameter.Parameter)

Parameter object that specifies a single fixed value.

#### *property* available_flags *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

List of boolean attributes that can be set on this parameter.

#### cardinality() → [float](https://docs.python.org/3/library/functions.html#float)

#### clone() → [FixedParameter](#ax.core.parameter.FixedParameter)

#### *property* dependents *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int), [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]]*

#### *property* domain_repr *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

Returns a string representation of the domain.

#### set_value(value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)) → [FixedParameter](#ax.core.parameter.FixedParameter)

#### validate(value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)) → [bool](https://docs.python.org/3/library/functions.html#bool)

Checks that the input is equal to the fixed value.

* **Parameters:**
  **value** – Value being checked.
* **Returns:**
  True if valid, False otherwise.

#### *property* value *: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)*

### *class* ax.core.parameter.Parameter

Bases: [`SortableBase`](utils.md#ax.utils.common.base.SortableBase)

#### *property* available_flags *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

List of boolean attributes that can be set on this parameter.

#### *abstract* cardinality() → [float](https://docs.python.org/3/library/functions.html#float)

#### cast(value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)) → [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)

#### clone() → [Parameter](#ax.core.parameter.Parameter)

#### *property* dependents *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int), [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]]*

#### *abstract property* domain_repr *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

Returns a string representation of the domain.

#### *property* is_fidelity *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### *property* is_hierarchical *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### *property* is_numeric *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### is_valid_type(value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)) → [bool](https://docs.python.org/3/library/functions.html#bool)

Whether a given value’s type is allowed by this parameter.

#### *property* name *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### *property* parameter_type *: [ParameterType](#ax.core.parameter.ParameterType)*

#### *property* python_type *: [type](https://docs.python.org/3/library/functions.html#type)[[int](https://docs.python.org/3/library/functions.html#int)] | [type](https://docs.python.org/3/library/functions.html#type)[[float](https://docs.python.org/3/library/functions.html#float)] | [type](https://docs.python.org/3/library/functions.html#type)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [type](https://docs.python.org/3/library/functions.html#type)[[bool](https://docs.python.org/3/library/functions.html#bool)]*

The python type for the corresponding ParameterType enum.

Used primarily for casting values of unknown type to conform
to that of the parameter.

#### *property* summary_dict *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int) | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]]*

#### *property* target_value *: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)*

#### *abstract* validate(value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)) → [bool](https://docs.python.org/3/library/functions.html#bool)

### *class* ax.core.parameter.ParameterType(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

#### BOOL *: [int](https://docs.python.org/3/library/functions.html#int)* *= 0*

#### FLOAT *: [int](https://docs.python.org/3/library/functions.html#int)* *= 2*

#### INT *: [int](https://docs.python.org/3/library/functions.html#int)* *= 1*

#### STRING *: [int](https://docs.python.org/3/library/functions.html#int)* *= 3*

#### *property* is_numeric *: [bool](https://docs.python.org/3/library/functions.html#bool)*

### *class* ax.core.parameter.RangeParameter(name: [str](https://docs.python.org/3/library/stdtypes.html#str), parameter_type: [ParameterType](#ax.core.parameter.ParameterType), lower: [float](https://docs.python.org/3/library/functions.html#float), upper: [float](https://docs.python.org/3/library/functions.html#float), log_scale: [bool](https://docs.python.org/3/library/functions.html#bool) = False, logit_scale: [bool](https://docs.python.org/3/library/functions.html#bool) = False, digits: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, is_fidelity: [bool](https://docs.python.org/3/library/functions.html#bool) = False, target_value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int) = None)

Bases: [`Parameter`](#ax.core.parameter.Parameter)

Parameter object that specifies a range of values.

#### *property* available_flags *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

List of boolean attributes that can be set on this parameter.

#### cardinality() → [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)

#### cast(value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)) → [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)

#### clone() → [RangeParameter](#ax.core.parameter.RangeParameter)

#### *property* digits *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

Number of digits to round values to for float type.

Upper and lower bound are re-cast after this property is changed.

#### *property* domain_repr *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

Returns a string representation of the domain.

#### is_valid_type(value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)) → [bool](https://docs.python.org/3/library/functions.html#bool)

Same as default except allows floats whose value is an int
for Int parameters.

#### *property* log_scale *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Whether the parameter’s random values should be sampled from log space.

#### *property* logit_scale *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Whether the parameter’s random values should be sampled from logit space.

#### *property* lower *: [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)*

Lower bound of the parameter range.

Value is cast to parameter type upon set and also validated
to ensure the bound is strictly less than upper bound.

#### set_digits(digits: [int](https://docs.python.org/3/library/functions.html#int)) → [RangeParameter](#ax.core.parameter.RangeParameter)

#### set_log_scale(log_scale: [bool](https://docs.python.org/3/library/functions.html#bool)) → [RangeParameter](#ax.core.parameter.RangeParameter)

#### set_logit_scale(logit_scale: [bool](https://docs.python.org/3/library/functions.html#bool)) → [RangeParameter](#ax.core.parameter.RangeParameter)

#### update_range(lower: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, upper: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [RangeParameter](#ax.core.parameter.RangeParameter)

Set the range to the given values.

If lower or upper is not provided, it will be left at its current value.

* **Parameters:**
  * **lower** – New value for the lower bound.
  * **upper** – New value for the upper bound.

#### *property* upper *: [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)*

Upper bound of the parameter range.

Value is cast to parameter type upon set and also validated
to ensure the bound is strictly greater than lower bound.

#### validate(value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int), tol: [float](https://docs.python.org/3/library/functions.html#float) = 1.5e-07) → [bool](https://docs.python.org/3/library/functions.html#bool)

Returns True if input is a valid value for the parameter.

Checks that value is of the right type and within
the valid range for the parameter. Returns False if value is None.

* **Parameters:**
  * **value** – Value being checked.
  * **tol** – Absolute tolerance for floating point comparisons.
* **Returns:**
  True if valid, False otherwise.

### ParameterConstraint

### *class* ax.core.parameter_constraint.OrderConstraint(lower_parameter: [Parameter](#ax.core.parameter.Parameter), upper_parameter: [Parameter](#ax.core.parameter.Parameter))

Bases: [`ParameterConstraint`](#ax.core.parameter_constraint.ParameterConstraint)

Constraint object for specifying one parameter to be smaller than another.

#### clone() → [OrderConstraint](#ax.core.parameter_constraint.OrderConstraint)

Clone.

#### clone_with_transformed_parameters(transformed_parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Parameter](#ax.core.parameter.Parameter)]) → [OrderConstraint](#ax.core.parameter_constraint.OrderConstraint)

Clone, but replace parameters with transformed versions.

#### *property* constraint_dict *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]*

Weights on parameters for linear constraint representation.

#### *property* lower_parameter *: [Parameter](#ax.core.parameter.Parameter)*

Parameter with lower value.

#### *property* parameters *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Parameter](#ax.core.parameter.Parameter)]*

Parameters.

#### *property* upper_parameter *: [Parameter](#ax.core.parameter.Parameter)*

Parameter with higher value.

### *class* ax.core.parameter_constraint.ParameterConstraint(constraint_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)], bound: [float](https://docs.python.org/3/library/functions.html#float))

Bases: [`SortableBase`](utils.md#ax.utils.common.base.SortableBase)

Base class for linear parameter constraints.

Constraints are expressed using a map from parameter name to weight
followed by a bound.

The constraint is satisfied if sum_i(w_i \* v_i) <= b where:
: w is the vector of parameter weights.
  v is a vector of parameter values.
  b is the specified bound.

#### *property* bound *: [float](https://docs.python.org/3/library/functions.html#float)*

Get bound of the inequality of the constraint.

#### check(parameter_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float)]) → [bool](https://docs.python.org/3/library/functions.html#bool)

Whether or not the set of parameter values satisfies the constraint.

Does a weighted sum of the parameter values based on the constraint_dict
and checks that the sum is less than the bound.

* **Parameters:**
  **parameter_dict** – Map from parameter name to parameter value.
* **Returns:**
  Whether the constraint is satisfied.

#### clone() → [ParameterConstraint](#ax.core.parameter_constraint.ParameterConstraint)

Clone.

#### clone_with_transformed_parameters(transformed_parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Parameter](#ax.core.parameter.Parameter)]) → [ParameterConstraint](#ax.core.parameter_constraint.ParameterConstraint)

Clone, but replaced parameters with transformed versions.

#### *property* constraint_dict *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]*

Get mapping from parameter names to weights.

### *class* ax.core.parameter_constraint.SumConstraint(parameters: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Parameter](#ax.core.parameter.Parameter)], is_upper_bound: [bool](https://docs.python.org/3/library/functions.html#bool), bound: [float](https://docs.python.org/3/library/functions.html#float))

Bases: [`ParameterConstraint`](#ax.core.parameter_constraint.ParameterConstraint)

Constraint on the sum of parameters being greater or less than a bound.

#### clone() → [SumConstraint](#ax.core.parameter_constraint.SumConstraint)

Clone.

To use the same constraint, we need to reconstruct the original bound.
We do this by re-applying the original bound weighting.

#### clone_with_transformed_parameters(transformed_parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Parameter](#ax.core.parameter.Parameter)]) → [SumConstraint](#ax.core.parameter_constraint.SumConstraint)

Clone, but replace parameters with transformed versions.

#### *property* constraint_dict *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]*

Weights on parameters for linear constraint representation.

#### *property* is_upper_bound *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Whether the bound is an upper or lower bound on the sum.

#### *property* op *: [ComparisonOp](#ax.core.types.ComparisonOp)*

Whether the sum is constrained by a <= or >= inequality.

#### *property* parameters *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Parameter](#ax.core.parameter.Parameter)]*

Parameters.

### ax.core.parameter_constraint.validate_constraint_parameters(parameters: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Parameter](#ax.core.parameter.Parameter)]) → [None](https://docs.python.org/3/library/constants.html#None)

Basic validation of parameters used in a constraint.

* **Parameters:**
  **parameters** – Parameters used in constraint.
* **Raises:**
  **ValueError if the parameters are not valid for use.** – 

### ParameterDistribution

### *class* ax.core.parameter_distribution.ParameterDistribution(parameters: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], distribution_class: [str](https://docs.python.org/3/library/stdtypes.html#str), distribution_parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None), multiplicative: [bool](https://docs.python.org/3/library/functions.html#bool) = False)

Bases: [`SortableBase`](utils.md#ax.utils.common.base.SortableBase)

A class for defining parameter distributions.

Intended for robust optimization use cases. This could be used to specify the
distribution of an environmental variable or the distribution of the input noise.

#### clone() → [ParameterDistribution](#ax.core.parameter_distribution.ParameterDistribution)

Clone.

#### *property* distribution *: rv_frozen*

Get the distribution object.

#### *property* distribution_class *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

The name of the scipy distribution class.

#### *property* distribution_parameters *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*

The parameters of the distribution.

#### is_environmental(search_space: RobustSearchSpace) → [bool](https://docs.python.org/3/library/functions.html#bool)

Check if the parameters are environmental variables of the given
search space.

* **Parameters:**
  **search_space** – The search space to check.
* **Returns:**
  A boolean denoting whether the parameters are environmental variables.

### RiskMeasure

### *class* ax.core.risk_measures.RiskMeasure(risk_measure: [str](https://docs.python.org/3/library/stdtypes.html#str), options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [bool](https://docs.python.org/3/library/functions.html#bool) | [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]])

Bases: [`SortableBase`](utils.md#ax.utils.common.base.SortableBase)

A class for defining risk measures.

This can be used with a RobustSearchSpace, to convert the predictions over

```
`
```

ParameterDistribution\`s to robust metrics, which then get used in candidate
generation to recommend robust candidates.

See ax/modelbridge/modelbridge_utils.py for RISK_MEASURE_NAME_TO_CLASS,
which lists the supported risk measures, and for extract_risk_measure
helper, which extracts the BoTorch risk measure.

#### clone() → [RiskMeasure](#ax.core.risk_measures.RiskMeasure)

Clone.

### Runner

### *class* ax.core.runner.Runner

Bases: [`Base`](utils.md#ax.utils.common.base.Base), [`SerializationMixin`](utils.md#ax.utils.common.serialization.SerializationMixin), [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

Abstract base class for custom runner classes

#### clone() → [Runner](#ax.core.runner.Runner)

Create a copy of this Runner.

#### poll_available_capacity() → [int](https://docs.python.org/3/library/functions.html#int)

Checks how much available capacity there is to schedule trial evaluations.
Required for runners used with Ax `Scheduler`.

NOTE: This method might be difficult to implement in some systems. Returns -1
if capacity of the system is “unlimited” or “unknown”
(meaning that the `Scheduler` should be trying to schedule as many trials
as is possible without violating scheduler settings). There is no need to
artificially force this method to limit capacity; `Scheduler` has other
limitations in place to limit number of trials running at once,
like the `SchedulerOptions.max_pending_trials` setting, or
more granular control in the form of the max_parallelism
setting in each of the GenerationStep\`s of a \`GenerationStrategy).

* **Returns:**
  An integer, representing how many trials there is available capacity for;
  -1 if capacity is “unlimited” or not possible to know in advance.

#### poll_exception(trial: [core.base_trial.BaseTrial](#ax.core.base_trial.BaseTrial)) → [str](https://docs.python.org/3/library/stdtypes.html#str)

Returns the exception from a trial.

* **Parameters:**
  **trial** – Trial to get exception for.
* **Returns:**
  Exception string.

#### poll_trial_status(trials: Iterable[[core.base_trial.BaseTrial](#ax.core.base_trial.BaseTrial)]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[core.base_trial.TrialStatus](#ax.core.base_trial.TrialStatus), [set](https://docs.python.org/3/library/stdtypes.html#set)[[int](https://docs.python.org/3/library/functions.html#int)]]

Checks the status of any non-terminal trials and returns their
indices as a mapping from TrialStatus to a list of indices. Required
for runners used with Ax `Scheduler`.

NOTE: Does not need to handle waiting between polling calls while trials
are running; this function should just perform a single poll.

* **Parameters:**
  **trials** – Trials to poll.
* **Returns:**
  A dictionary mapping TrialStatus to a list of trial indices that have
  the respective status at the time of the polling. This does not need to
  include trials that at the time of polling already have a terminal
  (ABANDONED, FAILED, COMPLETED) status (but it may).

#### *abstract* run(trial: [core.base_trial.BaseTrial](#ax.core.base_trial.BaseTrial)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Any]

Deploys a trial based on custom runner subclass implementation.

* **Parameters:**
  **trial** – The trial to deploy.
* **Returns:**
  Dict of run metadata from the deployment process.

#### *property* run_metadata_report_keys *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

A list of keys of the metadata dict returned by run() that are
relevant outside the runner-internal impolementation. These can e.g.
be reported in Scheduler.report_results().

#### run_multiple(trials: Iterable[[core.base_trial.BaseTrial](#ax.core.base_trial.BaseTrial)]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Any]]

Runs a single evaluation for each of the given trials. Useful when deploying
multiple trials at once is more efficient than deploying them one-by-one.
Used in Ax `Scheduler`.

NOTE: By default simply loops over run_trial. Should be overwritten
if deploying multiple trials in batch is preferable.

* **Parameters:**
  **trials** – Iterable of trials to be deployed, each containing arms with
  parameterizations to be evaluated. Can be a Trial
  if contains only one arm or a BatchTrial if contains
  multiple arms.
* **Returns:**
  Dict of trial index to the run metadata of that trial from the deployment
  process.

#### *property* staging_required *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Whether the trial goes to staged or running state once deployed.

#### stop(trial: [core.base_trial.BaseTrial](#ax.core.base_trial.BaseTrial), reason: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Any]

Stop a trial based on custom runner subclass implementation.

Optional method.

* **Parameters:**
  * **trial** – The trial to stop.
  * **reason** – A message containing information why the trial is to be stopped.
* **Returns:**
  A dictionary of run metadata from the stopping process.

### SearchSpace

### *class* ax.core.search_space.HierarchicalSearchSpace(parameters: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Parameter](#ax.core.parameter.Parameter)], parameter_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ParameterConstraint](#ax.core.parameter_constraint.ParameterConstraint)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: `SearchSpace`

#### cast_observation_features(observation_features: [ObservationFeatures](#ax.core.observation.ObservationFeatures)) → [ObservationFeatures](#ax.core.observation.ObservationFeatures)

Cast parameterization of given observation features to the hierarchical
structure of the given search space; return the newly cast observation features
with the full parameterization stored in `metadata` under
`Keys.FULL_PARAMETERIZATION`.

For each parameter in given parameterization, cast it to the proper type
specified in this search space and remove it from the parameterization if that
parameter should not be in the arm within the search space due to its
hierarchical structure.

#### check_membership(parameterization: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], raise_error: [bool](https://docs.python.org/3/library/functions.html#bool) = False, check_all_parameters_present: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [bool](https://docs.python.org/3/library/functions.html#bool)

Whether the given parameterization belongs in the search space.

Checks that the given parameter values have the same name/type as
search space parameters, are contained in the search space domain,
and satisfy the parameter constraints.

* **Parameters:**
  * **parameterization** – Dict from parameter name to value to validate.
  * **raise_error** – If true parameterization does not belong, raises an error
    with detailed explanation of why.
  * **check_all_parameters_present** – Ensure that parameterization specifies
    values for all parameters as expected by the search space and its
    hierarchical structure.
* **Returns:**
  Whether the parameterization is contained in the search space.

#### flatten() → SearchSpace

Returns a flattened `SearchSpace` with all the parameters in the
given `HierarchicalSearchSpace`; ignores their hierarchical structure.

#### flatten_observation_features(observation_features: [ObservationFeatures](#ax.core.observation.ObservationFeatures), inject_dummy_values_to_complete_flat_parameterization: [bool](https://docs.python.org/3/library/functions.html#bool) = False, use_random_dummy_values: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ObservationFeatures](#ax.core.observation.ObservationFeatures)

Flatten observation features that were previously cast to the hierarchical
structure of the given search space; return the newly flattened observation
features. This method re-injects parameter values that were removed from
observation features during casting (as they are saved in observation features
metadata).

* **Parameters:**
  * **observation_features** – Observation features corresponding to one point
    to flatten.
  * **inject_dummy_values_to_complete_flat_parameterization** – Whether to inject
    values for parameters that are not in the parameterization.
    This will be used to complete the parameterization after re-injecting
    the parameters that are recorded in the metadata (for parameters
    that were generated by Ax).
  * **use_random_dummy_values** – Whether to use random values for missing
    parameters. If False, we set the values to the middle of
    the corresponding parameter domain range.

#### *property* height *: [int](https://docs.python.org/3/library/functions.html#int)*

Height of the underlying tree structure of this hierarchical search space.

#### hierarchical_structure_str(parameter_names_only: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [str](https://docs.python.org/3/library/stdtypes.html#str)

String representation of the hierarchical structure.

* **Parameters:**
  **parameter_names_only** – Whether parameter should show up just as names
  (instead of full parameter strings), useful for a more concise
  representation.

#### *property* root *: [Parameter](#ax.core.parameter.Parameter)*

Root of the hierarchical search space tree, as identified during
`HierarchicalSearchSpace` construction.

### *class* ax.core.search_space.RobustSearchSpace(parameters: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Parameter](#ax.core.parameter.Parameter)], parameter_distributions: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ParameterDistribution](#ax.core.parameter_distribution.ParameterDistribution)], num_samples: [int](https://docs.python.org/3/library/functions.html#int), environmental_variables: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Parameter](#ax.core.parameter.Parameter)] | [None](https://docs.python.org/3/library/constants.html#None) = None, parameter_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ParameterConstraint](#ax.core.parameter_constraint.ParameterConstraint)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: `SearchSpace`

Search space for robust optimization that supports environmental variables
and input noise.

In addition to the usual search space properties, this allows specifying
environmental variables (parameters) and input noise distributions.

#### clone() → RobustSearchSpace

#### is_environmental_variable(parameter_name: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [bool](https://docs.python.org/3/library/functions.html#bool)

Check if a given parameter is an environmental variable.

* **Parameters:**
  **parameter** – A string denoting the name of the parameter.
* **Returns:**
  A boolean denoting whether the given parameter_name corresponds
  to an environmental variable of this search space.

#### *property* parameters *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Parameter](#ax.core.parameter.Parameter)]*

Get all parameters and environmental variables.

We include environmental variables here to support transform_search_space
and other similar functionality. It also helps avoid having to overwrite a
bunch of parent methods.

#### update_parameter(parameter: [Parameter](#ax.core.parameter.Parameter)) → [None](https://docs.python.org/3/library/constants.html#None)

### *class* ax.core.search_space.RobustSearchSpaceDigest(sample_param_perturbations: ~collections.abc.Callable[[], ~numpy.ndarray] | None = None, sample_environmental: ~collections.abc.Callable[[], ~numpy.ndarray] | None = None, environmental_variables: list[str] = <factory>, multiplicative: bool = False)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Container for lightweight representation of properties that are unique
to the RobustSearchSpace. This is used to append the SearchSpaceDigest.

NOTE: Both sample_param_perturbations and sample_environmental should
require no inputs and return a num_samples x d-dim array of samples from
the corresponding parameter distributions, where d is the number of
non-environmental parameters for distribution_sampler and the number of
environmental variables for environmental_sampler.

#### sample_param_perturbations

An optional callable for sampling from the
parameter distributions representing input perturbations.

* **Type:**
  [collections.abc.Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[], numpy.ndarray] | None

#### sample_environmental

An optional callable for sampling from the
distributions of the environmental variables.

* **Type:**
  [collections.abc.Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[], numpy.ndarray] | None

#### environmental_variables

A list of environmental variable names.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### multiplicative

Denotes whether the distribution is multiplicative.
Only relevant if paired with a distribution_sampler.

* **Type:**
  [bool](https://docs.python.org/3/library/functions.html#bool)

#### environmental_variables *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### multiplicative *: [bool](https://docs.python.org/3/library/functions.html#bool)* *= False*

#### sample_environmental *: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[], ndarray] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### sample_param_perturbations *: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[], ndarray] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

### *class* ax.core.search_space.SearchSpace(parameters: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Parameter](#ax.core.parameter.Parameter)], parameter_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ParameterConstraint](#ax.core.parameter_constraint.ParameterConstraint)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Base`](utils.md#ax.utils.common.base.Base)

Base object for SearchSpace object.

Contains a set of Parameter objects, each of which have a
name, type, and set of valid values. The search space also contains
a set of ParameterConstraint objects, which can be used to define
restrictions across parameters (e.g. p_a < p_b).

#### add_parameter(parameter: [Parameter](#ax.core.parameter.Parameter)) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_parameter_constraints(parameter_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ParameterConstraint](#ax.core.parameter_constraint.ParameterConstraint)]) → [None](https://docs.python.org/3/library/constants.html#None)

#### cast_arm(arm: [Arm](#ax.core.arm.Arm)) → [Arm](#ax.core.arm.Arm)

Cast parameterization of given arm to the types in this SearchSpace.

For each parameter in given arm, cast it to the proper type specified
in this search space. Throws if there is a mismatch in parameter names. This is
mostly useful for int/float, which user can be sloppy with when hand written.

* **Parameters:**
  **arm** – Arm to cast.
* **Returns:**
  New casted arm.

#### check_all_parameters_present(parameterization: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], raise_error: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [bool](https://docs.python.org/3/library/functions.html#bool)

Whether a given parameterization contains all the parameters in the
search space.

* **Parameters:**
  * **parameterization** – Dict from parameter name to value to validate.
  * **raise_error** – If true parameterization does not belong, raises an error
    with detailed explanation of why.
* **Returns:**
  Whether the parameterization is contained in the search space.

#### check_membership(parameterization: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], raise_error: [bool](https://docs.python.org/3/library/functions.html#bool) = False, check_all_parameters_present: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [bool](https://docs.python.org/3/library/functions.html#bool)

Whether the given parameterization belongs in the search space.

Checks that the given parameter values have the same name/type as
search space parameters, are contained in the search space domain,
and satisfy the parameter constraints.

* **Parameters:**
  * **parameterization** – Dict from parameter name to value to validate.
  * **raise_error** – If true parameterization does not belong, raises an error
    with detailed explanation of why.
  * **check_all_parameters_present** – Ensure that parameterization specifies
    values for all parameters as expected by the search space.
* **Returns:**
  Whether the parameterization is contained in the search space.

#### check_types(parameterization: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], allow_none: [bool](https://docs.python.org/3/library/functions.html#bool) = True, raise_error: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [bool](https://docs.python.org/3/library/functions.html#bool)

Checks that the given parameterization’s types match the search space.

* **Parameters:**
  * **parameterization** – Dict from parameter name to value to validate.
  * **allow_none** – Whether None is a valid parameter value.
  * **raise_error** – If true and parameterization does not belong, raises an error
    with detailed explanation of why.
* **Returns:**
  Whether the parameterization has valid types.

#### clone() → SearchSpace

#### construct_arm(parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [Arm](#ax.core.arm.Arm)

Construct new arm using given parameters and name. Any missing parameters
fallback to the experiment defaults, represented as None.

#### *property* is_hierarchical *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### *property* is_robust *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### out_of_design_arm() → [Arm](#ax.core.arm.Arm)

Create a default out-of-design arm.

An out of design arm contains values for some parameters which are
outside of the search space. In the modeling conversion, these parameters
are all stripped down to an empty dictionary, since the point is already
outside of the modeled space.

* **Returns:**
  New arm w/ null parameter values.

#### *property* parameter_constraints *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ParameterConstraint](#ax.core.parameter_constraint.ParameterConstraint)]*

#### *property* parameters *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Parameter](#ax.core.parameter.Parameter)]*

#### *property* range_parameters *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [RangeParameter](#ax.core.parameter.RangeParameter)]*

#### set_parameter_constraints(parameter_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ParameterConstraint](#ax.core.parameter_constraint.ParameterConstraint)]) → [None](https://docs.python.org/3/library/constants.html#None)

#### *property* summary_df *: pandas.DataFrame*

Creates a dataframe with information about each parameter in the given
search space. The resulting dataframe has one row per parameter, and the
following columns:

> - Name: the name of the parameter.
> - Type: the parameter subclass (Fixed, Range, Choice).
> - Domain: the parameter’s domain (e.g., “range=[0, 1]” or
>   “values=[‘a’, ‘b’]”).
> - Datatype: the datatype of the parameter (int, float, str, bool).
> - Flags: flags associated with the parameter, if any.
> - Target Value: the target value of the parameter, if applicable.
> - Dependent Parameters: for parameters in hierarchical search spaces,

> mapping from parameter value -> list of dependent parameter names.

#### *property* tunable_parameters *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Parameter](#ax.core.parameter.Parameter)]*

#### update_parameter(parameter: [Parameter](#ax.core.parameter.Parameter)) → [None](https://docs.python.org/3/library/constants.html#None)

#### validate_membership(parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]) → [None](https://docs.python.org/3/library/constants.html#None)

### *class* ax.core.search_space.SearchSpaceDigest(feature_names: list[str], bounds: list[tuple[int | float, int | float]], ordinal_features: list[int] = <factory>, categorical_features: list[int] = <factory>, discrete_choices: ~collections.abc.Mapping[int, list[int | float]] = <factory>, task_features: list[int] = <factory>, fidelity_features: list[int] = <factory>, target_values: dict[int, int | float] = <factory>, robust_digest: ~ax.core.search_space.RobustSearchSpaceDigest | None = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Container for lightweight representation of search space properties.

This is used for communicating between modelbridge and models. This is
an ephemeral object and not meant to be stored / serialized. It is typically
constructed from the transformed search space using extract_search_space_digest,
whose docstring explains how various fields are populated.

#### feature_names

A list of parameter names.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### bounds

A list [(l_0, u_0), …, (l_d, u_d)] of tuples representing the
lower and upper bounds on the respective parameter (both inclusive).

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float)]]

#### ordinal_features

A list of indices corresponding to the parameters
to be considered as ordinal discrete parameters. The corresponding
bounds are assumed to be integers, and parameter i is assumed
to take on values l_i, l_i+1, …, u_i.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]

#### categorical_features

A list of indices corresponding to the parameters
to be considered as categorical discrete parameters. The corresponding
bounds are assumed to be integers, and parameter i is assumed
to take on values l_i, l_i+1, …, u_i.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]

#### discrete_choices

A dictionary mapping indices of discrete (ordinal
or categorical) parameters to their respective sets of values
provided as a list.

* **Type:**
  [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[int](https://docs.python.org/3/library/functions.html#int), [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float)]]

#### task_features

A list of parameter indices to be considered as
task parameters.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]

#### fidelity_features

A list of parameter indices to be considered as
fidelity parameters.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]

#### target_values

A dictionary mapping parameter indices of fidelity or
task parameters to their respective target value.

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float)]

#### robust_digest

An optional RobustSearchSpaceDigest that carries the
additional attributes if using a RobustSearchSpace.

* **Type:**
  ax.core.search_space.RobustSearchSpaceDigest | None

#### bounds *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float)]]*

#### categorical_features *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]*

#### discrete_choices *: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[int](https://docs.python.org/3/library/functions.html#int), [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float)]]*

#### feature_names *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### fidelity_features *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]*

#### ordinal_features *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]*

#### robust_digest *: RobustSearchSpaceDigest | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### target_values *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float)]*

#### task_features *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]*

### Trial

### *class* ax.core.trial.Trial(experiment: [core.experiment.Experiment](#ax.core.experiment.Experiment), generator_run: [GeneratorRun](#ax.core.generator_run.GeneratorRun) | [None](https://docs.python.org/3/library/constants.html#None) = None, trial_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, ttl_seconds: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, index: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`BaseTrial`](#ax.core.base_trial.BaseTrial)

Trial that only has one attached arm and no arm weights.

* **Parameters:**
  * **experiment** – Experiment, to which this trial is attached.
  * **generator_run** – GeneratorRun, associated with this trial.
    Trial has only one generator run (of just one arm)
    attached to it. This can also be set later through add_arm
    or add_generator_run, but a trial’s associated genetor run is
    immutable once set.
  * **trial_type** – Type of this trial, if used in MultiTypeExperiment.
  * **ttl_seconds** – If specified, trials will be considered failed after
    this many seconds since the time the trial was ran, unless the
    trial is completed before then. Meant to be used to detect
    ‘dead’ trials, for which the evaluation process might have
    crashed etc., and which should be considered failed after
    their ‘time to live’ has passed.
  * **index** – If specified, the trial’s index will be set accordingly.
    This should generally not be specified, as in the index will be
    automatically determined based on the number of existing trials.
    This is only used for the purpose of loading from storage.

#### *property* abandoned_arms *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Arm](#ax.core.arm.Arm)]*

Abandoned arms attached to this trial.

#### add_arm(\*args, \*\*kwargs)

#### add_generator_run(\*args, \*\*kwargs)

#### *property* arm *: [Arm](#ax.core.arm.Arm) | [None](https://docs.python.org/3/library/constants.html#None)*

The arm associated with this batch.

#### *property* arms *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Arm](#ax.core.arm.Arm)]*

All arms attached to this trial.

* **Returns:**
  list of a single arm
  : attached to this trial if there is one, else None.
* **Return type:**
  arms

#### *property* arms_by_name *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Arm](#ax.core.arm.Arm)]*

Dictionary of all arms attached to this trial with their names
as keys.

* **Returns:**
  dictionary of a single
  : arm name to arm if one is attached to this trial,
    else None.
* **Return type:**
  arms

#### clone_to(experiment: [core.experiment.Experiment](#ax.core.experiment.Experiment) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [Trial](#ax.core.trial.Trial)

Clone the trial and attach it to the specified experiment.
If no experiment is provided, the original experiment will be used.

* **Parameters:**
  **experiment** – The experiment to which the cloned trial will belong.
  If unspecified, uses the current experiment.
* **Returns:**
  A new instance of the trial.

#### *property* generator_run *: [GeneratorRun](#ax.core.generator_run.GeneratorRun) | [None](https://docs.python.org/3/library/constants.html#None)*

Generator run attached to this trial.

#### *property* generator_runs *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[GeneratorRun](#ax.core.generator_run.GeneratorRun)]*

All generator runs associated with this trial.

#### get_metric_mean(metric_name: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [float](https://docs.python.org/3/library/functions.html#float)

Metric mean for the arm attached to this trial, retrieved from the
latest data available for the metric for the trial.

#### *property* objective_mean *: [float](https://docs.python.org/3/library/functions.html#float)*

Objective mean for the arm attached to this trial, retrieved from the
latest data available for the objective for the trial.

Note: the retrieved objective is the experiment-level objective at the
time of the call to objective_mean, which is not necessarily the
objective that was set at the time the trial was created or ran.

#### update_trial_data(raw_data: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]] | [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Hashable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]], metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str) | [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, sample_size: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, combine_with_last_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [str](https://docs.python.org/3/library/stdtypes.html#str)

Utility method that attaches data to a trial and
returns an update message.

* **Parameters:**
  * **raw_data** – Evaluation data for the trial. Can be a mapping from
    metric name to a tuple of mean and SEM, just a tuple of mean and
    SEM if only one metric in optimization, or just the mean if SEM is
    unknown (then Ax will infer observation noise level).
    Can also be a list of (fidelities, mapping from
    metric name to a tuple of mean and SEM).
  * **metadata** – Additional metadata to track about this run, optional.
  * **sample_size** – Number of samples collected for the underlying arm,
    optional.
  * **combine_with_last_data** – Whether to combine the given data with the
    data that was previously attached to the trial. See
    Experiment.attach_data for a detailed explanation.
* **Returns:**
  A string message summarizing the update.

#### validate_data_for_trial(data: Data) → [None](https://docs.python.org/3/library/constants.html#None)

Utility method to validate data before further processing.

## Core Types

### *class* ax.core.types.ComparisonOp(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Class for enumerating comparison operations.

#### GEQ *: [int](https://docs.python.org/3/library/functions.html#int)* *= 0*

#### LEQ *: [int](https://docs.python.org/3/library/functions.html#int)* *= 1*

### ax.core.types.merge_model_predict(predict: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]]]], predict_append: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]]]]) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]]]]

Append model predictions to an existing set of model predictions.

TModelPredict is of the form:
: {metric_name: [mean1, mean2, …],
  {metric_name: {metric_name: [var1, var2, …]}})

This will append the predictions

* **Parameters:**
  * **predict** – Initial set of predictions.
  * **other_predict** – Predictions to be appended.
* **Returns:**
  TModelPredict with the new predictions appended.

### ax.core.types.validate_evaluation_outcome(outcome: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]] | [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Hashable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]]) → [None](https://docs.python.org/3/library/constants.html#None)

Runtime validate that the supplied outcome has correct structure.

### ax.core.types.validate_fidelity_trial_evaluation(evaluation: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]]) → [None](https://docs.python.org/3/library/constants.html#None)

### ax.core.types.validate_floatlike(floatlike: [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer) → [None](https://docs.python.org/3/library/constants.html#None)

### ax.core.types.validate_map_dict(map_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Hashable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)]) → [None](https://docs.python.org/3/library/constants.html#None)

### ax.core.types.validate_map_trial_evaluation(evaluation: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Hashable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]]) → [None](https://docs.python.org/3/library/constants.html#None)

### ax.core.types.validate_param_value(param_value: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)) → [None](https://docs.python.org/3/library/constants.html#None)

### ax.core.types.validate_parameterization(parameterization: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]) → [None](https://docs.python.org/3/library/constants.html#None)

### ax.core.types.validate_single_metric_data(data: [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]) → [None](https://docs.python.org/3/library/constants.html#None)

### ax.core.types.validate_trial_evaluation(evaluation: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]) → [None](https://docs.python.org/3/library/constants.html#None)

## Core Utils

### *class* ax.core.utils.MissingMetrics(objective, outcome_constraints, tracking_metrics)

Bases: [`NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple)

#### objective *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [set](https://docs.python.org/3/library/stdtypes.html#set)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)]]]*

Alias for field number 0

#### outcome_constraints *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [set](https://docs.python.org/3/library/stdtypes.html#set)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)]]]*

Alias for field number 1

#### tracking_metrics *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [set](https://docs.python.org/3/library/stdtypes.html#set)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)]]]*

Alias for field number 2

### ax.core.utils.best_feasible_objective(optimization_config: [OptimizationConfig](#ax.core.optimization_config.OptimizationConfig), values: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), ndarray]) → ndarray

Compute the best feasible objective value found by each iteration.

* **Parameters:**
  * **optimization_config** – Optimization config.
  * **values** – Dictionary from metric name to array of value at each
    iteration. If optimization config contains outcome constraints, values
    for them must be present in values.

Returns: Array of cumulative best feasible value.

### ax.core.utils.extend_pending_observations(experiment: [Experiment](#ax.core.experiment.Experiment), pending_observations: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](#ax.core.observation.ObservationFeatures)]], generator_runs: [list](https://docs.python.org/3/library/stdtypes.html#list)[[GeneratorRun](#ax.core.generator_run.GeneratorRun)]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](#ax.core.observation.ObservationFeatures)]]

Extend given pending observations dict (from metric name to observations
that are pending for that metric), with arms in a given generator run.

* **Parameters:**
  * **experiment** – Experiment, for which the generation strategy is producing

    ```
    ``
    ```

    GeneratorRun\`\`s.
  * **pending_observations** – Dict from metric name to pending observations for
    that metric, used to avoid resuggesting arms that will be explored soon.
  * **generator_runs** – List of `GeneratorRun``s currently produced by the
    ``GenerationStrategy`.
* **Returns:**
  A new dictionary of pending observations to avoid in-place modification

### ax.core.utils.extract_pending_observations(experiment: [Experiment](#ax.core.experiment.Experiment), include_out_of_design_points: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](#ax.core.observation.ObservationFeatures)]] | [None](https://docs.python.org/3/library/constants.html#None)

Computes a list of pending observation features (corresponding to:
- arms that have been generated and run in the course of the experiment,
but have not been completed with data,
- arms that have been abandoned or belong to abandoned trials).

This function dispatches to:
- `get_pending_observation_features` if experiment is using
`BatchTrial`-s or has fewer than 100 trials,
- `get_pending_observation_features_based_on_trial_status` if
experiment is using  `Trial`-s and has more than 100 trials.

`get_pending_observation_features_based_on_trial_status` is a faster
way to compute pending observations, but it is not guaranteed to be
accurate for `BatchTrial` settings and makes assumptions, e.g.
arms in `COMPLETED` trial never being pending. See docstring of
that function for more details.

NOTE: Pending observation features are passed to the model to
instruct it to not generate the same points again.

### ax.core.utils.get_missing_metrics(data: Data, optimization_config: [OptimizationConfig](#ax.core.optimization_config.OptimizationConfig)) → [MissingMetrics](#ax.core.utils.MissingMetrics)

Return all arm_name, trial_index pairs, for which some of the
observatins of optimization config metrics are missing.

* **Parameters:**
  * **data** – Data to search.
  * **optimization_config** – provides metric_names to search for.
* **Returns:**
  A NamedTuple(missing_objective, Dict[str, missing_outcome_constraint])

### ax.core.utils.get_missing_metrics_by_name(data: Data, metric_names: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [set](https://docs.python.org/3/library/stdtypes.html#set)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)]]]

Return all arm_name, trial_index pairs missing some observations of
specified metrics.

* **Parameters:**
  * **data** – Data to search.
  * **metric_names** – list of metrics to search for.
* **Returns:**
  A Dict[str, missing_metrics], one entry for each metric_name.

### ax.core.utils.get_model_times(experiment: [Experiment](#ax.core.experiment.Experiment)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]

Get total times spent fitting the model and generating candidates in the
course of the experiment.

### ax.core.utils.get_model_trace_of_times(experiment: [Experiment](#ax.core.experiment.Experiment)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)], [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)]]

Get time spent fitting the model and generating candidates during each trial.
Not cumulative.

* **Returns:**
  List of fit times, list of gen times.

### ax.core.utils.get_pending_observation_features(experiment: [Experiment](#ax.core.experiment.Experiment), \*, include_out_of_design_points: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](#ax.core.observation.ObservationFeatures)]] | [None](https://docs.python.org/3/library/constants.html#None)

Computes a list of pending observation features (corresponding to:
- arms that have been generated in the course of the experiment,
but have not been completed with data,
- arms that have been abandoned or belong to abandoned trials).

NOTE: Pending observation features are passed to the model to
instruct it to not generate the same points again.

* **Parameters:**
  * **experiment** – Experiment, pending features on which we seek to compute.
  * **include_out_of_design_points** – By default, this function will not include
    “out of design” points (those that are not in the search space) among
    the pending points. This is because pending points are generally used to
    help the model avoid re-suggesting the same points again. For points
    outside of the search space, this will not happen, so they typically do
    not need to be included. However, if the user wants to include them,
    they can be included by setting this flag to `True`.
* **Returns:**
  An optional mapping from metric names to a list of observation features,
  pending for that metric (i.e. do not have evaluation data for that metric).
  If there are no pending features for any of the metrics, return is None.

### ax.core.utils.get_pending_observation_features_based_on_trial_status(experiment: [Experiment](#ax.core.experiment.Experiment), include_out_of_design_points: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](#ax.core.observation.ObservationFeatures)]] | [None](https://docs.python.org/3/library/constants.html#None)

A faster analogue of `get_pending_observation_features` that makes
assumptions about trials in experiment in order to speed up extraction
of pending points.

Assumptions:

* All arms in all trials in `CANDIDATE`, `STAGED`, `RUNNING` and `ABANDONED`
  statuses are to be considered pending for all outcomes.
* All arms in all trials in other statuses are to be considered not pending for
  all outcomes.

This entails:

* No actual data-fetching for trials to determine whether arms in them are pending
  for specific outcomes.
* Even if data is present for some outcomes in `RUNNING` trials, their arms will
  still be considered pending for those outcomes.

NOTE: This function should not be used to extract pending features in field
experiments, where arms in running trials should not be considered pending if
there is data for those arms.

* **Parameters:**
  **experiment** – Experiment, pending features on which we seek to compute.
* **Returns:**
  An optional mapping from metric names to a list of observation features,
  pending for that metric (i.e. do not have evaluation data for that metric).
  If there are no pending features for any of the metrics, return is None.

## Formatting Utils

### *class* ax.core.formatting_utils.DataType(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

#### DATA *= 1*

#### MAP_DATA *= 3*

### ax.core.formatting_utils.data_and_evaluations_from_raw_data(raw_data: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]] | [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Hashable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]]], metric_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], trial_index: [int](https://docs.python.org/3/library/functions.html#int), sample_sizes: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)], data_type: [DataType](#ax.core.formatting_utils.DataType), start_time: [int](https://docs.python.org/3/library/functions.html#int) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, end_time: [int](https://docs.python.org/3/library/functions.html#int) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]] | [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Hashable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]]], Data]

Transforms evaluations into Ax Data.

Each evaluation is either a trial evaluation: {metric_name -> (mean, SEM)}
or a fidelity trial evaluation for multi-fidelity optimizations:
[(fidelities, {metric_name -> (mean, SEM)})].

* **Parameters:**
  * **raw_data** – Mapping from arm name to raw_data.
  * **metric_names** – Names of metrics used to transform raw data to evaluations.
  * **trial_index** – Index of the trial, for which the evaluations are.
  * **sample_sizes** – Number of samples collected for each arm, may be empty
    if unavailable.
  * **start_time** – Optional start time of run of the trial that produced this
    data, in milliseconds or iso format.  Milliseconds will eventually be
    converted to iso format because iso format automatically works with the
    pandas column type Timestamp.
  * **end_time** – Optional end time of run of the trial that produced this
    data, in milliseconds or iso format.  Milliseconds will eventually be
    converted to iso format because iso format automatically works with the
    pandas column type Timestamp.

### ax.core.formatting_utils.raw_data_to_evaluation(raw_data: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]] | [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Hashable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]], metric_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]] | [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Hashable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer, [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | floating | integer | [None](https://docs.python.org/3/library/constants.html#None)]]]]

Format the trial evaluation data to a standard TTrialEvaluation
(mapping from metric names to a tuple of mean and SEM) representation, or
to a TMapTrialEvaluation.

Note: this function expects raw_data to be data for a Trial, not a
BatchedTrial.
