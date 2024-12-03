# ax.exceptions

## Constants

## Core

### *exception* ax.exceptions.core.AxError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception)

Base Ax exception.

All exceptions derived from AxError need to define a custom error message.
Additionally, exceptions can define a hint property that provides additional
guidance as to how to remedy the error.

### *exception* ax.exceptions.core.AxParameterWarning(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxWarning`](#ax.exceptions.core.AxWarning)

Ax warning used for concerns related to parameter setups.

### *exception* ax.exceptions.core.AxStorageWarning(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxWarning`](#ax.exceptions.core.AxWarning)

Ax warning used for storage related concerns.

### *exception* ax.exceptions.core.AxWarning(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`Warning`](https://docs.python.org/3/library/exceptions.html#Warning)

Base Ax warning.

All warnings derived from AxWarning need to define a custom warning message.
Additionally, warnings can define a hint property that provides additional
guidance as to how to remedy the warning.

### *exception* ax.exceptions.core.DataRequiredError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when more observed data is needed by the model to continue the
optimization.

Useful to distinguish when user needs to wait to request more trials until
more data is available.

### *exception* ax.exceptions.core.ExperimentNotFoundError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`ObjectNotFoundError`](#ax.exceptions.core.ObjectNotFoundError)

Raised when an experiment is not found in the database.

### *exception* ax.exceptions.core.ExperimentNotReadyError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '', exposures_unavailable: [bool](https://docs.python.org/3/library/functions.html#bool) = False)

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when failing to query data due to immature experiment.

Useful to distinguish data failure reasons in automated analyses.

### *exception* ax.exceptions.core.IncompatibleDependencyVersion(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raise when an imcompatible dependency version is installed.

### *exception* ax.exceptions.core.MetricDataNotReadyError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when trying to pull metric data from a trial that has
not finished running.

### *exception* ax.exceptions.core.MisconfiguredExperiment(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when experiment has incomplete or incorrect information.

### *exception* ax.exceptions.core.NoDataError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when no data is found for experiment in underlying data store.

Useful to distinguish data failure reasons in automated analyses.

### *exception* ax.exceptions.core.ObjectNotFoundError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError), [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)

Raised when an object is not found in the database.

This exception replaces ValueError raised by code when an objects is not
found in the database. In order to maintain backwards compatibility
ObjectNotFoundError inherits from ValueError. Dependency on ValueError
may be removed in the future.

### *exception* ax.exceptions.core.OptimizationComplete(message: [str](https://docs.python.org/3/library/stdtypes.html#str))

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when you hit SearchSpaceExhausted and GenerationStrategyComplete.

### *exception* ax.exceptions.core.OptimizationShouldStop(message: [str](https://docs.python.org/3/library/stdtypes.html#str))

Bases: [`OptimizationComplete`](#ax.exceptions.core.OptimizationComplete)

Raised when the Global Stopping Strategy suggests to stop the optimization.

### *exception* ax.exceptions.core.SearchSpaceExhausted(message: [str](https://docs.python.org/3/library/stdtypes.html#str))

Bases: [`OptimizationComplete`](#ax.exceptions.core.OptimizationComplete)

Raised when using an algorithm that deduplicates points and no more
new points can be sampled from the search space.

### *exception* ax.exceptions.core.UnsupportedError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when an unsupported request is made.

UnsupportedError may seem similar to NotImplementedError (NIE).
It differs in the following ways:

1. UnsupportedError is not used for abstract methods, which
   : is the official NIE use case.
2. UnsupportedError indicates an intentional and permanent lack of support.
   : It should not be used for TODO (another common use case of NIE).

### *exception* ax.exceptions.core.UnsupportedPlotError(message: [str](https://docs.python.org/3/library/stdtypes.html#str))

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when plotting functionality is not supported for the
given configurations.

### *exception* ax.exceptions.core.UserInputError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when the user passes in an invalid input

## Data

### *exception* ax.exceptions.data_provider.DataProviderError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), data_provider: [str](https://docs.python.org/3/library/stdtypes.html#str), data_provider_error: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

Bases: [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception)

Base Exception for Ax DataProviders.

The type of the data provider must be included.
The raw error is stored in the data_provider_error section,
and an Ax-friendly message is stored as the actual error message.

### *exception* ax.exceptions.data_provider.MissingDataError(missing_trial_indexes: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[int](https://docs.python.org/3/library/functions.html#int)])

Bases: [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception)

## Generation Strategy

### *exception* ax.exceptions.generation_strategy.AxGenerationException(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when there is an issue with the generation strategy.

### *exception* ax.exceptions.generation_strategy.GenerationStrategyCompleted(message: [str](https://docs.python.org/3/library/stdtypes.html#str))

Bases: [`OptimizationComplete`](#ax.exceptions.core.OptimizationComplete)

Special exception indicating that the generation strategy has been
completed.

### *exception* ax.exceptions.generation_strategy.GenerationStrategyMisconfiguredException(error_info: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None))

Bases: [`AxGenerationException`](#ax.exceptions.generation_strategy.AxGenerationException)

Special exception indicating that the generation strategy is misconfigured.

### *exception* ax.exceptions.generation_strategy.GenerationStrategyRepeatedPoints(message: [str](https://docs.python.org/3/library/stdtypes.html#str))

Bases: [`GenerationStrategyCompleted`](#ax.exceptions.generation_strategy.GenerationStrategyCompleted)

Special exception indicating that the generation strategy is repeatedly
suggesting previously sampled points.

### *exception* ax.exceptions.generation_strategy.MaxParallelismReachedException(model_name: [str](https://docs.python.org/3/library/stdtypes.html#str), num_running: [int](https://docs.python.org/3/library/functions.html#int), step_index: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, node_name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`AxGenerationException`](#ax.exceptions.generation_strategy.AxGenerationException)

Special exception indicating that maximum number of trials running in
parallel set on a given step (as GenerationStep.max_parallelism) has been
reached. Upon getting this exception, users should wait until more trials
are completed with data, to generate new trials.

### *exception* ax.exceptions.generation_strategy.OptimizationConfigRequired

Bases: [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)

Error indicating that candidate generation cannot be completed
because an optimization config was not provided.

## Model

### *exception* ax.exceptions.model.CVNotSupportedError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when cross validation is applied to a model which doesn’t
support it.

### *exception* ax.exceptions.model.ModelError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when an error occurs during modeling.

## Storage

### *exception* ax.exceptions.storage.ImmutabilityError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when an attempt is made to update an immutable object.

### *exception* ax.exceptions.storage.IncorrectDBConfigurationError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when an attempt is made to save and load an object, but
the current engine and session factory is setup up incorrectly to
process the call (e.g. current session factory will connect to a
wrong database for the call).

### *exception* ax.exceptions.storage.JSONDecodeError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when an error occurs during JSON decoding.

### *exception* ax.exceptions.storage.JSONEncodeError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when an error occurs during JSON encoding.

### *exception* ax.exceptions.storage.SQADecodeError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when an error occurs during SQA decoding.

### *exception* ax.exceptions.storage.SQAEncodeError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), hint: [str](https://docs.python.org/3/library/stdtypes.html#str) = '')

Bases: [`AxError`](#ax.exceptions.core.AxError)

Raised when an error occurs during SQA encoding.
