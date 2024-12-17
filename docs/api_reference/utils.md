# ax.utils

## Common

### Base

### *class* ax.utils.common.base.Base

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Metaclass for core Ax classes. Provides an equality check and db_id
property for SQA storage.

#### *property* db_id *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

### *class* ax.utils.common.base.SortableBase

Bases: [`Base`](#ax.utils.common.base.Base)

Extension to the base class that also provides an inequality check.

### Constants

### *class* ax.utils.common.constants.Keys(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: [`str`](https://docs.python.org/3/library/stdtypes.html#str), [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Enum of reserved keys in options dicts etc, alphabetized.

NOTE: Useful for keys in dicts that correspond to kwargs to
classes or functions and/or are used in multiple places.

#### ACQF_KWARGS *= 'acquisition_function_kwargs'*

#### AUTOSET_SURROGATE *= 'autoset_surrogate'*

#### BATCH_INIT_CONDITIONS *= 'batch_initial_conditions'*

#### CANDIDATE_SET *= 'candidate_set'*

#### CANDIDATE_SIZE *= 'candidate_size'*

#### COST_AWARE_UTILITY *= 'cost_aware_utility'*

#### COST_INTERCEPT *= 'cost_intercept'*

#### CURRENT_VALUE *= 'current_value'*

#### EXPAND *= 'expand'*

#### EXPECTED_ACQF_VAL *= 'expected_acquisition_value'*

#### EXPERIMENT_TOTAL_CONCURRENT_ARMS *= 'total_concurrent_arms'*

#### FIDELITY_FEATURES *= 'fidelity_features'*

#### FIDELITY_WEIGHTS *= 'fidelity_weights'*

#### FRAC_RANDOM *= 'frac_random'*

#### FULL_PARAMETERIZATION *= 'full_parameterization'*

#### IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF *= 'immutable_search_space_and_opt_config'*

#### MAXIMIZE *= 'maximize'*

#### METADATA *= 'metadata'*

#### METRIC_NAMES *= 'metric_names'*

#### NUM_FANTASIES *= 'num_fantasies'*

#### NUM_INNER_RESTARTS *= 'num_inner_restarts'*

#### NUM_RESTARTS *= 'num_restarts'*

#### NUM_TRACE_OBSERVATIONS *= 'num_trace_observations'*

#### OBJECTIVE *= 'objective'*

#### ONLY_SURROGATE *= 'only_surrogate'*

#### OPTIMIZER_KWARGS *= 'optimizer_kwargs'*

#### PAIRWISE_PREFERENCE_QUERY *= 'pairwise_pref_query'*

#### PREFERENCE_DATA *= 'preference_data'*

#### PRIMARY_SURROGATE *= 'primary'*

#### PROJECT *= 'project'*

#### QMC *= 'qmc'*

#### RAW_INNER_SAMPLES *= 'raw_inner_samples'*

#### RAW_SAMPLES *= 'raw_samples'*

#### RESUMED_FROM_STORAGE_TS *= 'resumed_from_storage_timestamps'*

#### SAMPLER *= 'sampler'*

#### SEED_INNER *= 'seed_inner'*

#### SEQUENTIAL *= 'sequential'*

#### STATE_DICT *= 'state_dict'*

#### SUBCLASS *= 'subclass'*

#### SUBSET_MODEL *= 'subset_model'*

#### TASK_FEATURES *= 'task_features'*

#### TRIAL_COMPLETION_TIMESTAMP *= 'trial_completion_timestamp'*

#### WARM_START_REFITTING *= 'warm_start_refitting'*

#### X_BASELINE *= 'X_baseline'*

### Decorator

### *class* ax.utils.common.decorator.ClassDecorator

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

Template for making a decorator work as a class level decorator.  That decorator
should extend ClassDecorator.  It must implement \_\_init_\_ and
decorate_callable.  See disable_logger.decorate_callable for an example.
decorate_callable should call self._call_func() instead of directly calling
func to handle static functions.
Note: \_call_func is still imperfect and unit tests should be used to ensure
everything is working properly.  There is a lot of complexity in detecting
classmethods and staticmethods and removing the self argument in the right
situations. For best results always use keyword args in the decorated class.

DECORATE_PRIVATE can be set to determine whether private methods should be
decorated. In the case of a logging decorator, you may only want to decorate things
the user calls. But in the case of a disable logging decorator, you may want to
decorate everything to ensure no logs escape.

#### DECORATE_PRIVATE *= True*

#### *abstract* decorate_callable(func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], T]) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], T]

#### decorate_class(klass: T) → T

### Deprecation

### Docutils

Support functions for sphinx et. al

### ax.utils.common.docutils.copy_doc(src: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[\_T], \_T]

A decorator that copies the docstring of another object

Since `sphinx` actually loads the python modules to grab the docstrings
this works with both `sphinx` and the `help` function.

```python
class Cat(Mamal):

  @property
  @copy_doc(Mamal.is_feline)
  def is_feline(self) -> true:
      ...
```

### Equality

### ax.utils.common.equality.dataframe_equals(df1: pandas.DataFrame, df2: pandas.DataFrame) → [bool](https://docs.python.org/3/library/functions.html#bool)

Compare equality of two pandas dataframes.

### ax.utils.common.equality.datetime_equals(dt1: [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime) | [None](https://docs.python.org/3/library/constants.html#None), dt2: [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime) | [None](https://docs.python.org/3/library/constants.html#None)) → [bool](https://docs.python.org/3/library/functions.html#bool)

Compare equality of two datetimes, ignoring microseconds.

### ax.utils.common.equality.equality_typechecker(eq_func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

A decorator to wrap all \_\_eq_\_ methods to ensure that the inputs
are of the right type.

### ax.utils.common.equality.is_ax_equal(one_val: [Any](https://docs.python.org/3/library/typing.html#typing.Any), other_val: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [bool](https://docs.python.org/3/library/functions.html#bool)

Check for equality of two values, handling lists, dicts, dfs, floats,
dates, and numpy arrays. This method and `same_elements` function
as a recursive unit.

Some special cases:
- For datetime objects, the equality is checked up to the second.

> Microseconds are ignored.
- For floats, `np.isclose` is used to check for almost-equality.
- For lists (and dict values), `same_elements` is used. This ignores
  the ordering of the elements, and checks that the two lists are subsets
  of each other (under the assumption that there are no duplicates).
- If the objects don’t fall into any of the special cases, we use simple
  equality check and cast the output to a boolean. If the comparison
  or cast fails, we return False. Example: the comparison of a float with
  a numpy array (with multiple elements) will return False.

### ax.utils.common.equality.object_attribute_dicts_equal(one_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], other_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], skip_db_id_check: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [bool](https://docs.python.org/3/library/functions.html#bool)

Utility to check if all items in attribute dicts of two Ax objects
are the same.

NOTE: Special-cases some Ax object attributes, like “_experiment” or
“_model”, where full equality is hard to check.

* **Parameters:**
  * **one_dict** – First object’s attribute dict (`obj.__dict__`).
  * **other_dict** – Second object’s attribute dict (`obj.__dict__`).
  * **skip_db_id_check** – If `True`, will exclude the `db_id` attributes from the
    equality check. Useful for ensuring that all attributes of an object are
    equal except the ids, with which one or both of them are saved to the
    database (e.g. if confirming an object before it was saved, to the version
    reloaded from the DB).

### ax.utils.common.equality.object_attribute_dicts_find_unequal_fields(one_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], other_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], fast_return: [bool](https://docs.python.org/3/library/functions.html#bool) = True, skip_db_id_check: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[Any](https://docs.python.org/3/library/typing.html#typing.Any), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[Any](https://docs.python.org/3/library/typing.html#typing.Any), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]]

Utility for finding out what attributes of two objects’ attribute dicts
are unequal.

* **Parameters:**
  * **one_dict** – First object’s attribute dict (`obj.__dict__`).
  * **other_dict** – Second object’s attribute dict (`obj.__dict__`).
  * **fast_return** – Boolean representing whether to return as soon as a
    single unequal attribute was found or to iterate over all attributes
    and collect all unequal ones.
  * **skip_db_id_check** – If `True`, will exclude the `db_id` attributes from the
    equality check. Useful for ensuring that all attributes of an object are
    equal except the ids, with which one or both of them are saved to the
    database (e.g. if confirming an object before it was saved, to the version
    reloaded from the DB).
* **Returns:**
  - attribute name to attribute values of unequal type (as a tuple),
  - attribute name to attribute values of unequal value (as a tuple).
* **Return type:**
  Two dictionaries

### ax.utils.common.equality.same_elements(list1: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)], list2: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [bool](https://docs.python.org/3/library/functions.html#bool)

Compare equality of two lists of core Ax objects.

Assumptions:
: – The contents of each list are types that implement \_\_eq_\_
  – The lists do not contain duplicates

Checking equality is then the same as checking that the lists are the same
length, and that both are subsets of the other.

### Executils

### ax.utils.common.executils.execute_with_timeout(partial_func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], T], timeout: [float](https://docs.python.org/3/library/functions.html#float)) → T

Execute a function in a thread that we can abandon if it takes too long.
The thread cannot actually be terminated, so the process will keep executing
after timeout, but not on the main thread.

* **Parameters:**
  * **partial_func** – A partial function to execute.  This should either be a
    function that takes no arguments, or a functools.partial function
    with all arguments bound.
  * **timeout** – The timeout in seconds.
* **Returns:**
  The return value of the partial function when called.

### ax.utils.common.executils.handle_exceptions_in_retries(no_retry_exceptions: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[type](https://docs.python.org/3/library/functions.html#type)[[Exception](https://docs.python.org/3/library/exceptions.html#Exception)], ...], retry_exceptions: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[type](https://docs.python.org/3/library/functions.html#type)[[Exception](https://docs.python.org/3/library/exceptions.html#Exception)], ...], suppress_errors: [bool](https://docs.python.org/3/library/functions.html#bool), check_message_contains: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None), last_retry: [bool](https://docs.python.org/3/library/functions.html#bool), logger: [Logger](https://docs.python.org/3/library/logging.html#logging.Logger) | [None](https://docs.python.org/3/library/constants.html#None), wrap_error_message_in: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)) → [Generator](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)[[None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None)]

### ax.utils.common.executils.retry_on_exception(exception_types: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[type](https://docs.python.org/3/library/functions.html#type)[[Exception](https://docs.python.org/3/library/exceptions.html#Exception)], ...] | [None](https://docs.python.org/3/library/constants.html#None) = None, no_retry_on_exception_types: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[type](https://docs.python.org/3/library/functions.html#type)[[Exception](https://docs.python.org/3/library/exceptions.html#Exception)], ...] | [None](https://docs.python.org/3/library/constants.html#None) = None, check_message_contains: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, retries: [int](https://docs.python.org/3/library/functions.html#int) = 3, suppress_all_errors: [bool](https://docs.python.org/3/library/functions.html#bool) = False, logger: [Logger](https://docs.python.org/3/library/logging.html#logging.Logger) | [None](https://docs.python.org/3/library/constants.html#None) = None, default_return_on_suppression: [Any](https://docs.python.org/3/library/typing.html#typing.Any) | [None](https://docs.python.org/3/library/constants.html#None) = None, wrap_error_message_in: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, initial_wait_seconds: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [Any](https://docs.python.org/3/library/typing.html#typing.Any) | [None](https://docs.python.org/3/library/constants.html#None)

A decorator for instance methods or standalone functions that makes them
retry on failure and allows to specify on which types of exceptions the
function should and should not retry.

NOTE: If the argument suppress_all_errors is supplied and set to True,
the error will be suppressed  and default value returned.

* **Parameters:**
  * **exception_types** – A tuple of exception(s) types to catch in the decorated
    function. If none is provided, baseclass Exception will be used.
  * **no_retry_on_exception_types** – Exception types to consider non-retryable even
    if their supertype appears in exception_types or the only exceptions to
    not retry on if no exception_types are specified.
  * **check_message_contains** – A list of strings, against which to match error
    messages. If the error message contains any one of these strings,
    the exception will cause a retry. NOTE: This argument works in
    addition to exception_types; if those are specified, only the
    specified types of exceptions will be caught and retried on if they
    contain the strings provided as check_message_contains.
  * **retries** – Number of retries to perform.
  * **suppress_all_errors** – If true, after all the retries are exhausted, the
    error will still be suppressed and default_return_on_suppresion
    will be returned from the function. NOTE: If using this argument,
    the decorated function may not actually get fully executed, if
    it consistently raises an exception.
  * **logger** – A handle for the logger to be used.
  * **default_return_on_suppression** – If the error is suppressed after all the
    retries, then this default value will be returned from the function.
    Defaults to None.
  * **wrap_error_message_in** – If raising the error message after all the retries,
    a string wrapper for the error message (useful for making error
    messages more user-friendly). NOTE: Format of resulting error will be:
    “<wrap_error_message_in>: <original_error_type>: <original_error_msg>”,
    with the stack trace of the original message.
  * **initial_wait_seconds** – Initial length of time to wait between failures,
    doubled after each failure up to a maximum of 10 minutes. If unspecified
    then there is no wait between retries.

### Kwargs

### ax.utils.common.kwargs.consolidate_kwargs(kwargs_iterable: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)], keywords: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Combine an iterable of kwargs into a single dict of kwargs, where kwargs
by duplicate keys that appear later in the iterable get priority over the
ones that appear earlier and only kwargs referenced in keywords will be
used. This allows to combine somewhat redundant sets of kwargs, where a
user-set kwarg, for instance, needs to override a default kwarg.

```pycon
>>> consolidate_kwargs(
...     kwargs_iterable=[{'a': 1, 'b': 2}, {'b': 3, 'c': 4, 'd': 5}],
...     keywords=['a', 'b', 'd']
... )
{'a': 1, 'b': 3, 'd': 5}
```

### ax.utils.common.kwargs.filter_kwargs(function: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Filter out kwargs that are not applicable for a given function.
Return a copy of given kwargs dict with only the required kwargs.

### ax.utils.common.kwargs.get_function_argument_names(function: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), omit: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

Extract parameter names from function signature.

### ax.utils.common.kwargs.get_function_default_arguments(function: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Extract default arguments from function signature.

### ax.utils.common.kwargs.warn_on_kwargs(callable_with_kwargs: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [None](https://docs.python.org/3/library/constants.html#None)

Log a warning when a decoder function receives unexpected kwargs.

NOTE: This mainly caters to the use case where an older version of Ax is
used to decode objects, serialized to JSON by a newer version of Ax (and
therefore potentially containing new fields). In that case, the decoding
function should not fail when encountering those additional fields, but
rather just ignore them and log a warning using this function.

### Logger

### *class* ax.utils.common.logger.AxOutputNameFilter(name='')

Bases: [`Filter`](https://docs.python.org/3/library/logging.html#logging.Filter)

This is a filter which sets the record’s output_name, if
not configured

#### filter(record: [LogRecord](https://docs.python.org/3/library/logging.html#logging.LogRecord)) → [bool](https://docs.python.org/3/library/functions.html#bool)

Determine if the specified record is to be logged.

Returns True if the record should be logged, or False otherwise.
If deemed appropriate, the record may be modified in-place.

### ax.utils.common.logger.build_file_handler(filepath: [str](https://docs.python.org/3/library/stdtypes.html#str), level: [int](https://docs.python.org/3/library/functions.html#int) = 20) → [StreamHandler](https://docs.python.org/3/library/logging.handlers.html#logging.StreamHandler)

Build a file handle that logs entries to the given file, using the
same formatting as the stream handler.

* **Parameters:**
  * **filepath** – Location of the file to log output to. If the file exists, output
    will be appended. If it does not exist, a new file will be created.
  * **level** – The log level. By default, sets level to INFO
* **Returns:**
  A logging.FileHandler instance

### ax.utils.common.logger.build_stream_handler(level: [int](https://docs.python.org/3/library/functions.html#int) = 20) → [StreamHandler](https://docs.python.org/3/library/logging.handlers.html#logging.StreamHandler)

Build the default stream handler used for most Ax logging. Sets
default level to INFO, instead of WARNING.

* **Parameters:**
  **level** – The log level. By default, sets level to INFO
* **Returns:**
  A logging.StreamHandler instance

### *class* ax.utils.common.logger.disable_logger(name: [str](https://docs.python.org/3/library/stdtypes.html#str), level: [int](https://docs.python.org/3/library/functions.html#int) = 40)

Bases: [`ClassDecorator`](#ax.utils.common.decorator.ClassDecorator)

#### decorate_callable(func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], T]) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], T]

### *class* ax.utils.common.logger.disable_loggers(names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], level: [int](https://docs.python.org/3/library/functions.html#int) = 40)

Bases: [`ClassDecorator`](#ax.utils.common.decorator.ClassDecorator)

#### decorate_callable(func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], T]) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], T]

### ax.utils.common.logger.get_logger(name: [str](https://docs.python.org/3/library/stdtypes.html#str), level: [int](https://docs.python.org/3/library/functions.html#int) = 20, force_name: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [Logger](https://docs.python.org/3/library/logging.html#logging.Logger)

Get an Axlogger.

To set a human-readable “output_name” that appears in logger outputs,
add {“output_name”: “[MY_OUTPUT_NAME]”} to the logger’s contextual
information. By default, we use the logger’s name

NOTE: To change the log level on particular outputs (e.g. STDERR logs),
set the proper log level on the relevant handler, instead of the logger
e.g. logger.handers[0].setLevel(INFO)

* **Parameters:**
  * **name** – The name of the logger.
  * **level** – The level at which to actually log.  Logs
    below this level of importance will be discarded
  * **force_name** – If set to false and the module specified
    is not ultimately a descendent of the ax module
    specified by name, “ax.” will be prepended to name
* **Returns:**
  The logging.Logger object.

### ax.utils.common.logger.make_indices_str(indices: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[int](https://docs.python.org/3/library/functions.html#int)]) → [str](https://docs.python.org/3/library/stdtypes.html#str)

Generate a string representation of an iterable of indices;
if indices are contiguous, returns a string formatted like like
‘<min_idx> - <max_idx>’, otherwise a string formatted like
‘[idx_1, idx_2, …, idx_n’].

### ax.utils.common.logger.set_stderr_log_level(level: [int](https://docs.python.org/3/library/functions.html#int)) → [None](https://docs.python.org/3/library/constants.html#None)

Set the log level for stream handler, such that logs of given level
are printed to STDERR by the root logger

### Mock Torch

### ax.utils.common.mock.mock_patch_method_original(mock_path: [str](https://docs.python.org/3/library/stdtypes.html#str), original_method: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], T]) → [MagicMock](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.MagicMock)

Context manager for patching a method returning type T on class C,
to track calls to it while still executing the original method. There
is not a native way to do this with mock.patch.

### Random

### ax.utils.common.random.set_rng_seed(seed: [int](https://docs.python.org/3/library/functions.html#int)) → [None](https://docs.python.org/3/library/constants.html#None)

Sets seeds for random number generators from numpy, pytorch,
and the native random module.

* **Parameters:**
  **seed** – The random number generator seed.

### ax.utils.common.random.with_rng_seed(seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)) → [Generator](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)[[None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None)]

Context manager that sets the random number generator seeds
to a given value and restores the previous state on exit.

If the seed is None, the context manager does nothing. This makes
it possible to use the context manager without having to change
the code based on whether the seed is specified.

* **Parameters:**
  **seed** – The random number generator seed.

### Result

### *class* ax.utils.common.result.Err(value: E)

Bases: [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`T`, `E`], [`Result`](#ax.utils.common.result.Result)[`T`, `E`]

Contains the error value.

#### *property* err *: E*

#### is_err() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### is_ok() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### map(op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[T], U]) → [Result](#ax.utils.common.result.Result)[U, E]

Maps a Result[T, E] to Result[U, E] by applying a function to a contained Ok
value, leaving an Err value untouched. This function can be used to compose
the results of two functions.

#### map_err(op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[E], F]) → [Result](#ax.utils.common.result.Result)[T, F]

Maps a Result[T, E] to Result[T, F] by applying a function to a contained Err
value, leaving an Ok value untouched. This function can be used to pass
through a successful result while handling an error.

#### map_or(default: U, op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[T], U]) → U

Returns the provided default (if Err), or applies a function to the contained
value (if Ok).

#### map_or_else(default_op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[], U], op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[T], U]) → U

Maps a Result[T, E] to U by applying fallback function default to a contained
Err value, or function op to a contained Ok value. This function can be used
to unpack a successful result while handling an error.

#### *property* ok *: [None](https://docs.python.org/3/library/constants.html#None)*

#### unwrap() → [NoReturn](https://docs.python.org/3/library/typing.html#typing.NoReturn)

Returns the contained Ok value.

Because this function may raise an UnwrapError, its use is generally
discouraged. Instead, prefer to handle the Err case explicitly, or call
unwrap_or, unwrap_or_else, or unwrap_or_default.

#### unwrap_err() → E

Returns the contained Err value.

Because this function may raise an UnwrapError, its use is generally
discouraged. Instead, prefer to handle the Err case explicitly, or call
unwrap_or, unwrap_or_else, or unwrap_or_default.

#### unwrap_or(default: T) → T

Returns the contained Ok value or a provided default.

#### unwrap_or_else(op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[E], T]) → T

Returns the contained Ok value or computes it from a Callable.

#### *property* value *: E*

### *class* ax.utils.common.result.ExceptionE(message: [str](https://docs.python.org/3/library/stdtypes.html#str), exception: [Exception](https://docs.python.org/3/library/exceptions.html#Exception))

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

A class that holds an Exception and can be used as the E type in Result[T, E].

#### exception *: [Exception](https://docs.python.org/3/library/exceptions.html#Exception)*

#### message *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### tb_str() → [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)

### *class* ax.utils.common.result.Ok(value: T)

Bases: [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`T`, `E`], [`Result`](#ax.utils.common.result.Result)[`T`, `E`]

Contains the success value.

#### *property* err *: [None](https://docs.python.org/3/library/constants.html#None)*

#### is_err() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### is_ok() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### map(op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[T], U]) → [Result](#ax.utils.common.result.Result)[U, E]

Maps a Result[T, E] to Result[U, E] by applying a function to a contained Ok
value, leaving an Err value untouched. This function can be used to compose
the results of two functions.

#### map_err(op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[E], F]) → [Result](#ax.utils.common.result.Result)[T, F]

Maps a Result[T, E] to Result[T, F] by applying a function to a contained Err
value, leaving an Ok value untouched. This function can be used to pass
through a successful result while handling an error.

#### map_or(default: U, op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[T], U]) → U

Returns the provided default (if Err), or applies a function to the contained
value (if Ok).

#### map_or_else(default_op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[], U], op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[T], U]) → U

Maps a Result[T, E] to U by applying fallback function default to a contained
Err value, or function op to a contained Ok value. This function can be used
to unpack a successful result while handling an error.

#### *property* ok *: T*

#### unwrap() → T

Returns the contained Ok value.

Because this function may raise an UnwrapError, its use is generally
discouraged. Instead, prefer to handle the Err case explicitly, or call
unwrap_or, unwrap_or_else, or unwrap_or_default.

#### unwrap_err() → [NoReturn](https://docs.python.org/3/library/typing.html#typing.NoReturn)

Returns the contained Err value.

Because this function may raise an UnwrapError, its use is generally
discouraged. Instead, prefer to handle the Err case explicitly, or call
unwrap_or, unwrap_or_else, or unwrap_or_default.

#### unwrap_or(default: U) → T

Returns the contained Ok value or a provided default.

#### unwrap_or_else(op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[E], T]) → T

Returns the contained Ok value or computes it from a Callable.

#### *property* value *: T*

### *class* ax.utils.common.result.Result

Bases: [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`T`, `E`], [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

A minimal implementation of a rusty Result monad.
See [https://doc.rust-lang.org/std/result/enum.Result.html](https://doc.rust-lang.org/std/result/enum.Result.html) for more information.

#### *abstract property* err *: E | [None](https://docs.python.org/3/library/constants.html#None)*

#### *abstract* is_err() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### *abstract* is_ok() → [bool](https://docs.python.org/3/library/functions.html#bool)

#### *abstract* map(op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[T], U]) → [Result](#ax.utils.common.result.Result)[U, E]

Maps a Result[T, E] to Result[U, E] by applying a function to a contained Ok
value, leaving an Err value untouched. This function can be used to compose
the results of two functions.

#### *abstract* map_err(op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[E], F]) → [Result](#ax.utils.common.result.Result)[T, F]

Maps a Result[T, E] to Result[T, F] by applying a function to a contained Err
value, leaving an Ok value untouched. This function can be used to pass
through a successful result while handling an error.

#### *abstract* map_or(default: U, op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[T], U]) → U

Returns the provided default (if Err), or applies a function to the contained
value (if Ok).

#### *abstract* map_or_else(default_op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[], U], op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[T], U]) → U

Maps a Result[T, E] to U by applying fallback function default to a contained
Err value, or function op to a contained Ok value. This function can be used
to unpack a successful result while handling an error.

#### *abstract property* ok *: T | [None](https://docs.python.org/3/library/constants.html#None)*

#### *abstract* unwrap() → T

Returns the contained Ok value.

Because this function may raise an UnwrapError, its use is generally
discouraged. Instead, prefer to handle the Err case explicitly, or call
unwrap_or, unwrap_or_else, or unwrap_or_default.

#### *abstract* unwrap_err() → E

Returns the contained Err value.

Because this function may raise an UnwrapError, its use is generally
discouraged. Instead, prefer to handle the Err case explicitly, or call
unwrap_or, unwrap_or_else, or unwrap_or_default.

#### *abstract* unwrap_or(default: T) → T

Returns the contained Ok value or a provided default.

#### *abstract* unwrap_or_else(op: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[E], T]) → T

Returns the contained Ok value or computes it from a Callable.

#### *abstract property* value *: T | E*

### *exception* ax.utils.common.result.UnwrapError

Bases: [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception)

Exception that indicates something has gone wrong in an unwrap call.

This should not happen in real world use and indicates a user has impropperly
or unsafely used the Result abstraction.

### Serialization

### *class* ax.utils.common.serialization.SerializationMixin

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Base class for Ax objects that define their JSON serialization and
deserialization logic at the class level, e.g. most commonly `Runner`
and `Metric` subclasses.

NOTE: Using this class for Ax objects that receive other Ax objects
as inputs, is recommended only iff the parent object (that would be
inheriting from this base class) is not enrolled into
CORE_ENCODER/DECODER_REGISTRY. Inheriting from this mixin with an Ax
object that is in CORE_ENCODER/DECODER_REGISTRY, will result in a
circular dependency, so such classes should inplement their encoding
and decoding logic within the json_store module and not on the classes.

For example, TransitionCriterion take TrialStatus as inputs and are defined
on the CORE_ENCODER/DECODER_REGISTRY, so TransitionCriterion should not inherit
from SerializationMixin and should define custom encoding/decoding logic within
the json_store module.

#### *classmethod* deserialize_init_args(args: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], decoder_registry: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [type](https://docs.python.org/3/library/functions.html#type)[T] | [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], T]] | [None](https://docs.python.org/3/library/constants.html#None) = None, class_decoder_registry: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]], [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Given a dictionary, deserialize the properties needed to initialize the
object. Used for storage.

#### *classmethod* serialize_init_args(obj: [SerializationMixin](#ax.utils.common.serialization.SerializationMixin)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Serialize the properties needed to initialize the object.
Used for storage.

### ax.utils.common.serialization.callable_from_reference(path: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Retrieves a callable by its path.

### ax.utils.common.serialization.callable_to_reference(callable: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) → [str](https://docs.python.org/3/library/stdtypes.html#str)

Obtains path to the callable of form <module>.<name>.

### ax.utils.common.serialization.extract_init_args(args: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], class_: [type](https://docs.python.org/3/library/functions.html#type)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Given a dictionary, extract the arguments required for the
given class’s constructor.

### ax.utils.common.serialization.named_tuple_to_dict(data: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Recursively convert NamedTuples to dictionaries.

### ax.utils.common.serialization.serialize_init_args(obj: [Any](https://docs.python.org/3/library/typing.html#typing.Any), exclude_fields: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Given an object, return a dictionary of the arguments that are
needed by its constructor.

### Testutils

Support functions for tests

### *class* ax.utils.common.testutils.TestCase(methodName: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'runTest')

Bases: `TestCase`

The base Ax test case, contains various helper functions to write unittests.

#### assertAxBaseEqual(first: [Base](#ax.utils.common.base.Base), second: [Base](#ax.utils.common.base.Base), msg: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, skip_db_id_check: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [None](https://docs.python.org/3/library/constants.html#None)

Check that two Ax objects that subclass `Base` are equal or raise
assertion error otherwise.

* **Parameters:**
  * **first** – `Base`-subclassing object to compare to `second`.
  * **second** – `Base`-subclassing object to compare to `first`.
  * **msg** – Message to put into the assertion error raised on inequality; if not
    specified, a default message is used.
  * **skip_db_id_check** – 

    If `True`, will exclude the `db_id` attributes from
    the equality check. Useful for ensuring that all attributes of an object
    are equal except the ids, with which one or both of them are saved to
    the database (e.g. if confirming an object before it was saved, to the
    > version reloaded from the DB).

#### assertDictsAlmostEqual(a: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], b: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], consider_nans_equal: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [None](https://docs.python.org/3/library/constants.html#None)

Testing utility that checks that
1) the keys of a and b are identical, and that
2) the values of a and b are almost equal if they have a floating point
type, considering NaNs as equal, and otherwise just equal.

* **Parameters:**
  * **test** – The test case object.
  * **a** – A dictionary.
  * **b** – Another dictionary.
  * **consider_nans_equal** – Whether to consider NaNs equal when comparing floating
    point numbers.

#### assertEqual(first: [Any](https://docs.python.org/3/library/typing.html#typing.Any), second: [Any](https://docs.python.org/3/library/typing.html#typing.Any), msg: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)

Fail if the two objects are unequal as determined by the ‘==’
operator.

#### assertRaisesOn(exc: [type](https://docs.python.org/3/library/functions.html#type)[[Exception](https://docs.python.org/3/library/exceptions.html#Exception)], line: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, regex: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [AbstractContextManager](https://docs.python.org/3/library/contextlib.html#contextlib.AbstractContextManager)[[None](https://docs.python.org/3/library/constants.html#None)]

Assert that an exception is raised on a specific line.

#### setUp() → [None](https://docs.python.org/3/library/constants.html#None)

Only show log messages of WARNING or higher while testing.

Ax prints a lot of INFO logs that are not relevant for unit tests.

Also silences a number of common warnings originating from Ax & BoTorch.

#### *static* silence_stderr() → [Generator](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)[[None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None)]

A context manager that silences stderr for part of a test.

If any exception passes through this context manager the stderr will be printed,
otherwise it will be discarded.

### ax.utils.common.testutils.setup_import_mocks(mocked_import_paths: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], mock_config_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)

This function mocks expensive modules used in tests. It must be called before
those modules are imported or it will not work.  Stubbing out these modules
will obviously affect the behavior of all tests that use it, so be sure modules
being mocked are not important to your test.  It will also mock all child modules.

* **Parameters:**
  * **mocked_import_paths** – List of module paths to mock.
  * **mock_config_dict** – Dictionary of attributes to mock on the modules being mocked.
    This is useful if the import is expensive, but there is still some
    functionality it has the test relies on.  These attributes will be
    set on all modules being mocked.

### Timeutils

### ax.utils.common.timeutils.current_timestamp_in_millis() → [int](https://docs.python.org/3/library/functions.html#int)

Grab current timestamp in milliseconds as an int.

### ax.utils.common.timeutils.timestamps_in_range(start: [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime), end: [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime), delta: [timedelta](https://docs.python.org/3/library/datetime.html#datetime.timedelta)) → [Generator](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)[[datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime), [None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None)]

Generator of timestamps in range [start, end], at intervals
delta.

### ax.utils.common.timeutils.to_ds(ts: [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime)) → [str](https://docs.python.org/3/library/stdtypes.html#str)

Convert a datetime to a DS string.

### ax.utils.common.timeutils.to_ts(ds: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime)

Convert a DS string to a datetime.

### ax.utils.common.timeutils.unixtime_to_pandas_ts(ts: [float](https://docs.python.org/3/library/functions.html#float)) → pandas.Timestamp

Convert float unixtime into pandas timestamp (UTC).

### Typeutils

### ax.utils.common.typeutils.checked_cast(typ: [type](https://docs.python.org/3/library/functions.html#type)[T], val: V, exception: [Exception](https://docs.python.org/3/library/exceptions.html#Exception) | [None](https://docs.python.org/3/library/constants.html#None) = None) → T

Cast a value to a type (with a runtime safety check).

Returns the value unchanged and checks its type at runtime. This signals to the
typechecker that the value has the designated type.

Like [typing.cast](https://docs.python.org/3/library/typing.html#typing.cast) `check_cast` performs no runtime conversion on its argument,
but, unlike `typing.cast`, `checked_cast` will throw an error if the value is
not of the expected type. The type passed as an argument should be a python class.

* **Parameters:**
  * **typ** – the type to cast to
  * **val** – the value that we are casting
  * **exception** – override exception to raise if  typecheck fails
* **Returns:**
  the `val` argument, unchanged

### ax.utils.common.typeutils.checked_cast_dict(key_typ: [type](https://docs.python.org/3/library/functions.html#type)[K], value_typ: [type](https://docs.python.org/3/library/functions.html#type)[V], d: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[X, Y]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[K, V]

Calls checked_cast on all keys and values in the dictionary.

### ax.utils.common.typeutils.checked_cast_list(typ: [type](https://docs.python.org/3/library/functions.html#type)[T], old_l: [list](https://docs.python.org/3/library/stdtypes.html#list)[V]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[T]

Calls checked_cast on all items in a list.

### ax.utils.common.typeutils.checked_cast_optional(typ: [type](https://docs.python.org/3/library/functions.html#type)[T], val: V | [None](https://docs.python.org/3/library/constants.html#None)) → T | [None](https://docs.python.org/3/library/constants.html#None)

Calls checked_cast only if value is not None.

### ax.utils.common.typeutils.checked_cast_to_tuple(typ: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[type](https://docs.python.org/3/library/functions.html#type)[V], ...], val: V) → T

Cast a value to a union of multiple types (with a runtime safety check).
This function is similar to checked_cast, but allows for the type to be
defined as a tuple of types, in which case the value is cast as a union of
the types in the tuple.

* **Parameters:**
  * **typ** – the tuple of types to cast to
  * **val** – the value that we are casting
* **Returns:**
  the `val` argument, unchanged

### ax.utils.common.typeutils.not_none(val: T | [None](https://docs.python.org/3/library/constants.html#None), message: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → T

Unbox an optional type.

* **Parameters:**
  * **val** – the value to cast to a non `None` type
  * **message** – optional override of the default error message
* **Returns:**
  `val` when `val` is not `None`
* **Return type:**
  V

Throws:
: ValueError if `val` is `None`

### Typeutils Non-Native

<a id="module-ax.utils.common.typeutils_nonnative"></a>

### ax.utils.common.typeutils_nonnative.numpy_type_to_python_type(value: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

If value is a Numpy int or float, coerce to a Python int or float.
This is necessary because some of our transforms return Numpy values.

### Typeutils Torch

### ax.utils.common.typeutils_torch.torch_type_from_str(identifier: [str](https://docs.python.org/3/library/stdtypes.html#str), type_name: [str](https://docs.python.org/3/library/stdtypes.html#str)) → dtype | device | Size

### ax.utils.common.typeutils_torch.torch_type_to_str(value: dtype | device | Size) → [str](https://docs.python.org/3/library/stdtypes.html#str)

Converts torch types, commonly used in Ax, to string representations.

## Flake8 Plugins

### Docstring Checker

### ax.utils.flake8_plugins.docstring_checker.A000(node: [AST](https://docs.python.org/3/library/ast.html#ast.AST)) → [Error](#ax.utils.flake8_plugins.docstring_checker.Error)

### *class* ax.utils.flake8_plugins.docstring_checker.DocstringChecker(tree, filename)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

A flake8 plug-in that makes sure all public functions have a docstring

#### fikename *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### name *: [str](https://docs.python.org/3/library/stdtypes.html#str)* *= 'docstring checker'*

#### run()

#### tree *: [Module](https://docs.python.org/3/library/ast.html#ast.Module)*

#### version *: [str](https://docs.python.org/3/library/stdtypes.html#str)* *= '1.0.0'*

### *class* ax.utils.flake8_plugins.docstring_checker.DocstringCheckerVisitor

Bases: [`NodeVisitor`](https://docs.python.org/3/library/ast.html#ast.NodeVisitor)

#### check_A000(node: [AST](https://docs.python.org/3/library/ast.html#ast.AST)) → [None](https://docs.python.org/3/library/constants.html#None)

#### errors *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Error](#ax.utils.flake8_plugins.docstring_checker.Error)]*

#### visit_AsyncFunctionDef(node: [ClassDef](https://docs.python.org/3/library/ast.html#ast.ClassDef)) → [None](https://docs.python.org/3/library/constants.html#None)

#### visit_ClassDef(node: [ClassDef](https://docs.python.org/3/library/ast.html#ast.ClassDef)) → [None](https://docs.python.org/3/library/constants.html#None)

#### visit_FunctionDef(node: [FunctionDef](https://docs.python.org/3/library/ast.html#ast.FunctionDef)) → [None](https://docs.python.org/3/library/constants.html#None)

### *class* ax.utils.flake8_plugins.docstring_checker.Error(lineno, col, message, type)

Bases: [`NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple)

#### col *: [int](https://docs.python.org/3/library/functions.html#int)*

Alias for field number 1

#### lineno *: [int](https://docs.python.org/3/library/functions.html#int)*

Alias for field number 0

#### message *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

Alias for field number 2

#### type *: [type](https://docs.python.org/3/library/functions.html#type)*

Alias for field number 3

### ax.utils.flake8_plugins.docstring_checker.is_copy_doc_call(c)

Tries to guess if this is a call to the `copy_doc` decorator. This is
a purely syntactic check so if the decorator was aliased as another name]
or wrapped in another function we will fail.

### ax.utils.flake8_plugins.docstring_checker.new_error(errorid: [str](https://docs.python.org/3/library/stdtypes.html#str), msg: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[AST](https://docs.python.org/3/library/ast.html#ast.AST)], [Error](#ax.utils.flake8_plugins.docstring_checker.Error)]

### ax.utils.flake8_plugins.docstring_checker.should_check(filename)

## Measurement

### Synthetic Functions

### *class* ax.utils.measurement.synthetic_functions.Aug_Branin

Bases: [`SyntheticFunction`](#ax.utils.measurement.synthetic_functions.SyntheticFunction)

Augmented Branin function (3-dimensional with infinitely many global minima).

### *class* ax.utils.measurement.synthetic_functions.Aug_Hartmann6

Bases: [`Hartmann6`](#ax.utils.measurement.synthetic_functions.Hartmann6)

Augmented Hartmann6 function (7-dimensional with 1 global minimum).

### *class* ax.utils.measurement.synthetic_functions.Branin

Bases: [`SyntheticFunction`](#ax.utils.measurement.synthetic_functions.SyntheticFunction)

Branin function (2-dimensional with 3 global minima).

### *class* ax.utils.measurement.synthetic_functions.FromBotorch(botorch_synthetic_function: SyntheticTestFunction)

Bases: [`SyntheticFunction`](#ax.utils.measurement.synthetic_functions.SyntheticFunction)

#### *property* name *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

### *class* ax.utils.measurement.synthetic_functions.Hartmann6

Bases: [`SyntheticFunction`](#ax.utils.measurement.synthetic_functions.SyntheticFunction)

Hartmann6 function (6-dimensional with 1 global minimum).

### *class* ax.utils.measurement.synthetic_functions.SyntheticFunction

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

#### *property* domain *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]]*

Domain on which function is evaluated.

The list is of the same length as the dimensionality of the inputs,
where each element of the list is a tuple corresponding to the min
and max of the domain for that dimension.

#### f(X: ndarray) → [float](https://docs.python.org/3/library/functions.html#float) | ndarray

Synthetic function implementation.

* **Parameters:**
  **X** (*numpy.ndarray*) – an n by d array, where n represents the number
  of observations and d is the dimensionality of the inputs.
* **Returns:**
  an n-dimensional array.
* **Return type:**
  numpy.ndarray

#### *property* fmax *: [float](https://docs.python.org/3/library/functions.html#float)*

Value at global minimum(s).

#### *property* fmin *: [float](https://docs.python.org/3/library/functions.html#float)*

Value at global minimum(s).

#### informative_failure_on_none(attr: T | [None](https://docs.python.org/3/library/constants.html#None)) → T

#### *property* maximums *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), ...]]*

List of global minimums.

Each element of the list is a d-tuple, where d is the dimensionality
of the inputs. There may be more than one global minimums.

#### *property* minimums *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), ...]]*

List of global minimums.

Each element of the list is a d-tuple, where d is the dimensionality
of the inputs. There may be more than one global minimums.

#### *property* name *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### *property* required_dimensionality *: [int](https://docs.python.org/3/library/functions.html#int)*

Required dimensionality of input to this function.

### ax.utils.measurement.synthetic_functions.from_botorch(botorch_synthetic_function: SyntheticTestFunction) → [SyntheticFunction](#ax.utils.measurement.synthetic_functions.SyntheticFunction)

Utility to generate Ax synthetic functions from BoTorch synthetic functions.

## Notebook

### Plotting

## Report

### Render

## Sensitivity

### Derivative GP

### ax.utils.sensitivity.derivative_gp.get_KXX_inv(gp: Model) → Tensor

Get the inverse matrix of K(X,X).
:param gp: Botorch model.

* **Returns:**
  The inverse of K(X,X).

### ax.utils.sensitivity.derivative_gp.get_KxX_dx(gp: Model, x: Tensor, kernel_type: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'rbf') → Tensor

Computes the analytic derivative of the kernel K(x,X) w.r.t. x.
:param gp: Botorch model.
:param x: (n x D) Test points.
:param kernel_type: Takes “rbf” or “matern”

* **Returns:**
  Tensor (n x D) The derivative of the kernel K(x,X) w.r.t. x.

### ax.utils.sensitivity.derivative_gp.get_Kxx_dx2(gp: Model, kernel_type: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'rbf') → Tensor

Computes the analytic second derivative of the kernel w.r.t. the training data
:param gp: Botorch model.
:param kernel_type: Takes “rbf” or “matern”

* **Returns:**
  Tensor (n x D x D) The second derivative of the kernel w.r.t. the training data.

### ax.utils.sensitivity.derivative_gp.posterior_derivative(gp: Model, x: Tensor, kernel_type: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'rbf') → MultivariateNormal

Computes the posterior of the derivative of the GP w.r.t. the given test
points x.
This follows the derivation used by GIBO in Sarah Muller, Alexander
von Rohr, Sebastian Trimpe. “Local policy search with Bayesian optimization”,
Advances in Neural Information Processing Systems 34, NeurIPS 2021.
:param gp: Botorch model
:param x: (n x D) Test points.
:param kernel_type: Takes “rbf” or “matern”

* **Returns:**
  A Botorch Posterior.

### Derivative Measures

### *class* ax.utils.sensitivity.derivative_measures.GpDGSMGpMean(model: Model, bounds: Tensor, derivative_gp: [bool](https://docs.python.org/3/library/functions.html#bool) = False, kernel_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, Y_scale: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, num_mc_samples: [int](https://docs.python.org/3/library/functions.html#int) = 10000, input_qmc: [bool](https://docs.python.org/3/library/functions.html#bool) = False, dtype: dtype = torch.float64, num_bootstrap_samples: [int](https://docs.python.org/3/library/functions.html#int) = 1, discrete_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

#### aggregation(transform_fun: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[Tensor], Tensor]) → Tensor

#### bootstrap_indices *: Tensor | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### gradient_absolute_measure() → Tensor

Computes the gradient absolute measure:

* **Returns:**
  if self.num_bootstrap_samples > 1
  : Tensor: (values, var_mc, stderr_mc) x dim

  else
  : Tensor: (values) x dim

#### gradient_measure() → Tensor

Computes the gradient measure:

* **Returns:**
  if self.num_bootstrap_samples > 1
  : Tensor: (values, var_mc, stderr_mc) x dim

  else
  : Tensor: (values) x dim

#### gradients_square_measure() → Tensor

Computes the gradient square measure:

* **Returns:**
  if num_bootstrap_samples > 1
  : Tensor: (values, var_mc, stderr_mc) x dim

  else
  : Tensor: (values) x dim

#### mean_gradients *: Tensor | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### mean_gradients_btsp *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

### *class* ax.utils.sensitivity.derivative_measures.GpDGSMGpSampling(model: Model, bounds: Tensor, num_gp_samples: [int](https://docs.python.org/3/library/functions.html#int), derivative_gp: [bool](https://docs.python.org/3/library/functions.html#bool) = False, kernel_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, Y_scale: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, num_mc_samples: [int](https://docs.python.org/3/library/functions.html#int) = 10000, input_qmc: [bool](https://docs.python.org/3/library/functions.html#bool) = False, gp_sample_qmc: [bool](https://docs.python.org/3/library/functions.html#bool) = False, dtype: dtype = torch.float64, num_bootstrap_samples: [int](https://docs.python.org/3/library/functions.html#int) = 1)

Bases: [`GpDGSMGpMean`](#ax.utils.sensitivity.derivative_measures.GpDGSMGpMean)

#### aggregation(transform_fun: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[Tensor], Tensor]) → Tensor

#### samples_gradients *: Tensor | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### samples_gradients_btsp *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

### ax.utils.sensitivity.derivative_measures.compute_derivatives_from_model_list(model_list: [list](https://docs.python.org/3/library/stdtypes.html#list)[Model], bounds: Tensor, discrete_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → Tensor

Computes average derivatives of a list of models on a bounded domain. Estimation
is according to the GP posterior mean function.

* **Parameters:**
  * **model_list** – A list of m botorch.models.model.Model types for which to compute
    the average derivative.
  * **bounds** – A 2 x d Tensor of lower and upper bounds of the domain of the models.
  * **discrete_features** – If specified, the inputs associated with the indices in
    this list are generated using an integer-valued uniform distribution,
    rather than the default (pseudo-)random continuous uniform distribution.
  * **kwargs** – Passed along to GpDGSMGpMean.
* **Returns:**
  A (m x d) tensor of gradient measures.

### ax.utils.sensitivity.derivative_measures.sample_discrete_parameters(input_mc_samples: Tensor, discrete_features: [None](https://docs.python.org/3/library/constants.html#None) | [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], bounds: Tensor, num_mc_samples: [int](https://docs.python.org/3/library/functions.html#int)) → Tensor

Samples the input parameters uniformly at random for the discrete features.

* **Parameters:**
  * **input_mc_samples** – The input mc samples tensor to be modified.
  * **discrete_features** – A list of integers (or None) of indices corresponding
    to discrete features.
  * **bounds** – The parameter bounds.
  * **num_mc_samples** – The number of Monte Carlo grid samples.
* **Returns:**
  A modified input mc samples tensor.

### Sobol Measures

### ax.utils.sensitivity.sobol_measures.GaussianLinkMean(mean: Tensor, var: Tensor) → Tensor

### ax.utils.sensitivity.sobol_measures.ProbitLinkMean(mean: Tensor, var: Tensor) → Tensor

### *class* ax.utils.sensitivity.sobol_measures.SobolSensitivity(bounds: Tensor, input_function: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[Tensor], Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, num_mc_samples: [int](https://docs.python.org/3/library/functions.html#int) = 10000, input_qmc: [bool](https://docs.python.org/3/library/functions.html#bool) = False, second_order: [bool](https://docs.python.org/3/library/functions.html#bool) = False, num_bootstrap_samples: [int](https://docs.python.org/3/library/functions.html#int) = 1, bootstrap_array: [bool](https://docs.python.org/3/library/functions.html#bool) = False, discrete_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

#### evalute_function(f_A_B_ABi: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)

evaluates the objective function and devides the evaluation into
: torch.Tensors needed for the indices computation.

* **Parameters:**
  **f_A_B_ABi** – Function evaluations on the entire grid of size M(d+2).

#### first_order_indices() → Tensor

Computes the first order Sobol indices:

* **Returns:**
  if num_bootstrap_samples>1
  : Tensor: (values,var_mc,stderr_mc)x dim

  else
  : Tensor: (values)x dim

#### generate_all_input_matrix() → Tensor

#### second_order_indices(first_order_idxs: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, first_order_idxs_btsp: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None) → Tensor

Computes the Second order Sobol indices:
:param first_order_idxs: Tensor of first order indices.
:param first_order_idxs_btsp: Tensor of all first order indices given by bootstrap.

* **Returns:**
  if num_bootstrap_samples>1
  : Tensor: (values,var_mc,stderr_mc)x dim

  else
  : Tensor: (values)x dim

#### total_order_indices() → Tensor

Computes the total Sobol indices:

* **Returns:**
  if num_bootstrap_samples>1
  : Tensor: (values,var_mc,stderr_mc)x dim

  else
  : Tensor: (values)x dim

### *class* ax.utils.sensitivity.sobol_measures.SobolSensitivityGPMean(model: ~botorch.models.model.Model, bounds: ~torch.Tensor, num_mc_samples: int = 10000, second_order: bool = False, input_qmc: bool = False, num_bootstrap_samples: int = 1, link_function: ~collections.abc.Callable[[~torch.Tensor, ~torch.Tensor], ~torch.Tensor] = <function GaussianLinkMean>, mini_batch_size: int = 128, discrete_features: list[int] | None = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

#### first_order_indices() → Tensor

Computes the first order Sobol indices:

* **Returns:**
  if num_bootstrap_samples>1
  : Tensor: (values,var_mc,stderr_mc)x dim

  else
  : Tensor: (values)x dim

#### second_order_indices() → Tensor

Computes the Second order Sobol indices:

* **Returns:**
  if num_bootstrap_samples>1
  : Tensor: (values,var_mc,stderr_mc)x dim(dim-1)/2

  else
  : Tensor: (values)x dim(dim-1)/2

#### total_order_indices() → Tensor

Computes the total Sobol indices:

* **Returns:**
  if num_bootstrap_samples>1
  : Tensor: (values,var_mc,stderr_mc)x dim

  else
  : Tensor: (values)x dim

### *class* ax.utils.sensitivity.sobol_measures.SobolSensitivityGPSampling(model: Model, bounds: Tensor, num_gp_samples: [int](https://docs.python.org/3/library/functions.html#int) = 1000, num_mc_samples: [int](https://docs.python.org/3/library/functions.html#int) = 10000, second_order: [bool](https://docs.python.org/3/library/functions.html#bool) = False, input_qmc: [bool](https://docs.python.org/3/library/functions.html#bool) = False, gp_sample_qmc: [bool](https://docs.python.org/3/library/functions.html#bool) = False, num_bootstrap_samples: [int](https://docs.python.org/3/library/functions.html#int) = 1, discrete_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

#### *property* dim *: [int](https://docs.python.org/3/library/functions.html#int)*

Returns the input dimensionality of self.model.

#### first_order_indices() → Tensor

Computes the first order Sobol indices:

* **Returns:**
  if num_bootstrap_samples>1
  : Tensor: (values, var_gp, stderr_gp, var_mc, stderr_mc) x dim

  else
  : Tensor: (values, var, stderr) x dim

#### second_order_indices() → Tensor

Computes the Second order Sobol indices:

* **Returns:**
  if num_bootstrap_samples>1
  : Tensor: (values, var_gp, stderr_gp, var_mc, stderr_mc) x dim(dim-1) / 2

  else
  : Tensor: (values, var, stderr) x dim(dim-1) / 2

#### total_order_indices() → Tensor

Computes the total Sobol indices:

* **Returns:**
  if num_bootstrap_samples>1
  : Tensor: (values, var_gp, stderr_gp, var_mc, stderr_mc) x dim

  else
  : Tensor: (values, var, stderr) x dim

### ax.utils.sensitivity.sobol_measures.ax_parameter_sens(model_bridge: [TorchModelBridge](modelbridge.md#ax.modelbridge.torch.TorchModelBridge), metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, order: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'first', signed: [bool](https://docs.python.org/3/library/functions.html#bool) = True, \*\*sobol_kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), ndarray]]

Compute sensitivity for all metrics on an TorchModelBridge.

Sobol measures are always positive regardless of the direction in which the
parameter influences f. If signed is set to True, then the Sobol measure for each
parameter will be given as its sign the sign of the average gradient with respect to
that parameter across the search space. Thus, important parameters that, when
increased, decrease f will have large and negative values; unimportant parameters
will have values close to 0.

* **Parameters:**
  * **model_bridge** – A ModelBridge object with models that were fit.
  * **metrics** – The names of the metrics and outcomes for which to compute
    sensitivities. This should preferably be metrics with a good model fit.
    Defaults to model_bridge.outcomes.
  * **order** – A string specifying the order of the Sobol indices to be computed.
    Supports “first” and “total” and defaults to “first”.
  * **signed** – A bool for whether the measure should be signed.
  * **sobol_kwargs** – keyword arguments passed on to SobolSensitivityGPMean, and if
    signed, GpDGSMGpMean.
* **Returns:**
  {‘parameter_name’: sensitivity_value}}, where the
  : sensitivity value is cast to a Numpy array in order to be compatible with
    plot_feature_importance_by_feature.
* **Return type:**
  Dictionary {‘metric_name’

### ax.utils.sensitivity.sobol_measures.compute_sobol_indices_from_model_list(model_list: [list](https://docs.python.org/3/library/stdtypes.html#list)[Model], bounds: Tensor, order: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'first', discrete_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*sobol_kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → Tensor

Computes Sobol indices of a list of models on a bounded domain.

* **Parameters:**
  * **model_list** – A list of botorch.models.model.Model types for which to compute
    the Sobol indices.
  * **bounds** – A 2 x d Tensor of lower and upper bounds of the domain of the models.
  * **order** – A string specifying the order of the Sobol indices to be computed.
    Supports “first” and “total” and defaults to “first”.
  * **discrete_features** – If specified, the inputs associated with the indices in
    this list are generated using an integer-valued uniform distribution,
    rather than the default (pseudo-)random continuous uniform distribution.
  * **sobol_kwargs** – keyword arguments passed on to SobolSensitivityGPMean.
* **Returns:**
  With m GPs, returns a (m x d) tensor of order-order Sobol indices.

## Stats

### Statstools

### Model Fit Metrics

## Testing

### Backend Scheduler

### Backend Simulator

### *class* ax.utils.testing.backend_simulator.BackendSimulator(options: [BackendSimulatorOptions](#ax.utils.testing.backend_simulator.BackendSimulatorOptions) | [None](https://docs.python.org/3/library/constants.html#None) = None, queued: [list](https://docs.python.org/3/library/stdtypes.html#list)[[SimTrial](#ax.utils.testing.backend_simulator.SimTrial)] | [None](https://docs.python.org/3/library/constants.html#None) = None, running: [list](https://docs.python.org/3/library/stdtypes.html#list)[[SimTrial](#ax.utils.testing.backend_simulator.SimTrial)] | [None](https://docs.python.org/3/library/constants.html#None) = None, failed: [list](https://docs.python.org/3/library/stdtypes.html#list)[[SimTrial](#ax.utils.testing.backend_simulator.SimTrial)] | [None](https://docs.python.org/3/library/constants.html#None) = None, completed: [list](https://docs.python.org/3/library/stdtypes.html#list)[[SimTrial](#ax.utils.testing.backend_simulator.SimTrial)] | [None](https://docs.python.org/3/library/constants.html#None) = None, verbose_logging: [bool](https://docs.python.org/3/library/functions.html#bool) = True)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Simulator for a backend deployment with concurrent dispatch and a queue.

#### *property* all_trials *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[SimTrial](#ax.utils.testing.backend_simulator.SimTrial)]*

All trials on the simulator.

#### *classmethod* from_state(state: [BackendSimulatorState](#ax.utils.testing.backend_simulator.BackendSimulatorState))

Construct a simulator from a state.

* **Parameters:**
  **state** – A `BackendSimulatorState` to set the simulator to.
* **Returns:**
  A `BackendSimulator` with the desired state.

#### get_sim_trial_by_index(trial_index: [int](https://docs.python.org/3/library/functions.html#int)) → [SimTrial](#ax.utils.testing.backend_simulator.SimTrial) | [None](https://docs.python.org/3/library/constants.html#None)

Get a `SimTrial` by `trial_index`.

* **Parameters:**
  **trial_index** – The index of the trial to return.
* **Returns:**
  A `SimTrial` with the index `trial_index` or None if not found.

#### lookup_trial_index_status(trial_index: [int](https://docs.python.org/3/library/functions.html#int)) → [TrialStatus](core.md#ax.core.base_trial.TrialStatus) | [None](https://docs.python.org/3/library/constants.html#None)

Lookup the trial status of a `trial_index`.

* **Parameters:**
  **trial_index** – The index of the trial to check.
* **Returns:**
  A `TrialStatus`.

#### new_trial(trial: [SimTrial](#ax.utils.testing.backend_simulator.SimTrial), status: [TrialStatus](core.md#ax.core.base_trial.TrialStatus)) → [None](https://docs.python.org/3/library/constants.html#None)

Register a trial into the simulator.

* **Parameters:**
  * **trial** – A new trial to add.
  * **status** – The status of the new trial, either STAGED (add to `self._queued`)
    or RUNNING (add to `self._running`).

#### *property* num_completed *: [int](https://docs.python.org/3/library/functions.html#int)*

The number of completed trials.

#### *property* num_failed *: [int](https://docs.python.org/3/library/functions.html#int)*

The number of failed trials.

#### *property* num_queued *: [int](https://docs.python.org/3/library/functions.html#int)*

The number of queued trials (to run as soon as capacity is available).

#### *property* num_running *: [int](https://docs.python.org/3/library/functions.html#int)*

The number of currently running trials.

#### reset() → [None](https://docs.python.org/3/library/constants.html#None)

Reset the simulator.

#### run_trial(trial_index: [int](https://docs.python.org/3/library/functions.html#int), runtime: [float](https://docs.python.org/3/library/functions.html#float)) → [None](https://docs.python.org/3/library/constants.html#None)

Run a simulated trial.

* **Parameters:**
  * **trial_index** – The index of the trial (usually the Ax trial index)
  * **runtime** – The runtime of the simulation. Typically sampled from the
    runtime model of a simulation model.

Internally, the runtime is scaled by the time_scaling factor, so that
the simulation can run arbitrarily faster than the underlying evaluation.

#### state() → [BackendSimulatorState](#ax.utils.testing.backend_simulator.BackendSimulatorState)

Return a `BackendSimulatorState` containing the state of the simulator.

#### status() → [SimStatus](#ax.utils.testing.backend_simulator.SimStatus)

Return the internal status of the simulator.

* **Returns:**
  A `SimStatus` object representing the current simulator state.

#### stop_trial(trial_index: [int](https://docs.python.org/3/library/functions.html#int)) → [None](https://docs.python.org/3/library/constants.html#None)

Stop a simulated trial by setting the completed time to the current time.

* **Parameters:**
  **trial_index** – The index of the trial to stop.

#### *property* time *: [float](https://docs.python.org/3/library/functions.html#float)*

The current time.

#### update() → [None](https://docs.python.org/3/library/constants.html#None)

Update the state of the simulator.

#### *property* use_internal_clock *: [bool](https://docs.python.org/3/library/functions.html#bool)*

Whether or not we are using the internal clock.

### *class* ax.utils.testing.backend_simulator.BackendSimulatorOptions(max_concurrency: [int](https://docs.python.org/3/library/functions.html#int) = 1, time_scaling: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, failure_rate: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, internal_clock: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, use_update_as_start_time: [bool](https://docs.python.org/3/library/functions.html#bool) = False)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Settings for the BackendSimulator.

* **Parameters:**
  * **max_concurrency** – The maximum number of trials that can be run
    in parallel.
  * **time_scaling** – The factor to scale down the runtime of the tasks by.
    If `runtime` is the actual runtime of a trial, the simulation
    time will be `runtime / time_scaling`.
  * **failure_rate** – The rate at which the trials are failing. For now, trials
    fail independently with at coin flip based on that rate.
  * **internal_clock** – The initial state of the internal clock. If None,
    the simulator uses `time.time()` as the clock.
  * **use_update_as_start_time** – Whether the start time of a new trial should be logged
    as the current time (at time of update) or end time of previous trial.
    This makes sense when using the internal clock and the BackendSimulator
    is simulated forward by an external process (such as Scheduler).

#### failure_rate *: [float](https://docs.python.org/3/library/functions.html#float)* *= 0.0*

#### internal_clock *: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### max_concurrency *: [int](https://docs.python.org/3/library/functions.html#int)* *= 1*

#### time_scaling *: [float](https://docs.python.org/3/library/functions.html#float)* *= 1.0*

#### use_update_as_start_time *: [bool](https://docs.python.org/3/library/functions.html#bool)* *= False*

### *class* ax.utils.testing.backend_simulator.BackendSimulatorState(options: [BackendSimulatorOptions](#ax.utils.testing.backend_simulator.BackendSimulatorOptions), verbose_logging: [bool](https://docs.python.org/3/library/functions.html#bool), queued: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)]], running: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)]], failed: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)]], completed: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)]])

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

State of the BackendSimulator.

* **Parameters:**
  * **options** – The BackendSimulatorOptions associated with this simulator.
  * **verbose_logging** – Whether the simulator is using verbose logging.
  * **queued** – Currently queued trials.
  * **running** – Currently running trials.
  * **failed** – Currently failed trials.
  * **completed** – Currently completed trials.

#### completed *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)]]*

#### failed *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)]]*

#### options *: [BackendSimulatorOptions](#ax.utils.testing.backend_simulator.BackendSimulatorOptions)*

#### queued *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)]]*

#### running *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)]]*

#### verbose_logging *: [bool](https://docs.python.org/3/library/functions.html#bool)*

### *class* ax.utils.testing.backend_simulator.SimStatus(queued: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], running: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], failed: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], time_remaining: [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)], completed: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)])

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Container for status of the simulation.

#### queued

List of indices of queued trials.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]

#### running

List of indices of running trials.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]

#### failed

List of indices of failed trials.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]

#### time_remaining

List of sim time remaining for running trials.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]

#### completed

List of indicies of completed trials.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]

#### completed *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]*

#### failed *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]*

#### queued *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]*

#### running *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]*

#### time_remaining *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]*

### *class* ax.utils.testing.backend_simulator.SimTrial(trial_index: [int](https://docs.python.org/3/library/functions.html#int), sim_runtime: [float](https://docs.python.org/3/library/functions.html#float), sim_start_time: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, sim_queued_time: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, sim_completed_time: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Container for the simulation tasks.

#### trial_index

The index of the trial (should match Ax trial index).

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int)

#### sim_runtime

The runtime of the trial (sampled at creation).

* **Type:**
  [float](https://docs.python.org/3/library/functions.html#float)

#### sim_start_time

When the trial started running (or exits queued state).

* **Type:**
  [float](https://docs.python.org/3/library/functions.html#float) | None

#### sim_queued_time

When the trial was initially queued.

* **Type:**
  [float](https://docs.python.org/3/library/functions.html#float) | None

#### sim_completed_time

When the trial was marked as completed. Currently,
this is used by an early-stopper via `stop_trial`.

* **Type:**
  [float](https://docs.python.org/3/library/functions.html#float) | None

#### sim_completed_time *: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### sim_queued_time *: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### sim_runtime *: [float](https://docs.python.org/3/library/functions.html#float)*

#### sim_start_time *: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### trial_index *: [int](https://docs.python.org/3/library/functions.html#int)*

### ax.utils.testing.backend_simulator.format(trial_list: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)]]) → [str](https://docs.python.org/3/library/stdtypes.html#str)

Helper function for formatting a list.

### Benchmark Stubs

### Core Stubs

### Modeling Stubs

### Preference Stubs

### Mocking

### ax.utils.testing.mock.fast_botorch_optimize(f: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Wraps f in the fast_botorch_optimize_context_manager for use as a decorator.

### ax.utils.testing.mock.fast_botorch_optimize_context_manager(force: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [Generator](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)[[None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None)]

A context manager to force botorch to speed up optimization. Currently, the
primary tactic is to force the underlying scipy methods to stop after just one
iteration.

> force: If True will not raise an AssertionError if no mocks are called.
> : USE RESPONSIBLY.

### ax.utils.testing.mock.skip_fit_gpytorch_mll(f: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Wraps f in the skip_fit_gpytorch_mll_context_manager for use as a decorator.

### ax.utils.testing.mock.skip_fit_gpytorch_mll_context_manager() → [Generator](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)[[None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None)]

A context manager that makes fit_gpytorch_mll a no-op.

This should only be used to speed up slow tests.

### Test Init Files

### *class* ax.utils.testing.test_init_files.InitTest(methodName: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'runTest')

Bases: [`TestCase`](#ax.utils.common.testutils.TestCase)

#### test_InitFiles() → [None](https://docs.python.org/3/library/constants.html#None)

\_\_init_\_.py files are necessary when not using buck targets

### Torch Stubs

### ax.utils.testing.torch_stubs.get_torch_test_data(dtype: dtype = torch.float32, cuda: [bool](https://docs.python.org/3/library/functions.html#bool) = False, constant_noise: [bool](https://docs.python.org/3/library/functions.html#bool) = True, task_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, offset: [float](https://docs.python.org/3/library/functions.html#float) = 0.0) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]], [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]]

### Utils

### ax.utils.testing.utils.generic_equals(first: [Any](https://docs.python.org/3/library/typing.html#typing.Any), second: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [bool](https://docs.python.org/3/library/functions.html#bool)

## Test Metrics

### Backend Simulator Map

### Branin Backend Map

## Tutorials

### Neural Net

### *class* ax.utils.tutorials.cnn_utils.CNN

Bases: `Module`

Convolutional Neural Network.

#### forward(x: Tensor) → Tensor

Define the computation performed at every call.

Should be overridden by all subclasses.

#### NOTE
Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

### ax.utils.tutorials.cnn_utils.evaluate(net: Module, data_loader: DataLoader, dtype: dtype, device: device) → [float](https://docs.python.org/3/library/functions.html#float)

Compute classification accuracy on provided dataset.

* **Parameters:**
  * **net** – trained model
  * **data_loader** – DataLoader containing the evaluation set
  * **dtype** – torch dtype
  * **device** – torch device
* **Returns:**
  classification accuracy
* **Return type:**
  [float](https://docs.python.org/3/library/functions.html#float)

### ax.utils.tutorials.cnn_utils.get_partition_data_loaders(train_valid_set: Dataset, test_set: Dataset, downsample_pct: [float](https://docs.python.org/3/library/functions.html#float) = 0.5, train_pct: [float](https://docs.python.org/3/library/functions.html#float) = 0.8, batch_size: [int](https://docs.python.org/3/library/functions.html#int) = 128, num_workers: [int](https://docs.python.org/3/library/functions.html#int) = 0, deterministic_partitions: [bool](https://docs.python.org/3/library/functions.html#bool) = False, downsample_pct_test: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[DataLoader, DataLoader, DataLoader]

Helper function for partitioning training data into training and validation sets,
: downsampling data, and initializing DataLoaders for each partition.

* **Parameters:**
  * **train_valid_set** – torch.dataset
  * **downsample_pct** – the proportion of the dataset to use for training, and
    validation
  * **train_pct** – the proportion of the downsampled data to use for training
  * **batch_size** – how many samples per batch to load
  * **num_workers** – number of workers (subprocesses) for loading data
  * **deterministic_partitions** – whether to partition data in a deterministic
    fashion
  * **downsample_pct_test** – the proportion of the dataset to use for test, default
    to be equal to downsample_pct
* **Returns:**
  training data
  DataLoader: validation data
  DataLoader: test data
* **Return type:**
  DataLoader

### ax.utils.tutorials.cnn_utils.load_mnist(downsample_pct: [float](https://docs.python.org/3/library/functions.html#float) = 0.5, train_pct: [float](https://docs.python.org/3/library/functions.html#float) = 0.8, data_path: [str](https://docs.python.org/3/library/stdtypes.html#str) = './data', batch_size: [int](https://docs.python.org/3/library/functions.html#int) = 128, num_workers: [int](https://docs.python.org/3/library/functions.html#int) = 0, deterministic_partitions: [bool](https://docs.python.org/3/library/functions.html#bool) = False, downsample_pct_test: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[DataLoader, DataLoader, DataLoader]

Load MNIST dataset (download if necessary) and split data into training,
: validation, and test sets.

* **Parameters:**
  * **downsample_pct** – the proportion of the dataset to use for training,
    validation, and test
  * **train_pct** – the proportion of the downsampled data to use for training
  * **data_path** – Root directory of dataset where MNIST/processed/training.pt
    and MNIST/processed/test.pt exist.
  * **batch_size** – how many samples per batch to load
  * **num_workers** – number of workers (subprocesses) for loading data
  * **deterministic_partitions** – whether to partition data in a deterministic
    fashion
  * **downsample_pct_test** – the proportion of the dataset to use for test, default
    to be equal to downsample_pct
* **Returns:**
  training data
  DataLoader: validation data
  DataLoader: test data
* **Return type:**
  DataLoader

### ax.utils.tutorials.cnn_utils.split_dataset(dataset: Dataset, lengths: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], deterministic_partitions: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [list](https://docs.python.org/3/library/stdtypes.html#list)[Dataset]

Split a dataset either randomly or deterministically.

* **Parameters:**
  * **dataset** – the dataset to split
  * **lengths** – the lengths of each partition
  * **deterministic_partitions** – deterministic_partitions: whether to partition
    data in a deterministic fashion
* **Returns:**
  split datasets
* **Return type:**
  List[Dataset]

### ax.utils.tutorials.cnn_utils.train(net: Module, train_loader: DataLoader, parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)], dtype: dtype, device: device) → Module

Train CNN on provided data set.

* **Parameters:**
  * **net** – initialized neural network
  * **train_loader** – DataLoader containing training set
  * **parameters** – dictionary containing parameters to be passed to the optimizer.
    - lr: default (0.001)
    - momentum: default (0.0)
    - weight_decay: default (0.0)
    - num_epochs: default (1)
  * **dtype** – torch dtype
  * **device** – torch device
* **Returns:**
  trained CNN.
* **Return type:**
  nn.Module
