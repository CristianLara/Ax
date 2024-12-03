# ax.models

## Base Models & Utilities

### ax.models.base

### *class* ax.models.base.Model

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Base class for an Ax model.

Note: the core methods each model has: fit, predict, gen,
cross_validate, and best_point are not present in this base class,
because the signatures for those methods vary based on the type of the model.
This class only contains the methods that all models have in common and for
which they all share the signature.

#### *classmethod* deserialize_state(serialized_state: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Restores model’s state from its serialized form, to the format it
expects to receive as kwargs.

#### feature_importances() → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

#### *classmethod* serialize_state(raw_state: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Serialized output of self._get_state to a JSON-ready dict.
This may involve storing part of state in files / external storage and
saving handles for that storage in the resulting serialized state.

### ax.models.discrete_base module

### *class* ax.models.discrete_base.DiscreteModel

Bases: [`Model`](#ax.models.base.Model)

This class specifies the interface for a model based on discrete parameters.

These methods should be implemented to have access to all of the features
of Ax.

#### best_point(n: [int](https://docs.python.org/3/library/functions.html#int), parameter_values: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]], objective_weights: ndarray | [None](https://docs.python.org/3/library/constants.html#None), outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, pending_observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]]] | [None](https://docs.python.org/3/library/constants.html#None) = None, model_gen_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None)

Obtains the point that has the best value according to the model
prediction and its model predictions.

* **Returns:**
  (1 x d) parameter value list representing the point with the best
  value according to the model prediction. None if this function
  is not implemented for the given model.

#### cross_validate(Xs_train: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]]], Ys_train: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], Yvars_train: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], X_test: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]], use_posterior_predictive: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray]

Do cross validation with the given training and test sets.

Training set is given in the same format as to fit. Test set is given
in the same format as to predict.

* **Parameters:**
  * **Xs_train** – A list of m lists X of parameterizations (each parameterization
    is a list of parameter values of length d), each of length k_i,
    for each outcome.
  * **Ys_train** – The corresponding list of m lists Y, each of length k_i, for
    each outcome.
  * **Yvars_train** – The variances of each entry in Ys, same shape.
  * **X_test** – List of the j parameterizations at which to make predictions.
  * **use_posterior_predictive** – A boolean indicating if the predictions
    should be from the posterior predictive (i.e. including
    observation noise).
* **Returns:**
  2-element tuple containing
  - (j x m) array of outcome predictions at X.
  - (j x m x m) array of predictive covariances at X.
    cov[j, m1, m2] is Cov[m1@j, m2@j].

#### fit(Xs: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]]], Ys: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], Yvars: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], parameter_values: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]], outcome_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [None](https://docs.python.org/3/library/constants.html#None)

Fit model to m outcomes.

* **Parameters:**
  * **Xs** – A list of m lists X of parameterizations (each parameterization
    is a list of parameter values of length d), each of length k_i,
    for each outcome.
  * **Ys** – The corresponding list of m lists Y, each of length k_i, for
    each outcome.
  * **Yvars** – The variances of each entry in Ys, same shape.
  * **parameter_values** – A list of possible values for each parameter.
  * **outcome_names** – A list of m outcome names.

#### gen(n: [int](https://docs.python.org/3/library/functions.html#int), parameter_values: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]], objective_weights: ndarray | [None](https://docs.python.org/3/library/constants.html#None), outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, pending_observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]]] | [None](https://docs.python.org/3/library/constants.html#None) = None, model_gen_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]], [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]

Generate new candidates.

* **Parameters:**
  * **n** – Number of candidates to generate.
  * **parameter_values** – A list of possible values for each parameter.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value during generation.
  * **pending_observations** – A list of m lists of parameterizations
    (each parameterization is a list of parameter values of length d),
    each of length k_i, for each outcome i.
  * **model_gen_options** – A config dictionary that can contain
    model-specific options.
* **Returns:**
  2-element tuple containing
  - List of n generated points, where each point is represented
    by a list of parameter values.
  - List of weights for each of the n points.

#### predict(X: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]]) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray]

Predict

* **Parameters:**
  **X** – List of the j parameterizations at which to make predictions.
* **Returns:**
  2-element tuple containing
  - (j x m) array of outcome predictions at X.
  - (j x m x m) array of predictive covariances at X.
    cov[j, m1, m2] is Cov[m1@j, m2@j].

### ax.models.torch_base module

### *class* ax.models.torch_base.TorchGenResults(points: ~torch.Tensor, weights: ~torch.Tensor, gen_metadata: dict[str, ~typing.Any] = <factory>, candidate_metadata: list[dict[str, ~typing.Any] | None] | None = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

points: (n x d) Tensor of generated points.
weights: n-tensor of weights for each point.
gen_metadata: Generation metadata
Dictionary of model-specific metadata for the given

> generation candidates

#### candidate_metadata *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### gen_metadata *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*

#### points *: Tensor*

#### weights *: Tensor*

### *class* ax.models.torch_base.TorchModel

Bases: [`Model`](#ax.models.base.Model)

This class specifies the interface for a torch-based model.

These methods should be implemented to have access to all of the features
of Ax.

#### best_point(search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig)) → Tensor | [None](https://docs.python.org/3/library/constants.html#None)

Identify the current best point, satisfying the constraints in the same
format as to gen.

Return None if no such point can be identified.

* **Parameters:**
  * **search_space_digest** – A SearchSpaceDigest object containing metadata
    about the search space (e.g. bounds, parameter types).
  * **torch_opt_config** – A TorchOptConfig object containing optimization
    arguments (e.g., objective weights, constraints).
* **Returns:**
  d-tensor of the best point.

#### cross_validate(datasets: [list](https://docs.python.org/3/library/stdtypes.html#list)[SupervisedDataset], X_test: Tensor, search_space_digest: SearchSpaceDigest, use_posterior_predictive: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

Do cross validation with the given training and test sets.

Training set is given in the same format as to fit. Test set is given
in the same format as to predict.

* **Parameters:**
  * **datasets** – A list of `SupervisedDataset` containers, each
    corresponding to the data of one metric (outcome).
  * **X_test** – (j x d) tensor of the j points at which to make predictions.
  * **search_space_digest** – A SearchSpaceDigest object containing
    metadata on the features in X.
  * **use_posterior_predictive** – A boolean indicating if the predictions
    should be from the posterior predictive (i.e. including
    observation noise).
* **Returns:**
  2-element tuple containing
  - (j x m) tensor of outcome predictions at X.
  - (j x m x m) tensor of predictive covariances at X.
    cov[j, m1, m2] is Cov[m1@j, m2@j].

#### device *: device | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### dtype *: dtype | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### evaluate_acquisition_function(X: Tensor, search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig), acq_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → Tensor

Evaluate the acquisition function on the candidate set X.

* **Parameters:**
  * **X** – (j x d) tensor of the j points at which to evaluate the acquisition
    function.
  * **search_space_digest** – A dataclass used to compactly represent a search space.
  * **torch_opt_config** – A TorchOptConfig object containing optimization
    arguments (e.g., objective weights, constraints).
  * **acq_options** – Keyword arguments used to contruct the acquisition function.
* **Returns:**
  A single-element tensor with the acquisition value for these points.

#### fit(datasets: [list](https://docs.python.org/3/library/stdtypes.html#list)[SupervisedDataset], search_space_digest: SearchSpaceDigest, candidate_metadata: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)]] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)

Fit model to m outcomes.

* **Parameters:**
  * **datasets** – A list of `SupervisedDataset` containers, each
    corresponding to the data of one metric (outcome).
  * **search_space_digest** – A `SearchSpaceDigest` object containing
    metadata on the features in the datasets.
  * **candidate_metadata** – Model-produced metadata for candidates, in
    the order corresponding to the Xs.

#### gen(n: [int](https://docs.python.org/3/library/functions.html#int), search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig)) → [TorchGenResults](#ax.models.torch_base.TorchGenResults)

Generate new candidates.

* **Parameters:**
  * **n** – Number of candidates to generate.
  * **search_space_digest** – A SearchSpaceDigest object containing metadata
    about the search space (e.g. bounds, parameter types).
  * **torch_opt_config** – A TorchOptConfig object containing optimization
    arguments (e.g., objective weights, constraints).
* **Returns:**
  A TorchGenResult container.

#### predict(X: Tensor) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

Predict

* **Parameters:**
  **X** – (j x d) tensor of the j points at which to make predictions.
* **Returns:**
  2-element tuple containing
  - (j x m) tensor of outcome predictions at X.
  - (j x m x m) tensor of predictive covariances at X.
    cov[j, m1, m2] is Cov[m1@j, m2@j].

#### update(datasets: [list](https://docs.python.org/3/library/stdtypes.html#list)[SupervisedDataset], metric_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], search_space_digest: SearchSpaceDigest, candidate_metadata: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)]] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)

Update the model.

Updating the model requires both existing and additional data.
The data passed into this method will become the new training data.

* **Parameters:**
  * **datasets** – A list of `SupervisedDataset` containers, each
    corresponding to the data of one metric (outcome). None
    means that there is no additional data for the corresponding
    outcome.
  * **metric_names** – A list of metric names, with the i-th metric
    corresponding to the i-th dataset.
  * **search_space_digest** – A SearchSpaceDigest object containing
    metadata on the features in X.
  * **candidate_metadata** – Model-produced metadata for candidates, in
    the order corresponding to the Xs.

### *class* ax.models.torch_base.TorchOptConfig(objective_weights: ~torch.Tensor, outcome_constraints: tuple[~torch.Tensor, ~torch.Tensor] | None = None, objective_thresholds: ~torch.Tensor | None = None, linear_constraints: tuple[~torch.Tensor, ~torch.Tensor] | None = None, fixed_features: dict[int, float] | None = None, pending_observations: list[~torch.Tensor] | None = None, model_gen_options: dict[str, int | float | str | ~botorch.acquisition.acquisition.AcquisitionFunction | list[str] | dict[int, ~typing.Any] | dict[str, ~typing.Any] | ~ax.core.optimization_config.OptimizationConfig | ~ax.models.winsorization_config.WinsorizationConfig | None] = <factory>, rounding_func: ~collections.abc.Callable[[~torch.Tensor], ~torch.Tensor] | None = None, opt_config_metrics: dict[str, ~ax.core.metric.Metric] | None = None, is_moo: bool = False, risk_measure: ~botorch.acquisition.risk_measures.RiskMeasureMCObjective | None = None, fit_out_of_design: bool = False)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Container for lightweight representation of optimization arguments.

This is used for communicating between modelbridge and models. This is
an ephemeral object and not meant to be stored / serialized.

#### objective_weights

If doing multi-objective optimization, these denote
which objectives should be maximized and which should be minimized.
Otherwise, the objective is to maximize a weighted sum of
the columns of f(x). These are the weights.

* **Type:**
  torch.Tensor

#### outcome_constraints

A tuple of (A, b). For k outcome constraints
and m outputs at f(x), A is (k x m) and b is (k x 1) such that
A f(x) <= b.

* **Type:**
  [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[torch.Tensor, torch.Tensor] | None

#### objective_thresholds

A tensor containing thresholds forming a
reference point from which to calculate pareto frontier hypervolume.
Points that do not dominate the objective_thresholds contribute
nothing to hypervolume.

* **Type:**
  torch.Tensor | None

#### linear_constraints

A tuple of (A, b). For k linear constraints on
d-dimensional x, A is (k x d) and b is (k x 1) such that
A x <= b for feasible x.

* **Type:**
  [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[torch.Tensor, torch.Tensor] | None

#### fixed_features

A map {feature_index: value} for features that
should be fixed to a particular value during generation.

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | None

#### pending_observations

A list of m (k_i x d) feature tensors X
for m outcomes and k_i pending observations for outcome i.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[torch.Tensor] | None

#### model_gen_options

A config dictionary that can contain
model-specific options. This commonly includes optimizer_kwargs,
which often specifies the optimizer options to be passed to the
optimizer while optimizing the acquisition function. These are
generally expected to mimic the signature of optimize_acqf,
though not all models may support all possible arguments and some
models may support additional arguments that are not passed to the
optimizer. While constructing a generation strategy, these options
can be passed in as follows:
>>> model_gen_kwargs = {
>>>     “model_gen_options”: {
>>>         “optimizer_kwargs”: {
>>>             “num_restarts”: 20,
>>>             “sequential”: False,
>>>             “options”: {
>>>                 “batch_limit: 5,
>>>                 “maxiter”: 200,
>>>             },
>>>         },
>>>     },
>>> }

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | botorch.acquisition.acquisition.AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), Any] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Any] | [ax.core.optimization_config.OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [ax.models.winsorization_config.WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | None]

#### rounding_func

A function that rounds an optimization result
appropriately (i.e., according to round-trip transformations).

* **Type:**
  [collections.abc.Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[torch.Tensor], torch.Tensor] | None

#### opt_config_metrics

A dictionary of metrics that are included in
the optimization config.

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), ax.core.metric.Metric] | None

#### is_moo

A boolean denoting whether this is for an MOO problem.

* **Type:**
  [bool](https://docs.python.org/3/library/functions.html#bool)

#### risk_measure

An optional risk measure, used for robust optimization.

* **Type:**
  botorch.acquisition.risk_measures.RiskMeasureMCObjective | None

#### fit_out_of_design *: [bool](https://docs.python.org/3/library/functions.html#bool)* *= False*

#### fixed_features *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### is_moo *: [bool](https://docs.python.org/3/library/functions.html#bool)* *= False*

#### linear_constraints *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### model_gen_options *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)]*

#### objective_thresholds *: Tensor | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### objective_weights *: Tensor*

#### opt_config_metrics *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Metric] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### outcome_constraints *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### pending_observations *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### risk_measure *: RiskMeasureMCObjective | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### rounding_func *: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[Tensor], Tensor] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

### ax.models.model_utils module

### *class* ax.models.model_utils.TorchModelLike(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A protocol that stands in for `TorchModel` like objects that
have a `predict` method.

#### predict(X: Tensor) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

Predicts outcomes given an input tensor.

* **Parameters:**
  **X** – A `n x d` tensor of input parameters.
* **Returns:**
  The predicted posterior mean as an `n x o`-dim tensor.
  Tensor: The predicted posterior covariance as a `n x o x o`-dim tensor.
* **Return type:**
  Tensor

### ax.models.model_utils.add_fixed_features(tunable_points: ndarray, d: [int](https://docs.python.org/3/library/functions.html#int), fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None), tunable_feature_indices: ndarray) → ndarray

Add fixed features to points in tunable space.

* **Parameters:**
  * **tunable_points** – Points in tunable space.
  * **d** – Dimension of parameter space.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value during generation.
  * **tunable_feature_indices** – Parameter indices (in d) which are tunable.
* **Returns:**
  Points in the full d-dimensional space, defined by bounds.
* **Return type:**
  points

### ax.models.model_utils.as_array(x: Tensor | ndarray | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor | ndarray, ...]) → ndarray | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ...]

Convert every item in a tuple of tensors/arrays into an array.

* **Parameters:**
  **x** – A tensor, array, or a tuple of potentially mixed tensors and arrays.
* **Returns:**
  x, with everything converted to array.

### ax.models.model_utils.best_in_sample_point(Xs: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor] | [list](https://docs.python.org/3/library/stdtypes.html#list)[ndarray], model: [TorchModelLike](#ax.models.model_utils.TorchModelLike), bounds: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]], objective_weights: Tensor | ndarray | [None](https://docs.python.org/3/library/constants.html#None), outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor | ndarray, Tensor | ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None, linear_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor | ndarray, Tensor | ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, risk_measure: RiskMeasureMCObjective | [None](https://docs.python.org/3/library/constants.html#None) = None, options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor | ndarray, [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None)

Select the best point that has been observed.

Implements two approaches to selecting the best point.

For both approaches, only points that satisfy parameter space constraints
(bounds, linear_constraints, fixed_features) will be returned. Points must
also be observed for all objective and constraint outcomes. Returned
points may violate outcome constraints, depending on the method below.

1: Select the point that maximizes the expected utility
(objective_weights^T posterior_objective_means - baseline) \* Prob(feasible)
Here baseline should be selected so that at least one point has positive
utility. It can be specified in the options dict, otherwise
min (objective_weights^T posterior_objective_means)
will be used, where the min is over observed points.

2: Select the best-objective point that is feasible with at least
probability p.

The following quantities may be specified in the options dict:

- best_point_method: ‘max_utility’ (default) or ‘feasible_threshold’
  to select between the two approaches described above.
- utility_baseline: Value for the baseline used in max_utility approach. If
  not provided, defaults to min objective value.
- probability_threshold: Threshold for the feasible_threshold approach.
  Defaults to p=0.95.
- feasibility_mc_samples: Number of MC samples used for estimating the
  probability of feasibility (defaults 10k).

* **Parameters:**
  * **Xs** – Training data for the points, among which to select the best.
  * **model** – A Torch model or Surrogate.
  * **bounds** – A list of (lower, upper) tuples for each feature.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b.
  * **linear_constraints** – A tuple of (A, b). For k linear constraints on
    d-dimensional x, A is (k x d) and b is (k x 1) such that
    A x <= b.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value in the best point.
  * **risk_measure** – An optional risk measure for reporting best robust point.
  * **options** – A config dictionary with settings described above.
* **Returns:**
  - d-array of the best point,
  - utility at the best point.
* **Return type:**
  A two-element tuple or None if no feasible point exist. In tuple

### ax.models.model_utils.best_observed_point(model: [TorchModelLike](#ax.models.model_utils.TorchModelLike), bounds: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]], objective_weights: Tensor | ndarray | [None](https://docs.python.org/3/library/constants.html#None), outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor | ndarray, Tensor | ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None, linear_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor | ndarray, Tensor | ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, risk_measure: RiskMeasureMCObjective | [None](https://docs.python.org/3/library/constants.html#None) = None, options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → Tensor | ndarray | [None](https://docs.python.org/3/library/constants.html#None)

Select the best point that has been observed.

Implements two approaches to selecting the best point.

For both approaches, only points that satisfy parameter space constraints
(bounds, linear_constraints, fixed_features) will be returned. Points must
also be observed for all objective and constraint outcomes. Returned
points may violate outcome constraints, depending on the method below.

1: Select the point that maximizes the expected utility
(objective_weights^T posterior_objective_means - baseline) \* Prob(feasible)
Here baseline should be selected so that at least one point has positive
utility. It can be specified in the options dict, otherwise
min (objective_weights^T posterior_objective_means)
will be used, where the min is over observed points.

2: Select the best-objective point that is feasible with at least
probability p.

The following quantities may be specified in the options dict:

- best_point_method: ‘max_utility’ (default) or ‘feasible_threshold’
  to select between the two approaches described above.
- utility_baseline: Value for the baseline used in max_utility approach. If
  not provided, defaults to min objective value.
- probability_threshold: Threshold for the feasible_threshold approach.
  Defaults to p=0.95.
- feasibility_mc_samples: Number of MC samples used for estimating the
  probability of feasibility (defaults 10k).

* **Parameters:**
  * **model** – A Torch model or Surrogate.
  * **bounds** – A list of (lower, upper) tuples for each feature.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b.
  * **linear_constraints** – A tuple of (A, b). For k linear constraints on
    d-dimensional x, A is (k x d) and b is (k x 1) such that
    A x <= b.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value in the best point.
  * **risk_measure** – An optional risk measure for reporting best robust point.
  * **options** – A config dictionary with settings described above.
* **Returns:**
  A d-array of the best point, or None if no feasible point exists.

### ax.models.model_utils.check_duplicate(point: ndarray, points: ndarray) → [bool](https://docs.python.org/3/library/functions.html#bool)

Check if a point exists in another array.

* **Parameters:**
  * **point** – Newly generated point to check.
  * **points** – Points previously generated.
* **Returns:**
  True if the point is contained in points, else False

### ax.models.model_utils.check_param_constraints(linear_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray], point: ndarray) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[bool](https://docs.python.org/3/library/functions.html#bool), ndarray]

Check if a point satisfies parameter constraints.

* **Parameters:**
  * **linear_constraints** – A tuple of (A, b). For k linear constraints on
    d-dimensional x, A is (k x d) and b is (k x 1) such that
    A x <= b.
  * **point** – A candidate point in d-dimensional space, as a (1 x d) matrix.
* **Returns:**
  2-element tuple containing
  - Flag that is True if all constraints are satisfied by the point.
  - Indices of constraints which are violated by the point.

### ax.models.model_utils.enumerate_discrete_combinations(discrete_choices: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[int](https://docs.python.org/3/library/functions.html#int), [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float)]]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]]

### ax.models.model_utils.filter_constraints_and_fixed_features(X: Tensor | ndarray, bounds: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]], linear_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor | ndarray, Tensor | ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → Tensor | ndarray

Filter points to those that satisfy bounds, linear_constraints, and
fixed_features.

* **Parameters:**
  * **X** – An tensor or array of points.
  * **bounds** – A list of (lower, upper) tuples for each feature.
  * **linear_constraints** – A tuple of (A, b). For k linear constraints on
    d-dimensional x, A is (k x d) and b is (k x 1) such that
    A x <= b.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value in the best point.
* **Returns:**
  Feasible points.

### ax.models.model_utils.get_observed(Xs: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor] | [list](https://docs.python.org/3/library/stdtypes.html#list)[ndarray], objective_weights: Tensor | ndarray, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor | ndarray, Tensor | ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None) → Tensor | ndarray

Filter points to those that are observed for objective outcomes and outcomes
that show up in outcome_constraints (if there are any).

* **Parameters:**
  * **Xs** – A list of m (k_i x d) feature matrices X. Number of rows k_i
    can vary from i=1,…,m.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b.
* **Returns:**
  Points observed for all objective outcomes and outcome constraints.

### ax.models.model_utils.mk_discrete_choices(ssd: SearchSpaceDigest, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[int](https://docs.python.org/3/library/functions.html#int), [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float)]]

### ax.models.model_utils.rejection_sample(gen_unconstrained: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int), ndarray, [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None)], ndarray], n: [int](https://docs.python.org/3/library/functions.html#int), d: [int](https://docs.python.org/3/library/functions.html#int), tunable_feature_indices: ndarray, linear_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None, deduplicate: [bool](https://docs.python.org/3/library/functions.html#bool) = False, max_draws: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, rounding_func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[ndarray], ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None, existing_points: ndarray | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, [int](https://docs.python.org/3/library/functions.html#int)]

Rejection sample in parameter space. Parameter space is typically
[0, 1] for all tunable parameters.

Models must implement a gen_unconstrained method in order to support
rejection sampling via this utility.

* **Parameters:**
  * **gen_unconstrained** – A callable that generates unconstrained points in
    the parameter space. This is typically the \_gen_unconstrained method
    of a RandomModel.
  * **n** – Number of samples to generate.
  * **d** – Dimensionality of the parameter space.
  * **tunable_feature_indices** – Indices of the tunable features in the
    parameter space.
  * **linear_constraints** – A tuple of (A, b). For k linear constraints on
    d-dimensional x, A is (k x d) and b is (k x 1) such that
    A x <= b.
  * **deduplicate** – If true, reject points that are duplicates of previously
    generated points. The points are deduplicated after applying the
    rounding function.
  * **max_draws** – Maximum number of attemped draws before giving up.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value during generation.
  * **rounding_func** – A function that rounds an optimization result
    appropriately (e.g., according to round-trip transformations).
  * **existing_points** – A set of previously generated points to use
    for deduplication. These should be provided in the parameter
    space model operates in.

### ax.models.model_utils.tunable_feature_indices(bounds: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]], fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → ndarray

Get the feature indices of tunable features.

* **Parameters:**
  * **bounds** – A list of (lower, upper) tuples for each column of X.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value during generation.
* **Returns:**
  The indices of tunable features.

### ax.models.model_utils.validate_bounds(bounds: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]], fixed_feature_indices: ndarray) → [None](https://docs.python.org/3/library/constants.html#None)

Ensure the requested space is [0,1]^d.

* **Parameters:**
  * **bounds** – A list of d (lower, upper) tuples for each column of X.
  * **fixed_feature_indices** – Indices of features which are fixed at a
    particular value.

### ax.models.types

### ax.models.winsorization_config module

### *class* ax.models.winsorization_config.WinsorizationConfig(lower_quantile_margin: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, upper_quantile_margin: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, lower_boundary: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, upper_boundary: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Dataclass for storing Winsorization configuration parameters

Attributes:
lower_quantile_margin: Winsorization will increase any metric value below this

> quantile to this quantile’s value.

upper_quantile_margin: Winsorization will decrease any metric value above this
: quantile to this quantile’s value. NOTE: this quantile will be inverted before
  any operations, e.g., a value of 0.2 will decrease values above the 80th
  percentile to the value of the 80th percentile.

lower_boundary: If this value is lesser than the metric value corresponding to
: `lower_quantile_margin`, set metric values below `lower_boundary` to
  `lower_boundary` and leave larger values unaffected.

upper_boundary: If this value is greater than the metric value corresponding to
: `upper_quantile_margin`, set metric values above `upper_boundary` to
  `upper_boundary` and leave smaller values unaffected.

#### lower_boundary *: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### lower_quantile_margin *: [float](https://docs.python.org/3/library/functions.html#float)* *= 0.0*

#### upper_boundary *: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### upper_quantile_margin *: [float](https://docs.python.org/3/library/functions.html#float)* *= 0.0*

## Discrete Models

### ax.models.discrete.eb_thompson module

### ax.models.discrete.full_factorial module

### *class* ax.models.discrete.full_factorial.FullFactorialGenerator(max_cardinality: [int](https://docs.python.org/3/library/functions.html#int) = 100, check_cardinality: [bool](https://docs.python.org/3/library/functions.html#bool) = True)

Bases: [`DiscreteModel`](#ax.models.discrete_base.DiscreteModel)

Generator for full factorial designs.

Generates arms for all possible combinations of parameter values,
each with weight 1.

The value of n supplied to gen will be ignored, as the number
of arms generated is determined by the list of parameter values.
To suppress this warning, use n = -1.

#### gen(n: [int](https://docs.python.org/3/library/functions.html#int), parameter_values: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]], objective_weights: ndarray | [None](https://docs.python.org/3/library/constants.html#None), outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, pending_observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]]] | [None](https://docs.python.org/3/library/constants.html#None) = None, model_gen_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]], [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]

Generate new candidates.

* **Parameters:**
  * **n** – Number of candidates to generate.
  * **parameter_values** – A list of possible values for each parameter.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value during generation.
  * **pending_observations** – A list of m lists of parameterizations
    (each parameterization is a list of parameter values of length d),
    each of length k_i, for each outcome i.
  * **model_gen_options** – A config dictionary that can contain
    model-specific options.
* **Returns:**
  2-element tuple containing
  - List of n generated points, where each point is represented
    by a list of parameter values.
  - List of weights for each of the n points.

### ax.models.discrete.thompson module

### *class* ax.models.discrete.thompson.ThompsonSampler(num_samples: [int](https://docs.python.org/3/library/functions.html#int) = 10000, min_weight: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, uniform_weights: [bool](https://docs.python.org/3/library/functions.html#bool) = False)

Bases: [`DiscreteModel`](#ax.models.discrete_base.DiscreteModel)

Generator for Thompson sampling.

The generator performs Thompson sampling on the data passed in via fit.
Arms are given weight proportional to the probability that they are
winners, according to Monte Carlo simulations.

#### fit(Xs: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]]], Ys: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], Yvars: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], parameter_values: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]], outcome_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [None](https://docs.python.org/3/library/constants.html#None)

Fit model to m outcomes.

* **Parameters:**
  * **Xs** – A list of m lists X of parameterizations (each parameterization
    is a list of parameter values of length d), each of length k_i,
    for each outcome.
  * **Ys** – The corresponding list of m lists Y, each of length k_i, for
    each outcome.
  * **Yvars** – The variances of each entry in Ys, same shape.
  * **parameter_values** – A list of possible values for each parameter.
  * **outcome_names** – A list of m outcome names.

#### gen(n: [int](https://docs.python.org/3/library/functions.html#int), parameter_values: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]], objective_weights: ndarray | [None](https://docs.python.org/3/library/constants.html#None), outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, pending_observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]]] | [None](https://docs.python.org/3/library/constants.html#None) = None, model_gen_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]], [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]

Generate new candidates.

* **Parameters:**
  * **n** – Number of candidates to generate.
  * **parameter_values** – A list of possible values for each parameter.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value during generation.
  * **pending_observations** – A list of m lists of parameterizations
    (each parameterization is a list of parameter values of length d),
    each of length k_i, for each outcome i.
  * **model_gen_options** – A config dictionary that can contain
    model-specific options.
* **Returns:**
  2-element tuple containing
  - List of n generated points, where each point is represented
    by a list of parameter values.
  - List of weights for each of the n points.

#### predict(X: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]]) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray]

Predict

* **Parameters:**
  **X** – List of the j parameterizations at which to make predictions.
* **Returns:**
  2-element tuple containing
  - (j x m) array of outcome predictions at X.
  - (j x m x m) array of predictive covariances at X.
    cov[j, m1, m2] is Cov[m1@j, m2@j].

## Random Models

### ax.models.random.base module

### *class* ax.models.random.base.RandomModel(deduplicate: [bool](https://docs.python.org/3/library/functions.html#bool) = True, seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, init_position: [int](https://docs.python.org/3/library/functions.html#int) = 0, generated_points: ndarray | [None](https://docs.python.org/3/library/constants.html#None) = None, fallback_to_sample_polytope: [bool](https://docs.python.org/3/library/functions.html#bool) = False)

Bases: [`Model`](#ax.models.base.Model)

This class specifies the basic skeleton for a random model.

As random generators do not make use of models, they do not implement
the fit or predict methods.

These models do not need data, or optimization configs.

To satisfy search space parameter constraints, these models can use
rejection sampling. To enable rejection sampling for a subclass, only
only \_gen_samples needs to be implemented, or alternatively,
\_gen_unconstrained/gen can be directly implemented.

#### deduplicate

If True (defaults to True), a single instantiation
of the model will not return the same point twice. This flag
is used in rejection sampling.

#### seed

An optional seed value for scrambling.

#### init_position

The initial state of the generator. This is the number
of samples to fast-forward before generating new samples.
Used to ensure that the re-loaded generator will continue generating
from the same sequence rather than starting from scratch.

#### generated_points

A set of previously generated points to use
for deduplication. These should be provided in the raw transformed
space the model operates in.

#### fallback_to_sample_polytope

If True, when rejection sampling fails,
we fall back to the HitAndRunPolytopeSampler.

#### gen(n: [int](https://docs.python.org/3/library/functions.html#int), bounds: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]], linear_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, model_gen_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None, rounding_func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[ndarray], ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray]

Generate new candidates.

* **Parameters:**
  * **n** – Number of candidates to generate.
  * **bounds** – A list of (lower, upper) tuples for each column of X.
    Defined on [0, 1]^d.
  * **linear_constraints** – A tuple of (A, b). For k linear constraints on
    d-dimensional x, A is (k x d) and b is (k x 1) such that
    A x <= b.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value during generation.
  * **model_gen_options** – A config dictionary that is passed along to the
    model.
  * **rounding_func** – A function that rounds an optimization result
    appropriately (e.g., according to round-trip transformations).
* **Returns:**
  2-element tuple containing
  - (n x d) array of generated points.
  - Uniform weights, an n-array of ones for each point.

### ax.models.random.uniform module

### *class* ax.models.random.uniform.UniformGenerator(deduplicate: [bool](https://docs.python.org/3/library/functions.html#bool) = True, seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, init_position: [int](https://docs.python.org/3/library/functions.html#int) = 0, generated_points: ndarray | [None](https://docs.python.org/3/library/constants.html#None) = None, fallback_to_sample_polytope: [bool](https://docs.python.org/3/library/functions.html#bool) = False)

Bases: [`RandomModel`](#ax.models.random.base.RandomModel)

This class specifies a uniform random generation algorithm.

As a uniform generator does not make use of a model, it does not implement
the fit or predict methods.

See base RandomModel for a description of model attributes.

### ax.models.random.sobol module

### *class* ax.models.random.sobol.SobolGenerator(deduplicate: [bool](https://docs.python.org/3/library/functions.html#bool) = True, seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, init_position: [int](https://docs.python.org/3/library/functions.html#int) = 0, scramble: [bool](https://docs.python.org/3/library/functions.html#bool) = True, generated_points: ndarray | [None](https://docs.python.org/3/library/constants.html#None) = None, fallback_to_sample_polytope: [bool](https://docs.python.org/3/library/functions.html#bool) = False)

Bases: [`RandomModel`](#ax.models.random.base.RandomModel)

This class specifies the generation algorithm for a Sobol generator.

As Sobol does not make use of a model, it does not implement
the fit or predict methods.

#### scramble

If True, permutes the parameter values among
the elements of the Sobol sequence. Default is True.

### See base \`RandomModel\` for a description of remaining attributes.

#### *property* engine *: SobolEngine | [None](https://docs.python.org/3/library/constants.html#None)*

Return a singleton SobolEngine.

#### gen(n: [int](https://docs.python.org/3/library/functions.html#int), bounds: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]], linear_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, model_gen_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None, rounding_func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[ndarray], ndarray] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray]

Generate new candidates.

* **Parameters:**
  * **n** – Number of candidates to generate.
  * **bounds** – A list of (lower, upper) tuples for each column of X.
  * **linear_constraints** – A tuple of (A, b). For k linear constraints on
    d-dimensional x, A is (k x d) and b is (k x 1) such that
    A x <= b.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value during generation.
  * **rounding_func** – A function that rounds an optimization result
    appropriately (e.g., according to round-trip transformations).
* **Returns:**
  2-element tuple containing
  - (n x d) array of generated points.
  - Uniform weights, an n-array of ones for each point.

#### init_engine(n_tunable_features: [int](https://docs.python.org/3/library/functions.html#int)) → SobolEngine

Initialize singleton SobolEngine, only on gen.

* **Parameters:**
  **n_tunable_features** – The number of features which can be
  searched over.
* **Returns:**
  SobolEngine, which can generate Sobol points.

## Torch Models & Utilities

### ax.models.torch.botorch module

### *class* ax.models.torch.botorch.BotorchModel(model_constructor: ~collections.abc.Callable[[list[~torch.Tensor], list[~torch.Tensor], list[~torch.Tensor], list[int], list[int], list[str], dict[str, ~torch.Tensor] | None, ~typing.Any], ~botorch.models.model.Model] = <function get_and_fit_model>, model_predictor: ~collections.abc.Callable[[~botorch.models.model.Model, ~torch.Tensor, bool], tuple[~torch.Tensor, ~torch.Tensor]] = <function predict_from_model>, acqf_constructor: ~ax.models.torch.botorch_defaults.TAcqfConstructor = <function get_qLogNEI>, acqf_optimizer: ~collections.abc.Callable[[~botorch.acquisition.acquisition.AcquisitionFunction, ~torch.Tensor, int, list[tuple[~torch.Tensor, ~torch.Tensor, float]] | None, list[tuple[~torch.Tensor, ~torch.Tensor, float]] | None, dict[int, float] | None, ~collections.abc.Callable[[~torch.Tensor], ~torch.Tensor] | None, ~typing.Any], tuple[~torch.Tensor, ~torch.Tensor]] = <function scipy_optimizer>, best_point_recommender: ~collections.abc.Callable[[~ax.models.torch_base.TorchModel, list[tuple[float, float]], ~torch.Tensor, tuple[~torch.Tensor, ~torch.Tensor] | None, tuple[~torch.Tensor, ~torch.Tensor] | None, dict[int, float] | None, dict[str, int | float | str | ~botorch.acquisition.acquisition.AcquisitionFunction | list[str] | dict[int, ~typing.Any] | dict[str, ~typing.Any] | ~ax.core.optimization_config.OptimizationConfig | ~ax.models.winsorization_config.WinsorizationConfig | None] | None, dict[int, float] | None], ~torch.Tensor | None] = <function recommend_best_observed_point>, refit_on_cv: bool = False, warm_start_refitting: bool = True, use_input_warping: bool = False, use_loocv_pseudo_likelihood: bool = False, prior: dict[str, ~typing.Any] | None = None, \*\*kwargs: ~typing.Any)

Bases: [`TorchModel`](#ax.models.torch_base.TorchModel)

Customizable botorch model.

By default, this uses a noisy Log Expected Improvement (qLogNEI) acquisition
function on top of a model made up of separate GPs, one for each outcome. This
behavior can be modified by providing custom implementations of the following
components:

- a model_constructor that instantiates and fits a model on data
- a model_predictor that predicts outcomes using the fitted model
- a acqf_constructor that creates an acquisition function from a fitted model
- a acqf_optimizer that optimizes the acquisition function
- a best_point_recommender that recommends a current “best” point (i.e.,
  : what the model recommends if the learning process ended now)

* **Parameters:**
  * **model_constructor** – A callable that instantiates and fits a model on data,
    with signature as described below.
  * **model_predictor** – A callable that predicts using the fitted model, with
    signature as described below.
  * **acqf_constructor** – A callable that creates an acquisition function from a
    fitted model, with signature as described below.
  * **acqf_optimizer** – A callable that optimizes the acquisition function, with
    signature as described below.
  * **best_point_recommender** – A callable that recommends the best point, with
    signature as described below.
  * **refit_on_cv** – If True, refit the model for each fold when performing
    cross-validation.
  * **warm_start_refitting** – If True, start model refitting from previous
    model parameters in order to speed up the fitting process.
  * **prior** – 

    An optional dictionary that contains the specification of GP model prior.
    Currently, the keys include:
    - covar_module_prior: prior on covariance matrix e.g.
    > {“lengthscale_prior”: GammaPrior(3.0, 6.0)}.
    - type: type of prior on task covariance matrix e.g.\`LKJCovariancePrior\`.
    - sd_prior: A scalar prior over nonnegative numbers, which is used for the
      : default LKJCovariancePrior task_covar_prior.
    - eta: The eta parameter on the default LKJ task_covar_prior.

Call signatures:

```default
model_constructor(
    Xs,
    Ys,
    Yvars,
    task_features,
    fidelity_features,
    metric_names,
    state_dict,
    **kwargs,
) -> model
```

Here Xs, Ys, Yvars are lists of tensors (one element per outcome),
task_features identifies columns of Xs that should be modeled as a task,
fidelity_features is a list of ints that specify the positions of fidelity
parameters in ‘Xs’, metric_names provides the names of each Y in Ys,
state_dict is a pytorch module state dict, and model is a BoTorch Model.
Optional kwargs are being passed through from the BotorchModel constructor.
This callable is assumed to return a fitted BoTorch model that has the same
dtype and lives on the same device as the input tensors.

```default
model_predictor(model, X) -> [mean, cov]
```

Here model is a fitted botorch model, X is a tensor of candidate points,
and mean and cov are the posterior mean and covariance, respectively.

```default
acqf_constructor(
    model,
    objective_weights,
    outcome_constraints,
    X_observed,
    X_pending,
    **kwargs,
) -> acq_function
```

Here model is a botorch Model, objective_weights is a tensor of weights
for the model outputs, outcome_constraints is a tuple of tensors describing
the (linear) outcome constraints, X_observed are previously observed points,
and X_pending are points whose evaluation is pending. acq_function is a
BoTorch acquisition function crafted from these inputs. For additional
details on the arguments, see get_qLogNEI.

```default
acqf_optimizer(
    acq_function,
    bounds,
    n,
    inequality_constraints,
    equality_constraints,
    fixed_features,
    rounding_func,
    **kwargs,
) -> candidates
```

Here acq_function is a BoTorch AcquisitionFunction, bounds is a tensor
containing bounds on the parameters, n is the number of candidates to be
generated, inequality_constraints are inequality constraints on parameter
values, fixed_features specifies features that should be fixed during
generation, and rounding_func is a callback that rounds an optimization
result appropriately. candidates is a tensor of generated candidates.
For additional details on the arguments, see scipy_optimizer.

```default
best_point_recommender(
    model,
    bounds,
    objective_weights,
    outcome_constraints,
    linear_constraints,
    fixed_features,
    model_gen_options,
    target_fidelities,
) -> candidates
```

Here model is a TorchModel, bounds is a list of tuples containing bounds
on the parameters, objective_weights is a tensor of weights for the model outputs,
outcome_constraints is a tuple of tensors describing the (linear) outcome
constraints, linear_constraints is a tuple of tensors describing constraints
on the design, fixed_features specifies features that should be fixed during
generation, model_gen_options is a config dictionary that can contain
model-specific options, and target_fidelities is a map from fidelity feature
column indices to their respective target fidelities, used for multi-fidelity
optimization problems. % TODO: refer to an example.

#### Xs *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor]*

#### Ys *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor]*

#### Yvars *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor]*

#### best_point(search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig)) → Tensor | [None](https://docs.python.org/3/library/constants.html#None)

Identify the current best point, satisfying the constraints in the same
format as to gen.

Return None if no such point can be identified.

* **Parameters:**
  * **search_space_digest** – A SearchSpaceDigest object containing metadata
    about the search space (e.g. bounds, parameter types).
  * **torch_opt_config** – A TorchOptConfig object containing optimization
    arguments (e.g., objective weights, constraints).
* **Returns:**
  d-tensor of the best point.

#### cross_validate(datasets: [list](https://docs.python.org/3/library/stdtypes.html#list)[SupervisedDataset], X_test: Tensor, use_posterior_predictive: [bool](https://docs.python.org/3/library/functions.html#bool) = False, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

Do cross validation with the given training and test sets.

Training set is given in the same format as to fit. Test set is given
in the same format as to predict.

* **Parameters:**
  * **datasets** – A list of `SupervisedDataset` containers, each
    corresponding to the data of one metric (outcome).
  * **X_test** – (j x d) tensor of the j points at which to make predictions.
  * **search_space_digest** – A SearchSpaceDigest object containing
    metadata on the features in X.
  * **use_posterior_predictive** – A boolean indicating if the predictions
    should be from the posterior predictive (i.e. including
    observation noise).
* **Returns:**
  2-element tuple containing
  - (j x m) tensor of outcome predictions at X.
  - (j x m x m) tensor of predictive covariances at X.
    cov[j, m1, m2] is Cov[m1@j, m2@j].

#### feature_importances() → ndarray

#### fit(datasets: [list](https://docs.python.org/3/library/stdtypes.html#list)[SupervisedDataset], search_space_digest: SearchSpaceDigest, candidate_metadata: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)]] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)

Fit model to m outcomes.

* **Parameters:**
  * **datasets** – A list of `SupervisedDataset` containers, each
    corresponding to the data of one metric (outcome).
  * **search_space_digest** – A `SearchSpaceDigest` object containing
    metadata on the features in the datasets.
  * **candidate_metadata** – Model-produced metadata for candidates, in
    the order corresponding to the Xs.

#### gen(n: [int](https://docs.python.org/3/library/functions.html#int), search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig)) → [TorchGenResults](#ax.models.torch_base.TorchGenResults)

Generate new candidates.

* **Parameters:**
  * **n** – Number of candidates to generate.
  * **search_space_digest** – A SearchSpaceDigest object containing metadata
    about the search space (e.g. bounds, parameter types).
  * **torch_opt_config** – A TorchOptConfig object containing optimization
    arguments (e.g., objective weights, constraints).
* **Returns:**
  A TorchGenResult container.

#### *property* model *: Model*

#### predict(X: Tensor) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

Predict

* **Parameters:**
  **X** – (j x d) tensor of the j points at which to make predictions.
* **Returns:**
  2-element tuple containing
  - (j x m) tensor of outcome predictions at X.
  - (j x m x m) tensor of predictive covariances at X.
    cov[j, m1, m2] is Cov[m1@j, m2@j].

#### *property* search_space_digest *: SearchSpaceDigest*

### ax.models.torch.botorch.get_feature_importances_from_botorch_model(model: Model | ModuleList | [None](https://docs.python.org/3/library/constants.html#None)) → ndarray

Get feature importances from a list of BoTorch models.

* **Parameters:**
  **models** – BoTorch model to get feature importances from.
* **Returns:**
  The feature importances as a numpy array where each row sums to 1.

### ax.models.torch.botorch.get_rounding_func(rounding_func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[Tensor], Tensor] | [None](https://docs.python.org/3/library/constants.html#None)) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[Tensor], Tensor] | [None](https://docs.python.org/3/library/constants.html#None)

### ax.models.torch.botorch_defaults module

### *class* ax.models.torch.botorch_defaults.TAcqfConstructor(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

### ax.models.torch.botorch_defaults.get_NEI() → [None](https://docs.python.org/3/library/constants.html#None)

TAcqfConstructor instantiating qNEI. See docstring of get_qEI for details.

### ax.models.torch.botorch_defaults.get_acqf(acquisition_function_name: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[], [None](https://docs.python.org/3/library/constants.html#None)]], [TAcqfConstructor](#ax.models.torch.botorch_defaults.TAcqfConstructor)]

Returns a decorator whose wrapper function instantiates an acquisition function.

NOTE: This is a decorator factory instead of a simple factory as serialization
of Botorch model kwargs requires callables to be have module-level paths, and
closures created by a simple factory do not have such paths. We solve this by
wrapping “empty” module-level functions with this decorator, we ensure that they
are serialized correctly, in addition to reducing code duplication.

### Example

```pycon
>>> @get_acqf("qEI")
... def get_qEI() -> None:
...     pass
>>> acqf = get_qEI(
...     model=model,
...     objective_weights=objective_weights,
...     outcome_constraints=outcome_constraints,
...     X_observed=X_observed,
...     X_pending=X_pending,
...     **kwargs,
... )
>>> type(acqf)
... botorch.acquisition.monte_carlo.qExpectedImprovement
```

* **Parameters:**
  **acquisition_function_name** – The name of the acquisition function to be
  instantiated by the returned function.
* **Returns:**
  A decorator whose wrapper function is a TAcqfConstructor, i.e. it requires a
  model, objective_weights, and optional outcome_constraints, X_observed,
  and X_pending as inputs, as well as kwargs, and returns an
  AcquisitionFunction instance that corresponds to acquisition_function_name.

### ax.models.torch.botorch_defaults.get_and_fit_model(Xs: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], Ys: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], Yvars: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], task_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], fidelity_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], metric_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], state_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, refit_model: [bool](https://docs.python.org/3/library/functions.html#bool) = True, use_input_warping: [bool](https://docs.python.org/3/library/functions.html#bool) = False, use_loocv_pseudo_likelihood: [bool](https://docs.python.org/3/library/functions.html#bool) = False, prior: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*, multitask_gp_ranks: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Prior | [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → GPyTorchModel

Instantiates and fits a botorch GPyTorchModel using the given data.
N.B. Currently, the logic for choosing ModelListGP vs other models is handled
using if-else statements in lines 96-137. In the future, this logic should be
taken care of by modular botorch.

* **Parameters:**
  * **Xs** – List of X data, one tensor per outcome.
  * **Ys** – List of Y data, one tensor per outcome.
  * **Yvars** – List of observed variance of Ys.
  * **task_features** – List of columns of X that are tasks.
  * **fidelity_features** – List of columns of X that are fidelity parameters.
  * **metric_names** – Names of each outcome Y in Ys.
  * **state_dict** – If provided, will set model parameters to this state
    dictionary. Otherwise, will fit the model.
  * **refit_model** – Flag for refitting model.
  * **prior** – 

    Optional[Dict]. A dictionary that contains the specification of
    GP model prior. Currently, the keys include:
    - covar_module_prior: prior on covariance matrix e.g.
    > {“lengthscale_prior”: GammaPrior(3.0, 6.0)}.
    - type: type of prior on task covariance matrix e.g.\`LKJCovariancePrior\`.
    - sd_prior: A scalar prior over nonnegative numbers, which is used for the
      : default LKJCovariancePrior task_covar_prior.
    - eta: The eta parameter on the default LKJ task_covar_prior.
  * **kwargs** – Passed to \_get_model.
* **Returns:**
  A fitted GPyTorchModel.

### ax.models.torch.botorch_defaults.get_qEI() → [None](https://docs.python.org/3/library/constants.html#None)

A TAcqfConstructor to instantiate a qEI acquisition function. The function body
is filled in by the decorator function get_acqf to simultaneously reduce code
duplication and allow serialization in Ax. TODO: Deprecate with legacy Ax model.

### ax.models.torch.botorch_defaults.get_qLogEI() → [None](https://docs.python.org/3/library/constants.html#None)

TAcqfConstructor instantiating qLogEI. See docstring of get_qEI for details.

### ax.models.torch.botorch_defaults.get_qLogNEI() → [None](https://docs.python.org/3/library/constants.html#None)

TAcqfConstructor instantiating qLogNEI. See docstring of get_qEI for details.

### ax.models.torch.botorch_defaults.get_warping_transform(d: [int](https://docs.python.org/3/library/functions.html#int), batch_shape: Size | [None](https://docs.python.org/3/library/constants.html#None) = None, task_feature: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None) → Warp

Construct input warping transform.

* **Parameters:**
  * **d** – The dimension of the input, including task features
  * **batch_shape** – The batch_shape of the model
  * **task_feature** – The index of the task feature
* **Returns:**
  The input warping transform.

### ax.models.torch.botorch_defaults.recommend_best_observed_point(model: [TorchModel](#ax.models.torch_base.TorchModel), bounds: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]], objective_weights: Tensor, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, linear_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, model_gen_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None, target_fidelities: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → Tensor | [None](https://docs.python.org/3/library/constants.html#None)

A wrapper around ax.models.model_utils.best_observed_point for TorchModel
that recommends a best point from previously observed points using either a
“max_utility” or “feasible_threshold” strategy.

* **Parameters:**
  * **model** – A TorchModel.
  * **bounds** – A list of (lower, upper) tuples for each column of X.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b.
  * **linear_constraints** – A tuple of (A, b). For k linear constraints on
    d-dimensional x, A is (k x d) and b is (k x 1) such that
    A x <= b.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value in the best point.
  * **model_gen_options** – A config dictionary that can contain
    model-specific options. See TorchOptConfig for details.
  * **target_fidelities** – A map {feature_index: value} of fidelity feature
    column indices to their respective target fidelities. Used for
    multi-fidelity optimization.
* **Returns:**
  A d-array of the best point, or None if no feasible point was observed.

### ax.models.torch.botorch_defaults.recommend_best_out_of_sample_point(model: [TorchModel](#ax.models.torch_base.TorchModel), bounds: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]], objective_weights: Tensor, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, linear_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, model_gen_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None, target_fidelities: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → Tensor | [None](https://docs.python.org/3/library/constants.html#None)

Identify the current best point by optimizing the posterior mean of the model.
This is “out-of-sample” because it considers un-observed designs as well.

Return None if no such point can be identified.

* **Parameters:**
  * **model** – A TorchModel.
  * **bounds** – A list of (lower, upper) tuples for each column of X.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b.
  * **linear_constraints** – A tuple of (A, b). For k linear constraints on
    d-dimensional x, A is (k x d) and b is (k x 1) such that
    A x <= b.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value in the best point.
  * **model_gen_options** – A config dictionary that can contain
    model-specific options. See TorchOptConfig for details.
  * **target_fidelities** – A map {feature_index: value} of fidelity feature
    column indices to their respective target fidelities. Used for
    multi-fidelity optimization.
* **Returns:**
  A d-array of the best point, or None if no feasible point exists.

### ax.models.torch.botorch_defaults.scipy_optimizer(acq_function: AcquisitionFunction, bounds: Tensor, n: [int](https://docs.python.org/3/library/functions.html#int), inequality_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor, [float](https://docs.python.org/3/library/functions.html#float)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, equality_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor, [float](https://docs.python.org/3/library/functions.html#float)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, rounding_func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[Tensor], Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*, num_restarts: [int](https://docs.python.org/3/library/functions.html#int) = 20, raw_samples: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, joint_optimization: [bool](https://docs.python.org/3/library/functions.html#bool) = False, options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int) | [str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

Optimizer using scipy’s minimize module on a numpy-adpator.

* **Parameters:**
  * **acq_function** – A botorch AcquisitionFunction.
  * **bounds** – A 2 x d-dim tensor, where bounds[0] (bounds[1]) are the
    lower (upper) bounds of the feasible hyperrectangle.
  * **n** – The number of candidates to generate.
  * **constraints** (*equality*) – A list of tuples (indices, coefficients, rhs),
    with each tuple encoding an inequality constraint of the form
    sum_i (X[indices[i]] \* coefficients[i]) >= rhs
  * **constraints** – A list of tuples (indices, coefficients, rhs),
    with each tuple encoding an equality constraint of the form
    sum_i (X[indices[i]] \* coefficients[i]) == rhs
  * **fixed_features** – A map {feature_index: value} for features that should
    be fixed to a particular value during generation.
  * **rounding_func** – A function that rounds an optimization result
    appropriately (i.e., according to round-trip transformations).
* **Returns:**
  2-element tuple containing
  - A n x d-dim tensor of generated candidates.
  - In the case of joint optimization, a scalar tensor containing
    the joint acquisition value of the n points. In the case of
    sequential optimization, a n-dim tensor of conditional acquisition
    values, where i-th element is the expected acquisition value
    conditional on having observed candidates 0,1,…,i-1.

### ax.models.torch.botorch_kg module

### *class* ax.models.torch.botorch_kg.KnowledgeGradient(cost_intercept: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, linear_truncated: [bool](https://docs.python.org/3/library/functions.html#bool) = True, use_input_warping: [bool](https://docs.python.org/3/library/functions.html#bool) = False, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

Bases: [`BotorchModel`](#ax.models.torch.botorch.BotorchModel)

The Knowledge Gradient with one shot optimization.

* **Parameters:**
  * **cost_intercept** – The cost intercept for the affine cost of the form
    cost_intercept + n, where n is the number of generated points.
    Only used for multi-fidelity optimzation (i.e., if fidelity_features
    are present).
  * **linear_truncated** – If False, use an alternate downsampling + exponential
    decay Kernel instead of the default LinearTruncatedFidelityKernel
    (only relevant for multi-fidelity optimization).
  * **kwargs** – Model-specific kwargs.

#### gen(n: [int](https://docs.python.org/3/library/functions.html#int), search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig)) → [TorchGenResults](#ax.models.torch_base.TorchGenResults)

Generate new candidates.

* **Parameters:**
  * **n** – Number of candidates to generate.
  * **search_space_digest** – A SearchSpaceDigest object containing metadata
    about the search space (e.g. bounds, parameter types).
  * **torch_opt_config** – A TorchOptConfig object containing optimization
    arguments (e.g., objective weights, constraints).
* **Returns:**
  A TorchGenResults container, containing
  - (n x d) tensor of generated points.
  - n-tensor of weights for each point.
  - Dictionary of model-specific metadata for the given
    : generation candidates.

### ax.models.torch.botorch_moo module

### *class* ax.models.torch.botorch_moo.MultiObjectiveBotorchModel(model_constructor: ~collections.abc.Callable[[list[~torch.Tensor], list[~torch.Tensor], list[~torch.Tensor], list[int], list[int], list[str], dict[str, ~torch.Tensor] | None, ~typing.Any], ~botorch.models.model.Model] = <function get_and_fit_model>, model_predictor: ~collections.abc.Callable[[~botorch.models.model.Model, ~torch.Tensor, bool], tuple[~torch.Tensor, ~torch.Tensor]] = <function predict_from_model>, acqf_constructor: ~ax.models.torch.botorch_defaults.TAcqfConstructor = <function get_qLogNEHVI>, acqf_optimizer: ~collections.abc.Callable[[~botorch.acquisition.acquisition.AcquisitionFunction, ~torch.Tensor, int, list[tuple[~torch.Tensor, ~torch.Tensor, float]] | None, list[tuple[~torch.Tensor, ~torch.Tensor, float]] | None, dict[int, float] | None, ~collections.abc.Callable[[~torch.Tensor], ~torch.Tensor] | None, ~typing.Any], tuple[~torch.Tensor, ~torch.Tensor]] = <function scipy_optimizer>, best_point_recommender: ~collections.abc.Callable[[~ax.models.torch_base.TorchModel, list[tuple[float, float]], ~torch.Tensor, tuple[~torch.Tensor, ~torch.Tensor] | None, tuple[~torch.Tensor, ~torch.Tensor] | None, dict[int, float] | None, dict[str, int | float | str | ~botorch.acquisition.acquisition.AcquisitionFunction | list[str] | dict[int, ~typing.Any] | dict[str, ~typing.Any] | ~ax.core.optimization_config.OptimizationConfig | ~ax.models.winsorization_config.WinsorizationConfig | None] | None, dict[int, float] | None], ~torch.Tensor | None] = <function recommend_best_observed_point>, frontier_evaluator: ~collections.abc.Callable[[~ax.models.torch_base.TorchModel, ~torch.Tensor, ~torch.Tensor | None, ~torch.Tensor | None, ~torch.Tensor | None, ~torch.Tensor | None, tuple[~torch.Tensor, ~torch.Tensor] | None], tuple[~torch.Tensor, ~torch.Tensor, ~torch.Tensor]] = <function pareto_frontier_evaluator>, refit_on_cv: bool = False, warm_start_refitting: bool = False, use_input_warping: bool = False, use_loocv_pseudo_likelihood: bool = False, prior: dict[str, ~typing.Any] | None = None, \*\*kwargs: ~typing.Any)

Bases: [`BotorchModel`](#ax.models.torch.botorch.BotorchModel)

Customizable multi-objective model.

By default, this uses an Expected Hypervolume Improvment function to find the
pareto frontier of a function with multiple outcomes. This behavior
can be modified by providing custom implementations of the following
components:

- a model_constructor that instantiates and fits a model on data
- a model_predictor that predicts outcomes using the fitted model
- a acqf_constructor that creates an acquisition function from a fitted model
- a acqf_optimizer that optimizes the acquisition function

* **Parameters:**
  * **model_constructor** – A callable that instantiates and fits a model on data,
    with signature as described below.
  * **model_predictor** – A callable that predicts using the fitted model, with
    signature as described below.
  * **acqf_constructor** – A callable that creates an acquisition function from a
    fitted model, with signature as described below.
  * **acqf_optimizer** – A callable that optimizes an acquisition
    function, with signature as described below.

Call signatures:

```default
model_constructor(
    Xs,
    Ys,
    Yvars,
    task_features,
    fidelity_features,
    metric_names,
    state_dict,
    **kwargs,
) -> model
```

Here Xs, Ys, Yvars are lists of tensors (one element per outcome),
task_features identifies columns of Xs that should be modeled as a task,
fidelity_features is a list of ints that specify the positions of fidelity
parameters in ‘Xs’, metric_names provides the names of each Y in Ys,
state_dict is a pytorch module state dict, and model is a BoTorch Model.
Optional kwargs are being passed through from the BotorchModel constructor.
This callable is assumed to return a fitted BoTorch model that has the same
dtype and lives on the same device as the input tensors.

```default
model_predictor(model, X) -> [mean, cov]
```

Here model is a fitted botorch model, X is a tensor of candidate points,
and mean and cov are the posterior mean and covariance, respectively.

```default
acqf_constructor(
    model,
    objective_weights,
    outcome_constraints,
    X_observed,
    X_pending,
    **kwargs,
) -> acq_function
```

Here model is a botorch Model, objective_weights is a tensor of weights
for the model outputs, outcome_constraints is a tuple of tensors describing
the (linear) outcome constraints, X_observed are previously observed points,
and X_pending are points whose evaluation is pending. acq_function is a
BoTorch acquisition function crafted from these inputs. For additional
details on the arguments, see get_qLogNEHVI.

```default
acqf_optimizer(
    acq_function,
    bounds,
    n,
    inequality_constraints,
    fixed_features,
    rounding_func,
    **kwargs,
) -> candidates
```

Here acq_function is a BoTorch AcquisitionFunction, bounds is a tensor
containing bounds on the parameters, n is the number of candidates to be
generated, inequality_constraints are inequality constraints on parameter
values, fixed_features specifies features that should be fixed during
generation, and rounding_func is a callback that rounds an optimization
result appropriately. candidates is a tensor of generated candidates.
For additional details on the arguments, see scipy_optimizer.

```default
frontier_evaluator(
    model,
    objective_weights,
    objective_thresholds,
    X,
    Y,
    Yvar,
    outcome_constraints,
)
```

Here model is a botorch Model, objective_thresholds is used in hypervolume
evaluations, objective_weights is a tensor of weights applied to the  objectives
(sign represents direction), X, Y, Yvar are tensors, outcome_constraints is
a tuple of tensors describing the (linear) outcome constraints.

#### Xs *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor]*

#### Ys *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor]*

#### Yvars *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor]*

#### gen(n: [int](https://docs.python.org/3/library/functions.html#int), search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig)) → [TorchGenResults](#ax.models.torch_base.TorchGenResults)

Generate new candidates.

* **Parameters:**
  * **n** – Number of candidates to generate.
  * **search_space_digest** – A SearchSpaceDigest object containing metadata
    about the search space (e.g. bounds, parameter types).
  * **torch_opt_config** – A TorchOptConfig object containing optimization
    arguments (e.g., objective weights, constraints).
* **Returns:**
  A TorchGenResult container.

### ax.models.torch.botorch_moo_defaults module

References

### ax.models.torch.botorch_moo_defaults.get_EHVI(model: Model, objective_weights: Tensor, objective_thresholds: Tensor, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, X_observed: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, X_pending: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, \*, mc_samples: [int](https://docs.python.org/3/library/functions.html#int) = 128, alpha: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None) → qExpectedHypervolumeImprovement

Instantiates a qExpectedHyperVolumeImprovement acquisition function.

* **Parameters:**
  * **model** – The underlying model which the acqusition function uses
    to estimate acquisition values of candidates.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **objective_thresholds** – A tensor containing thresholds forming a reference point
    from which to calculate pareto frontier hypervolume. Points that do not
    dominate the objective_thresholds contribute nothing to hypervolume.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b. (Not used by single task models)
  * **X_observed** – A tensor containing points observed for all objective
    outcomes and outcomes that appear in the outcome constraints (if
    there are any).
  * **X_pending** – A tensor containing points whose evaluation is pending (i.e.
    that have been submitted for evaluation) present for all objective
    outcomes and outcomes that appear in the outcome constraints (if
    there are any).
  * **mc_samples** – The number of MC samples to use (default: 512).
  * **alpha** – The hyperparameter controlling the approximate non-dominated
    partitioning. The default value of 0.0 means an exact partitioning
    is used. As the number of objectives m increases, consider increasing
    this parameter in order to limit computational complexity.
  * **seed** – The random seed for generating random starting points for optimization.
* **Returns:**
  The instantiated acquisition function.
* **Return type:**
  qExpectedHypervolumeImprovement

### ax.models.torch.botorch_moo_defaults.get_NEHVI(model: Model, objective_weights: Tensor, objective_thresholds: Tensor, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, X_observed: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, X_pending: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, \*, prune_baseline: [bool](https://docs.python.org/3/library/functions.html#bool) = True, mc_samples: [int](https://docs.python.org/3/library/functions.html#int) = 128, alpha: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, marginalize_dim: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, cache_root: [bool](https://docs.python.org/3/library/functions.html#bool) = True, seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None) → qNoisyExpectedHypervolumeImprovement

Instantiates a qNoisyExpectedHyperVolumeImprovement acquisition function.

* **Parameters:**
  * **model** – The underlying model which the acqusition function uses
    to estimate acquisition values of candidates.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b. (Not used by single task models)
  * **X_observed** – A tensor containing points observed for all objective
    outcomes and outcomes that appear in the outcome constraints (if
    there are any).
  * **X_pending** – A tensor containing points whose evaluation is pending (i.e.
    that have been submitted for evaluation) present for all objective
    outcomes and outcomes that appear in the outcome constraints (if
    there are any).
  * **prune_baseline** – If True, prune the baseline points for NEI (default: True).
  * **mc_samples** – The number of MC samples to use (default: 512).
  * **alpha** – The hyperparameter controlling the approximate non-dominated
    partitioning. The default value of 0.0 means an exact partitioning
    is used. As the number of objectives m increases, consider increasing
    this parameter in order to limit computational complexity (default: None).
  * **marginalize_dim** – The dimension along which to marginalize over, used for fully
    Bayesian models (default: None).
  * **cache_root** – If True, cache the root of the covariance matrix (default: True).
  * **seed** – The random seed for generating random starting points for optimization (
    default: None).
* **Returns:**
  The instantiated acquisition function.
* **Return type:**
  qNoisyExpectedHyperVolumeImprovement

### ax.models.torch.botorch_moo_defaults.get_qLogEHVI(model: Model, objective_weights: Tensor, objective_thresholds: Tensor, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, X_observed: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, X_pending: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, \*, mc_samples: [int](https://docs.python.org/3/library/functions.html#int) = 128, alpha: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None) → qLogExpectedHypervolumeImprovement

Instantiates a qLogExpectedHyperVolumeImprovement acquisition function.

* **Parameters:**
  * **model** – The underlying model which the acqusition function uses
    to estimate acquisition values of candidates.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **objective_thresholds** – A tensor containing thresholds forming a reference point
    from which to calculate pareto frontier hypervolume. Points that do not
    dominate the objective_thresholds contribute nothing to hypervolume.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b. (Not used by single task models)
  * **X_observed** – A tensor containing points observed for all objective
    outcomes and outcomes that appear in the outcome constraints (if
    there are any).
  * **X_pending** – A tensor containing points whose evaluation is pending (i.e.
    that have been submitted for evaluation) present for all objective
    outcomes and outcomes that appear in the outcome constraints (if
    there are any).
  * **mc_samples** – The number of MC samples to use (default: 512).
  * **alpha** – The hyperparameter controlling the approximate non-dominated
    partitioning. The default value of 0.0 means an exact partitioning
    is used. As the number of objectives m increases, consider increasing
    this parameter in order to limit computational complexity.
  * **seed** – The random seed for generating random starting points for optimization.
* **Returns:**
  The instantiated acquisition function.
* **Return type:**
  qLogExpectedHypervolumeImprovement

### ax.models.torch.botorch_moo_defaults.get_qLogNEHVI(model: Model, objective_weights: Tensor, objective_thresholds: Tensor, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, X_observed: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, X_pending: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, \*, prune_baseline: [bool](https://docs.python.org/3/library/functions.html#bool) = True, mc_samples: [int](https://docs.python.org/3/library/functions.html#int) = 128, alpha: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, marginalize_dim: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, cache_root: [bool](https://docs.python.org/3/library/functions.html#bool) = True, seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None) → qLogNoisyExpectedHypervolumeImprovement

Instantiates a qLogNoisyExpectedHyperVolumeImprovement acquisition function.

* **Parameters:**
  * **model** – The underlying model which the acqusition function uses
    to estimate acquisition values of candidates.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b. (Not used by single task models)
  * **X_observed** – A tensor containing points observed for all objective
    outcomes and outcomes that appear in the outcome constraints (if
    there are any).
  * **X_pending** – A tensor containing points whose evaluation is pending (i.e.
    that have been submitted for evaluation) present for all objective
    outcomes and outcomes that appear in the outcome constraints (if
    there are any).
  * **prune_baseline** – If True, prune the baseline points for NEI (default: True).
  * **mc_samples** – The number of MC samples to use (default: 512).
  * **alpha** – The hyperparameter controlling the approximate non-dominated
    partitioning. The default value of 0.0 means an exact partitioning
    is used. As the number of objectives m increases, consider increasing
    this parameter in order to limit computational complexity (default: None).
  * **marginalize_dim** – The dimension along which to marginalize over, used for fully
    Bayesian models (default: None).
  * **cache_root** – If True, cache the root of the covariance matrix (default: True).
  * **seed** – The random seed for generating random starting points for optimization (
    default: None).
* **Returns:**
  The instantiated acquisition function.
* **Return type:**
  qLogNoisyExpectedHyperVolumeImprovement

### ax.models.torch.botorch_moo_defaults.get_weighted_mc_objective_and_objective_thresholds(objective_weights: Tensor, objective_thresholds: Tensor) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[WeightedMCMultiOutputObjective, Tensor]

Construct weighted objective and apply the weights to objective thresholds.

* **Parameters:**
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **objective_thresholds** – A tensor containing thresholds forming a reference point
    from which to calculate pareto frontier hypervolume. Points that do not
    dominate the objective_thresholds contribute nothing to hypervolume.
* **Returns:**
  - The objective
  - The objective thresholds
* **Return type:**
  A two-element tuple with the objective and objective thresholds

### ax.models.torch.botorch_moo_defaults.infer_objective_thresholds(model: Model, objective_weights: Tensor, bounds: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, linear_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, subset_idcs: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, Xs: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, X_observed: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, objective_thresholds: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None) → Tensor

Infer objective thresholds.

This method uses the model-estimated Pareto frontier over the in-sample points
to infer absolute (not relativized) objective thresholds.

This uses a heuristic that sets the objective threshold to be a scaled nadir
point, where the nadir point is scaled back based on the range of each
objective across the current in-sample Pareto frontier.

See botorch.utils.multi_objective.hypervolume.infer_reference_point for
details on the heuristic.

* **Parameters:**
  * **model** – A fitted botorch Model.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights. These should not
    be subsetted.
  * **bounds** – A list of (lower, upper) tuples for each column of X.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b. These should not be subsetted.
  * **linear_constraints** – A tuple of (A, b). For k linear constraints on
    d-dimensional x, A is (k x d) and b is (k x 1) such that
    A x <= b.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value during generation.
  * **subset_idcs** – The indices of the outcomes that are modeled by the
    provided model. If subset_idcs not None, this method infers
    whether the model is subsetted.
  * **Xs** – A list of m (k_i x d) feature tensors X. Number of rows k_i can
    vary from i=1,…,m.
  * **X_observed** – A n x d-dim tensor of in-sample points to use for
    determining the current in-sample Pareto frontier.
  * **objective_thresholds** – Any known objective thresholds to pass to
    infer_reference_point heuristic. This should not be subsetted.
    If only a subset of the objectives have known thresholds, the
    remaining objectives should be NaN. If no objective threshold
    was provided, this can be None.
* **Returns:**
  A m-dim tensor of objective thresholds, where the objective
  : threshold is nan if the outcome is not an objective.

### ax.models.torch.botorch_moo_defaults.pareto_frontier_evaluator(model: [TorchModel](#ax.models.torch_base.TorchModel) | [None](https://docs.python.org/3/library/constants.html#None), objective_weights: Tensor, objective_thresholds: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, X: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, Y: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, Yvar: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor, Tensor]

Return outcomes predicted to lie on a pareto frontier.

Given a model and points to evaluate, use the model to predict which points
lie on the Pareto frontier.

* **Parameters:**
  * **model** – Model used to predict outcomes.
  * **objective_weights** – A m tensor of values indicating the weight to put
    on different outcomes. For pareto frontiers only the sign matters.
  * **objective_thresholds** – A tensor containing thresholds forming a reference point
    from which to calculate pareto frontier hypervolume. Points that do not
    dominate the objective_thresholds contribute nothing to hypervolume.
  * **X** – A n x d tensor of features to evaluate.
  * **Y** – A n x m tensor of outcomes to use instead of predictions.
  * **Yvar** – A n x m x m tensor of input covariances (NaN if unobserved).
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b.
* **Returns:**
  3-element tuple containing
  - A j x m tensor of outcome on the pareto frontier. j is the number
    : of frontier points.
  - A j x m x m tensor of predictive covariances.
    : cov[j, m1, m2] is Cov[m1@j, m2@j].
  - A j tensor of the index of each frontier point in the input Y.

### ax.models.torch.botorch_moo_defaults.scipy_optimizer_list(acq_function_list: [list](https://docs.python.org/3/library/stdtypes.html#list)[AcquisitionFunction], bounds: Tensor, inequality_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor, [float](https://docs.python.org/3/library/functions.html#float)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, rounding_func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[Tensor], Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, num_restarts: [int](https://docs.python.org/3/library/functions.html#int) = 20, raw_samples: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int) | [str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

Sequential optimizer using scipy’s minimize module on a numpy-adaptor.

The ith acquisition in the sequence uses the ith given acquisition_function.

* **Parameters:**
  * **acq_function_list** – A list of botorch AcquisitionFunctions,
    optimized sequentially.
  * **bounds** – A 2 x d-dim tensor, where bounds[0] (bounds[1]) are the
    lower (upper) bounds of the feasible hyperrectangle.
  * **n** – The number of candidates to generate.
  * **constraints** (*inequality*) – A list of tuples (indices, coefficients, rhs),
    with each tuple encoding an inequality constraint of the form
    sum_i (X[indices[i]] \* coefficients[i]) >= rhs
  * **fixed_features** – A map {feature_index: value} for features that should
    be fixed to a particular value during generation.
  * **rounding_func** – A function that rounds an optimization result
    appropriately (i.e., according to round-trip transformations).
* **Returns:**
  2-element tuple containing
  - A n x d-dim tensor of generated candidates.
  - A n-dim tensor of conditional acquisition
    values, where i-th element is the expected acquisition value
    conditional on having observed candidates 0,1,…,i-1.

### ax.models.torch.botorch_modular.acquisition module

### *class* ax.models.torch.botorch_modular.acquisition.Acquisition(surrogates: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Surrogate](#ax.models.torch.botorch_modular.surrogate.Surrogate)], search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig), botorch_acqf_class: [type](https://docs.python.org/3/library/functions.html#type)[AcquisitionFunction], options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Base`](utils.md#ax.utils.common.base.Base)

**All classes in ‘botorch_modular’ directory are under
construction, incomplete, and should be treated as alpha
versions only.**

Ax wrapper for BoTorch AcquisitionFunction, subcomponent
of BoTorchModel and is not meant to be used outside of it.

* **Parameters:**
  * **surrogates** – Dict of name => Surrogate model pairs, with which this acquisition
    function will be used.
  * **search_space_digest** – A SearchSpaceDigest object containing metadata
    about the search space (e.g. bounds, parameter types).
  * **torch_opt_config** – A TorchOptConfig object containing optimization
    arguments (e.g., objective weights, constraints).
  * **botorch_acqf_class** – Type of BoTorch AcquistitionFunction that
    should be used. Subclasses of Acquisition often specify
    these via default_botorch_acqf_class attribute, in which
    case specifying one here is not required.
  * **options** – Optional mapping of kwargs to the underlying Acquisition
    Function in BoTorch.

#### acqf *: AcquisitionFunction*

#### *property* botorch_acqf_class *: [type](https://docs.python.org/3/library/functions.html#type)[AcquisitionFunction]*

BoTorch `AcquisitionFunction` class underlying this `Acquisition`.

#### compute_model_dependencies(surrogates: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Surrogate](#ax.models.torch.botorch_modular.surrogate.Surrogate)], search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig), options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Computes inputs to acquisition function class based on the given
surrogate model.

NOTE: When subclassing Acquisition from a superclass where this
method returns a non-empty dictionary of kwargs to AcquisitionFunction,
call super().compute_model_dependencies and then update that
dictionary of options with the options for the subclass you are creating
(unless the superclass’ model dependencies should not be propagated to
the subclass). See MultiFidelityKnowledgeGradient.compute_model_dependencies
for an example.

* **Parameters:**
  * **surrogates** – Mapping from names to Surrogate objects containing BoTorch
    Model\`s, with which this \`Acquisition is to be used.
  * **search_space_digest** – A SearchSpaceDigest object containing metadata
    about the search space (e.g. bounds, parameter types).
  * **torch_opt_config** – A TorchOptConfig object containing optimization
    arguments (e.g., objective weights, constraints).
  * **options** – The options kwarg dict, passed on initialization of
    the Acquisition object.

Returns: A dictionary of surrogate model-dependent options, to be passed
: as kwargs to BoTorch\`AcquisitionFunction\` constructor.

#### *property* device *: device | [None](https://docs.python.org/3/library/constants.html#None)*

Torch device type of the tensors in the training data used in the model,
of which this `Acquisition` is a subcomponent.

#### *property* dtype *: dtype | [None](https://docs.python.org/3/library/constants.html#None)*

Torch data type of the tensors in the training data used in the model,
of which this `Acquisition` is a subcomponent.

#### evaluate(X: Tensor) → Tensor

Evaluate the acquisition function on the candidate set X.

* **Parameters:**
  **X** – A batch_shape x q x d-dim Tensor of t-batches with q d-dim design
  points each.
* **Returns:**
  A batch_shape’-dim Tensor of acquisition values at the given
  design points X, where batch_shape’ is the broadcasted batch shape of
  model and input X.

#### get_botorch_objective_and_transform(botorch_acqf_class: [type](https://docs.python.org/3/library/functions.html#type)[AcquisitionFunction], model: Model, objective_weights: Tensor, objective_thresholds: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, X_observed: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, risk_measure: RiskMeasureMCObjective | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[MCAcquisitionObjective | [None](https://docs.python.org/3/library/constants.html#None), PosteriorTransform | [None](https://docs.python.org/3/library/constants.html#None)]

#### *property* objective_thresholds *: Tensor | [None](https://docs.python.org/3/library/constants.html#None)*

The objective thresholds for all outcomes.

For non-objective outcomes, the objective thresholds are nans.

#### *property* objective_weights *: Tensor | [None](https://docs.python.org/3/library/constants.html#None)*

The objective weights for all outcomes.

#### optimize(n: [int](https://docs.python.org/3/library/functions.html#int), search_space_digest: SearchSpaceDigest, inequality_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor, [float](https://docs.python.org/3/library/functions.html#float)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, rounding_func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[Tensor], Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, optimizer_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor, Tensor]

Generate a set of candidates via multi-start optimization. Obtains
candidates and their associated acquisition function values.

* **Parameters:**
  * **n** – The number of candidates to generate.
  * **search_space_digest** – A `SearchSpaceDigest` object containing search space
    properties, e.g. `bounds` for optimization.
  * **inequality_constraints** – A list of tuples (indices, coefficients, rhs),
    with each tuple encoding an inequality constraint of the form
    `sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value during generation.
  * **rounding_func** – A function that post-processes an optimization
    result appropriately. This is typically passed down from
    ModelBridge to ensure compatibility of the candidates with
    with Ax transforms. For additional post processing, use
    post_processing_func option in optimizer_options.
  * **optimizer_options** – Options for the optimizer function, e.g. `sequential`
    or `raw_samples`. This can also include a post_processing_func
    which is applied to the candidates before the rounding_func.
    post_processing_func can be used to support more customized options
    that typically only exist in MBM, such as BoTorch transforms.
    See the docstring of TorchOptConfig for more information on passing
    down these options while constructing a generation strategy.
* **Returns:**
  A three-element tuple containing an n x d-dim tensor of generated
  candidates, a tensor with the associated acquisition values, and a tensor
  with the weight for each candidate.

#### options *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*

#### surrogates *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Surrogate](#ax.models.torch.botorch_modular.surrogate.Surrogate)]*

### ax.models.torch.randomforest module

### ax.models.torch.botorch_modular.model module

### *class* ax.models.torch.botorch_modular.model.BoTorchModel(surrogate_specs: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [SurrogateSpec](#ax.models.torch.botorch_modular.model.SurrogateSpec)] | [None](https://docs.python.org/3/library/constants.html#None) = None, surrogate: [Surrogate](#ax.models.torch.botorch_modular.surrogate.Surrogate) | [None](https://docs.python.org/3/library/constants.html#None) = None, acquisition_class: [type](https://docs.python.org/3/library/functions.html#type)[[Acquisition](#ax.models.torch.botorch_modular.acquisition.Acquisition)] | [None](https://docs.python.org/3/library/constants.html#None) = None, acquisition_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, botorch_acqf_class: [type](https://docs.python.org/3/library/functions.html#type)[AcquisitionFunction] | [None](https://docs.python.org/3/library/constants.html#None) = None, refit_on_cv: [bool](https://docs.python.org/3/library/functions.html#bool) = False, warm_start_refit: [bool](https://docs.python.org/3/library/functions.html#bool) = True)

Bases: [`TorchModel`](#ax.models.torch_base.TorchModel), [`Base`](utils.md#ax.utils.common.base.Base)

**All classes in ‘botorch_modular’ directory are under
construction, incomplete, and should be treated as alpha
versions only.**

Modular Model class for combining BoTorch subcomponents
in Ax. Specified via Surrogate and Acquisition, which wrap
BoTorch Model and AcquisitionFunction, respectively, for
convenient use in Ax.

* **Parameters:**
  * **acquisition_class** – Type of Acquisition to be used in
    this model, auto-selected based on experiment and data
    if not specified.
  * **acquisition_options** – Optional dict of kwargs, passed to
    the constructor of BoTorch AcquisitionFunction.
  * **botorch_acqf_class** – Type of AcquisitionFunction to be
    used in this model, auto-selected based on experiment
    and data if not specified.
  * **surrogate_specs** – Optional Mapping of names onto SurrogateSpecs, which specify
    how to initialize specific Surrogates to model specific outcomes. If None
    is provided a single Surrogate will be created and set up automatically
    based on the data provided.
  * **surrogate** – In liu of SurrogateSpecs, an instance of Surrogate may be
    provided to be used as the sole Surrogate for all outcomes
  * **refit_on_cv** – Whether to reoptimize model parameters during call to
    BoTorchmodel.cross_validate.
  * **warm_start_refit** – Whether to load parameters from either the provided
    state dict or the state dict of the current BoTorch Model during
    refitting. If False, model parameters will be reoptimized from
    scratch on refit. NOTE: This setting is ignored during
    cross_validate if the corresponding refit_on_… is False.

#### *property* Xs *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor]*

A list of tensors, each of shape `batch_shape x n_i x d`,
where n_i is the number of training inputs for the i-th model.

NOTE: This is an accessor for `self.surrogate.Xs`
and returns it unchanged.

#### acquisition_class *: [type](https://docs.python.org/3/library/functions.html#type)[[Acquisition](#ax.models.torch.botorch_modular.acquisition.Acquisition)]*

#### acquisition_options *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*

#### best_point(search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig)) → Tensor | [None](https://docs.python.org/3/library/constants.html#None)

Identify the current best point, satisfying the constraints in the same
format as to gen.

Return None if no such point can be identified.

* **Parameters:**
  * **search_space_digest** – A SearchSpaceDigest object containing metadata
    about the search space (e.g. bounds, parameter types).
  * **torch_opt_config** – A TorchOptConfig object containing optimization
    arguments (e.g., objective weights, constraints).
* **Returns:**
  d-tensor of the best point.

#### *property* botorch_acqf_class *: [type](https://docs.python.org/3/library/functions.html#type)[AcquisitionFunction]*

BoTorch `AcquisitionFunction` class, associated with this model.
Raises an error if one is not yet set.

#### cross_validate(datasets: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[SupervisedDataset], X_test: Tensor, search_space_digest: SearchSpaceDigest, use_posterior_predictive: [bool](https://docs.python.org/3/library/functions.html#bool) = False, \*\*additional_model_inputs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

Do cross validation with the given training and test sets.

Training set is given in the same format as to fit. Test set is given
in the same format as to predict.

* **Parameters:**
  * **datasets** – A list of `SupervisedDataset` containers, each
    corresponding to the data of one metric (outcome).
  * **X_test** – (j x d) tensor of the j points at which to make predictions.
  * **search_space_digest** – A SearchSpaceDigest object containing
    metadata on the features in X.
  * **use_posterior_predictive** – A boolean indicating if the predictions
    should be from the posterior predictive (i.e. including
    observation noise).
* **Returns:**
  2-element tuple containing
  - (j x m) tensor of outcome predictions at X.
  - (j x m x m) tensor of predictive covariances at X.
    cov[j, m1, m2] is Cov[m1@j, m2@j].

#### *property* device *: device*

Torch device type of the tensors in the training data used in the model,
of which this `Acquisition` is a subcomponent.

#### *property* dtype *: dtype*

Torch data type of the tensors in the training data used in the model,
of which this `Acquisition` is a subcomponent.

#### evaluate_acquisition_function(X: Tensor, search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig), acq_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → Tensor

Evaluate the acquisition function on the candidate set X.

* **Parameters:**
  * **X** – (j x d) tensor of the j points at which to evaluate the acquisition
    function.
  * **search_space_digest** – A dataclass used to compactly represent a search space.
  * **torch_opt_config** – A TorchOptConfig object containing optimization
    arguments (e.g., objective weights, constraints).
  * **acq_options** – Keyword arguments used to contruct the acquisition function.
* **Returns:**
  A single-element tensor with the acquisition value for these points.

#### feature_importances() → ndarray

Compute feature importances from the model.

Caveat: This assumes the following:
: 1. There is a single surrogate model (potentially a ModelList).
  2. We can get model lengthscales from covar_module.base_kernel.lengthscale

* **Returns:**
  The feature importances as a numpy array of size len(metrics) x 1 x dim
  where each row sums to 1.

#### fit(datasets: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[SupervisedDataset], search_space_digest: SearchSpaceDigest, candidate_metadata: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, state_dicts: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [OrderedDict](https://docs.python.org/3/library/collections.html#collections.OrderedDict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Tensor]] | [None](https://docs.python.org/3/library/constants.html#None) = None, refit: [bool](https://docs.python.org/3/library/functions.html#bool) = True, \*\*additional_model_inputs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [None](https://docs.python.org/3/library/constants.html#None)

Fit model to m outcomes.

* **Parameters:**
  * **datasets** – A list of `SupervisedDataset` containers, each
    corresponding to the data of one or more outcomes.
  * **search_space_digest** – A `SearchSpaceDigest` object containing
    metadata on the features in the datasets.
  * **candidate_metadata** – Model-produced metadata for candidates, in
    the order corresponding to the Xs.
  * **state_dicts** – Optional state dict to load by model label as passed in via
    surrogate_specs. If using a single, pre-instantiated model use

    ```
    `
    ```

    Keys.ONLY_SURROGATE.
  * **refit** – Whether to re-optimize model parameters.
  * **additional_model_inputs** – Additional kwargs to pass to the
    model input constructor in `Surrogate.fit`.

#### gen(n: [int](https://docs.python.org/3/library/functions.html#int), search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig)) → [TorchGenResults](#ax.models.torch_base.TorchGenResults)

Generate new candidates.

* **Parameters:**
  * **n** – Number of candidates to generate.
  * **search_space_digest** – A SearchSpaceDigest object containing metadata
    about the search space (e.g. bounds, parameter types).
  * **torch_opt_config** – A TorchOptConfig object containing optimization
    arguments (e.g., objective weights, constraints).
* **Returns:**
  A TorchGenResult container.

#### *property* outcomes_by_surrogate_label *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]]*

Returns a dictionary mapping from surrogate label to a list of outcomes.

#### *property* output_order *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]*

#### predict(X: Tensor) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

Predicts, potentially from multiple surrogates.

If predictions are from multiple surrogates, will stitch outputs together
in same order as input datasets, using self.output_order.

* **Parameters:**
  **X** – (n x d) Tensor of input locations.

Returns: Tuple of tensors: (n x m) mean, (n x m x m) covariance.

#### predict_from_surrogate(surrogate_label: [str](https://docs.python.org/3/library/stdtypes.html#str), X: Tensor, use_posterior_predictive: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

Predict from the Surrogate with the given label.

#### *property* search_space_digest *: SearchSpaceDigest*

#### *property* surrogate *: [Surrogate](#ax.models.torch.botorch_modular.surrogate.Surrogate)*

Surrogate, if there is only one.

#### surrogate_specs *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [SurrogateSpec](#ax.models.torch.botorch_modular.model.SurrogateSpec)]*

#### *property* surrogates *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Surrogate](#ax.models.torch.botorch_modular.surrogate.Surrogate)]*

Surrogates by label

### *class* ax.models.torch.botorch_modular.model.SurrogateSpec(botorch_model_class: type[~botorch.models.model.Model] | None = None, botorch_model_kwargs: dict[str, ~typing.Any] = <factory>, mll_class: type[~gpytorch.mlls.marginal_log_likelihood.MarginalLogLikelihood] = <class 'gpytorch.mlls.exact_marginal_log_likelihood.ExactMarginalLogLikelihood'>, mll_kwargs: dict[str, ~typing.Any] = <factory>, covar_module_class: type[~gpytorch.kernels.kernel.Kernel] | None = None, covar_module_kwargs: dict[str, ~typing.Any] | None = None, likelihood_class: type[~gpytorch.likelihoods.likelihood.Likelihood] | None = None, likelihood_kwargs: dict[str, ~typing.Any] | None = None, input_transform_classes: list[type[~botorch.models.transforms.input.InputTransform]] | None = None, input_transform_options: dict[str, dict[str, ~typing.Any]] | None = None, outcome_transform_classes: list[type[~botorch.models.transforms.outcome.OutcomeTransform]] | None = None, outcome_transform_options: dict[str, dict[str, ~typing.Any]] | None = None, allow_batched_models: bool = True, outcomes: list[str] = <factory>)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Fields in the SurrogateSpec dataclass correspond to arguments in
`Surrogate.__init__`, except for `outcomes` which is used to specify which
outcomes the Surrogate is responsible for modeling.
When `BotorchModel.fit` is called, these fields will be used to construct the
requisite Surrogate objects.
If `outcomes` is left empty then no outcomes will be fit to the Surrogate.

#### allow_batched_models *: [bool](https://docs.python.org/3/library/functions.html#bool)* *= True*

#### botorch_model_class *: [type](https://docs.python.org/3/library/functions.html#type)[Model] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### botorch_model_kwargs *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*

#### covar_module_class *: [type](https://docs.python.org/3/library/functions.html#type)[Kernel] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### covar_module_kwargs *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### input_transform_classes *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[type](https://docs.python.org/3/library/functions.html#type)[InputTransform]] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### input_transform_options *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### likelihood_class *: [type](https://docs.python.org/3/library/functions.html#type)[Likelihood] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### likelihood_kwargs *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### mll_class

alias of `ExactMarginalLogLikelihood`

#### mll_kwargs *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*

#### outcome_transform_classes *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[type](https://docs.python.org/3/library/functions.html#type)[OutcomeTransform]] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### outcome_transform_options *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### outcomes *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

### ax.models.torch.botorch_modular.model.single_surrogate_only(f: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], T]) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], T]

For use as a decorator on functions only implemented for BotorchModels with a
single Surrogate.

### ax.models.torch.botorch_modular.multi_fidelity module

### ax.models.torch.botorch_modular.optimizer_argparse module

### ax.models.torch.botorch_modular.sebo module

### ax.models.torch.botorch_modular.sebo.L1_norm_func(X: Tensor, init_point: Tensor) → Tensor

L1_norm takes in a a batch_shape x n x d-dim input tensor X
to a batch_shape x n x 1-dimensional L1 norm tensor. To be used
for constructing a GenericDeterministicModel.

### *class* ax.models.torch.botorch_modular.sebo.SEBOAcquisition(surrogates: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Surrogate](#ax.models.torch.botorch_modular.surrogate.Surrogate)], search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig), botorch_acqf_class: [type](https://docs.python.org/3/library/functions.html#type)[AcquisitionFunction], options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Acquisition`](#ax.models.torch.botorch_modular.acquisition.Acquisition)

Implement the acquisition function of Sparsity Exploring Bayesian
Optimization (SEBO).

The SEBO is a hyperparameter-free method to simultaneously maximize a target
objective and sparsity. When L0 norm is used, SEBO uses a novel differentiable
relaxation based on homotopy continuation to efficiently optimize for sparsity.

#### optimize(n: [int](https://docs.python.org/3/library/functions.html#int), search_space_digest: SearchSpaceDigest, inequality_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor, [float](https://docs.python.org/3/library/functions.html#float)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, rounding_func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[Tensor], Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, optimizer_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor, Tensor]

Generate a set of candidates via multi-start optimization. Obtains
candidates and their associated acquisition function values.

* **Parameters:**
  * **n** – The number of candidates to generate.
  * **search_space_digest** – A `SearchSpaceDigest` object containing search space
    properties, e.g. `bounds` for optimization.
  * **inequality_constraints** – A list of tuples (indices, coefficients, rhs),
    with each tuple encoding an inequality constraint of the form
    `sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
  * **fixed_features** – A map {feature_index: value} for features that
    should be fixed to a particular value during generation.
  * **rounding_func** – A function that post-processes an optimization
    result appropriately (i.e., according to round-trip
    transformations).
  * **optimizer_options** – Options for the optimizer function, e.g. `sequential`
    or `raw_samples`.
* **Returns:**
  A three-element tuple containing an n x d-dim tensor of generated
  candidates, a tensor with the associated acquisition values, and a tensor
  with the weight for each candidate.

### ax.models.torch.botorch_modular.sebo.clamp_candidates(X: Tensor, target_point: Tensor, clamp_tol: [float](https://docs.python.org/3/library/functions.html#float), \*\*tkwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → Tensor

Clamp generated candidates within the given ranges to the target point.

### ax.models.torch.botorch_modular.sebo.get_batch_initial_conditions(acq_function: AcquisitionFunction, raw_samples: [int](https://docs.python.org/3/library/functions.html#int), X_pareto: Tensor, target_point: Tensor, num_restarts: [int](https://docs.python.org/3/library/functions.html#int) = 20, \*\*tkwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → Tensor

Generate starting points for the SEBO acquisition function optimization.

### ax.models.torch.botorch_modular.surrogate module

### *class* ax.models.torch.botorch_modular.surrogate.Surrogate(botorch_model_class: type[~botorch.models.model.Model] | None = None, model_options: dict[str, ~typing.Any] | None = None, mll_class: type[~gpytorch.mlls.marginal_log_likelihood.MarginalLogLikelihood] = <class 'gpytorch.mlls.exact_marginal_log_likelihood.ExactMarginalLogLikelihood'>, mll_options: dict[str, ~typing.Any] | None = None, outcome_transform_classes: ~collections.abc.Sequence[type[~botorch.models.transforms.outcome.OutcomeTransform]] | None = None, outcome_transform_options: dict[str, dict[str, ~typing.Any]] | None = None, input_transform_classes: ~collections.abc.Sequence[type[~botorch.models.transforms.input.InputTransform]] | None = None, input_transform_options: dict[str, dict[str, ~typing.Any]] | None = None, covar_module_class: type[~gpytorch.kernels.kernel.Kernel] | None = None, covar_module_options: dict[str, ~typing.Any] | None = None, likelihood_class: type[~gpytorch.likelihoods.likelihood.Likelihood] | None = None, likelihood_options: dict[str, ~typing.Any] | None = None, allow_batched_models: bool = True)

Bases: [`Base`](utils.md#ax.utils.common.base.Base)

**All classes in ‘botorch_modular’ directory are under
construction, incomplete, and should be treated as alpha
versions only.**

Ax wrapper for BoTorch `Model`, subcomponent of `BoTorchModel`
and is not meant to be used outside of it.

* **Parameters:**
  * **botorch_model_class** – `Model` class to be used as the underlying
    BoTorch model. If None is provided a model class will be selected (either
    one for all outcomes or a ModelList with separate models for each outcome)
    will be selected automatically based off the datasets at construct time.
  * **model_options** – Dictionary of options / kwargs for the BoTorch
    `Model` constructed during `Surrogate.fit`.
    Note that the corresponding attribute will later be updated to include any
    additional kwargs passed into `BoTorchModel.fit`.
  * **mll_class** – `MarginalLogLikelihood` class to use for model-fitting.
  * **mll_options** – Dictionary of options / kwargs for the MLL.
  * **outcome_transform_classes** – List of BoTorch outcome transforms classes. Passed
    down to the BoTorch `Model`. Multiple outcome transforms can be chained
    together using `ChainedOutcomeTransform`.
  * **outcome_transform_options** – 

    Outcome transform classes kwargs. The keys are
    class string names and the values are dictionaries of outcome transform
    kwargs. For example,
    \`
    outcome_transform_classes = [Standardize]
    outcome_transform_options = {
    > ”Standardize”: {“m”: 1},

    \`
    For more options see botorch/models/transforms/outcome.py.
  * **input_transform_classes** – List of BoTorch input transforms classes.
    Passed down to the BoTorch `Model`. Multiple input transforms
    will be chained together using `ChainedInputTransform`.
  * **input_transform_options** – 

    Input transform classes kwargs. The keys are
    class string names and the values are dictionaries of input transform
    kwargs. For example,
    \`
    input_transform_classes = [Normalize, Round]
    input_transform_options = {
    > ”Normalize”: {“d”: 3},
    > “Round”: {“integer_indices”: [0], “categorical_features”: {1: 2}},

    For more input options see botorch/models/transforms/input.py.
  * **covar_module_class** – Covariance module class. This gets initialized after
    parsing the `covar_module_options` in `covar_module_argparse`,
    and gets passed to the model constructor as `covar_module`.
  * **covar_module_options** – Covariance module kwargs.
  * **likelihood** – `Likelihood` class. This gets initialized with
    `likelihood_options` and gets passed to the model constructor.
  * **likelihood_options** – Likelihood options.
  * **allow_batched_models** – Set to true to fit the models in a batch if supported.
    Set to false to fit individual models to each metric in a loop.

#### *property* Xs *: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor]*

#### best_in_sample_point(search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig), options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, [float](https://docs.python.org/3/library/functions.html#float)]

Finds the best observed point and the corresponding observed outcome
values.

#### best_out_of_sample_point(search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig), options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

Finds the best predicted point and the corresponding value of the
appropriate best point acquisition function.

* **Parameters:**
  * **search_space_digest** – A SearchSpaceDigest.
  * **torch_opt_config** – A TorchOptConfig; none-None fixed_features is
    not supported.
  * **options** – Optional. If present, seed_inner (default None) and qmc
    (default True) will be parsed from options; any other keys
    will be ignored.
* **Returns:**
  A two-tuple (candidate, acqf_value), where candidate is a 1d
  Tensor of the best predicted point and acqf_value is a scalar (0d)
  Tensor of the acquisition function value at the best point.

#### clone_reset() → [Surrogate](#ax.models.torch.botorch_modular.surrogate.Surrogate)

#### compute_diagnostics() → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Computes model diagnostics like cross-validation measure of fit, etc.

#### *property* device *: device*

#### *property* dtype *: dtype*

#### fit(datasets: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[SupervisedDataset], search_space_digest: SearchSpaceDigest, candidate_metadata: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, state_dict: [OrderedDict](https://docs.python.org/3/library/collections.html#collections.OrderedDict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, refit: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [None](https://docs.python.org/3/library/constants.html#None)

Fits the underlying BoTorch `Model` to `m` outcomes.

NOTE: `state_dict` and `refit` keyword arguments control how the
undelying BoTorch `Model` will be fit: whether its parameters will
be reoptimized and whether it will be warm-started from a given state.

There are three possibilities:

* `fit(state_dict=None)`: fit model from scratch (optimize model
  parameters and set its training data used for inference),
* `fit(state_dict=some_state_dict, refit=True)`: warm-start refit
  with a state dict of parameters (still re-optimize model parameters
  and set the training data),
* `fit(state_dict=some_state_dict, refit=False)`: load model parameters
  without refitting, but set new training data (used in cross-validation,
  for example).

* **Parameters:**
  * **datasets** – A list of `SupervisedDataset` containers, each
    corresponding to the data of one metric (outcome), to be passed
    to `Model.construct_inputs` in BoTorch.
  * **search_space_digest** – A `SearchSpaceDigest` object containing
    metadata on the features in the datasets.
  * **candidate_metadata** – Model-produced metadata for candidates, in
    the order corresponding to the Xs.
  * **state_dict** – Optional state dict to load.
  * **refit** – Whether to re-optimize model parameters.

#### *property* model *: Model*

#### *property* outcomes *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### pareto_frontier() → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

For multi-objective optimization, retrieve Pareto frontier instead
of best point.

Returns: A two-tuple of:
: - tensor of points in the feature space,
  - tensor of corresponding (multiple) outcomes.

#### predict(X: Tensor, use_posterior_predictive: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

Predicts outcomes given an input tensor.

* **Parameters:**
  * **X** – A `n x d` tensor of input parameters.
  * **use_posterior_predictive** – A boolean indicating if the predictions
    should be from the posterior predictive (i.e. including
    observation noise).
* **Returns:**
  The predicted posterior mean as an `n x o`-dim tensor.
  Tensor: The predicted posterior covariance as a `n x o x o`-dim tensor.
* **Return type:**
  Tensor

#### *property* training_data *: [list](https://docs.python.org/3/library/stdtypes.html#list)[SupervisedDataset]*

### ax.models.torch.botorch_modular.utils module

### ax.models.torch.botorch_modular.utils.check_outcome_dataset_match(outcome_names: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[str](https://docs.python.org/3/library/stdtypes.html#str)], datasets: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[SupervisedDataset], exact_match: [bool](https://docs.python.org/3/library/functions.html#bool)) → [None](https://docs.python.org/3/library/constants.html#None)

Check that the given outcome names match those of datasets.

Based on exact_match we either require that outcome names are
a subset of all outcomes or require the them to be the same.

Also checks that there are no duplicates in outcome names.

* **Parameters:**
  * **outcome_names** – A list of outcome names.
  * **datasets** – A list of SupervisedDataset objects.
  * **exact_match** – If True, outcome_names must be the same as the union of
    outcome names of the datasets. Otherwise, we check that the
    outcome_names are a subset of all outcomes.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError) – If there is no match.

### ax.models.torch.botorch_modular.utils.choose_botorch_acqf_class(pending_observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, linear_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, objective_thresholds: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, objective_weights: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None) → [type](https://docs.python.org/3/library/functions.html#type)[AcquisitionFunction]

Chooses a BoTorch AcquisitionFunction class.

### ax.models.torch.botorch_modular.utils.choose_model_class(datasets: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[SupervisedDataset], search_space_digest: SearchSpaceDigest) → [type](https://docs.python.org/3/library/functions.html#type)[Model]

Chooses a BoTorch Model using the given data (currently just Yvars)
and its properties (information about task and fidelity features).

* **Parameters:**
  * **Yvars** – List of tensors, each representing observation noise for a
    given outcome, where outcomes are in the same order as in Xs.
  * **task_features** – List of columns of X that are tasks.
  * **fidelity_features** – List of columns of X that are fidelity parameters.
* **Returns:**
  A BoTorch Model class.

### ax.models.torch.botorch_modular.utils.construct_acquisition_and_optimizer_options(acqf_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)], model_gen_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)]]

Extract acquisition and optimizer options from model_gen_options.

### ax.models.torch.botorch_modular.utils.convert_to_block_design(datasets: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[SupervisedDataset], force: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [list](https://docs.python.org/3/library/stdtypes.html#list)[SupervisedDataset]

### ax.models.torch.botorch_modular.utils.fit_botorch_model(model: Model, mll_class: [type](https://docs.python.org/3/library/functions.html#type)[MarginalLogLikelihood], mll_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)

Fit a BoTorch model.

### ax.models.torch.botorch_modular.utils.get_post_processing_func(rounding_func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[Tensor], Tensor] | [None](https://docs.python.org/3/library/constants.html#None), optimizer_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[Tensor], Tensor] | [None](https://docs.python.org/3/library/constants.html#None)

Get the post processing function by combining the rounding function
with the post processing function provided as part of the optimizer
options. If both are given, the post processing function is applied before
applying the rounding function. If only one of them is given, then
it is used as the post processing function.

### ax.models.torch.botorch_modular.utils.get_subset_datasets(datasets: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[SupervisedDataset], subset_outcome_names: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[SupervisedDataset]

Get the list of datasets corresponding to the given subset of
outcome names. This is used to separate out datasets that are
used by one surrogate.

* **Parameters:**
  * **datasets** – A list of SupervisedDataset objects.
  * **subset_outcome_names** – A list of outcome names to get datasets for.
* **Returns:**
  A list of SupervisedDataset objects corresponding to the given
  subset of outcome names.

### ax.models.torch.botorch_modular.utils.subset_state_dict(state_dict: [OrderedDict](https://docs.python.org/3/library/collections.html#collections.OrderedDict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Tensor], submodel_index: [int](https://docs.python.org/3/library/functions.html#int)) → [OrderedDict](https://docs.python.org/3/library/collections.html#collections.OrderedDict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Tensor]

Get the state dict for a submodel from the state dict of a model list.

* **Parameters:**
  * **state_dict** – A state dict.
  * **submodel_index** – The index of the submodel to extract.
* **Returns:**
  The state dict for the submodel.

### ax.models.torch.botorch_modular.utils.use_model_list(datasets: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[SupervisedDataset], botorch_model_class: [type](https://docs.python.org/3/library/functions.html#type)[Model], allow_batched_models: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [bool](https://docs.python.org/3/library/functions.html#bool)

### ax.models.torch.botorch_modular.kernels module

### *class* ax.models.torch.botorch_modular.kernels.ScaleMaternKernel(ard_num_dims: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, batch_shape: Size | [None](https://docs.python.org/3/library/constants.html#None) = None, lengthscale_prior: Prior | [None](https://docs.python.org/3/library/constants.html#None) = None, outputscale_prior: Prior | [None](https://docs.python.org/3/library/constants.html#None) = None, lengthscale_constraint: Interval | [None](https://docs.python.org/3/library/constants.html#None) = None, outputscale_constraint: Interval | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

Bases: `ScaleKernel`

### *class* ax.models.torch.botorch_modular.kernels.TemporalKernel(dim: [int](https://docs.python.org/3/library/functions.html#int), temporal_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], matern_ard_num_dims: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, batch_shape: Size | [None](https://docs.python.org/3/library/constants.html#None) = None, lengthscale_prior: Prior | [None](https://docs.python.org/3/library/constants.html#None) = None, temporal_lengthscale_prior: Prior | [None](https://docs.python.org/3/library/constants.html#None) = None, period_length_prior: Prior | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_period_length: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, outputscale_prior: Prior | [None](https://docs.python.org/3/library/constants.html#None) = None, lengthscale_constraint: Interval | [None](https://docs.python.org/3/library/constants.html#None) = None, outputscale_constraint: Interval | [None](https://docs.python.org/3/library/constants.html#None) = None, temporal_lengthscale_constraint: Interval | [None](https://docs.python.org/3/library/constants.html#None) = None, period_length_constraint: Interval | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

Bases: `ScaleKernel`

A product kernel of a periodic kernel and a Matern kernel.

The periodic kernel computes the similarity between temporal
features such as the time of day.

The Matern kernel computes the similarity between the tunable
parameters.

### ax.models.torch.botorch_modular.input_constructors.covar_modules module

### ax.models.torch.botorch_modular.input_constructors.input_transforms module

### ax.models.torch.botorch_modular.input_constructors.outcome_transform module

### ax.models.torch.cbo_lcea module

### *class* ax.models.torch.cbo_lcea.LCEABO(decomposition: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]], cat_feature_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None, embs_feature_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None, context_weight_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None, embs_dim_list: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, gp_model_args: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`BotorchModel`](#ax.models.torch.botorch.BotorchModel)

Does Bayesian optimization with Latent Context Embedding Additive (LCE-A) GP.
The parameter space decomposition must be provided.

* **Parameters:**
  * **decomposition** – Keys are context names. Values are the lists of parameter
    names belong to the context, e.g.
    {‘context1’: [‘p1_c1’, ‘p2_c1’],’context2’: [‘p1_c2’, ‘p2_c2’]}.
  * **gp_model_args** – Dictionary of kwargs to pass to GP model training.
    - train_embedding: Boolen. If true, we will train context embedding;
    otherwise, we use pre-trained embeddings from embds_feature_dict only.
    Default is True.

#### best_point(search_space_digest: SearchSpaceDigest, torch_opt_config: [TorchOptConfig](#ax.models.torch_base.TorchOptConfig)) → Tensor | [None](https://docs.python.org/3/library/constants.html#None)

Identify the current best point, satisfying the constraints in the same
format as to gen.

Return None if no such point can be identified.

* **Parameters:**
  * **search_space_digest** – A SearchSpaceDigest object containing metadata
    about the search space (e.g. bounds, parameter types).
  * **torch_opt_config** – A TorchOptConfig object containing optimization
    arguments (e.g., objective weights, constraints).
* **Returns:**
  d-tensor of the best point.

#### fit(datasets: [list](https://docs.python.org/3/library/stdtypes.html#list)[SupervisedDataset], search_space_digest: SearchSpaceDigest, candidate_metadata: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)]] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)

Fit model to m outcomes.

* **Parameters:**
  * **datasets** – A list of `SupervisedDataset` containers, each
    corresponding to the data of one metric (outcome).
  * **search_space_digest** – A `SearchSpaceDigest` object containing
    metadata on the features in the datasets.
  * **candidate_metadata** – Model-produced metadata for candidates, in
    the order corresponding to the Xs.

#### get_and_fit_model(Xs: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], Ys: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], Yvars: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], task_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], fidelity_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], metric_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], state_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, fidelity_model_id: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → GPyTorchModel

Get a fitted LCEAGP model for each outcome.
:param Xs: X for each outcome.
:param Ys: Y for each outcome.
:param Yvars: Noise variance of Y for each outcome.

Returns: Fitted LCEAGP model.

#### *property* model *: LCEAGP | ModelListGP*

### ax.models.torch.cbo_lcea.get_map_model(train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor, decomposition: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]], train_embedding: [bool](https://docs.python.org/3/library/functions.html#bool) = True, cat_feature_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None, embs_feature_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None, embs_dim_list: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, context_weight_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[LCEAGP, ExactMarginalLogLikelihood]

Obtain MAP fitting of Latent Context Embedding Additive (LCE-A) GP.

### ax.models.torch.cbo_lcem module

### *class* ax.models.torch.cbo_lcem.LCEMBO(context_cat_feature: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, context_emb_feature: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, embs_dim_list: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`BotorchModel`](#ax.models.torch.botorch.BotorchModel)

Does Bayesian optimization with LCE-M GP.

#### get_and_fit_model(Xs: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], Ys: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], Yvars: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], task_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], fidelity_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], metric_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], state_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, fidelity_model_id: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → ModelListGP

Get a fitted multi-task contextual GP model for each outcome.
:param Xs: List of X data, one tensor per outcome.
:param Ys: List of Y data, one tensor per outcome.
:param Yvars: List of Noise variance of Yvar data, one tensor per outcome.
:param task_features: List of columns of X that are tasks.

Returns: ModeListGP that each model is a fitted LCEM GP model.

### ax.models.torch.cbo_sac module

### *class* ax.models.torch.cbo_sac.SACBO(decomposition: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]])

Bases: [`BotorchModel`](#ax.models.torch.botorch.BotorchModel)

Does Bayesian optimization with structural additive contextual GP (SACGP).
The parameter space decomposition must be provided.

* **Parameters:**
  **decomposition** – Keys are context names. Values are the lists of parameter
  names belong to the context, e.g.
  {‘context1’: [‘p1_c1’, ‘p2_c1’],’context2’: [‘p1_c2’, ‘p2_c2’]}.

#### fit(datasets: [list](https://docs.python.org/3/library/stdtypes.html#list)[SupervisedDataset], search_space_digest: SearchSpaceDigest, candidate_metadata: [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)]] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)

Fit model to m outcomes.

* **Parameters:**
  * **datasets** – A list of `SupervisedDataset` containers, each
    corresponding to the data of one metric (outcome).
  * **search_space_digest** – A `SearchSpaceDigest` object containing
    metadata on the features in the datasets.
  * **candidate_metadata** – Model-produced metadata for candidates, in
    the order corresponding to the Xs.

#### get_and_fit_model(Xs: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], Ys: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], Yvars: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], task_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], fidelity_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], metric_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], state_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, fidelity_model_id: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → GPyTorchModel

Get a fitted StructuralAdditiveContextualGP model for each outcome.
:param Xs: X for each outcome.
:param Ys: Y for each outcome.
:param Yvars: Noise variance of Y for each outcome.

Returns: Fitted StructuralAdditiveContextualGP model.

### ax.models.torch.cbo_sac.generate_model_space_decomposition(decomposition: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]], feature_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]]

### ax.models.torch.fully_bayesian module

### ax.models.torch.fully_bayesian_model_utils module

### ax.models.torch.utils module

### *class* ax.models.torch.utils.SubsetModelData(model: botorch.models.model.Model, objective_weights: torch.Tensor, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[torch.Tensor, torch.Tensor] | [None](https://docs.python.org/3/library/constants.html#None), objective_thresholds: torch.Tensor | [None](https://docs.python.org/3/library/constants.html#None), indices: torch.Tensor)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

#### indices *: Tensor*

#### model *: Model*

#### objective_thresholds *: Tensor | [None](https://docs.python.org/3/library/constants.html#None)*

#### objective_weights *: Tensor*

#### outcome_constraints *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None)*

### ax.models.torch.utils.get_botorch_objective_and_transform(botorch_acqf_class: [type](https://docs.python.org/3/library/functions.html#type)[AcquisitionFunction], model: Model, objective_weights: Tensor, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, X_observed: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None, risk_measure: RiskMeasureMCObjective | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[MCAcquisitionObjective | [None](https://docs.python.org/3/library/constants.html#None), PosteriorTransform | [None](https://docs.python.org/3/library/constants.html#None)]

Constructs a BoTorch AcquisitionObjective object.

* **Parameters:**
  * **botorch_acqf_class** – The acquisition function class the objective
    and posterior transform are to be used with. This is mainly
    used to determine whether to construct a multi-output or a
    single-output objective.
  * **model** – A BoTorch Model.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b. (Not used by single task models)
  * **X_observed** – Observed points that are feasible and appear in the
    objective or the constraints. None if there are no such points.
  * **risk_measure** – An optional risk measure for robust optimization.
* **Returns:**
  A two-tuple containing (optionally) an MCAcquisitionObjective and
  (optionally) a PosteriorTransform.

### ax.models.torch.utils.get_out_of_sample_best_point_acqf(model: Model, Xs: [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor], X_observed: Tensor, objective_weights: Tensor, mc_samples: [int](https://docs.python.org/3/library/functions.html#int) = 512, fixed_features: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, fidelity_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, target_fidelities: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, seed_inner: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, qmc: [bool](https://docs.python.org/3/library/functions.html#bool) = True, risk_measure: RiskMeasureMCObjective | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[AcquisitionFunction, [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None)]

Picks an appropriate acquisition function to find the best
out-of-sample (predicted by the given surrogate model) point
and instantiates it.

NOTE: Typically the appropriate function is the posterior mean,
but can differ to account for fidelities etc.

### ax.models.torch.utils.is_noiseless(model: Model) → [bool](https://docs.python.org/3/library/functions.html#bool)

Check if a given (single-task) botorch model is noiseless

### ax.models.torch.utils.normalize_indices(indices: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)], d: [int](https://docs.python.org/3/library/functions.html#int)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]

Normalize a list of indices to ensure that they are positive.

* **Parameters:**
  * **indices** – A list of indices (may contain negative indices for indexing
    “from the back”).
  * **d** – The dimension of the tensor to index.
* **Returns:**
  A normalized list of indices such that each index is between 0 and d-1.

### ax.models.torch.utils.pick_best_out_of_sample_point_acqf_class(outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, mc_samples: [int](https://docs.python.org/3/library/functions.html#int) = 512, qmc: [bool](https://docs.python.org/3/library/functions.html#bool) = True, seed_inner: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, risk_measure: RiskMeasureMCObjective | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[type](https://docs.python.org/3/library/functions.html#type)[AcquisitionFunction], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]

### ax.models.torch.utils.predict_from_model(model: Model, X: Tensor, use_posterior_predictive: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor]

Predicts outcomes given a model and input tensor.

For a GaussianMixturePosterior we currently use a Gaussian approximation where we
compute the mean and variance of the Gaussian mixture. This should ideally be
changed to compute quantiles instead when Ax supports non-Gaussian distributions.

* **Parameters:**
  * **model** – A botorch Model.
  * **X** – A n x d tensor of input parameters.
  * **use_posterior_predictive** – A boolean indicating if the predictions
    should be from the posterior predictive (i.e. including
    observation noise).
* **Returns:**
  The predicted posterior mean as an n x o-dim tensor.
  Tensor: The predicted posterior covariance as a n x o x o-dim tensor.
* **Return type:**
  Tensor

### ax.models.torch.utils.randomize_objective_weights(objective_weights: Tensor, random_scalarization_distribution: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'simplex') → Tensor

Generate a random weighting based on acquisition function settings.

* **Parameters:**
  * **objective_weights** – Base weights to multiply by random values.
  * **random_scalarization_distribution** – “simplex” or “hypersphere”.
* **Returns:**
  A normalized list of indices such that each index is between 0 and d-1.

### ax.models.torch.utils.subset_model(model: Model, objective_weights: Tensor, outcome_constraints: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None) = None, objective_thresholds: Tensor | [None](https://docs.python.org/3/library/constants.html#None) = None) → [SubsetModelData](#ax.models.torch.utils.SubsetModelData)

Subset a botorch model to the outputs used in the optimization.

* **Parameters:**
  * **model** – A BoTorch Model. If the model does not implement the
    subset_outputs method, this function is a null-op and returns the
    input arguments.
  * **objective_weights** – The objective is to maximize a weighted sum of
    the columns of f(x). These are the weights.
  * **objective_thresholds** – The m-dim tensor of objective thresholds. There
    is one for each modeled metric.
  * **outcome_constraints** – A tuple of (A, b). For k outcome constraints
    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
    A f(x) <= b. (Not used by single task models)
* **Returns:**
  A SubsetModelData dataclass containing the model, objective_weights,
  outcome_constraints, objective thresholds, all subset to only those
  outputs that appear in either the objective weights or the outcome
  constraints, along with the indices of the outputs.

### ax.models.torch.utils.tensor_callable_to_array_callable(tensor_func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[Tensor], Tensor], device: device) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[ndarray], ndarray]

transfer a tensor callable to an array callable
