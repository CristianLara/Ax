# ax.modelbridge

## Generation Strategy, Registry, and Factory

### Generation Strategy

### Generation Node

### External Generation Node

Transition Criterion
.. automodule:: ax.modelbridge.transition_criterion

> * **members:**
> * **undoc-members:**
> * **show-inheritance:**

Generation Node Input Constructors
.. automodule:: ax.modelbridge.generation_node_input_constructors

> * **members:**
> * **undoc-members:**
> * **show-inheritance:**

### Registry

### Factory

### ModelSpec

## Model Bridges

### Base Model Bridge

### *class* ax.modelbridge.base.BaseGenArgs(search_space: ax.core.search_space.SearchSpace, optimization_config: [ax.core.optimization_config.OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None), pending_observations: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[ax.core.observation.ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]], fixed_features: [ax.core.observation.ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None))

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

#### fixed_features *: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None)*

#### optimization_config *: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)*

#### pending_observations *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]]*

#### search_space *: SearchSpace*

### *class* ax.modelbridge.base.GenResults(observation_features: list[ax.core.observation.ObservationFeatures], weights: list[float], best_observation_features: ax.core.observation.ObservationFeatures | None = None, gen_metadata: dict[str, typing.Any] = <factory>)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

#### best_observation_features *: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### gen_metadata *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*

#### observation_features *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]*

#### weights *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]*

### *class* ax.modelbridge.base.ModelBridge(search_space: SearchSpace, model: [Any](https://docs.python.org/3/library/typing.html#typing.Any), transforms: [list](https://docs.python.org/3/library/stdtypes.html#list)[[type](https://docs.python.org/3/library/functions.html#type)[[Transform](#ax.modelbridge.transforms.base.Transform)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, experiment: [Experiment](core.md#ax.core.experiment.Experiment) | [None](https://docs.python.org/3/library/constants.html#None) = None, data: Data | [None](https://docs.python.org/3/library/constants.html#None) = None, transform_configs: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, status_quo_name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, status_quo_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, fit_out_of_design: [bool](https://docs.python.org/3/library/functions.html#bool) = False, fit_abandoned: [bool](https://docs.python.org/3/library/functions.html#bool) = False, fit_tracking_metrics: [bool](https://docs.python.org/3/library/functions.html#bool) = True, fit_on_init: [bool](https://docs.python.org/3/library/functions.html#bool) = True)

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

The main object for using models in Ax.

ModelBridge specifies 3 methods for using models:

- predict: Make model predictions. This method is not optimized for
  speed and so should be used primarily for plotting or similar tasks
  and not inside an optimization loop.
- gen: Use the model to generate new candidates.
- cross_validate: Do cross validation to assess model predictions.

ModelBridge converts Ax types like Data and Arm to types that are
meant to be consumed by the models. The data sent to the model will depend
on the implementation of the subclass, which will specify the actual API
for external model.

This class also applies a sequence of transforms to the input data and
problem specification which can be used to ensure that the external model
receives appropriate inputs.

Subclasses will implement what is here referred to as the “terminal
transform,” which is a transform that changes types of the data and problem
specification.

#### cross_validate(cv_training_data: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)], cv_test_points: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)], use_posterior_predictive: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationData](core.md#ax.core.observation.ObservationData)]

Make a set of cross-validation predictions.

* **Parameters:**
  * **cv_training_data** – The training data to use for cross validation.
  * **cv_test_points** – The test points at which predictions will be made.
  * **use_posterior_predictive** – A boolean indicating if the predictions
    should be from the posterior predictive (i.e. including
    observation noise).
* **Returns:**
  A list of predictions at the test points.

#### feature_importances(metric_name: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]

Computes feature importances for a single metric.

Depending on the type of the model, this method will approach sensitivity
analysis (calculating the sensitivity of the metric to changes in the search
space’s parameters, a.k.a. features) differently.

For Bayesian optimization models (BoTorch models), this method uses parameter
inverse lengthscales to compute normalized feature importances.

NOTE: Currently, this is only implemented for GP models.

* **Parameters:**
  **metric_name** – Name of metric to compute feature importances for.
* **Returns:**
  A dictionary mapping parameter names to their corresponding feature
  importances.

#### gen(n: [int](https://docs.python.org/3/library/functions.html#int), search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, pending_observations: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None, model_gen_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [GeneratorRun](core.md#ax.core.generator_run.GeneratorRun)

Generate new points from the underlying model according to
search_space, optimization_config and other parameters.

* **Parameters:**
  * **n** – Number of points to generate
  * **search_space** – Search space
  * **optimization_config** – Optimization config
  * **pending_observations** – A map from metric name to pending
    observations for that metric.
  * **fixed_features** – An ObservationFeatures object containing any
    features that should be fixed at specified values during
    generation.
  * **model_gen_options** – A config dictionary that is passed along to the
    model. See TorchOptConfig for details.
* **Returns:**
  A GeneratorRun object that contains the generated points and other metadata.

#### get_training_data() → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]

A copy of the (untransformed) data with which the model was fit.

#### *property* metric_names *: [set](https://docs.python.org/3/library/stdtypes.html#set)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

Metric names present in training data.

#### *property* model_space *: SearchSpace*

SearchSpace used to fit model.

#### predict(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]]]]

Make model predictions (mean and covariance) for the given
observation features.

Predictions are made for all outcomes.
If an out-of-design observation can successfully be transformed,
the predicted value will be returned.
Othwerise, we will attempt to find that observation in the training data
and return the raw value.

* **Parameters:**
  **observation_features** – observation features
* **Returns:**
  2-element tuple containing
  - Dictionary from metric name to list of mean estimates, in same
    order as observation_features.
  - Nested dictionary with cov[‘metric1’][‘metric2’] a list of
    cov(metric1@x, metric2@x) for x in observation_features.

#### *property* status_quo *: [Observation](core.md#ax.core.observation.Observation) | [None](https://docs.python.org/3/library/constants.html#None)*

Observation corresponding to status quo, if any.

#### *property* status_quo_data_by_trial *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [ObservationData](core.md#ax.core.observation.ObservationData)] | [None](https://docs.python.org/3/library/constants.html#None)*

A map of trial index to the status quo observation data of each trial

#### *property* statuses_to_fit *: [set](https://docs.python.org/3/library/stdtypes.html#set)[[TrialStatus](core.md#ax.core.base_trial.TrialStatus)]*

Statuses to fit the model on.

#### *property* statuses_to_fit_map_metric *: [set](https://docs.python.org/3/library/stdtypes.html#set)[[TrialStatus](core.md#ax.core.base_trial.TrialStatus)]*

Statuses to fit the model on.

#### *property* training_in_design *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[bool](https://docs.python.org/3/library/functions.html#bool)]*

For each observation in the training data, a bool indicating if it
is in-design for the model.

#### transform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Applies transforms to given observation features and returns them in the
model space.

* **Parameters:**
  **observation_features** – ObservationFeatures to be transformed.
* **Returns:**
  Transformed values. This could be e.g. a torch Tensor, depending
  on the ModelBridge subclass.

#### transform_observations(observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Applies transforms to given observation features and returns them in the
model space.

* **Parameters:**
  **observation_features** – ObservationFeatures to be transformed.
* **Returns:**
  Transformed values. This could be e.g. a torch Tensor, depending
  on the ModelBridge subclass.

#### update(new_data: Data, experiment: [Experiment](core.md#ax.core.experiment.Experiment)) → [None](https://docs.python.org/3/library/constants.html#None)

Update the model bridge and the underlying model with new data. This
method should be used instead of fit, in cases where the underlying
model does not need to be re-fit from scratch, but rather updated.

Note: update expects only new data (obtained since the model initialization
or last update) to be passed in, not all data in the experiment.

* **Parameters:**
  * **new_data** – Data from the experiment obtained since the last call to
    update.
  * **experiment** – Experiment, in which this data was obtained.

### ax.modelbridge.base.clamp_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)], search_space: SearchSpace) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

### ax.modelbridge.base.gen_arms(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)], arms_by_signature: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Arm](core.md#ax.core.arm.Arm)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[Arm](core.md#ax.core.arm.Arm)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None)]

Converts observation features to a tuple of arms list and candidate metadata
dict, where arm signatures are mapped to their respective candidate metadata.

### ax.modelbridge.base.unwrap_observation_data(observation_data: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationData](core.md#ax.core.observation.ObservationData)]) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]]]]

Converts observation data to the format for model prediction outputs.
That format assumes each observation data has the same set of metrics.

### Discrete Model Bridge

### *class* ax.modelbridge.discrete.DiscreteModelBridge(search_space: SearchSpace, model: [Any](https://docs.python.org/3/library/typing.html#typing.Any), transforms: [list](https://docs.python.org/3/library/stdtypes.html#list)[[type](https://docs.python.org/3/library/functions.html#type)[[Transform](#ax.modelbridge.transforms.base.Transform)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, experiment: [Experiment](core.md#ax.core.experiment.Experiment) | [None](https://docs.python.org/3/library/constants.html#None) = None, data: Data | [None](https://docs.python.org/3/library/constants.html#None) = None, transform_configs: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, status_quo_name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, status_quo_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, fit_out_of_design: [bool](https://docs.python.org/3/library/functions.html#bool) = False, fit_abandoned: [bool](https://docs.python.org/3/library/functions.html#bool) = False, fit_tracking_metrics: [bool](https://docs.python.org/3/library/functions.html#bool) = True, fit_on_init: [bool](https://docs.python.org/3/library/functions.html#bool) = True)

Bases: [`ModelBridge`](#ax.modelbridge.base.ModelBridge)

A model bridge for using models based on discrete parameters.

Requires that all parameters have been transformed to ChoiceParameters.

#### model *: [DiscreteModel](models.md#ax.models.discrete_base.DiscreteModel)*

#### outcomes *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### parameters *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### search_space *: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None)*

### Random Model Bridge

### *class* ax.modelbridge.random.RandomModelBridge(search_space: SearchSpace, model: [Any](https://docs.python.org/3/library/typing.html#typing.Any), transforms: [list](https://docs.python.org/3/library/stdtypes.html#list)[[type](https://docs.python.org/3/library/functions.html#type)[[Transform](#ax.modelbridge.transforms.base.Transform)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, experiment: [Experiment](core.md#ax.core.experiment.Experiment) | [None](https://docs.python.org/3/library/constants.html#None) = None, data: Data | [None](https://docs.python.org/3/library/constants.html#None) = None, transform_configs: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, status_quo_name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, status_quo_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, fit_out_of_design: [bool](https://docs.python.org/3/library/functions.html#bool) = False, fit_abandoned: [bool](https://docs.python.org/3/library/functions.html#bool) = False, fit_tracking_metrics: [bool](https://docs.python.org/3/library/functions.html#bool) = True, fit_on_init: [bool](https://docs.python.org/3/library/functions.html#bool) = True)

Bases: [`ModelBridge`](#ax.modelbridge.base.ModelBridge)

A model bridge for using purely random ‘models’.
Data and optimization configs are not required.

This model bridge interfaces with RandomModel.

#### model

A RandomModel used to generate candidates
(note: this an awkward use of the word ‘model’).

* **Type:**
  [ax.models.random.base.RandomModel](models.md#ax.models.random.base.RandomModel)

#### parameters

Params found in search space on modelbridge init.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### model *: [RandomModel](models.md#ax.models.random.base.RandomModel)*

#### parameters *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

### Torch Model Bridge

### *class* ax.modelbridge.torch.TorchModelBridge(experiment: [Experiment](core.md#ax.core.experiment.Experiment), search_space: SearchSpace, data: Data, model: [TorchModel](models.md#ax.models.torch_base.TorchModel), transforms: [list](https://docs.python.org/3/library/stdtypes.html#list)[[type](https://docs.python.org/3/library/functions.html#type)[[Transform](#ax.modelbridge.transforms.base.Transform)]], transform_configs: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, torch_dtype: dtype | [None](https://docs.python.org/3/library/constants.html#None) = None, torch_device: device | [None](https://docs.python.org/3/library/constants.html#None) = None, status_quo_name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, status_quo_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, fit_out_of_design: [bool](https://docs.python.org/3/library/functions.html#bool) = False, fit_abandoned: [bool](https://docs.python.org/3/library/functions.html#bool) = False, fit_tracking_metrics: [bool](https://docs.python.org/3/library/functions.html#bool) = True, fit_on_init: [bool](https://docs.python.org/3/library/functions.html#bool) = True, default_model_gen_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`ModelBridge`](#ax.modelbridge.base.ModelBridge)

A model bridge for using torch-based models.

Specifies an interface that is implemented by TorchModel. In particular,
model should have methods fit, predict, and gen. See TorchModel for the
API for each of these methods.

Requires that all parameters have been transformed to RangeParameters
or FixedParameters with float type and no log scale.

This class converts Ax parameter types to torch tensors before passing
them to the model.

#### evaluate_acquisition_function(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]], search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, pending_observations: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None, acq_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]

Evaluate the acquisition function for given set of observation
features.

* **Parameters:**
  * **observation_features** – Either a list or a list of lists of observation
    features, representing parameterizations, for which to evaluate the
    acquisition function. If a single list is passed, the acquisition
    function is evaluated for each observation feature. If a list of lists
    is passed each element (itself a list of observation features)
    represents a batch of points for which to evaluate the joint acquisition
    value.
  * **search_space** – Search space for fitting the model.
  * **optimization_config** – Optimization config defining how to optimize
    the model.
  * **pending_observations** – A map from metric name to pending observations for
    that metric.
  * **fixed_features** – An ObservationFeatures object containing any features that
    should be fixed at specified values during generation.
  * **acq_options** – Keyword arguments used to contruct the acquisition function.
* **Returns:**
  A list of acquisition function values, in the same order as the
  input observation features.

#### feature_importances(metric_name: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]

Computes feature importances for a single metric.

Depending on the type of the model, this method will approach sensitivity
analysis (calculating the sensitivity of the metric to changes in the search
space’s parameters, a.k.a. features) differently.

For Bayesian optimization models (BoTorch models), this method uses parameter
inverse lengthscales to compute normalized feature importances.

NOTE: Currently, this is only implemented for GP models.

* **Parameters:**
  **metric_name** – Name of metric to compute feature importances for.
* **Returns:**
  A dictionary mapping parameter names to their corresponding feature
  importances.

#### infer_objective_thresholds(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[ObjectiveThreshold]

Infer objective thresholds.

This method is only applicable for Multi-Objective optimization problems.

This method uses the model-estimated Pareto frontier over the in-sample points
to infer absolute (not relativized) objective thresholds.

This uses a heuristic that sets the objective threshold to be a scaled nadir
point, where the nadir point is scaled back based on the range of each
objective across the current in-sample Pareto frontier.

#### model *: [TorchModel](models.md#ax.models.torch_base.TorchModel) | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### model_best_point(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, pending_observations: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None, model_gen_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[Arm](core.md#ax.core.arm.Arm), [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]] | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None)

#### outcomes *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### parameters *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

### ax.modelbridge.torch.validate_optimization_config(optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig), outcomes: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [None](https://docs.python.org/3/library/constants.html#None)

Validate optimization config against model fitted outcomes.

* **Parameters:**
  * **optimization_config** – Config to validate.
  * **outcomes** – List of metric names w/ valid model fits.
* **Raises:**
  **ValueError if** – 
  1. Relative constraints are found
         2. Optimization metrics are not present in model fitted outcomes.

### Pairwise Model Bridge

### Map Torch Model Bridge

## Utilities

### General Utilities

### ax.modelbridge.modelbridge_utils.array_to_observation_data(f: ndarray, cov: ndarray, outcomes: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationData](core.md#ax.core.observation.ObservationData)]

Convert arrays of model predictions to a list of ObservationData.

* **Parameters:**
  * **f** – An (n x m) array
  * **cov** – An (n x m x m) array
  * **outcomes** – A list of d outcome names

Returns: A list of n ObservationData

### ax.modelbridge.modelbridge_utils.check_has_multi_objective_and_data(experiment: [Experiment](core.md#ax.core.experiment.Experiment), data: Data, optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)

Raise an error if not using a MultiObjective or if the data is empty.

### ax.modelbridge.modelbridge_utils.extract_objective_thresholds(objective_thresholds: [list](https://docs.python.org/3/library/stdtypes.html#list)[ObjectiveThreshold], objective: Objective, outcomes: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → ndarray | [None](https://docs.python.org/3/library/constants.html#None)

Extracts objective thresholds’ values, in the order of outcomes.

Will return None if no objective thresholds, otherwise the extracted tensor
will be the same length as outcomes.

Outcomes that are not part of an objective and the objectives that do no have
a corresponding objective threshold will be given a threshold of NaN. We will
later infer appropriate threshold values for the objectives that are given a
threshold of NaN.

* **Parameters:**
  * **objective_thresholds** – Objective thresholds to extract values from.
  * **objective** – The corresponding Objective, for validation purposes.
  * **outcomes** – n-length list of names of metrics.
* **Returns:**
  (n,) array of thresholds

### ax.modelbridge.modelbridge_utils.extract_objective_weights(objective: Objective, outcomes: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → ndarray

Extract a weights for objectives.

Weights are for a maximization problem.

Give an objective weight to each modeled outcome. Outcomes that are modeled
but not part of the objective get weight 0.

In the single metric case, the objective is given either +/- 1, depending
on the minimize flag.

In the multiple metric case, each objective is given the input weight,
multiplied by the minimize flag.

* **Parameters:**
  * **objective** – Objective to extract weights from.
  * **outcomes** – n-length list of names of metrics.
* **Returns:**
  n-length array of weights.

### ax.modelbridge.modelbridge_utils.extract_outcome_constraints(outcome_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint], outcomes: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray] | [None](https://docs.python.org/3/library/constants.html#None)

### ax.modelbridge.modelbridge_utils.extract_parameter_constraints(parameter_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ParameterConstraint](core.md#ax.core.parameter_constraint.ParameterConstraint)], param_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray] | [None](https://docs.python.org/3/library/constants.html#None)

Convert Ax parameter constraints into a tuple of NumPy arrays representing the
system of linear inequality constraints.

* **Parameters:**
  * **parameter_constraints** – A list of parameter constraint objects.
  * **param_names** – A list of parameter names.
* **Returns:**
  An optional tuple of NumPy arrays (A, b) representing the system of linear
  inequality constraints A x < b.

### ax.modelbridge.modelbridge_utils.extract_risk_measure(risk_measure: [RiskMeasure](core.md#ax.core.risk_measures.RiskMeasure)) → RiskMeasureMCObjective

Extracts the BoTorch risk measure objective from an Ax RiskMeasure.

* **Parameters:**
  **risk_measure** – The RiskMeasure object.
* **Returns:**
  The corresponding RiskMeasureMCObjective object.

### ax.modelbridge.modelbridge_utils.extract_robust_digest(search_space: SearchSpace, param_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → RobustSearchSpaceDigest | [None](https://docs.python.org/3/library/constants.html#None)

Extracts the RobustSearchSpaceDigest.

* **Parameters:**
  * **search_space** – A SearchSpace to digest.
  * **param_names** – A list of names of the parameters that are used in optimization.
    If environmental variables are present, these should be the last entries
    in param_names.
* **Returns:**
  If the search_space is not a RobustSearchSpace, this returns None.
  Otherwise, it returns a RobustSearchSpaceDigest with entries populated
  from the properties of the search_space. In particular, this constructs
  two optional callables, sample_param_perturbations and sample_environmental,
  that require no inputs and return a num_samples x d-dim array of samples
  from the corresponding parameter distributions, where d is the number of
  environmental variables for environmental_sampler and the number of
  non-environmental parameters in \`param_names for distribution_sampler.

### ax.modelbridge.modelbridge_utils.extract_search_space_digest(search_space: SearchSpace, param_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → SearchSpaceDigest

Extract basic parameter properties from a search space.

This is typically called with the transformed search space and makes certain
assumptions regarding the parameters being transformed.

For ChoiceParameters:
\* The choices are assumed to be numerical. ChoiceToNumericChoice
and OrderedChoiceToIntegerRange
transforms handle this.
\* If is_task, its index is added to task_features.
\* If ordered, its index is added to ordinal_features.
\* Otherwise, its index is added to categorical_features.
\* In all cases, the choices are added to discrete_choices.
\* The minimum and maximum value are added to the bounds.
\* The target_value is added to target_values.

For RangeParameters:
\* They’re assumed not to be in the log_scale. The Log transform handles this.
\* If integer, its index is added to ordinal_features and the choices are added
to discrete_choices.
\* The minimum and maximum value are added to the bounds.

If a parameter is_fidelity:
\* Its target_value is assumed to be numerical.
\* The target_value is added to target_values.
\* Its index is added to fidelity_features.

### ax.modelbridge.modelbridge_utils.feasible_hypervolume(optimization_config: [MultiObjectiveOptimizationConfig](core.md#ax.core.optimization_config.MultiObjectiveOptimizationConfig), values: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), ndarray]) → ndarray

Compute the feasible hypervolume each iteration.

* **Parameters:**
  * **optimization_config** – Optimization config.
  * **values** – Dictionary from metric name to array of value at each
    iteration (each array is n-dim). If optimization config contains
    outcome constraints, values for them must be present in values.

Returns: Array of feasible hypervolumes.

### ax.modelbridge.modelbridge_utils.get_fixed_features(fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None), param_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None)

Reformat a set of fixed_features.

### ax.modelbridge.modelbridge_utils.get_fixed_features_from_experiment(experiment: [Experiment](core.md#ax.core.experiment.Experiment)) → [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)

### ax.modelbridge.modelbridge_utils.get_pareto_frontier_and_configs(modelbridge: modelbridge_module.torch.TorchModelBridge, observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)], observation_data: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationData](core.md#ax.core.observation.ObservationData)] | [None](https://docs.python.org/3/library/constants.html#None) = None, objective_thresholds: TRefPoint | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [MultiObjectiveOptimizationConfig](core.md#ax.core.optimization_config.MultiObjectiveOptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, arm_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None, use_model_predictions: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)], Tensor, Tensor, Tensor | [None](https://docs.python.org/3/library/constants.html#None)]

Helper that applies transforms and calls `frontier_evaluator`.

Returns the `frontier_evaluator` configs in addition to the Pareto
observations.

* **Parameters:**
  * **modelbridge** – `Modelbridge` used to predict metrics outcomes.
  * **observation_features** – Observation features to consider for the Pareto
    frontier.
  * **observation_data** – Data for computing the Pareto front, unless
    `observation_features` are provided and `model_predictions is True`.
  * **objective_thresholds** – Metric values bounding the region of interest in
    the objective outcome space; used to override objective thresholds
    specified in `optimization_config`, if necessary.
  * **optimization_config** – Multi-objective optimization config.
  * **arm_names** – Arm names for each observation in `observation_features`.
  * **use_model_predictions** – If `True`, will use model predictions at
    `observation_features` to compute Pareto front. If `False`,
    will use `observation_data` directly to compute Pareto front, ignoring
    `observation_features`.

Returns: Four-item tuple of:
: - frontier_observations: Observations of points on the pareto frontier,
  - f: n x m tensor representation of the Pareto frontier values where n is the
    length of frontier_observations and m is the number of metrics,
  - obj_w: m tensor of objective weights,
  - obj_t: m tensor of objective thresholds corresponding to Y, or None if no
    objective thresholds used.

### ax.modelbridge.modelbridge_utils.hypervolume(modelbridge: modelbridge_module.torch.TorchModelBridge, observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)], objective_thresholds: TRefPoint | [None](https://docs.python.org/3/library/constants.html#None) = None, observation_data: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationData](core.md#ax.core.observation.ObservationData)] | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [MultiObjectiveOptimizationConfig](core.md#ax.core.optimization_config.MultiObjectiveOptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, selected_metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, use_model_predictions: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [float](https://docs.python.org/3/library/functions.html#float)

Helper function that computes (feasible) hypervolume.

* **Parameters:**
  * **modelbridge** – The modelbridge.
  * **observation_features** – The observation features for the in-sample arms.
  * **objective_thresholds** – The objective thresholds to be used for computing
    the hypervolume. If None, these are extracted from the optimization
    config.
  * **observation_data** – The observed outcomes for the in-sample arms.
  * **optimization_config** – The optimization config specifying the objectives,
    objectives thresholds, and outcome constraints.
  * **selected_metrics** – A list of objective metric names specifying which
    objectives to use in hypervolume computation. By default, all
    objectives are used.
  * **use_model_predictions** – A boolean indicating whether to use model predictions
    for determining the in-sample Pareto frontier instead of the raw observed
    values.
* **Returns:**
  The (feasible) hypervolume.

### ax.modelbridge.modelbridge_utils.observation_data_to_array(outcomes: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], observation_data: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationData](core.md#ax.core.observation.ObservationData)]) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray]

Convert a list of Observation data to arrays.

Any missing mean or covariance values will be returned as NaNs.

* **Parameters:**
  * **outcomes** – A list of m outcomes to extract observation data for.
  * **observation_data** – A list of n `ObservationData` objects.
* **Returns:**
  An (n x m) array of mean observations.
  - cov: An (n x m x m) array of covariance observations.
* **Return type:**
  - means

### ax.modelbridge.modelbridge_utils.observation_features_to_array(parameters: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], obsf: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → ndarray

Convert a list of Observation features to arrays.

### ax.modelbridge.modelbridge_utils.observed_hypervolume(modelbridge: modelbridge_module.torch.TorchModelBridge, objective_thresholds: TRefPoint | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [MultiObjectiveOptimizationConfig](core.md#ax.core.optimization_config.MultiObjectiveOptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, selected_metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [float](https://docs.python.org/3/library/functions.html#float)

Calculate hypervolume of a pareto frontier based on observed data.

Given observed data, return the hypervolume of the pareto frontier formed from
those outcomes.

* **Parameters:**
  * **modelbridge** – Modelbridge that holds previous training data.
  * **objective_thresholds** – Point defining the origin of hyperrectangles that
    can contribute to hypervolume. Note that if this is None,
    objective_thresholds must be present on the
    modelbridge.optimization_config.
  * **observation_features** – observation features to predict. Model’s training
    data used by default if unspecified.
  * **optimization_config** – Optimization config
  * **selected_metrics** – If specified, hypervolume will only be evaluated on
    the specified subset of metrics. Otherwise, all metrics will be used.
* **Returns:**
  (float) calculated hypervolume.

### ax.modelbridge.modelbridge_utils.observed_pareto_frontier(modelbridge: modelbridge_module.torch.TorchModelBridge, objective_thresholds: TRefPoint | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [MultiObjectiveOptimizationConfig](core.md#ax.core.optimization_config.MultiObjectiveOptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]

Generate a pareto frontier based on observed data. Given observed data
(sourced from model training data), return points on the Pareto frontier
as Observation-s.

* **Parameters:**
  * **modelbridge** – `Modelbridge` that holds previous training data.
  * **objective_thresholds** – Metric values bounding the region of interest in
    the objective outcome space; used to override objective thresholds
    in the optimization config, if needed.
  * **optimization_config** – Multi-objective optimization config.
* **Returns:**
  Data representing points on the pareto frontier.

### ax.modelbridge.modelbridge_utils.pareto_frontier(modelbridge: modelbridge_module.torch.TorchModelBridge, observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)], observation_data: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationData](core.md#ax.core.observation.ObservationData)] | [None](https://docs.python.org/3/library/constants.html#None) = None, objective_thresholds: TRefPoint | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [MultiObjectiveOptimizationConfig](core.md#ax.core.optimization_config.MultiObjectiveOptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, arm_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None, use_model_predictions: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]

Compute the list of points on the Pareto frontier as Observation-s
in the untransformed search space.

* **Parameters:**
  * **modelbridge** – `Modelbridge` used to predict metrics outcomes.
  * **observation_features** – Observation features to consider for the Pareto
    frontier.
  * **observation_data** – Data for computing the Pareto front, unless
    `observation_features` are provided and `model_predictions is True`.
  * **objective_thresholds** – Metric values bounding the region of interest in
    the objective outcome space; used to override objective thresholds
    specified in `optimization_config`, if necessary.
  * **optimization_config** – Multi-objective optimization config.
  * **arm_names** – Arm names for each observation in `observation_features`.
  * **use_model_predictions** – If `True`, will use model predictions at
    `observation_features` to compute Pareto front. If `False`,
    will use `observation_data` directly to compute Pareto front, ignoring
    `observation_features`.

Returns: Points on the Pareto frontier as Observation-s in order of descending
: individual hypervolume if possible.

### ax.modelbridge.modelbridge_utils.parse_observation_features(X: ndarray, param_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], candidate_metadata: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Re-format raw model-generated candidates into ObservationFeatures.

* **Parameters:**
  * **param_names** – List of param names.
  * **X** – Raw np.ndarray of candidate values.
  * **candidate_metadata** – Model’s metadata for candidates it produced.
* **Returns:**
  List of candidates, represented as ObservationFeatures.

### ax.modelbridge.modelbridge_utils.pending_observations_as_array_list(pending_observations: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]], outcome_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], param_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[ndarray] | [None](https://docs.python.org/3/library/constants.html#None)

Re-format pending observations.

* **Parameters:**
  * **pending_observations** – List of raw numpy pending observations.
  * **outcome_names** – List of outcome names.
  * **param_names** – List fitted param names.
* **Returns:**
  Filtered pending observations data, by outcome and param names.

### ax.modelbridge.modelbridge_utils.predicted_hypervolume(modelbridge: modelbridge_module.torch.TorchModelBridge, objective_thresholds: TRefPoint | [None](https://docs.python.org/3/library/constants.html#None) = None, observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)] | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [MultiObjectiveOptimizationConfig](core.md#ax.core.optimization_config.MultiObjectiveOptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None, selected_metrics: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [float](https://docs.python.org/3/library/functions.html#float)

Calculate hypervolume of a pareto frontier based on the posterior means of
given observation features.

Given a model and features to evaluate calculate the hypervolume of the pareto
frontier formed from their predicted outcomes.

* **Parameters:**
  * **modelbridge** – Modelbridge used to predict metrics outcomes.
  * **objective_thresholds** – point defining the origin of hyperrectangles that
    can contribute to hypervolume.
  * **observation_features** – observation features to predict. Model’s training
    data used by default if unspecified.
  * **optimization_config** – Optimization config
  * **selected_metrics** – If specified, hypervolume will only be evaluated on
    the specified subset of metrics. Otherwise, all metrics will be used.
* **Returns:**
  calculated hypervolume.

### ax.modelbridge.modelbridge_utils.predicted_pareto_frontier(modelbridge: modelbridge_module.torch.TorchModelBridge, objective_thresholds: TRefPoint | [None](https://docs.python.org/3/library/constants.html#None) = None, observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)] | [None](https://docs.python.org/3/library/constants.html#None) = None, optimization_config: [MultiObjectiveOptimizationConfig](core.md#ax.core.optimization_config.MultiObjectiveOptimizationConfig) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]

Generate a Pareto frontier based on the posterior means of given
observation features. Given a model and optionally features to evaluate
(will use model training data if not specified), use the model to predict
which points lie on the Pareto frontier.

* **Parameters:**
  * **modelbridge** – `Modelbridge` used to predict metrics outcomes.
  * **observation_features** – Observation features to predict, if provided and
    `use_model_predictions is True`.
  * **objective_thresholds** – Metric values bounding the region of interest in
    the objective outcome space; used to override objective thresholds
    specified in `optimization_config`, if necessary.
  * **optimization_config** – Multi-objective optimization config.
* **Returns:**
  Observations representing points on the Pareto frontier.

### ax.modelbridge.modelbridge_utils.process_contextual_datasets(datasets: [list](https://docs.python.org/3/library/stdtypes.html#list)[SupervisedDataset], outcomes: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], parameter_decomposition: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]], metric_decomposition: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[ContextualDataset]

Contruct a list of ContextualDataset.

* **Parameters:**
  * **datasets** – A list of Dataset objects.
  * **outcomes** – The names of the outcomes to extract observations for.
  * **parameter_decomposition** – Keys are context names. Values are the lists
    of parameter names belonging to the context, e.g.
    {‘context1’: [‘p1_c1’, ‘p2_c1’],’context2’: [‘p1_c2’, ‘p2_c2’]}.
  * **metric_decomposition** – 

    Context breakdown metrics. Keys are context names.
    Values are the lists of metric names belonging to the context:
    {
    > ’context1’: [‘m1_c1’, ‘m2_c1’, ‘m3_c1’],
    > ‘context2’: [‘m1_c2’, ‘m2_c2’, ‘m3_c2’],

    }

Returns: A list of ContextualDataset objects. Order generally will not be that of
: outcomes.

### ax.modelbridge.modelbridge_utils.transform_callback(param_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], transforms: [MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Transform](#ax.modelbridge.transforms.base.Transform)]) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[ndarray], ndarray]

A closure for performing the round trip transformations.

The function round points by de-transforming points back into
the original space (done by applying transforms in reverse), and then
re-transforming them.
This function is specifically for points which are formatted as numpy
arrays. This function is passed to \_model_gen.

* **Parameters:**
  * **param_names** – Names of parameters to transform.
  * **transforms** – Ordered set of transforms which were applied to the points.
* **Returns:**
  a function with for performing the roundtrip transform.

### ax.modelbridge.modelbridge_utils.transform_search_space(search_space: SearchSpace, transforms: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[type](https://docs.python.org/3/library/functions.html#type)[[Transform](#ax.modelbridge.transforms.base.Transform)]], transform_configs: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → SearchSpace

Apply all given transforms to a copy of the SearchSpace iteratively.

### ax.modelbridge.modelbridge_utils.validate_and_apply_final_transform(objective_weights: ~numpy.ndarray, outcome_constraints: tuple[~numpy.ndarray, ~numpy.ndarray] | None, linear_constraints: tuple[~numpy.ndarray, ~numpy.ndarray] | None, pending_observations: list[~numpy.ndarray] | None, objective_thresholds: ~numpy.ndarray | None = None, final_transform: ~collections.abc.Callable[[~numpy.ndarray], ~torch.Tensor] = <built-in method tensor of type object>) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None), [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[Tensor, Tensor] | [None](https://docs.python.org/3/library/constants.html#None), [list](https://docs.python.org/3/library/stdtypes.html#list)[Tensor] | [None](https://docs.python.org/3/library/constants.html#None), Tensor | [None](https://docs.python.org/3/library/constants.html#None)]

### Prediction Utilities

### Cross Validation

### Model Selection

### Dispatch Utilities

## Transforms

### ax.modelbridge.transforms.deprecated_transform_mixin

### ax.modelbridge.transforms.base

### *class* ax.modelbridge.transforms.base.Transform(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: TConfig | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Defines the API for a transform that is applied to search_space,
observation_features, observation_data, and optimization_config.

Transforms are used to adapt the search space and data into the types
and structures expected by the model. When Transforms are used (for
instance, in ModelBridge), it is always assumed that they may potentially
mutate the transformed object in-place.

Forward transforms are defined for all four of those quantities. Reverse
transforms are defined for observation_data and observation.

The forward transform for observation features must accept a partial
observation with not all features recorded.

Forward and reverse transforms for observation data accept a list of
observation features as an input, but they will not be mutated.

The forward transform for optimization config accepts the modelbridge and
fixed features as inputs, but they will not be mutated.

This class provides an identify transform.

#### config *: TConfig*

#### modelbridge *: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None)*

#### transform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Transform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features

Returns: transformed observation features

#### transform_observations(observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]

Transform observations.

Typically done in place. By default, the effort is split into separate
transformations of the features and the data.

* **Parameters:**
  **observations** – Observations.

Returns: transformed observations.

#### transform_optimization_config(optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig), modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig)

Transform optimization config.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **optimization_config** – The optimization config

Returns: transformed optimization config.

#### transform_search_space(search_space: SearchSpace) → SearchSpace

Transform search space.

The transforms are typically done in-place. This calls two private methods,
\_transform_search_space, which transforms the core search space attributes,
and \_transform_parameter_distributions, which transforms the distributions
when using a RobustSearchSpace.

* **Parameters:**
  **search_space** – The search space

Returns: transformed search space.

#### untransform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Untransform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features in the transformed space

Returns: observation features in the original space

#### untransform_observations(observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]

Untransform observations.

Typically done in place. By default, the effort is split into separate
backwards transformations of the features and the data.

* **Parameters:**
  **observations** – Observations.

Returns: untransformed observations.

#### untransform_outcome_constraints(outcome_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint], fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint]

Untransform outcome constraints.

If outcome constraints are modified in transform_optimization_config,
this method should reverse the portion of that transformation that was
applied to the outcome constraints.

### ax.modelbridge.transforms.cast

### *class* ax.modelbridge.transforms.cast.Cast(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Cast each param value to the respective parameter’s type/format and
to a flattened version of the hierarchical search space, if applicable.

This is a default transform that should run across all models.

NOTE: In case where searh space is hierarchical and this transform is
configured to flatten it:

> * All calls to Cast.transform_… transform Ax objects defined in
>   terms of hierarchical search space, to their definitions in terms of
>   flattened search space.
> * All calls to Cast.untransform_… cast Ax objects back to a
>   hierarchical search space.
> * The hierarchical search space is seen as the “original” search space,
>   and the flattened search space –– as “transformed”.

Transform is done in-place for casting types, but objects are copied
during flattening of- and casting to the hierarchical search space.

#### transform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Transform observation features by adding parameter values that
were removed during casting of observation features to hierarchical
search space.

* **Parameters:**
  **observation_features** – Observation features

Returns: transformed observation features

#### untransform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Untransform observation features by casting parameter values to their
expected types and removing parameter values that are not applicable given
the values of other parameters and the hierarchical structure of the search
space.

* **Parameters:**
  **observation_features** – Observation features in the transformed space

Returns: observation features in the original space

### ax.modelbridge.transforms.cap_parameter

### *class* ax.modelbridge.transforms.cap_parameter.CapParameter(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Cap parameter range(s) to given values. Expects a configuration of form
{ parameter_name -> new_upper_range_value }.

This transform only transforms the search space.

### ax.modelbridge.transforms.centered_unit_x

### *class* ax.modelbridge.transforms.centered_unit_x.CenteredUnitX(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`UnitX`](#ax.modelbridge.transforms.unit_x.UnitX)

Map X to [-1, 1]^d for RangeParameter of type float and not log scale.

Transform is done in-place.

#### target_lb *: [float](https://docs.python.org/3/library/functions.html#float)* *= -1.0*

#### target_range *: [float](https://docs.python.org/3/library/functions.html#float)* *= 2.0*

### ax.modelbridge.transforms.choice_encode

### *class* ax.modelbridge.transforms.choice_encode.ChoiceEncode(\*args: [Any](https://docs.python.org/3/library/typing.html#typing.Any), \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

Bases: `DeprecatedTransformMixin`, [`ChoiceToNumericChoice`](#ax.modelbridge.transforms.choice_encode.ChoiceToNumericChoice)

Deprecated alias for ChoiceToNumericChoice.

### *class* ax.modelbridge.transforms.choice_encode.ChoiceToNumericChoice(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Convert general ChoiceParameters to integer or float ChoiceParameters.

If the parameter type is numeric (int, float) and the parameter is ordered,
then the values are normalized to the unit interval while retaining relative
spacing. If the parameter type is unordered (categorical) or ordered but
non-numeric, this transform uses an integer encoding to 0, 1, …, n_choices - 1.
The resulting choice parameter will be considered ordered iff the original
parameter is.

In the inverse transform, parameters will be mapped back onto the original domain.

This transform does not transform task parameters
(use TaskChoiceToIntTaskChoice for this).

Note that this behavior is different from that of OrderedChoiceToIntegerRange, which
transforms (ordered) ChoiceParameters to integer RangeParameters (rather than
ChoiceParameters).

Transform is done in-place.

#### transform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Transform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features

Returns: transformed observation features

#### untransform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Untransform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features in the transformed space

Returns: observation features in the original space

### *class* ax.modelbridge.transforms.choice_encode.OrderedChoiceEncode(\*args: [Any](https://docs.python.org/3/library/typing.html#typing.Any), \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

Bases: `DeprecatedTransformMixin`, [`OrderedChoiceToIntegerRange`](#ax.modelbridge.transforms.choice_encode.OrderedChoiceToIntegerRange)

Deprecated alias for OrderedChoiceToIntegerRange.

### *class* ax.modelbridge.transforms.choice_encode.OrderedChoiceToIntegerRange(search_space: SearchSpace, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)], modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`ChoiceToNumericChoice`](#ax.modelbridge.transforms.choice_encode.ChoiceToNumericChoice)

Convert ordered ChoiceParameters to integer RangeParameters.

Parameters will be transformed to an integer RangeParameters, mapped from the
original choice domain to a contiguous range 0, 1, …, n_choices - 1
of integers. Does not transform task parameters.

In the inverse transform, parameters will be mapped back onto the original domain.

In order to encode all ChoiceParameters (not just ordered ChoiceParameters),
use ChoiceToNumericChoice instead.

Transform is done in-place.

### ax.modelbridge.transforms.choice_encode.transform_choice_values(p: [ChoiceParameter](core.md#ax.core.parameter.ChoiceParameter)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, [ParameterType](core.md#ax.core.parameter.ParameterType)]

Transforms the choice values and returns the new parameter type.

If the choices were numeric (int or float) and ordered, then they’re cast
to float and rescaled to [0, 1]. Otherwise, they’re cast to integers
0, 1, …, n_choices - 1.

### ax.modelbridge.transforms.convert_metric_names

### *class* ax.modelbridge.transforms.convert_metric_names.ConvertMetricNames(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Convert all metric names to canonical name as specified on a
multi_type_experiment.

For example, a multi-type experiment may have an offline simulator which attempts to
approximate observations from some online system. We want to map the offline
metric names to the corresponding online ones so the model can associate them.

This is done by replacing metric names in the data with the corresponding
online metric names.

In the inverse transform, data will be mapped back onto the original metric names.
By default, this transform is turned off. It can be enabled by passing the
“perform_untransform” flag to the config.

#### untransform_observations(observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]

Untransform observations.

Typically done in place. By default, the effort is split into separate
backwards transformations of the features and the data.

* **Parameters:**
  **observations** – Observations.

Returns: untransformed observations.

### ax.modelbridge.transforms.convert_metric_names.convert_mt_observations(observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)], experiment: [MultiTypeExperiment](core.md#ax.core.multi_type_experiment.MultiTypeExperiment)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]

Apply ConvertMetricNames transform to observations for a MT experiment.

### ax.modelbridge.transforms.convert_metric_names.tconfig_from_mt_experiment(experiment: [MultiTypeExperiment](core.md#ax.core.multi_type_experiment.MultiTypeExperiment)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)]

Generate the TConfig for this transform given a multi_type_experiment.

* **Parameters:**
  **experiment** – The experiment from which to generate the config.
* **Returns:**
  The transform config to pass into the ConvertMetricNames constructor.

### ax.modelbridge.transforms.derelativize

### *class* ax.modelbridge.transforms.derelativize.Derelativize(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: TConfig | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Changes relative constraints to not-relative constraints using a plug-in
estimate of the status quo value.

If status quo is in-design, uses model estimate at status quo. If not, uses
raw observation at status quo.

Will raise an error if status quo is in-design and model fails to predict
for it, unless the flag “use_raw_status_quo” is set to True in the
transform config, in which case it will fall back to using the observed
value in the training data.

Transform is done in-place.

#### transform_optimization_config(optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig), modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig)

Transform optimization config.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **optimization_config** – The optimization config

Returns: transformed optimization config.

#### untransform_outcome_constraints(outcome_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint], fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint]

Untransform outcome constraints.

If outcome constraints are modified in transform_optimization_config,
this method should reverse the portion of that transformation that was
applied to the outcome constraints.

### ax.modelbridge.transforms.int_range_to_choice

### *class* ax.modelbridge.transforms.int_range_to_choice.IntRangeToChoice(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Convert a RangeParameter of type int to a ordered ChoiceParameter.

Transform is done in-place.

### ax.modelbridge.transforms.int_to_float

### *class* ax.modelbridge.transforms.int_to_float.IntToFloat(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Convert a RangeParameter of type int to type float.

Uses either randomized_rounding or default python rounding,
depending on ‘rounding’ flag.

The min_choices config can be used to transform only the parameters
with cardinality greater than or equal to min_choices; with the exception
of log_scale parameters, which are always transformed.

Transform is done in-place.

#### transform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Transform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features

Returns: transformed observation features

#### untransform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Untransform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features in the transformed space

Returns: observation features in the original space

### ax.modelbridge.transforms.ivw

### *class* ax.modelbridge.transforms.ivw.IVW(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: TConfig | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

If an observation data contains multiple observations of a metric, they
are combined using inverse variance weighting.

### ax.modelbridge.transforms.ivw.ivw_metric_merge(obsd: [ObservationData](core.md#ax.core.observation.ObservationData), conflicting_noiseless: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'warn') → [ObservationData](core.md#ax.core.observation.ObservationData)

Merge multiple observations of a metric with inverse variance weighting.

Correctly updates the covariance of the new merged estimates:
ybar1 = Sum_i w_i \* y_i
ybar2 = Sum_j w_j \* y_j
cov[ybar1, ybar2] = Sum_i Sum_j w_i \* w_j \* cov[y_i, y_j]

w_i will be infinity if any variance is 0. If one variance is 0., then
the IVW estimate is the corresponding mean. If there are multiple
measurements with 0 variance but means are all the same, then IVW estimate
is that mean. If there are multiple measurements and means differ, behavior
depends on argument conflicting_noiseless. “ignore” and “warn” will use
the first of the measurements as the IVW estimate. “warn” will additionally
log a warning. “raise” will raise an exception.

* **Parameters:**
  * **obsd** – An ObservationData object
  * **conflicting_noiseless** – “warn”, “ignore”, or “raise”

### ax.modelbridge.transforms.inverse_gaussian_cdf_y

### *class* ax.modelbridge.transforms.inverse_gaussian_cdf_y.InverseGaussianCdfY(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: base_modelbridge.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Apply inverse CDF transform to Y.

This means that we model uniform distributions as gaussian-distributed.

### ax.modelbridge.transforms.log

### *class* ax.modelbridge.transforms.log.Log(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Apply log base 10 to a float RangeParameter domain.

Transform is done in-place.

#### transform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Transform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features

Returns: transformed observation features

#### untransform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Untransform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features in the transformed space

Returns: observation features in the original space

### ax.modelbridge.transforms.log_y

### *class* ax.modelbridge.transforms.log_y.LogY(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: base_modelbridge.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: TConfig | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Apply (natural) log-transform to Y.

This essentially means that we are model the observations as log-normally
distributed. If config specifies match_ci_width=True, use a matching
procedure based on the width of the CIs, otherwise (the default), use the
delta method,

Transform is applied only for the metrics specified in the transform config.
Transform is done in-place.

#### transform_optimization_config(optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig), modelbridge: base_modelbridge.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig)

Transform optimization config.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **optimization_config** – The optimization config

Returns: transformed optimization config.

#### untransform_outcome_constraints(outcome_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint], fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint]

Untransform outcome constraints.

If outcome constraints are modified in transform_optimization_config,
this method should reverse the portion of that transformation that was
applied to the outcome constraints.

### ax.modelbridge.transforms.log_y.lognorm_to_norm(mu_ln: ndarray, Cov_ln: ndarray) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray]

Compute mean and covariance of a MVN from those of the associated log-MVN

If Y is log-normal with mean mu_ln and covariance Cov_ln, then
X ~ N(mu_n, Cov_n) with

> Cov_n_{ij} = log(1 + Cov_ln_{ij} / (mu_ln_{i} \* mu_n_{j}))
> mu_n_{i} = log(mu_ln_{i}) - 0.5 \* log(1 + Cov_ln_{ii} / mu_ln_{i}\*\*2)

### ax.modelbridge.transforms.log_y.match_ci_width(mean: ndarray, variance: ndarray, transform: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[ndarray], ndarray], level: [float](https://docs.python.org/3/library/functions.html#float) = 0.95) → ndarray

### ax.modelbridge.transforms.log_y.norm_to_lognorm(mu_n: ndarray, Cov_n: ndarray) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ndarray, ndarray]

Compute mean and covariance of a log-MVN from its MVN sufficient statistics

If X ~ N(mu_n, Cov_n) and Y = exp(X), then Y is log-normal with

> mu_ln_{i} = exp(mu_n_{i}) + 0.5 \* Cov_n_{ii}
> Cov_ln_{ij} = exp(mu_n_{i} + mu_n_{j} + 0.5 \* (Cov_n_{ii} + Cov_n_{jj})) \*
> (exp(Cov_n_{ij}) - 1)

### ax.modelbridge.transforms.logit

### *class* ax.modelbridge.transforms.logit.Logit(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Apply logit transform to a float RangeParameter domain.

Transform is done in-place.

#### transform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Transform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features

Returns: transformed observation features

#### untransform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Untransform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features in the transformed space

Returns: observation features in the original space

### ax.modelbridge.transforms.map_unit_x

### *class* ax.modelbridge.transforms.map_unit_x.MapUnitX(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: TConfig | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`UnitX`](#ax.modelbridge.transforms.unit_x.UnitX)

A UnitX transform for map parameters in observation_features, identified
as those that are not part of the search space. Since they are not part of the
search space, the bounds are inferred from the set of observation features. Only
observation features are transformed; all other objects undergo identity transform.

#### target_lb *: [float](https://docs.python.org/3/library/functions.html#float)* *= 0.0*

#### target_range *: [float](https://docs.python.org/3/library/functions.html#float)* *= 1.0*

#### untransform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Untransform if the parameter exists in the observation feature. Note the
extra existence check from UnitX.untransform_observation_features because
when map key features are used, they may not exist after generation or best
point computations.

### ax.modelbridge.transforms.merge_repeated_measurements

### *class* ax.modelbridge.transforms.merge_repeated_measurements.MergeRepeatedMeasurements(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: [ModelBridge](#ax.modelbridge.base.ModelBridge) | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Merge repeated measurements for to obtain one observation per arm.

Repeated measurements are merged via inverse variance weighting (e.g. over
different trials). This intentionally ignores the trial index and assumes
stationarity.

TODO: Support inverse variance weighting correlated outcomes (full covariance).

Note: this is not reversible.

#### transform_observations(observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]

Transform observations.

Typically done in place. By default, the effort is split into separate
transformations of the features and the data.

* **Parameters:**
  **observations** – Observations.

Returns: transformed observations.

### ax.modelbridge.transforms.metrics_as_task

### *class* ax.modelbridge.transforms.metrics_as_task.MetricsAsTask(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Convert metrics to a task parameter.

For each metric to be used as a task, the config must specify a list of the
target metrics for that particular task metric. So,

config = {
: ‘metric_task_map’: {
  : ‘metric1’: [‘metric2’, ‘metric3’],
    ‘metric2’: [‘metric3’],
  <br/>
  }

}

means that metric2 will be given additional task observations of metric1,
and metric3 will be given additional task observations of both metric1 and
metric2. Note here that metric2 and metric3 are the target tasks, and this
map is from base tasks to target tasks.

#### transform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

If transforming features without data, map them to the target.

#### transform_observations(observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]

Transform observations.

Typically done in place. By default, the effort is split into separate
transformations of the features and the data.

* **Parameters:**
  **observations** – Observations.

Returns: transformed observations.

#### untransform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Untransform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features in the transformed space

Returns: observation features in the original space

#### untransform_observations(observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]

Untransform observations.

Typically done in place. By default, the effort is split into separate
backwards transformations of the features and the data.

* **Parameters:**
  **observations** – Observations.

Returns: untransformed observations.

### ax.modelbridge.transforms.one_hot

### *class* ax.modelbridge.transforms.one_hot.OneHot(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Convert categorical parameters (unordered ChoiceParameters) to
one-hot-encoded parameters.

Does not convert task parameters.

Parameters will be one-hot-encoded, yielding a set of RangeParameters,
of type float, on [0, 1]. If there are two values, one single RangeParameter
will be yielded, otherwise there will be a new RangeParameter for each
ChoiceParameter value.

In the reverse transform, floats can be converted to a one-hot encoded vector
using one of two methods:

Strict rounding: Choose the maximum value. With levels [‘a’, ‘b’, ‘c’] and
: float values [0.2, 0.4, 0.3], the restored parameter would be set to ‘b’.
  Ties are broken randomly, so values [0.2, 0.4, 0.4] is randomly set to ‘b’
  or ‘c’.

Randomized rounding: Sample from the distribution. Float values
: [0.2, 0.4, 0.3] are transformed to ‘a’ w.p.
  0.2/0.9, ‘b’ w.p. 0.4/0.9, or ‘c’ w.p. 0.3/0.9.

Type of rounding can be set using transform_config[‘rounding’] to either
‘strict’ or ‘randomized’. Defaults to strict.

Transform is done in-place.

#### transform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Transform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features

Returns: transformed observation features

#### untransform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Untransform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features in the transformed space

Returns: observation features in the original space

### *class* ax.modelbridge.transforms.one_hot.OneHotEncoder(values: [list](https://docs.python.org/3/library/stdtypes.html#list)[[None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)])

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

OneHot encodes a list of labels.

#### inverse_transform(encoded_label: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]) → [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)

Inverse transorm a one hot encoded label.

#### transform(label: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]

One hot encode a given label.

### ax.modelbridge.transforms.percentile_y

### *class* ax.modelbridge.transforms.percentile_y.PercentileY(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Map Y values to percentiles based on their empirical CDF.

### ax.modelbridge.transforms.power_transform_y

### ax.modelbridge.transforms.remove_fixed

### *class* ax.modelbridge.transforms.remove_fixed.RemoveFixed(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Remove fixed parameters.

Fixed parameters should not be included in the SearchSpace.
This transform removes these parameters, leaving only tunable parameters.

Transform is done in-place for observation features.

#### transform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Transform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features

Returns: transformed observation features

#### untransform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Untransform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features in the transformed space

Returns: observation features in the original space

### ax.modelbridge.transforms.rounding

### ax.modelbridge.transforms.rounding.contains_constrained_integer(search_space: SearchSpace, transform_parameters: [set](https://docs.python.org/3/library/stdtypes.html#set)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [bool](https://docs.python.org/3/library/functions.html#bool)

Check if any integer parameters are present in parameter_constraints.

Order constraints are ignored since strict rounding preserves ordering.

### ax.modelbridge.transforms.rounding.randomized_onehot_round(x: ndarray) → ndarray

Randomized rounding of x to a one-hot vector.
x should be 0 <= x <= 1. If x includes negative values,
they will be rounded to zero.

### ax.modelbridge.transforms.rounding.randomized_round(x: [float](https://docs.python.org/3/library/functions.html#float)) → [int](https://docs.python.org/3/library/functions.html#int)

Randomized round of x

### ax.modelbridge.transforms.rounding.randomized_round_parameters(parameters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], transform_parameters: [set](https://docs.python.org/3/library/stdtypes.html#set)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)]

### ax.modelbridge.transforms.rounding.strict_onehot_round(x: ndarray) → ndarray

Round x to a one-hot vector by selecting the max element.
Ties broken randomly.

### ax.modelbridge.transforms.search_space_to_choice

### *class* ax.modelbridge.transforms.search_space_to_choice.SearchSpaceToChoice(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Replaces the search space with a single choice parameter, whose values
are the signatures of the arms observed in the data.

This transform is meant to be used with ThompsonSampler.

Choice parameter will be unordered unless config[“use_ordered”] specifies
otherwise.

Transform is done in-place.

#### transform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Transform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features

Returns: transformed observation features

#### untransform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Untransform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features in the transformed space

Returns: observation features in the original space

### ax.modelbridge.transforms.search_space_to_float

---

<a id="module-ax.modelbridge.transforms.standardize_y"></a>

### *class* ax.modelbridge.transforms.standardize_y.StandardizeY(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: base_modelbridge.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Standardize Y, separately for each metric.

Transform is done in-place.

#### transform_optimization_config(optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig), modelbridge: base_modelbridge.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig)

Transform optimization config.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **optimization_config** – The optimization config

Returns: transformed optimization config.

#### untransform_outcome_constraints(outcome_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint], fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint]

Untransform outcome constraints.

If outcome constraints are modified in transform_optimization_config,
this method should reverse the portion of that transformation that was
applied to the outcome constraints.

### ax.modelbridge.transforms.standardize_y.compute_standardization_parameters(Ys: [defaultdict](https://docs.python.org/3/library/collections.html#collections.defaultdict)[[str](https://docs.python.org/3/library/stdtypes.html#str) | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bool](https://docs.python.org/3/library/functions.html#bool) | [float](https://docs.python.org/3/library/functions.html#float) | [int](https://docs.python.org/3/library/functions.html#int)], [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]]) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str) | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [float](https://docs.python.org/3/library/functions.html#float)], [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str) | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [float](https://docs.python.org/3/library/functions.html#float)]]

Compute mean and std. dev of Ys.

### ax.modelbridge.transforms.stratified_standardize_y

### *class* ax.modelbridge.transforms.stratified_standardize_y.StratifiedStandardizeY(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Standardize Y, separately for each metric and for each value of a
ChoiceParameter.

The name of the parameter by which to stratify the standardization can be
specified in config[“parameter_name”]. If not specified, will use a task
parameter if search space contains exactly 1 task parameter, and will raise
an exception otherwise.

The stratification parameter must be fixed during generation if there are
outcome constraints, in order to apply the standardization to the
constraints.

Transform is done in-place.

#### transform_observations(observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]

Transform observations.

Typically done in place. By default, the effort is split into separate
transformations of the features and the data.

* **Parameters:**
  **observations** – Observations.

Returns: transformed observations.

#### transform_optimization_config(optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig), modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig)

Transform optimization config.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **optimization_config** – The optimization config

Returns: transformed optimization config.

#### untransform_observations(observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)]

Untransform observations.

Typically done in place. By default, the effort is split into separate
backwards transformations of the features and the data.

* **Parameters:**
  **observations** – Observations.

Returns: untransformed observations.

#### untransform_outcome_constraints(outcome_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint], fixed_features: [ObservationFeatures](core.md#ax.core.observation.ObservationFeatures) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[OutcomeConstraint]

Untransform outcome constraints.

If outcome constraints are modified in transform_optimization_config,
this method should reverse the portion of that transformation that was
applied to the outcome constraints.

### ax.modelbridge.transforms.task_encode

### *class* ax.modelbridge.transforms.task_encode.TaskChoiceToIntTaskChoice(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`OrderedChoiceToIntegerRange`](#ax.modelbridge.transforms.choice_encode.OrderedChoiceToIntegerRange)

Convert task ChoiceParameters to integer-valued ChoiceParameters.

Parameters will be transformed to an integer ChoiceParameter with
property is_task=True, mapping values from the original choice domain to a
contiguous range integers 0, 1, …, n_choices-1.

In the inverse transform, parameters will be mapped back onto the original domain.

Transform is done in-place.

### *class* ax.modelbridge.transforms.task_encode.TaskEncode(\*args: [Any](https://docs.python.org/3/library/typing.html#typing.Any), \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

Bases: `DeprecatedTransformMixin`, [`TaskChoiceToIntTaskChoice`](#ax.modelbridge.transforms.task_encode.TaskChoiceToIntTaskChoice)

Deprecated alias for TaskChoiceToIntTaskChoice.

### ax.modelbridge.transforms.time_as_feature

### ax.modelbridge.transforms.transform_to_new_sq

### ax.modelbridge.transforms.trial_as_task

### *class* ax.modelbridge.transforms.trial_as_task.TrialAsTask(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Convert trial to one or more task parameters.

How trial is mapped to parameter is specified with a map like
{parameter_name: {trial_index: level name}}.
For example,
{“trial_param1”: {0: “level1”, 1: “level1”, 2: “level2”},}
will create choice parameters “trial_param1” with is_task=True.
Observations with trial 0 or 1 will have “trial_param1” set to “level1”,
and those with trial 2 will have “trial_param1” set to “level2”. Multiple
parameter names and mappings can be specified in this dict.

The trial level mapping can be specified in config[“trial_level_map”]. If
not specified, defaults to a parameter with a level for every trial index.

For the reverse transform, if there are multiple mappings in the transform
the trial will not be set.

The created parameter will be given a target value that will default to the
lowest trial index in the mapping, or can be provided in config[“target_trial”].

Will raise if trial not specified for every point in the training data.

Transform is done in-place.

#### transform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Transform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features

Returns: transformed observation features

#### untransform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Untransform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features in the transformed space

Returns: observation features in the original space

### ax.modelbridge.transforms.unit_x

### *class* ax.modelbridge.transforms.unit_x.UnitX(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Map X to [0, 1]^d for RangeParameter of type float and not log scale.

Uses bounds l <= x <= u, sets x_tilde_i = (x_i - l_i) / (u_i - l_i).
Constraints wTx <= b are converted to gTx_tilde <= h, where
g_i = w_i (u_i - l_i) and h = b - wTl.

Transform is done in-place.

#### target_lb *: [float](https://docs.python.org/3/library/functions.html#float)* *= 0.0*

#### target_range *: [float](https://docs.python.org/3/library/functions.html#float)* *= 1.0*

#### transform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Transform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features

Returns: transformed observation features

#### untransform_observation_features(observation_features: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationFeatures](core.md#ax.core.observation.ObservationFeatures)]

Untransform observation features.

This is typically done in-place. This class implements the identity
transform (does nothing).

* **Parameters:**
  **observation_features** – Observation features in the transformed space

Returns: observation features in the original space

### ax.modelbridge.transforms.utils

### *class* ax.modelbridge.transforms.utils.ClosestLookupDict(\*args: [Any](https://docs.python.org/3/library/typing.html#typing.Any), \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

Bases: [`dict`](https://docs.python.org/3/library/stdtypes.html#dict)

A dictionary with numeric keys that looks up the closest key.

### ax.modelbridge.transforms.utils.construct_new_search_space(search_space: SearchSpace, parameters: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Parameter](core.md#ax.core.parameter.Parameter)], parameter_constraints: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ParameterConstraint](core.md#ax.core.parameter_constraint.ParameterConstraint)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → SearchSpace

Construct a search space with the transformed arguments.

If the search_space is a RobustSearchSpace, this will use its
environmental variables and distributions, and remove the environmental
variables from parameters before constructing.

* **Parameters:**
  * **parameters** – List of transformed parameter objects.
  * **parameter_constraints** – List of parameter constraints.
* **Returns:**
  The new search space instance.

### ax.modelbridge.transforms.utils.derelativize_optimization_config_with_raw_status_quo(optimization_config: [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig), modelbridge: modelbridge_module.base.ModelBridge, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None)) → [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig)

Derelativize optimization_config using raw status-quo values

### ax.modelbridge.transforms.utils.get_data(observation_data: [list](https://docs.python.org/3/library/stdtypes.html#list)[[ObservationData](core.md#ax.core.observation.ObservationData)], metric_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, raise_on_non_finite_data: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [list](https://docs.python.org/3/library/stdtypes.html#list)[[float](https://docs.python.org/3/library/functions.html#float)]]

Extract all metrics if metric_names is None.

Raises a value error if any data is non-finite.

* **Parameters:**
  * **observation_data** – List of observation data.
  * **metric_names** – List of metric names.
  * **raise_on_non_finite_data** – If true, raises an exception on nan/inf.
* **Returns:**
  A dictionary mapping metric names to lists of metric values.

### ax.modelbridge.transforms.utils.match_ci_width_truncated(mean: [float](https://docs.python.org/3/library/functions.html#float), variance: [float](https://docs.python.org/3/library/functions.html#float), transform: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[float](https://docs.python.org/3/library/functions.html#float)], [float](https://docs.python.org/3/library/functions.html#float)], level: [float](https://docs.python.org/3/library/functions.html#float) = 0.95, margin: [float](https://docs.python.org/3/library/functions.html#float) = 0.001, lower_bound: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, upper_bound: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, clip_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]

Estimate a transformed variance using the match ci width method.

See log_y transform for the original. Here, bounds are forced to lie
within a [lower_bound, upper_bound] interval after transformation.

### ax.modelbridge.winsorize

### *class* ax.modelbridge.transforms.winsorize.Winsorize(search_space: SearchSpace | [None](https://docs.python.org/3/library/constants.html#None) = None, observations: [list](https://docs.python.org/3/library/stdtypes.html#list)[[Observation](core.md#ax.core.observation.Observation)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modelbridge: modelbridge_module.base.ModelBridge | [None](https://docs.python.org/3/library/constants.html#None) = None, config: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [str](https://docs.python.org/3/library/stdtypes.html#str) | AcquisitionFunction | [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[int](https://docs.python.org/3/library/functions.html#int), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [OptimizationConfig](core.md#ax.core.optimization_config.OptimizationConfig) | [WinsorizationConfig](models.md#ax.models.winsorization_config.WinsorizationConfig) | [None](https://docs.python.org/3/library/constants.html#None)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`Transform`](#ax.modelbridge.transforms.base.Transform)

Clip the mean values for each metric to lay within the limits provided in
the config. The config can contain either or both of two keys:
- `"winsorization_config"`, corresponding to either a single

> `WinsorizationConfig`, which, if provided will be used for all metrics; or
> a mapping `Dict[str, WinsorizationConfig]` between each metric name and its
> `WinsorizationConfig`.
- `"derelativize_with_raw_status_quo"`, indicating whether to use the raw
  : status-quo value for any derelativization. Note this defaults to `False`,
    which is unsupported and simply fails if derelativization is necessary. The
    user must specify `derelativize_with_raw_status_quo = True` in order for
    derelativization to succeed. Note that this must match the use_raw_status_quo
    value in the `Derelativize` config if used.

For example,
`{"winsorization_config": WinsorizationConfig(lower_quantile_margin=0.3)}`
will specify the same 30% winsorization from below for all metrics, whereas

```
``
```

\`
{

> “winsorization_config”:
> {

> > “metric_1”: WinsorizationConfig(lower_quantile_margin=0.2),
> > “metric_2”: WinsorizationConfig(upper_quantile_margin=0.1),

> }

#### }

will winsorize 20% from below for metric_1 and 10% from above from metric_2.
Additional metrics won’t be winsorized.

You can also determine the winsorization cutoffs automatically without having an
`OptimizationConfig` by passing in AUTO_WINS_QUANTILE for the quantile you want
to winsorize. For example, to automatically winsorize large values:

> `"m1": WinsorizationConfig(upper_quantile_margin=AUTO_WINS_QUANTILE)`.

This may be useful when fitting models in a notebook where there is no corresponding
`OptimizationConfig`.

Additionally, you can pass in winsorization boundaries `lower_boundary` and
`upper_boundary``that specify a maximum allowable amount of winsorization. This
is discouraged and will eventually be deprecated as we strongly encourage
that users allow ``Winsorize` to automatically infer these boundaries from
the optimization config.

#### cutoffs *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]]*

### ax.modelbridge.transforms.relativize
