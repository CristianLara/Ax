#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from typing import Any, Dict, Optional, Tuple, Type

from ax.modelbridge.registry import Models

# Ax data tranformation layer
from ax.models.torch.botorch_modular.acquisition import Acquisition

# Ax wrappers for BoTorch components
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate

# Experiment examination utilities
from ax.service.utils.report_utils import exp_to_df

# Test Ax objects
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_data_multi_objective,
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.models.gp_regression import SingleTaskGP

# BoTorch components
from botorch.models.model import Model
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


# # Setup and Usage of BoTorch Models in Ax
# 
# Ax provides a set of flexible wrapper abstractions to mix-and-match BoTorch components like `Model` and `AcquisitionFunction` and combine them into a single `Model` object in Ax. The wrapper abstractions: `Surrogate`, `Acquisition`, and `BoTorchModel` – are located in `ax/models/torch/botorch_modular` directory and aim to encapsulate boilerplate code that interfaces between Ax and BoTorch. This functionality is in beta-release and still evolving.
# 
# This tutorial walks through setting up a custom combination of BoTorch components in Ax in following steps:
# 
# 1. **Quick-start example of `BoTorchModel` use**
# 1. **`BoTorchModel` = `Surrogate` + `Acquisition` (overview)**
#    1. Example with minimal options that uses the defaults
#    2. Example showing all possible options
#    3. Surrogate and Acquisition Q&A
# 2. **I know which Botorch Model and AcquisitionFunction I'd like to combine in Ax. How do set this up?**
#    1. Making a `Surrogate` from BoTorch `Model`
#    2. Using an arbitrary BoTorch `AcquisitionFunction` in Ax
# 3. **Using `Models.BOTORCH_MODULAR`** (convenience wrapper that enables storage and resumability)
# 4. **Utilizing `BoTorchModel` in generation strategies** (abstraction that allows to chain models together and use them in Ax Service API etc.)
#    1. Specifying `pending_observations` to avoid the model re-suggesting points that are part of `RUNNING` or `ABANDONED` trials.
# 5. **Customizing a `Surrogate` or `Acquisition`** (for cases where existing subcomponent classes are not sufficient)

# ## 1. Quick-start example
# 
# Here we set up a `BoTorchModel` with `SingleTaskGP` with `qLogNoisyExpectedImprovement`, one of the most popular combinations in Ax:

# In[ ]:


experiment = get_branin_experiment(with_trial=True)
data = get_branin_data(trials=[experiment.trials[0]])


# In[ ]:


# `Models` automatically selects a model + model bridge combination.
# For `BOTORCH_MODULAR`, it will select `BoTorchModel` and `TorchModelBridge`.
model_bridge_with_GPEI = Models.BOTORCH_MODULAR(
    experiment=experiment,
    data=data,
    surrogate=Surrogate(SingleTaskGP),  # Optional, will use default if unspecified
    botorch_acqf_class=qLogNoisyExpectedImprovement,  # Optional, will use default if unspecified
)


# Now we can use this model to generate candidates (`gen`), predict outcome at a point (`predict`), or evaluate acquisition function value at a given point (`evaluate_acquisition_function`).

# In[ ]:


generator_run = model_bridge_with_GPEI.gen(n=1)
generator_run.arms[0]


# -----
# Before you read the rest of this tutorial:
# 
# - Note that the concept of ‘model’ is Ax is somewhat a misnomer; we use ['model'](https://ax.dev/docs/glossary.html#model) to refer to an optimization setup capable of producing candidate points for optimization (and often capable of being fit to data, with exception for quasi-random generators). See [Models documentation page](https://ax.dev/docs/models.html) for more information.
# - Learn about `ModelBridge` in Ax, as users should rarely be interacting with a `Model` object directly (more about ModelBridge, a data transformation layer in Ax, [here](https://ax.dev/docs/models.html#deeper-dive-organization-of-the-modeling-stack)).

# ## 2. BoTorchModel = Surrogate + Acquisition
# 
# A `BoTorchModel` in Ax consists of two main subcomponents: a surrogate model and an acquisition function. A surrogate model is represented as an instance of Ax’s `Surrogate` class, which is a wrapper around BoTorch's `Model` class. The acquisition function is represented as an instance of Ax’s `Acquisition` class, a wrapper around BoTorch's `AcquisitionFunction` class.

# ### 2A. Example that uses defaults and requires no options
# 
# BoTorchModel does not always require surrogate and acquisition specification. If instantiated without one or both components specified, defaults are selected based on properties of experiment and data (see Appendix 2 for auto-selection logic).

# In[ ]:


# The surrogate is not specified, so it will be auto-selected
# during `model.fit`.
GPEI_model = BoTorchModel(botorch_acqf_class=qLogExpectedImprovement)

# The acquisition class is not specified, so it will be
# auto-selected during `model.gen` or `model.evaluate_acquisition`
GPEI_model = BoTorchModel(surrogate=Surrogate(SingleTaskGP))

# Both the surrogate and acquisition class will be auto-selected.
GPEI_model = BoTorchModel()


# ### 2B. Example with all the options
# Below are the full set of configurable settings of a `BoTorchModel` with their descriptions:

# In[ ]:


model = BoTorchModel(
    # Optional `Surrogate` specification to use instead of default
    surrogate=Surrogate(
        # BoTorch `Model` type
        botorch_model_class=SingleTaskGP,
        # Optional, MLL class with which to optimize model parameters
        mll_class=ExactMarginalLogLikelihood,
        # Optional, dictionary of keyword arguments to underlying
        # BoTorch `Model` constructor
        model_options={},
    ),
    # Optional BoTorch `AcquisitionFunction` to use instead of default
    botorch_acqf_class=qLogExpectedImprovement,
    # Optional dict of keyword arguments, passed to the input
    # constructor for the given BoTorch `AcquisitionFunction`
    acquisition_options={},
    # Optional Ax `Acquisition` subclass (if the given BoTorch
    # `AcquisitionFunction` requires one, which is rare)
    acquisition_class=None,
    # Less common model settings shown with default values, refer
    # to `BoTorchModel` documentation for detail
    refit_on_cv=False,
    warm_start_refit=True,
)


# ## 2C. `Surrogate` and `Acquisition` Q&A
# 
# **Why is the `surrogate` argument expected to be an instance, but `botorch_acqf_class` –– a class?** Because a BoTorch `AcquisitionFunction` object (and therefore its Ax wrapper, `Acquisition`) is ephemeral: it is constructed, immediately used, and destroyed during `BoTorchModel.gen`, so there is no reason to keep around an `Acquisition` instance. A `Surrogate`, on another hand, is kept in memory as long as its parent `BoTorchModel` is.
# 
# **How to know when to use specify acquisition_class (and thereby a non-default Acquisition type) instead of just passing in botorch_acqf_class?** In short, custom `Acquisition` subclasses are needed when a given `AcquisitionFunction` in BoTorch needs some non-standard subcomponents or inputs (e.g. a custom BoTorch `MCAcquisitionObjective`). <TODO>
# 
# **Please post any other questions you have to our dedicated issue on Github: https://github.com/facebook/Ax/issues/363.** This functionality is in beta-release and your feedback will be of great help to us!

# ## 3. I know which Botorch `Model` and `AcquisitionFunction` I'd like to combine in Ax. How do set this up?

# ### 3a. Making a `Surrogate` from BoTorch `Model`:
# Most models should work with base `Surrogate` in Ax, except for BoTorch `ModelListGP`. `ModelListGP` is a special case because its purpose is to combine multiple sub-models into a single `Model` in BoTorch. It is most commonly used for multi-objective and constrained optimization. Whether or not `ModelListGP` is used is determined automatically based on the `Model` class and the data being used via the `ax.models.torch.botorch_modular.utils.use_model_list` function.
# 
# If your `Model` is not a `ModelListGP`, the steps to set it up as a `Surrogate` are:
# 1. Implement a [`construct_inputs` class method](https://github.com/pytorch/botorch/blob/main/botorch/models/model.py#L143). The purpose of this method is to produce arguments to a particular model from a standardized set of inputs passed to BoTorch `Model`-s from [`Surrogate.construct`](https://github.com/facebook/Ax/blob/main/ax/models/torch/botorch_modular/surrogate.py#L148) in Ax. It should accept training data in form of a `SupervisedDataset` container and optionally other keyword arguments and produce a dictionary of arguments to `__init__` of the `Model`. See [`SingleTaskMultiFidelityGP.construct_inputs`](https://github.com/pytorch/botorch/blob/5b3172f3daa22f6ea2f6f4d1d0a378a9518dcd8d/botorch/models/gp_regression_fidelity.py#L131) for an example.
# 2. Pass any additional needed keyword arguments for the `Model` constructor (that cannot be constructed from the training data and other arguments to `construct_inputs`) via `model_options` argument to `Surrogate`.

# In[ ]:


from botorch.models.model import Model
from botorch.utils.datasets import SupervisedDataset


class MyModelClass(Model):

    ...  # Implementation of `MyModelClass`

    @classmethod
    def construct_inputs(
        cls, training_data: SupervisedDataset, **kwargs
    ) -> Dict[str, Any]:
        fidelity_features = kwargs.get("fidelity_features")
        if fidelity_features is None:
            raise ValueError(f"Fidelity features required for {cls.__name__}.")

        return {
            **super().construct_inputs(training_data=training_data, **kwargs),
            "fidelity_features": fidelity_features,
        }


surrogate = Surrogate(
    botorch_model_class=MyModelClass,  # Must implement `construct_inputs`
    # Optional dict of additional keyword arguments to `MyModelClass`
    model_options={},
)


# NOTE: if you run into a case where base `Surrogate` does not work with your BoTorch `Model`, please let us know in this Github issue: https://github.com/facebook/Ax/issues/363, so we can find the right solution and augment this tutorial.

# ### 3B. Using an arbitrary BoTorch `AcquisitionFunction` in Ax

# Steps to set up any `AcquisitionFunction` in Ax are:
# 1. Define an input constructor function. The purpose of this method is to produce arguments to a acquisition function from a standardized set of inputs passed to BoTorch `AcquisitionFunction`-s from `Acquisition.__init__` in Ax. For example, see [`construct_inputs_qEHVI`](https://github.com/pytorch/botorch/blob/main/botorch/acquisition/input_constructors.py#L477), which creates a fairly complex set of arguments needed by `qExpectedHypervolumeImprovement` –– a popular multi-objective optimization acquisition function offered in Ax and BoTorch. For more examples, see this collection in BoTorch: [botorch/acquisition/input_constructors.py](https://github.com/pytorch/botorch/blob/main/botorch/acquisition/input_constructors.py) 
#    1. Note that the new input constructor needs to be decorated with `@acqf_input_constructor(AcquisitionFunctionClass)` to register it.
# 2. (Optional) If a given `AcquisitionFunction` requires specific options passed to the BoTorch `optimize_acqf`, it's possible to add default optimizer options for a given `AcquisitionFunction` to avoid always manually passing them via `acquisition_options`.
# 3. Specify the BoTorch `AcquisitionFunction` class as `botorch_acqf_class` to `BoTorchModel`
# 4. (Optional) Pass any additional keyword arguments to acquisition function constructor or to the optimizer function via `acquisition_options` argument to `BoTorchModel`.

# In[ ]:


from ax.models.torch.botorch_modular.optimizer_argparse import optimizer_argparse
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import acqf_input_constructor, MaybeDict
from botorch.utils.datasets import SupervisedDataset
from torch import Tensor


class MyAcquisitionFunctionClass(AcquisitionFunction):
    ...  # Actual contents of the acquisition function class.


# 1. Add input constructor
@acqf_input_constructor(MyAcquisitionFunctionClass)
def construct_inputs_my_acqf(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective_thresholds: Tensor,
    **kwargs: Any,
) -> Dict[str, Any]:
    pass


# 2. Register default optimizer options
@optimizer_argparse.register(MyAcquisitionFunctionClass)
def _argparse_my_acqf(
    acqf: MyAcquisitionFunctionClass, sequential: bool = True
) -> dict:
    return {
        "sequential": sequential
    }  # default to sequentially optimizing batches of queries


# 3-4. Specifying `botorch_acqf_class` and `acquisition_options`
BoTorchModel(
    botorch_acqf_class=MyAcquisitionFunctionClass,
    acquisition_options={
        "alpha": 10**-6,
        # The sub-dict by the key "optimizer_options" can be passed
        # to propagate options to `optimize_acqf`, used in
        # `Acquisition.optimize`, to add/override the default
        # optimizer options registered above.
        "optimizer_options": {"sequential": False},
    },
)


# See section 2A for combining the resulting `Surrogate` instance and `Acquisition` type into a `BoTorchModel`. You can also leverage `Models.BOTORCH_MODULAR` for ease of use; more on it in section 4 below or in section 1 quick-start example.

# ## 4. Using `Models.BOTORCH_MODULAR` 
# 
# To simplify the instantiation of an Ax ModelBridge and its undelying Model, Ax provides a [`Models` registry enum](https://github.com/facebook/Ax/blob/main/ax/modelbridge/registry.py#L355). When calling entries of that enum (e.g. `Models.BOTORCH_MODULAR(experiment, data)`), the inputs are automatically distributed between a `Model` and a `ModelBridge` for a given setup. A call to a `Model` enum member yields a model bridge with an underlying model, ready for use to generate candidates.
# 
# Here we use `Models.BOTORCH_MODULAR` to set up a model with all-default subcomponents:

# In[ ]:


model_bridge_with_GPEI = Models.BOTORCH_MODULAR(
    experiment=experiment,
    data=data,
)
model_bridge_with_GPEI.gen(1)


# In[ ]:


model_bridge_with_GPEI.model.botorch_acqf_class


# In[ ]:


model_bridge_with_GPEI.model.surrogate.botorch_model_class


# We can use the same `Models.BOTORCH_MODULAR` to set up a model for multi-objective optimization:

# In[ ]:


model_bridge_with_EHVI = Models.BOTORCH_MODULAR(
    experiment=get_branin_experiment_with_multi_objective(
        has_objective_thresholds=True, with_batch=True
    ),
    data=get_branin_data_multi_objective(),
)
model_bridge_with_EHVI.gen(1)


# In[ ]:


model_bridge_with_EHVI.model.botorch_acqf_class


# In[ ]:


model_bridge_with_EHVI.model.surrogate.botorch_model_class


# Furthermore, the quick-start example at the top of this tutorial shows how to specify surrogate and acquisition subcomponents to `Models.BOTORCH_MODULAR`. 

# ## 5. Utilizing `BoTorchModel` in generation strategies
# 
# Generation strategy is a key concept in Ax, enabling use of Service API (a.k.a. `AxClient`) and many other higher-level abstractions. A `GenerationStrategy` allows to chain multiple models in Ax and thereby automate candidate generation. Refer to the "Generation Strategy" tutorial for more detail in generation strategies.
# 
# An example generation stategy with the modular `BoTorchModel` would look like this:

# In[ ]:


from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features

gs = GenerationStrategy(
    steps=[
        GenerationStep(  # Initialization step
            # Which model to use for this step
            model=Models.SOBOL,
            # How many generator runs (each of which is then made a trial)
            # to produce with this step
            num_trials=5,
            # How many trials generated from this step must be `COMPLETED`
            # before the next one
            min_trials_observed=5,
        ),
        GenerationStep(  # BayesOpt step
            model=Models.BOTORCH_MODULAR,
            # No limit on how many generator runs will be produced
            num_trials=-1,
            model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
                "surrogate": Surrogate(SingleTaskGP),
                "botorch_acqf_class": qLogNoisyExpectedImprovement,
            },
        ),
    ]
)


# Set up an experiment and generate 10 trials in it, adding synthetic data to experiment after each one:

# In[ ]:


experiment = get_branin_experiment(minimize=True)

assert len(experiment.trials) == 0
experiment.search_space


# ## 5a. Specifying `pending_observations`
# Note that it's important to **specify pending observations** to the call to `gen` to avoid getting the same points re-suggested. Without `pending_observations` argument, Ax models are not aware of points that should be excluded from generation. Points are considered "pending" when they belong to `STAGED`, `RUNNING`, or `ABANDONED` trials (with the latter included so model does not re-suggest points that are considered "bad" and should not be re-suggested).
# 
# If the call to `get_pending_observation_features` becomes slow in your setup (since it performs data-fetching etc.), you can opt for `get_pending_observation_features_based_on_trial_status` (also from `ax.modelbridge.modelbridge_utils`), but note the limitations of that utility (detailed in its docstring).

# In[ ]:


for _ in range(10):
    # Produce a new generator run and attach it to experiment as a trial
    generator_run = gs.gen(
        experiment=experiment,
        n=1,
        pending_observations=get_pending_observation_features(experiment=experiment),
    )
    trial = experiment.new_trial(generator_run)

    # Mark the trial as 'RUNNING' so we can mark it 'COMPLETED' later
    trial.mark_running(no_runner_required=True)

    # Attach data for the new trial and mark it 'COMPLETED'
    experiment.attach_data(get_branin_data(trials=[trial]))
    trial.mark_completed()

    print(f"Completed trial #{trial.index}, suggested by {generator_run._model_key}.")


# Now we examine the experiment and observe the trials that were added to it and produced by the generation strategy:

# In[ ]:


exp_to_df(experiment)


# ## 6. Customizing a `Surrogate` or `Acquisition`
# 
# We expect the base `Surrogate` and `Acquisition` classes to work with most BoTorch components, but there could be a case where you would need to subclass one of aforementioned abstractions to handle a given BoTorch component. If you run into a case like this, feel free to open an issue on our [Github issues page](https://github.com/facebook/Ax/issues) –– it would be very useful for us to know 
# 
# One such example would be a need for a custom `MCAcquisitionObjective` or posterior transform. To subclass `Acquisition` accordingly, one would override the `get_botorch_objective_and_transform` method:

# In[ ]:


from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.acquisition.risk_measures import RiskMeasureMCObjective


class CustomObjectiveAcquisition(Acquisition):
    def get_botorch_objective_and_transform(
        self,
        botorch_acqf_class: Type[AcquisitionFunction],
        model: Model,
        objective_weights: Tensor,
        objective_thresholds: Optional[Tensor] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        X_observed: Optional[Tensor] = None,
        risk_measure: Optional[RiskMeasureMCObjective] = None,
    ) -> Tuple[Optional[MCAcquisitionObjective], Optional[PosteriorTransform]]:
        ...  # Produce the desired `MCAcquisitionObjective` and `PosteriorTransform` instead of the default


# Then to use the new subclass in `BoTorchModel`, just specify `acquisition_class` argument along with `botorch_acqf_class` (to `BoTorchModel` directly or to `Models.BOTORCH_MODULAR`, which just passes the relevant arguments to `BoTorchModel` under the hood, as discussed in section 4):

# In[ ]:


Models.BOTORCH_MODULAR(
    experiment=experiment,
    data=data,
    acquisition_class=CustomObjectiveAcquisition,
    botorch_acqf_class=MyAcquisitionFunctionClass,
)


# To use a custom `Surrogate` subclass, pass the `surrogate` argument of that type:
# ```
# Models.BOTORCH_MODULAR(
#     experiment=experiment, 
#     data=data,
#     surrogate=CustomSurrogate(botorch_model_class=MyModelClass),
# )
# ```

# ------

# ## Appendix 1: Methods available on `BoTorchModel`
# 
# Note that usually all these methods are used through `ModelBridge` –– a convertion and transformation layer that adapts Ax abstractions to inputs required by the given model.
# 
# **Core methods on `BoTorchModel`:**
# * `fit` selects a surrogate if needed and fits the surrogate model to data via `Surrogate.fit`,
# * `predict` estimates metric values at a given point via `Surrogate.predict`,
# * `gen` instantiates an acquisition function via `Acquisition.__init__` and optimizes it to generate candidates.
# 
# **Other methods on `BoTorchModel`:**
# * `update` updates surrogate model with training data and optionally reoptimizes model parameters via `Surrogate.update`,
# * `cross_validate` re-fits the surrogate model to subset of training data and makes predictions for test data,
# * `evaluate_acquisition_function` instantiates an acquisition function and evaluates it for a given point.
# ------
# 

# ## Appendix 2: Default surrogate models and acquisition functions
# 
# By default, the chosen surrogate model will be:
# * if fidelity parameters are present in search space: `SingleTaskMultiFidelityGP`,
# * if task parameters are present: a set of `MultiTaskGP`  wrapped in a `ModelListGP` and each modeling one task,
# * `SingleTaskGP` otherwise.
# 
# The chosen acquisition function will be:
# * for multi-objective settings: `qLogExpectedHypervolumeImprovement`,
# * for single-objective settings: `qLogNoisyExpectedImprovement`.
# ----

# ## Appendix 3: Handling storage errors that arise from objects that don't have serialization logic in A
# 
# Attempting to store a generator run produced via `Models.BOTORCH_MODULAR` instance that included options without serization logic with will produce an error like: `"Object <SomeAcquisitionOption object> passed to 'object_to_json' (of type <class SomeAcquisitionOption'>) is not registered with a corresponding encoder in ENCODER_REGISTRY."`

# The two options for handling this error are:
# 1. disabling storage of `BoTorchModel`'s options by passing `no_model_options_storage=True` to `Models.BOTORCH_MODULAR(...)` call –– this will prevent model options from being stored on the generator run, so a generator run can be saved but cannot be used to restore the model that produced it,
# 2. specifying serialization logic for a given object that needs to occur among the `Model` or `AcquisitionFunction` options. Tutorial for this is in the works, but in the meantime you can [post an issue on the Ax GitHub](https://github.com/facebook/Ax/issues) to get help with this.
