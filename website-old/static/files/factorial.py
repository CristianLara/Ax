#!/usr/bin/env python
# coding: utf-8

# # Factorial design with empirical Bayes and Thompson Sampling

# <markdowncell>
# This tutorial illustrates how to run a factorial experiment. In such an experiment, each parameter (factor) can be assigned one of multiple discrete values (levels). A full-factorial experiment design explores all possible combinations of factors and levels.
# 
# For instance, consider a banner with a title and an image. We are considering two different titles and three different images. A full-factorial experiment will compare all 2*3=6 possible combinations of title and image, to see which version of the banner performs the best.
# 
# In this example, we first run an exploratory batch to collect data on all possible combinations. Then we use empirical Bayes to model the data and shrink noisy estimates toward the mean. Next, we use Thompson Sampling to suggest a set of arms (combinations of factors and levels) on which to collect more data. We repeat the process until we have identified the best performing combination(s).

# In[ ]:


import numpy as np
import pandas as pd
import sklearn as skl
from typing import Dict, Optional, Tuple, Union
from ax import (
    Arm,
    ChoiceParameter,
    Models,
    ParameterType,
    SearchSpace,
    Experiment,
    OptimizationConfig,
    Objective,
)
from ax.plot.scatter import plot_fitted
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.stats.statstools import agresti_coull_sem


# In[ ]:


init_notebook_plotting()


# ## 1. Define the search space

# <markdowncell>
# First, we define our search space. A factorial search space contains a ChoiceParameter for each factor, where the values of the parameter are its levels.

# In[ ]:


search_space = SearchSpace(
    parameters=[
        ChoiceParameter(
            name="factor1",
            parameter_type=ParameterType.STRING,
            values=["level11", "level12", "level13"],
        ),
        ChoiceParameter(
            name="factor2",
            parameter_type=ParameterType.STRING,
            values=["level21", "level22"],
        ),
        ChoiceParameter(
            name="factor3",
            parameter_type=ParameterType.STRING,
            values=["level31", "level32", "level33", "level34"],
        ),
    ]
)


# ## 2. Define a custom metric

# Second, we define a custom metric, which is responsible for computing
# the mean and standard error of a given arm.
# 
# In this example, each possible parameter value is given a coefficient. The higher the level, the higher the coefficient, and the higher the coefficients, the greater the mean.
# 
# The standard error of each arm is determined by the weight passed into the evaluation function, which represents the size of the population on which this arm was evaluated. The higher the weight, the greater the sample size, and thus the lower the standard error.

# In[ ]:


from ax import Data, Metric
from ax.utils.common.result import Ok
import pandas as pd
from random import random


one_hot_encoder = skl.preprocessing.OneHotEncoder(
    categories=[par.values for par in search_space.parameters.values()],
)


class FactorialMetric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            batch_size = 10000
            noise_level = 0.0
            weight = trial.normalized_arm_weights().get(arm, 1.0)
            coefficients = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.2, 0.3, 0.4])
            features = np.array(list(params.values())).reshape(1, -1)
            encoded_features = one_hot_encoder.fit_transform(features)
            z = (
                coefficients @ encoded_features.T
                + np.sqrt(noise_level) * np.random.randn()
            )
            p = np.exp(z) / (1 + np.exp(z))
            plays = np.random.binomial(batch_size, weight)
            successes = np.random.binomial(plays, p)
            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "trial_index": trial.index,
                    "mean": float(successes) / plays,
                    "sem": agresti_coull_sem(successes, plays),
                }
            )
        return Ok(value=Data(df=pd.DataFrame.from_records(records)))


# ## 3. Define the experiment

# <markdowncell>
# We now set up our experiment and define the status quo arm, in which each parameter is assigned to the lowest level.

# In[ ]:


from ax import Runner


class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata


exp = Experiment(
    name="my_factorial_closed_loop_experiment",
    search_space=search_space,
    optimization_config=OptimizationConfig(
        objective=Objective(metric=FactorialMetric(name="success_metric"), minimize=False)
    ),
    runner=MyRunner(),
)
exp.status_quo = Arm(
    parameters={"factor1": "level11", "factor2": "level21", "factor3": "level31"}
)


# ## 4. Run an exploratory batch

# <markdowncell>
# We then generate an a set of arms that covers the full space of the factorial design, including the status quo. There are three parameters, with two, three, and four values, respectively, so there are 24 possible arms.

# In[ ]:


factorial = Models.FACTORIAL(search_space=exp.search_space)
factorial_run = factorial.gen(
    n=-1
)  # Number of arms to generate is derived from the search space.
print(len(factorial_run.arms))


# Now we create a trial including all of these arms, so that we can collect data and evaluate the performance of each.

# In[ ]:


trial = exp.new_batch_trial(optimize_for_power=True).add_generator_run(
    factorial_run, multiplier=1
)


# By default, the weight of each arm in `factorial_run` will be 1. However, to optimize for power on the contrasts of `k` groups against the status quo, the status quo should be `sqrt(k)` larger than any of the treatment groups. Since we have 24 different arms in our search space, the status quo should be roughly five times larger. That larger weight is automatically set by Ax under the hood if `optimize_for_power` kwarg is set to True on new batched trial creation.

# In[ ]:


trial._status_quo_weight_override


# ## 5. Iterate using Thompson Sampling

# <markdowncell>
# Next, we run multiple trials (iterations of the experiment) to hone in on the optimal arm(s). 
# 
# In each iteration, we first collect data about all arms in that trial by calling `trial.run()` and `trial.mark_complete()`. Then we run Thompson Sampling, which assigns a weight to each arm that is proportional to the probability of that arm being the best. Arms whose weight exceed `min_weight` are added to the next trial, so that we can gather more data on their performance.

# In[ ]:


models = []
for i in range(4):
    print(f"Running trial {i+1}...")
    trial.run()
    trial.mark_completed()
    thompson = Models.THOMPSON(experiment=exp, data=trial.fetch_data(), min_weight=0.01)
    models.append(thompson)
    thompson_run = thompson.gen(n=-1)
    trial = exp.new_batch_trial(optimize_for_power=True).add_generator_run(thompson_run)


# ## Plot 1: Predicted outcomes for each arm in initial trial

# <markdowncell>
# The plot below shows the mean and standard error for each arm in the first trial. We can see that the standard error for the status quo is the smallest, since this arm was assigned 5x weight.

# In[ ]:


render(plot_fitted(models[0], metric="success_metric", rel=False))


# ## Plot 2: Predicted outcomes for arms in last trial

# The following plot below shows the mean and standard error for each arm that made it to the last trial (as well as the status quo, which appears throughout). 

# In[ ]:


render(plot_fitted(models[-1], metric="success_metric", rel=False))


# <markdowncell>
# As expected given our evaluation function, arms with higher levels
# perform better and are given higher weight. Below we see the arms
# that made it to the final trial.

# In[ ]:


results = pd.DataFrame(
    [
        {"values": ",".join(arm.parameters.values()), "weight": weight}
        for arm, weight in trial.normalized_arm_weights().items()
    ]
)
print(results)


# ## Plot 3: Rollout Process

# We can also visualize the progression of the experience in the following rollout chart. Each bar represents a trial, and the width of the bands within a bar are proportional to the weight of the arms in that trial. 
# 
# In the first trial, all arms appear with equal weight, except for the status quo. By the last trial, we have narrowed our focus to only four arms, with arm 0_22 (the arm with the highest levels) having the greatest weight.

# In[ ]:


from ax.plot.bandit_rollout import plot_bandit_rollout
from ax.utils.notebook.plotting import render

render(plot_bandit_rollout(exp))


# ## Plot 4: Marginal Effects

# Finally, we can examine which parameter values had the greatest effect on the overall arm value. As we see in the diagram below, arms whose parameters were assigned the lower level values (such as `levell1`, `levell2`, `level31` and `level32`) performed worse than average, whereas arms with higher levels performed better than average.

# In[ ]:


from ax.plot.marginal_effects import plot_marginal_effects

render(plot_marginal_effects(models[0], "success_metric"))

