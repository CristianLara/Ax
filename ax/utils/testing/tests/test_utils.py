#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import torch
from ax.adapter.registry import Generators
from ax.generation_strategy.generation_strategy import (
    GenerationNode,
    GenerationStrategy,
)
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.utils import generic_equals, run_trials_with_gs


class TestUtils(TestCase):
    def test_generic_equals(self) -> None:
        # Basics.
        self.assertTrue(generic_equals(5, 5))
        self.assertFalse(generic_equals(5, 1))
        self.assertTrue(generic_equals("abc", "abc"))
        self.assertFalse(generic_equals("abc", "abcd"))
        self.assertFalse(generic_equals("abc", 5))
        # With tensors.
        self.assertTrue(generic_equals(torch.ones(2), torch.ones(2)))
        self.assertFalse(generic_equals(torch.ones(2), torch.zeros(2)))
        self.assertFalse(generic_equals(torch.ones(2), [0, 0]))
        # Dictionaries.
        self.assertTrue(generic_equals({"a": torch.ones(2)}, {"a": torch.ones(2)}))
        self.assertFalse(generic_equals({"a": torch.ones(2)}, {"a": torch.zeros(2)}))
        self.assertFalse(
            generic_equals({"a": torch.ones(2)}, {"a": torch.ones(2), "b": 5})
        )
        self.assertFalse(generic_equals({"a": torch.ones(2)}, [torch.ones(2)]))
        self.assertTrue(
            generic_equals({"a": torch.ones(2), "b": 2}, {"b": 2, "a": torch.ones(2)})
        )
        # Tuple / list.
        self.assertTrue(generic_equals([3, 2], [3, 2]))
        self.assertTrue(generic_equals([3, (2, 3)], [3, (2, 3)]))
        self.assertFalse(generic_equals([3, (2, 3)], [3, (2, 4)]))
        self.assertFalse(generic_equals([0, 1], range(2)))
        # Other.
        self.assertTrue(generic_equals(range(2), range(2)))
        self.assertTrue(generic_equals(np.ones(2), np.ones(2)))
        self.assertFalse(generic_equals(np.ones(2), np.zeros(2)))
        self.assertTrue(generic_equals({1, 2}, {1, 2}))
        self.assertFalse(generic_equals({1, 2}, {1, 2, 3}))

    @mock_botorch_optimize
    def test_run_trials_with_gs(self) -> None:
        experiment = get_experiment_with_observations(
            observations=[[1.0, 5.0], [2.0, 4.0]]
        )
        # There are 2 trials with observations for 2 metrics.
        self.assertEqual(len(experiment.trials), 2)
        self.assertEqual(len(experiment.lookup_data().df), 4)
        gs = GenerationStrategy(
            nodes=[
                GenerationNode(
                    node_name="MBM",
                    model_specs=[
                        GeneratorSpec(
                            model_enum=Generators.BOTORCH_MODULAR,
                        )
                    ],
                )
            ]
        )
        run_trials_with_gs(experiment=experiment, gs=gs, num_trials=2)
        self.assertEqual(len(experiment.trials), 4)
        self.assertEqual(len(experiment.lookup_data().df), 8)
        for idx in [2, 3]:
            self.assertEqual(
                experiment.trials[idx].generator_runs[0]._generation_node_name, "MBM"
            )
