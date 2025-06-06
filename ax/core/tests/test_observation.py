#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.map_data import MapData, MapKeyInfo
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric
from ax.core.observation import (
    _filter_data_on_status,
    Observation,
    ObservationData,
    ObservationFeatures,
    observations_from_data,
    recombine_observations,
    separate_observations,
)
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.core.trial_status import TrialStatus
from ax.core.types import TParameterization
from ax.utils.common.testutils import TestCase
from pyre_extensions import assert_is_instance, none_throws


class ObservationsTest(TestCase):
    def test_ObservationFeatures(self) -> None:
        t = np.datetime64("now")
        attrs = {
            "parameters": {"x": 0, "y": "a"},
            "trial_index": 2,
            "start_time": t,
            "end_time": t,
            "random_split": 1,
        }
        # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool, float,
        #  int, str]]` but got `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[Dict[str, typing.Any]]`
        #  but got `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[int64]` but got
        #  `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[Timestamp]` but got
        #  `Union[Dict[str, Union[int, str]], int, datetime64]`.
        obsf = ObservationFeatures(**attrs)
        for k, v in attrs.items():
            self.assertEqual(getattr(obsf, k), v)
        printstr = "ObservationFeatures(parameters={'x': 0, 'y': 'a'}, "
        printstr += "trial_index=2, "
        printstr += "start_time={t}, end_time={t}, ".format(t=t)
        printstr += "random_split=1)"
        self.assertEqual(repr(obsf), printstr)
        # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool, float,
        #  int, str]]` but got `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[Dict[str, typing.Any]]`
        #  but got `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[int64]` but got
        #  `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[Timestamp]` but got
        #  `Union[Dict[str, Union[int, str]], int, datetime64]`.
        obsf2 = ObservationFeatures(**attrs)
        self.assertEqual(hash(obsf), hash(obsf2))
        a = {obsf, obsf2}
        self.assertEqual(len(a), 1)
        self.assertEqual(obsf, obsf2)
        attrs.pop("trial_index")
        # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool, float,
        #  int, str]]` but got `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[Dict[str, typing.Any]]`
        #  but got `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[int64]` but got
        #  `Union[Dict[str, Union[int, str]], int, datetime64]`.
        # pyre-fixme[6]: For 1st param expected `Optional[Timestamp]` but got
        #  `Union[Dict[str, Union[int, str]], int, datetime64]`.
        obsf3 = ObservationFeatures(**attrs)
        self.assertNotEqual(obsf, obsf3)
        self.assertFalse(obsf == 1)

    def test_Clone(self) -> None:
        # Test simple cloning.
        arm = Arm({"x": 0, "y": "a"})
        obsf = ObservationFeatures.from_arm(arm, trial_index=3)
        self.assertIsNot(obsf, obsf.clone())
        self.assertEqual(obsf, obsf.clone())

        # Test cloning with swapping parameters.
        clone_with_new_params = obsf.clone(replace_parameters={"x": 1, "y": "b"})
        self.assertNotEqual(obsf, clone_with_new_params)
        obsf.parameters = {"x": 1, "y": "b"}
        self.assertEqual(obsf, clone_with_new_params)

    def test_ObservationFeaturesFromArm(self) -> None:
        arm = Arm({"x": 0, "y": "a"})
        obsf = ObservationFeatures.from_arm(arm, trial_index=3)
        self.assertIsNot(arm.parameters, obsf.parameters)
        self.assertEqual(obsf.parameters, arm.parameters)
        self.assertEqual(obsf.trial_index, 3)

    def test_UpdateFeatures(self) -> None:
        parameters = {"x": 0, "y": "a"}
        new_parameters = {"z": "foo"}

        # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool, float,
        #  int, str]]` but got `Dict[str, Union[int, str]]`.
        # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
        obsf = ObservationFeatures(parameters=parameters, trial_index=3)

        # Ensure None trial_index doesn't override existing value
        obsf.update_features(ObservationFeatures(parameters={}))
        self.assertEqual(obsf.trial_index, 3)

        # Test override
        new_obsf = ObservationFeatures(
            # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Dict[str, str]`.
            parameters=new_parameters,
            trial_index=4,
            start_time=pd.Timestamp("2005-02-25"),
            end_time=pd.Timestamp("2005-02-26"),
            random_split=7,
        )
        obsf.update_features(new_obsf)
        self.assertEqual(obsf.parameters, {**parameters, **new_parameters})
        self.assertEqual(obsf.trial_index, 4)
        self.assertEqual(obsf.random_split, 7)
        self.assertEqual(obsf.start_time, pd.Timestamp("2005-02-25"))
        self.assertEqual(obsf.end_time, pd.Timestamp("2005-02-26"))

    def test_ObservationData(self) -> None:
        attrs = {
            "metric_names": ["a", "b"],
            "means": np.array([4.0, 5.0]),
            "covariance": np.array([[1.0, 4.0], [3.0, 6.0]]),
        }
        # pyre-fixme[6]: For 1st param expected `List[str]` but got
        #  `Union[List[str], ndarray]`.
        # pyre-fixme[6]: For 1st param expected `ndarray` but got `Union[List[str],
        #  ndarray]`.
        obsd = ObservationData(**attrs)
        self.assertEqual(obsd.metric_names, attrs["metric_names"])
        self.assertTrue(np.array_equal(obsd.means, attrs["means"]))
        self.assertTrue(np.array_equal(obsd.covariance, attrs["covariance"]))
        # use legacy printing for numpy (<= 1.13 add spaces in front of floats;
        # to get around tests failing on older versions, peg version to 1.13)
        if np.__version__ >= "1.14":
            np.set_printoptions(legacy="1.13")
        printstr = "ObservationData(metric_names=['a', 'b'], means=[ 4.  5.], "
        printstr += "covariance=[[ 1.  4.]\n [ 3.  6.]])"
        self.assertEqual(repr(obsd), printstr)
        self.assertEqual(obsd.means_dict, {"a": 4.0, "b": 5.0})
        self.assertEqual(
            obsd.covariance_matrix,
            {"a": {"a": 1.0, "b": 4.0}, "b": {"a": 3.0, "b": 6.0}},
        )

    def test_ObservationDataValidation(self) -> None:
        with self.assertRaises(ValueError):
            ObservationData(
                metric_names=["a", "b"],
                means=np.array([4.0]),
                covariance=np.array([[1.0, 4.0], [3.0, 6.0]]),
            )
        with self.assertRaises(ValueError):
            ObservationData(
                metric_names=["a", "b"],
                means=np.array([4.0, 5.0]),
                covariance=np.array([1.0, 4.0]),
            )

    def test_ObservationDataEq(self) -> None:
        od1 = ObservationData(
            metric_names=["a", "b"],
            means=np.array([4.0, 5.0]),
            covariance=np.array([[1.0, 4.0], [3.0, 6.0]]),
        )
        od2 = ObservationData(
            metric_names=["a", "b"],
            means=np.array([4.0, 5.0]),
            covariance=np.array([[1.0, 4.0], [3.0, 6.0]]),
        )
        od3 = ObservationData(
            metric_names=["a", "b"],
            means=np.array([4.0, 5.0]),
            covariance=np.array([[2.0, 4.0], [3.0, 6.0]]),
        )
        self.assertEqual(od1, od2)
        self.assertNotEqual(od1, od3)
        self.assertFalse(od1 == 1)

    def test_Observation(self) -> None:
        obs = Observation(
            features=ObservationFeatures(parameters={"x": 20}),
            data=ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
            arm_name="0_0",
        )
        self.assertEqual(obs.features, ObservationFeatures(parameters={"x": 20}))
        self.assertEqual(
            obs.data,
            ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
        )
        self.assertEqual(obs.arm_name, "0_0")
        obs2 = Observation(
            features=ObservationFeatures(parameters={"x": 20}),
            data=ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
            arm_name="0_0",
        )
        self.assertEqual(obs, obs2)
        obs3 = Observation(
            features=ObservationFeatures(parameters={"x": 10}),
            data=ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
            arm_name="0_0",
        )
        self.assertNotEqual(obs, obs3)
        self.assertNotEqual(obs, 1)

    def test_ObservationsFromData(self) -> None:
        truth = [
            {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a"},
                "mean": 2.0,
                "sem": 2.0,
                "trial_index": 1,
                "metric_name": "a",
            },
            {
                "arm_name": "0_1",
                "parameters": {"x": 1, "y": "b"},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 2,
                "metric_name": "a",
            },
            {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a"},
                "mean": 4.0,
                "sem": 4.0,
                "trial_index": 1,
                "metric_name": "b",
            },
        ]
        arms = {
            # pyre-fixme[6]: For 1st param expected `Optional[str]` but got
            #  `Union[Dict[str, Union[int, str]], float, str]`.
            # pyre-fixme[6]: For 2nd param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Union[Dict[str, Union[int, str]], float,
            #  str]`.
            obs["arm_name"]: Arm(name=obs["arm_name"], parameters=obs["parameters"])
            for obs in truth
        }
        experiment = Mock()
        experiment._trial_indices_by_status = {status: set() for status in TrialStatus}
        trials = {
            obs["trial_index"]: Trial(
                experiment, GeneratorRun(arms=[arms[obs["arm_name"]]])
            )
            for obs in truth
        }
        type(experiment).arms_by_name = PropertyMock(return_value=arms)
        type(experiment).trials = PropertyMock(return_value=trials)

        df = pd.DataFrame(truth)[
            ["arm_name", "trial_index", "mean", "sem", "metric_name"]
        ]
        data = Data(df=df)

        with self.assertRaisesRegex(ValueError, "`metric_name` column is missing"):
            observations = _filter_data_on_status(
                df=df.drop(columns="metric_name"),
                experiment=experiment,
                trial_status=None,
                is_arm_abandoned=False,
                statuses_to_include=set(),
                statuses_to_include_map_metric=set(),
            )

        type(experiment).metrics = PropertyMock(return_value={"a": "a", "b": "b"})
        observations = observations_from_data(experiment, data)
        self.assertEqual(len(observations), 2)
        self.assertListEqual(observations[0].data.metric_names, ["a", "b"])
        self.assertListEqual(observations[1].data.metric_names, ["a"])

        # Get them in the order we want for tests below
        if observations[0].features.parameters["x"] == 1:
            observations.reverse()

        obsd_truth = {
            "metric_names": [["a", "b"], ["a"]],
            "means": [np.array([2.0, 4.0]), np.array([3])],
            "covariance": [np.diag([4.0, 16.0]), np.array([[9.0]])],
        }
        cname_truth = ["0_0", "0_1"]

        for i, obs in enumerate(observations):
            self.assertEqual(obs.features.parameters, truth[i]["parameters"])
            self.assertEqual(obs.features.trial_index, truth[i]["trial_index"])
            self.assertEqual(obs.data.metric_names, obsd_truth["metric_names"][i])
            self.assertTrue(np.array_equal(obs.data.means, obsd_truth["means"][i]))
            self.assertTrue(
                np.array_equal(obs.data.covariance, obsd_truth["covariance"][i])
            )
            self.assertEqual(obs.arm_name, cname_truth[i])

    def test_ObservationsFromDataWithFidelities(self) -> None:
        truth = {
            0.5: {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 2.0,
                "sem": 2.0,
                "trial_index": 1,
                "metric_name": "a",
                "fidelities": json.dumps({"z": 0.5}),
                "updated_parameters": {"x": 0, "y": "a", "z": 0.5},
                "mean_t": np.array([2.0]),
                "covariance_t": np.array([[4.0]]),
            },
            0.25: {
                "arm_name": "0_1",
                "parameters": {"x": 1, "y": "b", "z": 0.5},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 2,
                "metric_name": "a",
                "fidelities": json.dumps({"z": 0.25}),
                "updated_parameters": {"x": 1, "y": "b", "z": 0.25},
                "mean_t": np.array([3.0]),
                "covariance_t": np.array([[9.0]]),
            },
            1: {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 4.0,
                "sem": 4.0,
                "trial_index": 1,
                "metric_name": "b",
                "fidelities": json.dumps({"z": 1}),
                "updated_parameters": {"x": 0, "y": "a", "z": 1},
                "mean_t": np.array([4.0]),
                "covariance_t": np.array([[16.0]]),
            },
        }
        arms = {
            # pyre-fixme[6]: For 1st param expected `Optional[str]` but got
            #  `Union[Dict[str, Union[float, str]], Dict[str, Union[int, str]], float,
            #  ndarray, str]`.
            # pyre-fixme[6]: For 2nd param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Union[Dict[str, Union[float, str]],
            #  Dict[str, Union[int, str]], float, ndarray, str]`.
            obs["arm_name"]: Arm(name=obs["arm_name"], parameters=obs["parameters"])
            for _, obs in truth.items()
        }
        experiment = Mock()
        experiment._trial_indices_by_status = {status: set() for status in TrialStatus}
        trials = {
            obs["trial_index"]: Trial(
                experiment, GeneratorRun(arms=[arms[obs["arm_name"]]])
            )
            for _, obs in truth.items()
        }
        type(experiment).arms_by_name = PropertyMock(return_value=arms)
        type(experiment).trials = PropertyMock(return_value=trials)
        type(experiment).metrics = PropertyMock(return_value={"a": "a", "b": "b"})

        df = pd.DataFrame(list(truth.values()))[
            ["arm_name", "trial_index", "mean", "sem", "metric_name", "fidelities"]
        ]
        data = Data(df=df)
        observations = observations_from_data(experiment, data)

        self.assertEqual(len(observations), 3)
        for obs in observations:
            # pyre-fixme[6]: For 1st param expected `float` but got `Union[None,
            #  bool, float, int, str]`.
            t = truth[obs.features.parameters["z"]]
            self.assertEqual(obs.features.parameters, t["updated_parameters"])
            self.assertEqual(obs.features.trial_index, t["trial_index"])
            self.assertEqual(obs.data.metric_names, [t["metric_name"]])
            # pyre-fixme[6]: For 2nd argument expected `Union[_SupportsArray[dtype[ty...
            self.assertTrue(np.array_equal(obs.data.means, t["mean_t"]))
            # pyre-fixme[6]: For 2nd argument expected `Union[_SupportsArray[dtype[ty...
            self.assertTrue(np.array_equal(obs.data.covariance, t["covariance_t"]))
            self.assertEqual(obs.arm_name, t["arm_name"])

    def test_ObservationsFromMapData(self) -> None:
        truth = [
            {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 2.0,
                "sem": 2.0,
                "trial_index": 0,
                "metric_name": "a",
                "mean_t": np.array([2.0]),
                "covariance_t": np.array([[4.0]]),
                "step": 0.5,
                "timestamp": 50,
            },
            {
                "arm_name": "0_1",
                "parameters": {"x": 1, "y": "b", "z": 0.5},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 1,
                "metric_name": "a",
                "mean_t": np.array([3.0]),
                "covariance_t": np.array([[9.0]]),
                "step": 0.25,
                "timestamp": 25,
            },
            {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 4.0,
                "sem": 4.0,
                "trial_index": 2,
                "metric_name": "b",
                "mean_t": np.array([4.0]),
                "covariance_t": np.array([[16.0]]),
                "step": 1,
                "timestamp": 100,
            },
        ]
        arms = [
            Arm(
                name=assert_is_instance(obs["arm_name"], str),
                # pyre-fixme[6]: For 2nd param expected `Dict[str, Union[None, bool,...
                parameters=obs["parameters"],
            )
            for obs in truth
        ]
        parameters = [
            RangeParameter(
                name="x", parameter_type=ParameterType.INT, lower=0, upper=1
            ),
            ChoiceParameter(
                name="y", parameter_type=ParameterType.STRING, values=["a", "b"]
            ),
            RangeParameter(
                name="z", parameter_type=ParameterType.FLOAT, lower=0.25, upper=1
            ),
        ]
        experiment = Experiment(
            search_space=SearchSpace(parameters=parameters),
            tracking_metrics=[Metric(name="a"), Metric(name="b")],
        )
        for arm in arms:
            experiment.new_trial(generator_run=GeneratorRun(arms=[arm]))

        df = pd.DataFrame(truth)[
            [
                "arm_name",
                "trial_index",
                "mean",
                "sem",
                "metric_name",
                "step",
                "timestamp",
            ]
        ]
        data = MapData(
            df=df,
            map_key_infos=[
                MapKeyInfo(key="step", default_value=0.0),
                MapKeyInfo(key="timestamp", default_value=0.0),
            ],
        )
        observations = observations_from_data(experiment=experiment, data=data)
        self.assertEqual(len(observations), 3)

        truth_reordered = [truth[0], truth[2], truth[1]]
        for t, obs in zip(truth_reordered, observations, strict=True):
            self.assertEqual(obs.features.parameters, t["parameters"])
            self.assertEqual(obs.features.trial_index, t["trial_index"])
            self.assertEqual(obs.data.metric_names, [t["metric_name"]])
            # pyre-fixme[6]: For 2nd argument expected `Union[_SupportsArray[dtype[ty...
            self.assertTrue(np.array_equal(obs.data.means, t["mean_t"]))
            # pyre-fixme[6]: For 2nd argument expected `Union[_SupportsArray[dtype[ty...
            self.assertTrue(np.array_equal(obs.data.covariance, t["covariance_t"]))
            self.assertEqual(obs.arm_name, t["arm_name"])
            self.assertEqual(
                obs.features.metadata, {"step": t["step"], "timestamp": t["timestamp"]}
            )

        # testing that we can handle empty data with latest_rows_per_group
        empty_data = MapData()
        observations = observations_from_data(
            experiment,
            empty_data,
            latest_rows_per_group=1,
        )

    def test_ObservationsFromDataAbandoned(self) -> None:
        truth = [
            {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 2.0,
                "sem": 2.0,
                "trial_index": 0,
                "metric_name": "a",
                "updated_parameters": {"x": 0, "y": "a", "z": 0.5},
                "mean_t": np.array([2.0]),
                "covariance_t": np.array([[4.0]]),
                "z": 0.5,
                "timestamp": 50,
            },
            {
                "arm_name": "1_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 4.0,
                "sem": 4.0,
                "trial_index": 1,
                "metric_name": "a",
                "updated_parameters": {"x": 0, "y": "a", "z": 1},
                "mean_t": np.array([4.0]),
                "covariance_t": np.array([[16.0]]),
                "z": 1,
                "timestamp": 100,
            },
            {
                "arm_name": "1_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 4.0,
                "sem": 4.0,
                "trial_index": 1,
                "metric_name": "b",
                "updated_parameters": {"x": 0, "y": "a", "z": 1},
                "mean_t": np.array([4.0]),
                "covariance_t": np.array([[16.0]]),
                "z": 1,
                "timestamp": 100,
            },
            {
                "arm_name": "1_0",
                "parameters": {"x": 0, "y": "a", "z": 1},
                "mean": 4.0,
                "sem": 4.0,
                "trial_index": 1,
                "metric_name": "c",
                "updated_parameters": {"x": 0, "y": "a", "z": 1},
                "mean_t": np.array([4.0]),
                "covariance_t": np.array([[16.0]]),
                "z": 1,
                "timestamp": 100,
            },
            {
                "arm_name": "2_0",
                "parameters": {"x": 1, "y": "a", "z": 0.5},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 2,
                "metric_name": "a",
                "updated_parameters": {"x": 1, "y": "b", "z": 0.25},
                "mean_t": np.array([3.0]),
                "covariance_t": np.array([[9.0]]),
                "z": 0.25,
                "timestamp": 25,
            },
            {
                "arm_name": "2_1",
                "parameters": {"x": 1, "y": "b", "z": 0.75},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 2,
                "metric_name": "a",
                "updated_parameters": {"x": 1, "y": "b", "z": 0.75},
                "mean_t": np.array([3.0]),
                "covariance_t": np.array([[9.0]]),
                "z": 0.75,
                "timestamp": 25,
            },
        ]
        arms = {
            # pyre-fixme[6]: For 1st param expected `Optional[str]` but got
            #  `Union[Dict[str, Union[float, str]], Dict[str, Union[int, str]], float,
            #  ndarray, str]`.
            # pyre-fixme[6]: For 2nd param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Union[Dict[str, Union[float, str]],
            #  Dict[str, Union[int, str]], float, ndarray, str]`.
            obs["arm_name"]: Arm(name=obs["arm_name"], parameters=obs["parameters"])
            for obs in truth
        }
        experiment = Mock()
        experiment._trial_indices_by_status = {status: set() for status in TrialStatus}
        trials = {
            obs["trial_index"]: (
                Trial(experiment, GeneratorRun(arms=[arms[obs["arm_name"]]]))
            )
            for obs in truth[:-1]
            # pyre-fixme[16]: Item `Dict` of `Union[Dict[str, typing.Union[float,
            #  str]], Dict[str, typing.Union[int, str]], float, ndarray, str]` has no
            #  attribute `startswith`.
            if not obs["arm_name"].startswith("2")
        }
        batch = BatchTrial(experiment, GeneratorRun(arms=[arms["2_0"], arms["2_1"]]))
        # pyre-fixme[6]: For 1st param expected
        #  `SupportsKeysAndGetItem[Union[Dict[str, Union[float, str]], Dict[str,
        #  Union[int, str]], float, ndarray, str], Trial]` but got `Dict[int,
        #  BatchTrial]`.
        trials.update({2: batch})
        # pyre-fixme[16]: Optional type has no attribute `mark_abandoned`.
        trials.get(1).mark_abandoned()
        # pyre-fixme[16]: Optional type has no attribute `mark_arm_abandoned`.
        trials.get(2).mark_arm_abandoned(arm_name="2_1")
        type(experiment).arms_by_name = PropertyMock(return_value=arms)
        type(experiment).trials = PropertyMock(return_value=trials)
        type(experiment).metrics = PropertyMock(
            return_value={"a": "a", "b": MapMetric(name="b")}
        )

        df = pd.DataFrame(truth)[
            ["arm_name", "trial_index", "mean", "sem", "metric_name"]
        ]
        data = Data(df=df)

        # Data includes metric "c" not attached to the experiment.
        with patch("ax.core.observation.logger.exception") as mock_logger:
            observations_from_data(experiment, data)
        mock_logger.assert_called_once()
        call_str = mock_logger.call_args.args[0]
        self.assertIn("Data contains metric c that has not been", call_str)

        # Add "c" to the experiment
        type(experiment).metrics = PropertyMock(
            return_value={"a": "a", "b": MapMetric(name="b"), "c": "c"}
        )
        # 1 arm is abandoned and 1 trial is abandoned, so only 2 observations should be
        # included.
        obs_no_abandoned = observations_from_data(experiment, data)
        self.assertEqual(len(obs_no_abandoned), 2)

        # Including all statuses for non-map metrics should yield all metrics except b
        obs_with_abandoned = observations_from_data(
            experiment, data, statuses_to_include=set(TrialStatus)
        )
        self.assertEqual(len(obs_with_abandoned), 4)
        for obs in obs_with_abandoned:
            if obs.arm_name == "1_0":
                self.assertEqual(set(obs.data.metric_names), {"a", "c"})

        # Including all statuses for all metrics should yield all metrics
        obs_with_abandoned = observations_from_data(
            experiment,
            data,
            statuses_to_include=set(TrialStatus),
            statuses_to_include_map_metric=set(TrialStatus),
        )
        self.assertEqual(len(obs_with_abandoned), 4)
        for obs in obs_with_abandoned:
            if obs.arm_name == "1_0":
                self.assertEqual(set(obs.data.metric_names), {"a", "b", "c"})

    def test_ObservationsFromDataWithSomeMissingTimes(self) -> None:
        truth = [
            {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a"},
                "mean": 2.0,
                "sem": 2.0,
                "trial_index": 1,
                "metric_name": "a",
                "start_time": 0,
            },
            {
                "arm_name": "0_1",
                "parameters": {"x": 1, "y": "b"},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 2,
                "metric_name": "a",
                "start_time": 0,
            },
            {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a"},
                "mean": 4.0,
                "sem": 4.0,
                "trial_index": 1,
                "metric_name": "b",
                "start_time": None,
            },
            {
                "arm_name": "0_1",
                "parameters": {"x": 1, "y": "b"},
                "mean": 5.0,
                "sem": 5.0,
                "trial_index": 2,
                "metric_name": "b",
                "start_time": None,
            },
        ]
        arms = {
            # pyre-fixme[6]: For 1st param expected `Optional[str]` but got
            #  `Union[None, Dict[str, Union[int, str]], float, str]`.
            # pyre-fixme[6]: For 2nd param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Union[None, Dict[str, Union[int, str]],
            #  float, str]`.
            obs["arm_name"]: Arm(name=obs["arm_name"], parameters=obs["parameters"])
            for obs in truth
        }
        experiment = Mock()
        experiment._trial_indices_by_status = {status: set() for status in TrialStatus}
        trials = {
            obs["trial_index"]: Trial(
                experiment, GeneratorRun(arms=[arms[obs["arm_name"]]])
            )
            for obs in truth
        }
        type(experiment).arms_by_name = PropertyMock(return_value=arms)
        type(experiment).trials = PropertyMock(return_value=trials)
        type(experiment).metrics = PropertyMock(return_value={"a": "a", "b": "b"})

        df = pd.DataFrame(truth)[
            ["arm_name", "trial_index", "mean", "sem", "metric_name", "start_time"]
        ]
        data = Data(df=df)
        observations = observations_from_data(experiment, data)

        self.assertEqual(len(observations), 2)
        # Get them in the order we want for tests below
        if observations[0].features.parameters["x"] == 1:
            observations.reverse()

        obsd_truth = {
            "metric_names": [["a", "b"], ["a", "b"]],
            "means": [np.array([2.0, 4.0]), np.array([3.0, 5.0])],
            "covariance": [np.diag([4.0, 16.0]), np.diag([9.0, 25.0])],
        }
        cname_truth = ["0_0", "0_1"]

        for i, obs in enumerate(observations):
            self.assertEqual(obs.features.parameters, truth[i]["parameters"])
            self.assertEqual(obs.features.trial_index, truth[i]["trial_index"])
            self.assertEqual(obs.data.metric_names, obsd_truth["metric_names"][i])
            self.assertTrue(np.array_equal(obs.data.means, obsd_truth["means"][i]))
            self.assertTrue(
                np.array_equal(obs.data.covariance, obsd_truth["covariance"][i])
            )
            self.assertEqual(obs.arm_name, cname_truth[i])

    def test_ObservationsFromDataWithDifferentTimesSingleTrial(
        self, with_nat: bool = False
    ) -> None:
        params0: TParameterization = {"x": 0, "y": "a"}
        params1: TParameterization = {"x": 1, "y": "a"}
        truth = [
            {
                "arm_name": "0_0",
                "parameters": params0,
                "mean": 2.0,
                "sem": 2.0,
                "trial_index": 0,
                "metric_name": "a",
                "start_time": "2024-03-20 08:45:00",
                "end_time": pd.NaT if with_nat else "2024-03-20 08:47:00",
            },
            {
                "arm_name": "0_0",
                "parameters": params0,
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 0,
                "metric_name": "b",
                "start_time": "2024-03-20 08:45:00",
                "end_time": pd.NaT if with_nat else "2024-03-20 08:46:00",
            },
            {
                "arm_name": "0_1",
                "parameters": params1,
                "mean": 4.0,
                "sem": 4.0,
                "trial_index": 0,
                "metric_name": "a",
                "start_time": "2024-03-20 08:43:00",
                "end_time": pd.NaT if with_nat else "2024-03-20 08:46:00",
            },
            {
                "arm_name": "0_1",
                "parameters": params1,
                "mean": 5.0,
                "sem": 5.0,
                "trial_index": 0,
                "metric_name": "b",
                "start_time": "2024-03-20 08:45:00",
                "end_time": pd.NaT if with_nat else "2024-03-20 08:46:00",
            },
        ]
        arms_by_name = {
            "0_0": Arm(name="0_0", parameters=params0),
            "0_1": Arm(name="0_1", parameters=params1),
        }
        experiment = Mock()
        experiment._trial_indices_by_status = {status: set() for status in TrialStatus}
        trials = {
            0: BatchTrial(experiment, GeneratorRun(arms=list(arms_by_name.values())))
        }
        type(experiment).arms_by_name = PropertyMock(return_value=arms_by_name)
        type(experiment).trials = PropertyMock(return_value=trials)
        type(experiment).metrics = PropertyMock(return_value={"a": "a", "b": "b"})
        df = pd.DataFrame(truth)[
            [
                "arm_name",
                "trial_index",
                "mean",
                "sem",
                "metric_name",
                "start_time",
                "end_time",
            ]
        ]
        data = Data(df=df)
        observations = observations_from_data(experiment, data)

        self.assertEqual(len(observations), 2)
        # Get them in the order we want for tests below
        if observations[0].features.parameters["x"] == 1:
            observations.reverse()

        obs_truth = {
            "arm_name": ["0_0", "0_1"],
            "parameters": [{"x": 0, "y": "a"}, {"x": 1, "y": "a"}],
            "metric_names": [["a", "b"], ["a", "b"]],
            "means": [np.array([2.0, 3.0]), np.array([4.0, 5.0])],
            "covariance": [np.diag([4.0, 9.0]), np.diag([16.0, 25.0])],
        }

        for i, obs in enumerate(observations):
            self.assertEqual(obs.features.parameters, obs_truth["parameters"][i])
            self.assertEqual(
                obs.features.trial_index,
                0,
            )
            self.assertEqual(obs.data.metric_names, obs_truth["metric_names"][i])
            # pyre-fixme[6]: For 2nd argument expected `Union[_SupportsArray[dtype[ty...
            self.assertTrue(np.array_equal(obs.data.means, obs_truth["means"][i]))
            self.assertTrue(
                # pyre-fixme[6]: For 2nd argument expected `Union[_SupportsArray[dtyp...
                np.array_equal(obs.data.covariance, obs_truth["covariance"][i])
            )
            self.assertEqual(obs.arm_name, obs_truth["arm_name"][i])
            self.assertEqual(obs.arm_name, obs_truth["arm_name"][i])
            if i == 0:
                self.assertEqual(
                    none_throws(obs.features.start_time).strftime("%Y-%m-%d %X"),
                    "2024-03-20 08:45:00",
                )
                self.assertIsNone(obs.features.end_time)
            else:
                self.assertIsNone(obs.features.start_time)
                if with_nat:
                    self.assertIsNone(obs.features.end_time)
                else:
                    self.assertEqual(
                        none_throws(obs.features.end_time).strftime("%Y-%m-%d %X"),
                        "2024-03-20 08:46:00",
                    )

    def test_observations_from_dataframe_with_nat_timestamps(self) -> None:
        self.test_ObservationsFromDataWithDifferentTimesSingleTrial(with_nat=True)

    def test_SeparateObservations(self) -> None:
        obs_arm_name = "0_0"
        obs = Observation(
            features=ObservationFeatures(parameters={"x": 20}),
            data=ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
            arm_name=obs_arm_name,
        )
        obs_feats, obs_data = separate_observations(observations=[obs])
        self.assertEqual(obs.features, ObservationFeatures(parameters={"x": 20}))
        self.assertEqual(
            obs.data,
            ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
        )
        with self.assertRaises(ValueError):
            recombine_observations(observation_features=obs_feats, observation_data=[])
        with self.assertRaises(ValueError):
            recombine_observations(
                observation_features=obs_feats, observation_data=obs_data, arm_names=[]
            )
        new_obs = recombine_observations(obs_feats, obs_data, [obs_arm_name])[0]
        self.assertEqual(new_obs.features, obs.features)
        self.assertEqual(new_obs.data, obs.data)
        self.assertEqual(new_obs.arm_name, obs_arm_name)
        obs_feats, obs_data = separate_observations(observations=[obs], copy=True)
        self.assertEqual(obs.features, ObservationFeatures(parameters={"x": 20}))
        self.assertEqual(
            obs.data,
            ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
        )

    def test_ObservationsWithCandidateMetadata(self) -> None:
        SOME_METADATA_KEY = "metadatum"
        truth = [
            {
                "arm_name": "0_0",
                "parameters": {"x": 0, "y": "a"},
                "mean": 2.0,
                "sem": 2.0,
                "trial_index": 0,
                "metric_name": "a",
            },
            {
                "arm_name": "1_0",
                "parameters": {"x": 1, "y": "b"},
                "mean": 3.0,
                "sem": 3.0,
                "trial_index": 1,
                "metric_name": "a",
            },
        ]
        arms = {
            # pyre-fixme[6]: For 1st param expected `Optional[str]` but got
            #  `Union[Dict[str, Union[int, str]], float, str]`.
            # pyre-fixme[6]: For 2nd param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Union[Dict[str, Union[int, str]], float,
            #  str]`.
            obs["arm_name"]: Arm(name=obs["arm_name"], parameters=obs["parameters"])
            for obs in truth
        }
        experiment = Mock()
        experiment._trial_indices_by_status = {status: set() for status in TrialStatus}
        trials = {
            obs["trial_index"]: Trial(
                experiment,
                GeneratorRun(
                    arms=[arms[obs["arm_name"]]],
                    candidate_metadata_by_arm_signature={
                        arms[obs["arm_name"]].signature: {
                            SOME_METADATA_KEY: f"value_{obs['trial_index']}"
                        }
                    },
                ),
            )
            for obs in truth
        }
        type(experiment).arms_by_name = PropertyMock(return_value=arms)
        type(experiment).trials = PropertyMock(return_value=trials)
        type(experiment).metrics = PropertyMock(return_value={"a": "a", "b": "b"})

        df = pd.DataFrame(truth)[
            ["arm_name", "trial_index", "mean", "sem", "metric_name"]
        ]
        data = Data(df=df)
        observations = observations_from_data(experiment, data)
        for observation in observations:
            self.assertEqual(
                # pyre-fixme[16]: Optional type has no attribute `get`.
                observation.features.metadata.get(SOME_METADATA_KEY),
                f"value_{observation.features.trial_index}",
            )

    def test_observation_repr(self) -> None:
        obs = Observation(
            features=ObservationFeatures(parameters={"x": 20}),
            data=ObservationData(
                means=np.array([1]), covariance=np.array([[2]]), metric_names=["a"]
            ),
            arm_name="0_0",
        )
        expected = (
            "Observation(\n"
            "    features=ObservationFeatures(parameters={'x': 20}),\n"
            "    data=ObservationData(metric_names=['a'], "
            "means=[1], covariance=[[2]]),\n"
            "    arm_name='0_0',\n"
            ")"
        )
        self.assertEqual(repr(obs), expected)
