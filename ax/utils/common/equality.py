#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Callable

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from ax.utils.common.typeutils_nonnative import numpy_type_to_python_type


def equality_typechecker(eq_func: Callable) -> Callable:
    """A decorator to wrap all __eq__ methods to ensure that the inputs
    are of the right type.
    """

    # no type annotation for now; breaks sphinx-autodoc-typehints
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _type_safe_equals(self, other):
        if not isinstance(other, self.__class__):
            return False
        return eq_func(self, other)

    return _type_safe_equals


# pyre-fixme[2]: Parameter annotation cannot contain `Any`.
def same_elements(list1: list[Any], list2: list[Any]) -> bool:
    """Compare equality of two lists of core Ax objects.

    Assumptions:
        -- The contents of each list are types that implement __eq__
        -- The lists do not contain duplicates

    Checking equality is then the same as checking that the lists are the same
    length, and that both are subsets of the other.
    """

    if len(list1) != len(list2):
        return False

    matched = [False for _ in list2]
    for item1 in list1:
        matched_this_item = False
        for i, item2 in enumerate(list2):
            if not matched[i] and is_ax_equal(item1, item2):
                matched[i] = True
                matched_this_item = True
                break
        if not matched_this_item:
            return False
    return all(matched)


# pyre-fixme[2]: Parameter annotation cannot contain `Any`.
def is_ax_equal(one_val: Any, other_val: Any) -> bool:
    """Check for equality of two values, handling lists, dicts, dfs, floats,
    dates, and numpy arrays. This method and ``same_elements`` function
    as a recursive unit.

    Some special cases:
    - For datetime objects, the equality is checked up to a tolerance of one second.
    - For floats, ``np.isclose`` is used to check for almost-equality.
    - For lists (and dict values), ``same_elements`` is used. This ignores
      the ordering of the elements, and checks that the two lists are subsets
      of each other (under the assumption that there are no duplicates).
    - If the objects don't fall into any of the special cases, we use simple
      equality check and cast the output to a boolean. If the comparison
      or cast fails, we return False. Example: the comparison of a float with
      a numpy array (with multiple elements) will return False.
    """
    if isinstance(one_val, list) and isinstance(other_val, list):
        return same_elements(one_val, other_val)
    elif isinstance(one_val, dict) and isinstance(other_val, dict):
        return sorted(one_val.keys()) == sorted(other_val.keys()) and same_elements(
            list(one_val.values()), list(other_val.values())
        )
    elif isinstance(one_val, np.ndarray) and isinstance(other_val, np.ndarray):
        return np.array_equal(one_val, other_val, equal_nan=True)
    elif isinstance(one_val, datetime):
        return datetime_equals(one_val, other_val)
    elif isinstance(one_val, float) and isinstance(other_val, float):
        return np.isclose(one_val, other_val, equal_nan=True)
    elif isinstance(one_val, pd.DataFrame) and isinstance(other_val, pd.DataFrame):
        return dataframe_equals(one_val, other_val)
    else:
        try:
            return bool(one_val == other_val)
        except Exception:
            return False


def datetime_equals(dt1: datetime | None, dt2: datetime | None) -> bool:
    """Compare equality of two datetimes, up to a difference of one second."""
    if not dt1 and not dt2:
        return True
    if not (dt1 and dt2):
        return False
    return (dt1 - dt2).total_seconds() < 1.0


def dataframe_equals(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """Compare equality of two pandas dataframes."""
    try:
        if df1.empty and df2.empty:
            equal = True
        else:
            pd.testing.assert_frame_equal(
                df1.sort_index(axis=1), df2.sort_index(axis=1), check_exact=False
            )
            equal = True
    except AssertionError:
        equal = False

    return equal


def object_attribute_dicts_equal(
    one_dict: dict[str, Any], other_dict: dict[str, Any], skip_db_id_check: bool = False
) -> bool:
    """Utility to check if all items in attribute dicts of two Ax objects
    are the same.


    NOTE: Special-cases some Ax object attributes, like "_experiment" or
    "_model", where full equality is hard to check.

    Args:
        one_dict: First object's attribute dict (``obj.__dict__``).
        other_dict: Second object's attribute dict (``obj.__dict__``).
        skip_db_id_check: If ``True``, will exclude the ``db_id`` attributes from the
            equality check. Useful for ensuring that all attributes of an object are
            equal except the ids, with which one or both of them are saved to the
            database (e.g. if confirming an object before it was saved, to the version
            reloaded from the DB).
    """
    unequal_type, unequal_value = object_attribute_dicts_find_unequal_fields(
        one_dict=one_dict, other_dict=other_dict, skip_db_id_check=skip_db_id_check
    )
    return not bool(unequal_type or unequal_value)


# pyre-fixme[3]: Return annotation cannot contain `Any`.
def object_attribute_dicts_find_unequal_fields(
    one_dict: dict[str, Any],
    other_dict: dict[str, Any],
    fast_return: bool = True,
    skip_db_id_check: bool = False,
) -> tuple[dict[str, tuple[Any, Any]], dict[str, tuple[Any, Any]]]:
    """Utility for finding out what attributes of two objects' attribute dicts
    are unequal.

    Args:
        one_dict: First object's attribute dict (``obj.__dict__``).
        other_dict: Second object's attribute dict (``obj.__dict__``).
        fast_return: Boolean representing whether to return as soon as a
            single unequal attribute was found or to iterate over all attributes
            and collect all unequal ones.
        skip_db_id_check: If ``True``, will exclude the ``db_id`` attributes from the
            equality check. Useful for ensuring that all attributes of an object are
            equal except the ids, with which one or both of them are saved to the
            database (e.g. if confirming an object before it was saved, to the version
            reloaded from the DB).

    Returns:
        Two dictionaries:
            - attribute name to attribute values of unequal type (as a tuple),
            - attribute name to attribute values of unequal value (as a tuple).
    """
    unequal_type, unequal_value = {}, {}
    for field in one_dict:
        one_val = one_dict.get(field)
        other_val = other_dict.get(field)
        one_val = numpy_type_to_python_type(one_val)
        other_val = numpy_type_to_python_type(other_val)
        skip_type_check = skip_db_id_check and field == "_db_id"
        if not skip_type_check and (type(one_val) is not type(other_val)):
            unequal_type[field] = (one_val, other_val)
            if fast_return:
                return unequal_type, unequal_value

        if field == "_experiment":
            # Prevent infinite loop when checking equality of Trials (on Experiment,
            # with back-pointer), GenSteps (on GenerationStrategy), AnalysisRun-s
            # (on AnalysisScheduler).
            if one_val is None or other_val is None:
                equal = one_val is None and other_val is None
            else:
                # We compare `_name` because `name` attribute errors if not set.
                equal = one_val._name == other_val._name
        elif field == "_generation_strategy":
            # Prevent infinite loop when checking equality of Trials (on Experiment,
            # with back-pointer), GenSteps (on GenerationStrategy), AnalysisRun-s
            # (on AnalysisScheduler).
            if one_val is None or other_val is None:
                equal = one_val is None and other_val is None
            else:
                # We compare `name` because it is set dynamically in
                # some cases (see `GenerationStrategy.name` attribute).
                equal = one_val.name == other_val.name
        elif field == "analysis_scheduler":
            # prevent infinite loop when checking equality of analysis runs
            equal = one_val is other_val is None or (one_val.db_id == other_val.db_id)
        elif field == "_db_id":
            equal = skip_db_id_check or one_val == other_val
        elif field == "_model":
            # TODO[T52643706]: replace with per-`Adapter` method like
            # `equivalent_models`, to compare models more meaningfully.
            if not hasattr(one_val, "model") or not hasattr(other_val, "model"):
                equal = not hasattr(other_val, "model") and not hasattr(
                    other_val, "model"
                )
            else:
                # If adapters have a `model` attribute, the types of the
                # values of those attributes should be equal if the model
                # adapter is the same.
                equal = (
                    hasattr(one_val, "model")
                    and hasattr(other_val, "model")
                    and isinstance(one_val.model, type(other_val.model))
                )
        else:
            equal = is_ax_equal(one_val, other_val)

        if not equal:
            unequal_value[field] = (one_val, other_val)
            if fast_return:
                return unequal_type, unequal_value
    return unequal_type, unequal_value
