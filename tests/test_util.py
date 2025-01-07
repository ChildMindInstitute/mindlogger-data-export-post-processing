"""Tests for the util module."""

import pandas as pd

from mindlogger_data_export import util


def test_val_score_mapping():
    options = "1: 0 (score: 0), 2: 1 (score: 2), 3: 2 (score: 3), 4: 3 (score: 4), 5: 4 (score: 5)"
    response = "value: 2"
    data = pd.DataFrame({"options": [options], "response": [response]})

    result = util.val_score_mapping(data)

    assert (pd.Series(["3"]) == result).all()
