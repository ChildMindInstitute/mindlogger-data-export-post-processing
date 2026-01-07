"""Test ResponseParser."""

from datetime import date, time, timedelta

import pytest

from mindlogger_data_export.parsers import (
    DateResponseParser,
    FullResponseParser,
    GeoResponseParser,
    OptionsParser,
    SingleSelectResponseParser,
    SubscaleResponseParser,
    TextResponseParser,
)
from mindlogger_data_export.schema import RESPONSE_VALUE_DICT_SCHEMA


@pytest.mark.parametrize(
    ("response_field", "expected"),
    [
        pytest.param("10", {"type": "raw_value", "raw_value": "10"}, id="raw_value"),
        pytest.param(
            "10.456", {"type": "raw_value", "raw_value": "10.456"}, id="raw_value"
        ),
        pytest.param(
            "text: Some text here",
            {"type": "text", "text": "Some text here"},
            id="text",
        ),
        pytest.param(
            "text: Some multiline\ntext here",
            {"type": "text", "text": "Some multiline\ntext here"},
            id="text_multiline",
        ),
        pytest.param(
            "value: null", {"type": "null_value", "null_value": True}, id="raw_value"
        ),
        pytest.param("value: 2", {"type": "value", "value": ["2"]}, id="value"),
        pytest.param(
            "value: 0.95", {"type": "value", "value": ["0.95"]}, id="value_frac"
        ),
        pytest.param(
            "value: 1, 2, 3",
            {"type": "value", "value": ["1", "2", "3"]},
            id="multivalue",
        ),
        pytest.param(
            "abcd-b3j5.mp4",
            {"type": "file", "file": "abcd-b3j5.mp4"},
            id="file",
        ),
        pytest.param(
            "./path/to/file.mp4",
            {"type": "file", "file": "./path/to/file.mp4"},
            id="filepath",
        ),
        pytest.param(
            "value: 1 | text: Some text here",
            {
                "type": "value",
                "value": ["1"],
                "optional_text": "Some text here",
            },
            id="value_with_text",
        ),
        pytest.param(
            "date: 1/2/21", {"type": "date", "date": date(2021, 2, 1)}, id="date"
        ),
        pytest.param(
            "date: 04/05/2021",
            {"type": "date", "date": date(2021, 5, 4)},
            id="date_padded",
        ),
        pytest.param(
            "date: 26/11/2024", {"type": "date", "date": date(2024, 11, 26)}, id="date2"
        ),
        pytest.param(
            "time: hr 12 min 30", {"type": "time", "time": time(12, 30)}, id="time"
        ),
        pytest.param(
            "time_range: from hr 9 min 30 / to hr 12 min 5",
            {"type": "time_range", "time_range": timedelta(hours=3, minutes=-25)},
            id="timerange",
        ),
        pytest.param(
            "time_range: from (hr 16, min 0) / to (hr 16, min 2)",
            {"type": "time_range", "time_range": timedelta(hours=0, minutes=2)},
            id="timerange_parens",
        ),
        pytest.param(
            "geo: lat 40.7128 long -74.0060",
            {"type": "geo", "geo": {"latitude": 40.7128, "longitude": -74.0060}},
            id="geo",
        ),
        pytest.param(
            "geo: lat (40.7128) / long (-74.0060)",
            {"type": "geo", "geo": {"latitude": 40.7128, "longitude": -74.0060}},
            id="geo_parens",
        ),
        pytest.param(
            "row1: 1\nrow2: 2",
            {
                "type": "matrix",
                "matrix": [
                    {"row": "row1", "value": ["1"]},
                    {"row": "row2", "value": ["2"]},
                ],
            },
            id="singleperrow",
        ),
        pytest.param(
            "row1: 1, 2\nrow2: 3, 4",
            {
                "type": "matrix",
                "matrix": [
                    {"row": "row1", "value": ["1", "2"]},
                    {"row": "row2", "value": ["3", "4"]},
                ],
            },
            id="multiperrow",
        ),
        pytest.param(
            "row1: c1, c2\nrow2: c3, c4",
            {
                "type": "matrix",
                "matrix": [
                    {"row": "row1", "value": ["c1", "c2"]},
                    {"row": "row2", "value": ["c3", "c4"]},
                ],
            },
            id="multiperrow_str",
        ),
    ],
)
def test_response_parser(response_field, expected):
    """Test ResponsePreprocessor."""
    parser = FullResponseParser()
    schema = RESPONSE_VALUE_DICT_SCHEMA
    assert parser.parse(response_field) == schema | expected


@pytest.mark.parametrize(
    ("options_field", "expected"),
    [
        pytest.param(
            "1: 0 (score: 3)", [{"name": "1", "value": 0, "score": 3}], id="score"
        ),
        pytest.param(
            "1: 0 (score: -2)",
            [{"name": "1", "value": 0, "score": -2}],
            id="negative-score",
        ),
        pytest.param(
            "Max: 2, Min: 0",
            [
                {"name": 0, "value": 0, "score": 0},
                {"name": 1, "value": 1, "score": 1},
                {"name": 2, "value": 2, "score": 2},
            ],
            id="max_min",
        ),
        pytest.param(
            "Min: 1, Max: 3",
            [
                {"name": 1, "value": 1, "score": 1},
                {"name": 2, "value": 2, "score": 2},
                {"name": 3, "value": 3, "score": 3},
            ],
            id="min_max",
        ),
        pytest.param(
            "1: 0, 2: 1, 3: 2",
            [
                {"name": "1", "value": 0, "score": None},
                {"name": "2", "value": 1, "score": None},
                {"name": "3", "value": 2, "score": None},
            ],
            id="values",
        ),
        pytest.param(
            "1: 0 (score: 5), 2: 1 (score: 4), 3: 2 (score: 1)",
            [
                {"name": "1", "value": 0, "score": 5},
                {"name": "2", "value": 1, "score": 4},
                {"name": "3", "value": 2, "score": 1},
            ],
        ),
    ],
)
def test_options_parser(options_field, expected):
    parser = OptionsParser()
    assert parser.parse(options_field) == expected


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        pytest.param(
            "value: 10",
            {"null_value": None, "single_value": 10, "optional_text": None},
            id="value_only",
        ),
        pytest.param(
            "value: 1 | text: optional text",
            {"null_value": None, "single_value": 1, "optional_text": "optional text"},
            id="with_optional_text",
        ),
        pytest.param(
            "value: null",
            {"null_value": True, "single_value": None, "optional_text": None},
            id="null_value",
        ),
    ],
)
def test_single_select_response_parser(input_str, expected):
    parser = SingleSelectResponseParser()
    parsed = parser.parse(input_str)
    assert parsed is not None
    assert parsed == RESPONSE_VALUE_DICT_SCHEMA | expected


def test_multiselect_response_parser():
    pass


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        pytest.param(
            "Some text here",
            {"type": "text", "text": "Some text here"},
            id="text",
        ),
        pytest.param(
            "Some multiline\ntext here",
            {"type": "text", "text": "Some multiline\ntext here"},
            id="text_multiline",
        ),
    ],
)
def test_text_response_parser(input_str, expected):
    parser = TextResponseParser()
    parsed = parser.parse(input_str)
    assert parsed is not None
    assert parsed == RESPONSE_VALUE_DICT_SCHEMA | expected


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        pytest.param(
            "4.2",
            {"type": "subscale", "subscale": 4.2},
            id="text",
        )
    ],
)
def test_subscale_response_parser(input_str, expected):
    parser = SubscaleResponseParser()
    parsed = parser.parse(input_str)
    assert parsed is not None
    assert parsed == RESPONSE_VALUE_DICT_SCHEMA | expected


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        pytest.param(
            "geo: lat 40.7128 long -74.0060",
            {"type": "geo", "geo": {"latitude": 40.7128, "longitude": -74.0060}},
            id="geo",
        ),
        pytest.param(
            "geo: lat (40.7128) / long (-74.0060)",
            {"type": "geo", "geo": {"latitude": 40.7128, "longitude": -74.0060}},
            id="geo_parens",
        ),
    ],
)
def test_geo_response_parser(input_str, expected):
    parser = GeoResponseParser()
    parsed = parser.parse(input_str)
    assert parsed is not None
    assert parsed == RESPONSE_VALUE_DICT_SCHEMA | expected


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        pytest.param(
            "date: 1/2/21", {"type": "date", "date": date(2021, 2, 1)}, id="date"
        ),
        pytest.param(
            "date: 04/05/2021",
            {"type": "date", "date": date(2021, 5, 4)},
            id="date_padded",
        ),
        pytest.param(
            "date: 26/11/2024", {"type": "date", "date": date(2024, 11, 26)}, id="date2"
        ),
    ],
)
def test_date_response_parser(input_str, expected):
    parser = DateResponseParser()
    parsed = parser.parse(input_str)
    assert parsed is not None
    assert parsed == RESPONSE_VALUE_DICT_SCHEMA | expected


def test_typed_response_parser():
    pass
