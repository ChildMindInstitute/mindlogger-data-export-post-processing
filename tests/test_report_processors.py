"""Test report preprocessors."""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import polars.selectors as cs
import pytest
from polars.testing.asserts import assert_frame_equal

from mindlogger_data_export import (
    DateTimeProcessor,
    OptionsStructProcessor,
    ScoredTypedData,
    StructResponseProcessor,
    UnnestingResponseProcessor,
)
from mindlogger_data_export.parsers import ResponseTransformer

NYC_TZ = ZoneInfo("America/New_York")


def test_datetime_preprocessor_init():
    """Test DateTimePreprocessor."""
    preprocessor = DateTimeProcessor()
    assert preprocessor.NAME == "DateTime"


@pytest.mark.parametrize(
    ("field_name", "timestamps", "expected"),
    [
        pytest.param(
            "activity_scheduled_time",
            ["1733927671657", "not scheduled"],
            [
                datetime(2024, 12, 11, 14, 34, 31, 657000, ZoneInfo("UTC")),
                None,
            ],
            id="activity_scheduled_time",
        ),
        pytest.param(
            "activity_start_time",
            [1733927671657, 1726245913260],
            [
                datetime(2024, 12, 11, 14, 34, 31, 657000, ZoneInfo("UTC")),
                datetime(2024, 9, 13, 16, 45, 13, 260000, ZoneInfo("UTC")),
            ],
            id="activity_start_time",
        ),
    ],
)
def test_date_preprocessor(field_name, timestamps, expected):
    """Test DateTimePreprocessor."""
    preprocessor = DateTimeProcessor()
    report = pl.DataFrame({field_name: timestamps})
    processed_report = preprocessor._run(report)
    assert processed_report[field_name + "_dt"].to_list() == expected


@pytest.mark.parametrize(
    ("response_field", "expected"),
    [
        pytest.param("10", {"type": "raw_value", "raw_value": "10"}, id="raw_value"),
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
        pytest.param("value: 2", {"type": "value", "value": [2]}, id="value"),
        pytest.param(
            "value: 1, 2, 3",
            {"type": "value", "value": [1, 2, 3]},
            id="multivalue",
        ),
        pytest.param(
            "./path/to/file.mp4",
            {"type": "file", "file": "./path/to/file.mp4"},
            id="file",
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
            "time: hr 12 min 30", {"type": "time", "time": time(12, 30)}, id="time"
        ),
        pytest.param(
            "time_range: from hr 9 min 30 / to hr 12 min 5",
            {"type": "time_range", "time_range": timedelta(hours=3, minutes=-25)},
            id="timerange",
        ),
        pytest.param(
            "geo: lat 40.7128 long -74.0060",
            {"type": "geo", "geo": {"latitude": 40.7128, "longitude": -74.0060}},
            id="geo",
        ),
        pytest.param(
            "row1: 1\nrow2: 2",
            {
                "type": "matrix",
                "matrix": [
                    {"row": "row1", "value": [1]},
                    {"row": "row2", "value": [2]},
                ],
            },
            id="singleperrow",
        ),
        pytest.param(
            "row1: 1, 2\nrow2: 3, 4",
            {
                "type": "matrix",
                "matrix": [
                    {"row": "row1", "value": [1, 2]},
                    {"row": "row2", "value": [3, 4]},
                ],
            },
            id="multiperrow",
        ),
    ],
)
def test_response_preprocessor_single_row(response_field, expected):
    """Test ResponsePreprocessor."""
    preprocessor = StructResponseProcessor()
    schema = ResponseTransformer().DEFAULT_SCHEMA
    responses = [response_field]
    processed_expected = schema | expected

    processed_responses = [processed_expected]

    report = pl.DataFrame(
        {"response": responses},
    )
    processed_report = preprocessor._run(report)
    assert processed_report["parsed_response"].to_list() == processed_responses


@pytest.mark.parametrize(
    ("responses", "expected"),
    [
        pytest.param(
            [
                "10",
                "text: Some text here",
                "text: Some multiline\ntext here",
                "value: null",
                "value: 2",
                "value: 1, 2, 3",
                "./path/to/file.mp4",
                "date: 1/2/21",
                "date: 04/05/2021",
                "time: hr 12 min 30",
                "time_range: from hr 9 min 30 / to hr 12 min 5",
                "geo: lat 40.7128 long -74.0060",
                "row1: 1\nrow2: 2",
                "row1: 1, 2\nrow2: 3, 4",
            ],
            [
                {"type": "raw_value", "raw_value": "10"},
                {"type": "text", "text": "Some text here"},
                {"type": "text", "text": "Some multiline\ntext here"},
                {"type": "null_value", "null_value": True},
                {"type": "value", "value": [2]},
                {"type": "value", "value": [1, 2, 3]},
                {"type": "file", "file": "./path/to/file.mp4"},
                {"type": "date", "date": date(2021, 2, 1)},
                {"type": "date", "date": date(2021, 5, 4)},
                {"type": "time", "time": time(12, 30)},
                {"type": "time_range", "time_range": timedelta(hours=3, minutes=-25)},
                {"type": "geo", "geo": {"latitude": 40.7128, "longitude": -74.0060}},
                {
                    "type": "matrix",
                    "matrix": [
                        {"row": "row1", "value": [1]},
                        {"row": "row2", "value": [2]},
                    ],
                },
                {
                    "type": "matrix",
                    "matrix": [
                        {"row": "row1", "value": [1, 2]},
                        {"row": "row2", "value": [3, 4]},
                    ],
                },
            ],
            id="multirow",
        ),
    ],
)
def test_response_preprocessor_multi_row(responses, expected):
    """Test ResponsePreprocessor on data with multiple rows."""
    preprocessor = StructResponseProcessor()
    schema = ResponseTransformer().DEFAULT_SCHEMA
    expected = [schema | e for e in expected]
    report = pl.DataFrame(
        {"response": responses},
    )
    processed_report = preprocessor._run(report)
    assert processed_report["parsed_response"].to_list() == expected


@pytest.mark.parametrize(
    ("options_field", "expected"),
    [
        pytest.param(
            "Max: 2, Min: 0",
            [
                {"name": "0", "value": 0, "score": 0},
                {"name": "1", "value": 1, "score": 1},
                {"name": "2", "value": 2, "score": 2},
            ],
            id="max_min",
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
    ],
)
def test_options_preprocessor(options_field, expected):
    preprocessor = OptionsStructProcessor()
    options = [options_field]

    processed_options = [expected]

    report = pl.DataFrame(
        {"options": options},
    )
    processed_report = preprocessor._run(report)
    assert processed_report["parsed_options"].to_list() == processed_options


def test_unnesting_response_preprocessor():
    """Test UnnestingResponsePreprocessor on data with multiple rows."""
    preprocessor = UnnestingResponseProcessor()
    report = pl.DataFrame(
        {
            "response": [
                "10",
                "text: Some text here",
                "text: Some multiline\ntext here",
                "value: null",
                "value: 2",
                "value: 1, 2, 3",
                "./path/to/file.mp4",
                "date: 1/2/21",
                "date: 04/05/2021",
                "time: hr 12 min 30",
                "time_range: from hr 9 min 30 / to hr 12 min 5",
                "geo: lat 40.7128 long -74.0060",
                "row1: 1\nrow2: 2",
                "row1: 1, 2\nrow2: 3, 4",
            ],
        },
    )
    expected_df = {
        "response_raw_value": ["10"] + [None] * 19,
        "response_text": [None]
        + ["Some text here", "Some multiline\ntext here"]
        + [None] * 17,
        "response_null_value": [None] * 3 + [True] + [None] * 16,
        "response_value": [None] * 4 + [2, 1, 2, 3] + [None] * 12,
        "response_file": [None] * 8 + ["./path/to/file.mp4"] + [None] * 11,
        "response_date": [None] * 9 + [date(2021, 2, 1), date(2021, 5, 4)] + [None] * 9,
        "response_time": [None] * 11 + [time(12, 30)] + [None] * 8,
        "response_time_range": [None] * 12
        + [timedelta(hours=3, minutes=-25)]
        + [None] * 7,
        "response_geo_latitude": [None] * 13 + [40.7128] + [None] * 6,
        "response_geo_longitude": [None] * 13 + [-74.0060] + [None] * 6,
        "response_matrix_row": [None] * 14
        + ["row1", "row2"]
        + ["row1", "row1", "row2", "row2"],  # [None] * 4,
        "response_matrix_value": [None] * 14 + [1, 2] + [1, 2, 3, 4],  # [None] * 4,
        "response_type": [
            "raw_value",
            "text",
            "text",
            "null_value",
            "value",
            "value",
            "value",
            "value",
            "file",
            "date",
            "date",
            "time",
            "time_range",
            "geo",
            "matrix",
            "matrix",
            "matrix",
            "matrix",
            "matrix",
            "matrix",
        ],
    }
    expected_df = pl.DataFrame(
        expected_df,
        schema={
            "response_type": pl.String,
            "response_raw_value": pl.String,
            "response_text": pl.String,
            "response_null_value": pl.Boolean,
            "response_file": pl.String,
            "response_value": pl.Int64,
            "response_date": pl.Date,
            "response_time": pl.Time,
            "response_time_range": pl.Duration,
            "response_geo_latitude": pl.Float64,
            "response_geo_longitude": pl.Float64,
            "response_matrix_row": pl.String,
            "response_matrix_value": pl.Int64,
        },
    )
    processed_report = preprocessor.process(report)
    assert_frame_equal(
        processed_report.select(cs.starts_with(preprocessor.COLUMN_PREFIX)),
        expected_df,
        check_column_order=False,
    )


def test_score_value_mapping_processor():
    """Test ScoreValueMappingProcessor."""
    preprocessor = ScoredTypedData()
    item_id_cols = [
        "version",
        "activity_flow_id",
        "activity_flow_name",
        "activity_id",
        "activity_name",
        "item_id",
        "item",
        "prompt",
    ]
    report = pl.DataFrame(
        {
            "version": ["1.0", "1.0", "1.0"],
            "activity_flow_id": [
                "ACTIVITY_FLOW_ID_1",
                "ACTIVITY_FLOW_ID_2",
                "ACTIVITY_FLOW_ID_3",
            ],
            "activity_flow_name": [
                "ACTIVITY_FLOW_NAME_1",
                "ACTIVITY_FLOW_NAME_2",
                "ACTIVITY_FLOW_NAME_3",
            ],
            "activity_id": ["ACTIVITY_ID_1", "ACTIVITY_ID_2", "ACTIVITY_ID_3"],
            "activity_name": ["ACTIVITY_NAME_1", "ACTIVITY_NAME_2", "ACTIVITY_NAME_3"],
            "item_id": ["ITEM_ID_1", "ITEM_ID_2", "ITEM_ID_3"],
            "item": ["ITEM_1", "ITEM_2", "ITEM_3"],
            "prompt": ["PROMPT_1", "PROMPT_2", "PROMPT_3"],
            "options": [
                "Max: 2, Min: 0",
                "1: 0, 2: 1, 3: 2",
                "1: 0 (score: 3), 2: 1 (score: 4), 3: 2 (score: 5)",
            ],
            "response": ["value: 1", "value: 2", "value: 2"],
        },
    )
    expected_df = report.with_columns(
        option_name=pl.Series(["1", "3", "3"]),
        option_score=pl.Series([1, None, 5]),
    ).drop("options", "response")
    processed_report = preprocessor.process(report)

    processed_report = processed_report.select(
        item_id_cols + ["option_name", "option_score"]
    )
    assert_frame_equal(
        processed_report,
        expected_df,
        check_column_order=False,
    )
