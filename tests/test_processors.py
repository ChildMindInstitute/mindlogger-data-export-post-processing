"""Test report preprocessors."""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mindlogger_data_export.parsers import ResponseTransformer
from mindlogger_data_export.processors import (
    DateTimeProcessor,
    OptionsStructProcessor,
    ResponseStructProcessor,
    SubscaleProcessor,
)

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
        pytest.param("value: 2", {"type": "value", "value": ["2"]}, id="value"),
        pytest.param(
            "value: 1, 2, 3",
            {"type": "value", "value": ["1", "2", "3"]},
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
    ],
)
def test_response_preprocessor_single_row(response_field, expected):
    """Test ResponsePreprocessor."""
    preprocessor = ResponseStructProcessor()
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
                {"type": "value", "value": ["2"]},
                {"type": "value", "value": ["1", "2", "3"]},
                {"type": "file", "file": "./path/to/file.mp4"},
                {"type": "date", "date": date(2021, 2, 1)},
                {"type": "date", "date": date(2021, 5, 4)},
                {"type": "time", "time": time(12, 30)},
                {"type": "time_range", "time_range": timedelta(hours=3, minutes=-25)},
                {"type": "geo", "geo": {"latitude": 40.7128, "longitude": -74.0060}},
                {
                    "type": "matrix",
                    "matrix": [
                        {"row": "row1", "value": ["1"]},
                        {"row": "row2", "value": ["2"]},
                    ],
                },
                {
                    "type": "matrix",
                    "matrix": [
                        {"row": "row1", "value": ["1", "2"]},
                        {"row": "row2", "value": ["3", "4"]},
                    ],
                },
            ],
            id="multirow",
        ),
    ],
)
def test_response_preprocessor_multi_row(responses, expected):
    """Test ResponsePreprocessor on data with multiple rows."""
    preprocessor = ResponseStructProcessor()
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


def test_subscale_processor():
    """Test SubscaleProcessor."""
    report = pl.DataFrame(
        {
            "activity_submission_id": [
                "activity_submission_id_1",
                "activity_submission_id_2",
            ],
            "activity_flow_submission_id": ["", ""],
            "activity_scheduled_time": ["", ""],
            "activity_start_time": ["", ""],
            "activity_end_time": ["", ""],
            "flag": ["", ""],
            "secret_user_id": ["", ""],
            "userId": ["", ""],
            "source_user_subject_id": ["", ""],
            "source_user_secret_id": ["", ""],
            "source_user_nickname": ["", ""],
            "source_user_relation": ["", ""],
            "source_user_tag": ["", ""],
            "target_user_subject_id": ["", ""],
            "target_user_secret_id": ["", ""],
            "target_user_nickname": ["", ""],
            "target_user_tag": ["", ""],
            "input_user_subject_id": ["", ""],
            "input_user_secret_id": ["", ""],
            "input_user_nickname": ["", ""],
            "activity_id": ["", ""],
            "activity_name": ["", ""],
            "activity_flow_id": ["", ""],
            "activity_flow_name": ["", ""],
            "item": ["item_1", "item_2"],
            "item_id": ["item_id_1", "item_id_2"],
            "response": ["r_1", "r_2"],
            "prompt": ["prompt_1", "prompt_2"],
            "options": ["options_1", "options_2"],
            "version": ["version_1", "version_2"],
            "rawScore": ["3", "5"],
            "reviewing_id": ["reviewing_id_1", "reviewing_id_2"],
            "event_id": ["event_id_1", "event_id_2"],
            "timezone_offset": ["timezone_offset_1", "timezone_offset_2"],
            "Final SubScale Score": [1, 5],
            "Optional text for Final SubScale Score": ["t_1", "t_2"],
            "AVG": [2, 6],
            "Optional text for AVG": ["t_3", "t_4"],
        }
    )
    preprocessor = SubscaleProcessor()
    processed_report = preprocessor._run(report)
    assert "Final SubScale Score" not in processed_report.columns
    assert "Optional text for Final SubScale Score" not in processed_report.columns
    assert "AVG" not in processed_report.columns
    assert "Optional text for AVG" not in processed_report.columns
    assert processed_report.width == report.width - (2 * 2)
    assert processed_report.height == report.height + (4 * 2)
    expected = pl.DataFrame(
        {
            "activity_submission_id": (
                ["activity_submission_id_1"] * 5 + ["activity_submission_id_2"] * 5
            ),
            "item": [
                "item_1",
                "subscale__AVG",
                "activity_score",
                "subscale_text__AVG",
                "activity_score_text",
                "item_2",
                "subscale__AVG",
                "activity_score",
                "subscale_text__AVG",
                "activity_score_text",
            ],
            "response": ["r_1", "2", "1", "t_3", "t_1", "r_2", "6", "5", "t_4", "t_2"],
        }
    )
    assert_frame_equal(
        processed_report.select("activity_submission_id", "item", "response"),
        expected,
        check_row_order=False,
    )
