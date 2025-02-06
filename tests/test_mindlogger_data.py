"""Test MindloggerData object."""
# ruff: noqa

from pathlib import Path

from datetime import date, time, timedelta
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mindlogger_data_export.parsers import ResponseParser
from mindlogger_data_export import (
    MindloggerData,
    UserType,
)

FIXTURE_DIR = Path(__file__).parent.resolve() / "data"
WITH_REPORT = pytest.mark.datafiles(FIXTURE_DIR / "report.csv")


def test_mindlogger_data_create_nonexistent_raises_error():
    """MindloggerData.create should raise error for nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        MindloggerData.create(Path("nonexistent"))


def test_mindlogger_data_create_not_a_directory_raises_error(tmp_path: Path):
    """MindloggerData.create should raise error for non-directory."""
    file_path = tmp_path / "file.txt"
    file_path.touch()
    with pytest.raises(NotADirectoryError):
        MindloggerData.create(file_path)


def test_mindlogger_data_create_empty_raises_error(tmp_path: Path):
    """MindloggerData.create should raise error for empty directory."""
    with pytest.raises(FileNotFoundError):
        MindloggerData.create(tmp_path)


@WITH_REPORT
def test_mindlogger_source_users(datafiles: Path):
    """Test MindloggerData.source_users."""
    mindlogger_report = datafiles / "report.csv"
    mindlogger_data = MindloggerData(pl.read_csv(mindlogger_report))
    source_users = mindlogger_data.source_users
    assert len(source_users) == 1
    assert source_users[0].user_type == UserType.SOURCE
    assert source_users[0].subject_id == "ab64e77e-60b0-4725-8d93-079cceb8fc03"


@WITH_REPORT
def test_mindlogger_target_users(datafiles: Path):
    """Test MindloggerData.target_users."""
    mindlogger_report = datafiles / "report.csv"
    mindlogger_data = MindloggerData(pl.read_csv(mindlogger_report))
    target_users = mindlogger_data.target_users
    assert len(target_users) == 1
    assert target_users[0].user_type == UserType.TARGET
    assert target_users[0].subject_id == "ab64e77e-60b0-4725-8d93-079cceb8fc03"


@WITH_REPORT
def test_mindlogger_input_users(datafiles: Path):
    """Test MindloggerData.input_users."""
    mindlogger_report = datafiles / "report.csv"
    mindlogger_data = MindloggerData(pl.read_csv(mindlogger_report))
    input_users = mindlogger_data.input_users
    assert len(input_users) == 1
    assert input_users[0].user_type == UserType.INPUT
    assert input_users[0].subject_id == "ab64e77e-60b0-4725-8d93-079cceb8fc03"


@WITH_REPORT
def test_mindlogger_account_users(datafiles: Path):
    """Test MindloggerData.account_users."""
    mindlogger_report = datafiles / "report.csv"
    mindlogger_data = MindloggerData(pl.read_csv(mindlogger_report))
    account_users = mindlogger_data.account_users
    assert len(account_users) == 1
    assert account_users[0].user_type == UserType.ACCOUNT
    assert account_users[0].subject_id == "645e8cc0-a67a-c10f-93b4-50e000000000"


def test_long_response():
    """Test UnnestingResponsePreprocessor on data with multiple rows."""
    report = pl.DataFrame(
        {
            "parsed_response": [
                {"type": "raw_value", "raw_value": "10"},
                {"type": "text", "text": "Some text here"},
                {"type": "text", "text": "Some multiline\ntext here"},
                {"type": "null", "null_value": True},
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
        },
        schema={
            "parsed_response": pl.Struct(
                {
                    "type": pl.String,
                    "raw_value": pl.String,
                    "null_value": pl.Boolean,
                    "value": pl.List(pl.Int64),
                    "text": pl.String,
                    "file": pl.String,
                    "date": date,
                    "time": time,
                    "time_range": timedelta,
                    "geo": pl.Struct({"latitude": pl.Float64, "longitude": pl.Float64}),
                    "matrix": pl.List(
                        pl.Struct({"row": pl.String, "value": pl.List(pl.Int64)})
                    ),
                }
            )
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
            "null",
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
    expanded_report = MindloggerData.expand_responses(report).drop("parsed_response")
    assert_frame_equal(
        expanded_report,
        expected_df,
        check_column_order=False,
    )


# def test_score_value_mapping_processor():
#     """Test ScoreValueMappingProcessor."""
#     preprocessor = ScoredTypedData()
#     item_id_cols = [
#         "version",
#         "activity_flow_id",
#         "activity_flow_name",
#         "activity_id",
#         "activity_name",
#         "item_id",
#         "item",
#         "prompt",
#     ]
#     report = pl.DataFrame(
#         {
#             "version": ["1.0", "1.0", "1.0"],
#             "activity_flow_id": [
#                 "ACTIVITY_FLOW_ID_1",
#                 "ACTIVITY_FLOW_ID_2",
#                 "ACTIVITY_FLOW_ID_3",
#             ],
#             "activity_flow_name": [
#                 "ACTIVITY_FLOW_NAME_1",
#                 "ACTIVITY_FLOW_NAME_2",
#                 "ACTIVITY_FLOW_NAME_3",
#             ],
#             "activity_id": ["ACTIVITY_ID_1", "ACTIVITY_ID_2", "ACTIVITY_ID_3"],
#             "activity_name": ["ACTIVITY_NAME_1", "ACTIVITY_NAME_2", "ACTIVITY_NAME_3"],
#             "item_id": ["ITEM_ID_1", "ITEM_ID_2", "ITEM_ID_3"],
#             "item": ["ITEM_1", "ITEM_2", "ITEM_3"],
#             "prompt": ["PROMPT_1", "PROMPT_2", "PROMPT_3"],
#             "options": [
#                 "Max: 2, Min: 0",
#                 "1: 0, 2: 1, 3: 2",
#                 "1: 0 (score: 3), 2: 1 (score: 4), 3: 2 (score: 5)",
#             ],
#             "response": ["value: 1", "value: 2", "value: 2"],
#         },
#     )
#     expected_df = report.with_columns(
#         option_name=pl.Series(["1", "3", "3"]),
#         option_score=pl.Series([1, None, 5]),
#     ).drop("options", "response")
#     processed_report = preprocessor.process(report)

#     processed_report = processed_report.select(
#         item_id_cols + ["option_name", "option_score"]
#     )
#     assert_frame_equal(
#         processed_report,
#         expected_df,
#         check_column_order=False,
#     )


# def test_mindlogger_items(mindlogger_export_config: MindloggerExportConfig):
#     """Test MindloggerData.items."""
#     mindlogger_report = mindlogger_export_config.input_dir / "report.csv"  # noqa: ERA001
#     mindlogger_data = MindloggerData(pl.read_csv(mindlogger_report))
#     # items = mindlogger_data.items
#     # assert len(items) == 23

#     # 15 Item IDs
#     item_ids = {i.id for i in items}
#     assert len(item_ids) == 15
#     assert item_ids == {
#         "4260fed8-d266-4f13-a543-817ca946c47d",
#         "d95159b5-f44c-4975-ae24-1d26022afe9c",
#         "f197953f-aa8a-4ac0-97a2-87bc7b634306",
#         "5d48d463-5fb7-48a6-8d77-a864e66efa6e",
#         "a9b58769-7473-4127-8c39-813c0c3ecf4a",
#         "a44a3ca9-19d0-48cc-b200-293f454597b7",
#         "1b91619e-cf50-4743-b7e6-381a768bb68d",
#         "9d9f8dda-d6ca-496b-b20c-b992d74bd91f",
#         "ce9424bd-5fce-4926-96f2-63a2dec27dfe",
#         "12d9f51a-3988-4515-9b3d-df5a13035917",
#         "4285ee68-6905-4d9b-be4d-940f3a805027",
#         "cd926b89-06a9-4de4-956e-6879d55e2258",
#         "57179d77-244a-4132-95bd-d29609ccfd68",
#         "6ce16878-2261-458f-b746-7cb6bbd0173f",
#         "76c0b654-a4c6-4dd7-8270-33f4ee06d57b",
#     }

#     # 20 Item names
#     item_names = {i.name for i in items}
#     assert len(item_names) == 20
#     assert item_names == {
#         "Item4-Text",
#         "Itemms",
#         "Item2-Multiple_Selection",
#         "Item3-Slider",
#         "age_screen",
#         "Itemns",
#         "Itemsl",
#         "Itemss",
#         "slider_alert_item",
#         "Item5-Number_Selection",
#         "suicide_alert",
#         "gender_screen",
#         "q4",
#         "Itemst",
#         "Date",
#         "Item2_test",
#         "Item1",
#         "Item1-Single_Selection",
#         "q2",
#         "q6",
#     }

#     item_prompts = {i.prompt for i in items}
#     assert len(item_prompts) == 11
#     assert item_prompts == {
#         "date",
#         "select",
#         "Itemms",
#         "Itemns",
#         "How do you describe yourself?<br><br>*Please provide your response as accurately as possible. The information you provide is important for ensuring the accuracy of your results. If you have any concerns about how your information will be used, please refer to our Terms of Service.*",
#         "How old are you?<br><br>*Please provide your response as accurately as possible. The information you provide is important for ensuring the accuracy of your results. If you have any concerns about how your information will be used, please refer to our Terms of Service.*",
#         "Itemsl",
#         "suicide alert",
#         "Itemss",
#         "Itemst",
#         "slider_alert",
#     }

#     item_options = {i.options for i in items}
#     assert len(item_options) == 14
#     assert item_options == {
#         "0: 0 (score: 1), 1: 1 (score: 2), 2: 2 (score: 3), 3: 3 (score: 4), 4: 4 (score: 5), 5: 5 (score: 6), 6: 6 (score: 7), 7: 7 (score: 8), 8: 8 (score: 9), 9: 9 (score: 10), 10: 10 (score: 11), 11: 11 (score: 12), 12: 12 (score: 13)",
#         None,
#         "Male: 0, Female: 1",
#         "Min: 0, Max: 10",
#         "4: 0 (score: 4), 8: 1 (score: 8), None: 2 (score: 0)",
#         "1: 0 (score: 1), 2: 1 (score: 2), 3: 2 (score: 3)",
#         "1: 0 (score: 0), 2: 1 (score: 2), 3: 2 (score: 3), 4: 3 (score: 4), 5: 4 (score: 5)",
#         "4: 0, 8: 1, None: 2",
#         "1: 0 (score: 1), 2: 1 (score: 2), 3: 2 (score: 3), 4: 3 (score: 4)",
#         "1: 0, 2: 1, 3: 2",
#         "0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12",
#         "No: 0 (score: 0), Yes: 1 (score: 1)",
#         "0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10",
#         "Yes: 0, No: 1",
#     }
