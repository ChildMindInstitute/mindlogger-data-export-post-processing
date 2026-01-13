"""Test MindloggerData object."""

# ruff: noqa
from pandas.tests.arrays.boolean.test_arithmetic import data

from pathlib import Path

from datetime import date, time, timedelta, datetime
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mindlogger_data_export.parsers import FullResponseParser
from mindlogger_data_export import schema
from mindlogger_data_export import (
    MindloggerData,
    UserType,
)

FIXTURE_DIR = Path(__file__).parent.resolve() / "data"
WITH_REPORT = pytest.mark.datafiles(FIXTURE_DIR)


def test_mindlogger_data_create_nonexistent_raises_error():
    """MindloggerData.create should raise error for nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        MindloggerData.create(Path("nonexistent"))


def test_mindlogger_data_create_empty_raises_error(tmp_path: Path):
    """MindloggerData.create should raise error for empty directory."""
    with pytest.raises(ValueError):
        MindloggerData.create(tmp_path)


@WITH_REPORT
def test_mindlogger_source_users(datafiles: Path):
    """Test MindloggerData.source_users."""
    mindlogger_data = MindloggerData.create(datafiles)
    source_users = mindlogger_data.source_users
    source_user_ids = set(user.id for user in source_users)
    assert len(source_users) == 2
    assert all(user.user_type == UserType.SOURCE for user in source_users)
    assert source_users[0].id == "1e15e0bf-1b81-418e-9b80-20b0cb4cac33"
    assert source_users[1].id == "1e15e0bf-1b81-418e-9b80-20b0cb4cac33"


@WITH_REPORT
def test_mindlogger_target_users(datafiles: Path):
    """Test MindloggerData.target_users."""
    mindlogger_data = MindloggerData.create(datafiles)
    target_users = mindlogger_data.target_users
    target_user_ids = set(user.id for user in target_users)
    assert len(target_users) == 2
    assert all(user.user_type == UserType.TARGET for user in target_users)
    assert "1e15e0bf-1b81-418e-9b80-20b0cb4cac33" in target_user_ids
    assert "096cec52-0723-460d-a40e-3fcc1961b1b8" in target_user_ids


@WITH_REPORT
def test_mindlogger_input_users(datafiles: Path):
    """Test MindloggerData.input_users."""
    mindlogger_data = MindloggerData.create(datafiles)
    input_users = mindlogger_data.input_users
    assert len(input_users) == 1
    assert input_users[0].user_type == UserType.INPUT
    assert input_users[0].id == "1e15e0bf-1b81-418e-9b80-20b0cb4cac33"


@WITH_REPORT
def test_mindlogger_account_users(datafiles: Path):
    """Test MindloggerData.account_users."""
    mindlogger_data = MindloggerData.create(datafiles)
    account_users = mindlogger_data.account_users
    assert len(account_users) == 1
    assert account_users[0].user_type == UserType.ACCOUNT
    assert account_users[0].id == "6056fc79-931a-412e-b15b-d5798c826a23"


@pytest.fixture
def report():
    """Input general report."""
    return pl.DataFrame(
        {
            "applet_version": ["0.1.1"],
            "utc_timezone_offset": [timedelta(minutes=-300)],
            "target_user": [
                {
                    "id": "U1",
                    "secret_id": "SECU1",
                    "nickname": "NICK1",
                    "relation": "RELREL",
                    "tag": "TAG1",
                }
            ],
            "source_user": [
                {
                    "id": "U1",
                    "secret_id": "SECU1",
                    "nickname": "NICK1",
                    "relation": "RELREL",
                    "tag": "TAG1",
                }
            ],
            "input_user": [
                {
                    "id": "U1",
                    "secret_id": "SECU1",
                    "nickname": "NICK1",
                    "relation": "RELREL",
                    "tag": "TAG1",
                }
            ],
            "account_user": [
                {
                    "id": "U1",
                    "secret_id": "SECU1",
                    "nickname": "NICK1",
                    "relation": "RELREL",
                    "tag": "TAG1",
                }
            ],
            "item": [
                {
                    "id": "ItemId1",
                    "name": "ItemName1",
                    "prompt": "Prompt1",
                    "type": "singleSelect",
                    "raw_options": "",
                    "response_options": [
                        {
                            "name": "Option1",
                            "value": 0,
                            "score": 1,
                        },
                        {
                            "name": "Option2",
                            "value": 1,
                            "score": 2,
                        },
                        {
                            "name": "Option3",
                            "value": 2,
                            "score": 3,
                        },
                    ],
                }
            ],
            "response": [
                {
                    "status": "completed",
                    "value": {"value": [0, 1]},
                    "raw_score": 1,
                }
            ],
            "activity_flow": [
                {
                    "id": "FLOW1",
                    "name": "FlowName1",
                    "submission_id": "FlowSubmissionId1",
                }
            ],
            "activity": [{"id": "ActivityId1", "name": "ActivityName1"}],
            "activity_time": [
                {
                    "start_time": datetime(2012, 1, 2, 12, 10, 11),
                    "end_time": datetime(2012, 1, 2, 12, 15, 15),
                }
            ],
            "activity_schedule": [
                {
                    "id": "ActivityScheduleId1",
                    "history_id": "ActivityHistoryId",
                    "start_time": datetime(2012, 1, 1),
                }
            ],
        },
        schema=schema.INTERNAL_SCHEMA,
    )


def test_expand_options(report):
    _df = MindloggerData.expand_options(report)
    assert _df is not None


def test_expand_responses(report):
    _df = MindloggerData.expand_responses(report)
    assert _df is not None


def test_data_dictionary(report):
    _data = MindloggerData(report)
    assert len(list(_data.data_dictionary)) != 0
