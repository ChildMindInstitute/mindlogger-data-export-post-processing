"""Test MindloggerData object."""
# ruff: noqa

import shutil
from pathlib import Path

import polars as pl
import pytest

from mindlogger_data_export import (
    MindloggerData,
    MindloggerExportConfig,
    UserType,
)

FIXTURE_DIR = Path(__file__).parent.resolve() / "data"
WITH_REPORT = pytest.mark.datafiles(FIXTURE_DIR / "mindlogger_report.csv")


@pytest.fixture
def mindlogger_export_config(tmp_path: Path) -> MindloggerExportConfig:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (tmp_path / "output").mkdir()
    shutil.copy(
        Path(__file__).parent.resolve() / "data/report.csv",
        input_dir / "report.csv",
    )
    return MindloggerExportConfig(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
    )


def test_mindlogger_data_create_nonexistent_raises_error():
    """MindloggerData.create should raise error for nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        MindloggerData.create(
            MindloggerExportConfig(
                input_dir=Path("nonexistent"), output_dir=Path("nonexistent")
            )
        )


def test_mindlogger_data_create_not_a_directory_raises_error(tmp_path: Path):
    """MindloggerData.create should raise error for non-directory."""
    file_path = tmp_path / "file.txt"
    file_path.touch()
    with pytest.raises(NotADirectoryError):
        MindloggerData.create(
            MindloggerExportConfig(input_dir=file_path, output_dir=tmp_path)
        )


def test_mindlogger_data_create_empty_raises_error(tmp_path: Path):
    """MindloggerData.create should raise error for empty directory."""
    with pytest.raises(FileNotFoundError):
        MindloggerData.create(
            MindloggerExportConfig(input_dir=tmp_path, output_dir=tmp_path)
        )


def test_mindlogger_source_users(mindlogger_export_config: MindloggerExportConfig):
    """Test MindloggerData.source_users."""
    mindlogger_report = mindlogger_export_config.input_dir / "report.csv"
    mindlogger_data = MindloggerData(pl.read_csv(mindlogger_report))
    source_users = mindlogger_data.source_users
    assert len(source_users) == 1
    assert source_users[0].user_type == UserType.SOURCE
    assert source_users[0].subject_id == "ab64e77e-60b0-4725-8d93-079cceb8fc03"


def test_mindlogger_target_users(mindlogger_export_config: MindloggerExportConfig):
    """Test MindloggerData.target_users."""
    mindlogger_report = mindlogger_export_config.input_dir / "report.csv"
    mindlogger_data = MindloggerData(pl.read_csv(mindlogger_report))
    target_users = mindlogger_data.target_users
    assert len(target_users) == 1
    assert target_users[0].user_type == UserType.TARGET
    assert target_users[0].subject_id == "ab64e77e-60b0-4725-8d93-079cceb8fc03"


def test_mindlogger_input_users(mindlogger_export_config: MindloggerExportConfig):
    """Test MindloggerData.input_users."""
    mindlogger_report = mindlogger_export_config.input_dir / "report.csv"
    mindlogger_data = MindloggerData(pl.read_csv(mindlogger_report))
    input_users = mindlogger_data.input_users
    assert len(input_users) == 1
    assert input_users[0].user_type == UserType.INPUT
    assert input_users[0].subject_id == "ab64e77e-60b0-4725-8d93-079cceb8fc03"


def test_mindlogger_account_users(mindlogger_export_config: MindloggerExportConfig):
    """Test MindloggerData.account_users."""
    mindlogger_report = mindlogger_export_config.input_dir / "report.csv"
    mindlogger_data = MindloggerData(pl.read_csv(mindlogger_report))
    account_users = mindlogger_data.account_users
    assert len(account_users) == 1
    assert account_users[0].user_type == UserType.ACCOUNT
    assert account_users[0].subject_id == "645e8cc0-a67a-c10f-93b4-50e000000000"


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
