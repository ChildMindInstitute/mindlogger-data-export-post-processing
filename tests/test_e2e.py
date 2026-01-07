"""E2E tests."""

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent.resolve() / "data"
WITH_SUBSCALE_TEST_DATA = pytest.mark.datafiles(FIXTURE_DIR / "subscale")
WITH_RESPONSE_TYPES_DATA = pytest.mark.datafiles(FIXTURE_DIR / "all_response_types")


# @WITH_SUBSCALE_TEST_DATA
# def test_long_report(datafiles: Path):
#     """Test long report."""
#     data = MindloggerData.create(datafiles)
#     long_report = data.report.drop(cs.ends_with("_dt") | cs.starts_with("parsed_"))
#     expected = pl.read_csv(
#         datafiles / "long.csv",
#         schema_overrides={"rawScore": pl.String},
#     )

#     assert_frame_equal(
#         long_report,
#         expected,
#         check_column_order=False,
#         check_row_order=False,
#     )


# @WITH_RESPONSE_TYPES_DATA
# def test_data_dictionary(datafiles: Path):
#     """Test data dictionary."""
#     data = MindloggerData.create(datafiles)

#     output = DataDictionaryFormat().produce(data)

#     expected = pl.read_csv(datafiles / "applet_data_dict.csv")
#     assert_frame_equal(output[0].output, expected, check_row_order=False)
