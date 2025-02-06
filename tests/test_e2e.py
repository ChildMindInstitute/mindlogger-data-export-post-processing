"""E2E tests."""

from pathlib import Path

import polars as pl
import polars.selectors as cs
import pytest
from polars.testing import assert_frame_equal

from mindlogger_data_export import MindloggerData

FIXTURE_DIR = Path(__file__).parent.resolve() / "data"


@pytest.mark.datafiles(FIXTURE_DIR / "subscale")
def test_long_report(datafiles: Path):
    """Test long report."""
    data = MindloggerData.create(datafiles)
    long_report = data.report.drop(cs.ends_with("_dt") | cs.starts_with("parsed_"))
    expected = pl.read_csv(datafiles / "long.csv")
    assert_frame_equal(
        long_report, expected, check_column_order=False, check_row_order=False
    )
