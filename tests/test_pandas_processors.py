"""Tests for PandasReportProcessor."""

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from mindlogger_data_export import PandasReportProcessor


class PandasReportProcessorForTst(PandasReportProcessor):
    """Test PandasReportProcessor."""

    NAME = "PandasTest"

    def _run_pd(self, report: pd.DataFrame) -> pd.DataFrame:
        """Convert Pandas DataFrame to Polars DataFrame."""
        report["test_pd"] = "and back"
        return report


DATA_DIR = Path(__file__).parent.resolve() / "data"
WITH_REPORT = pytest.mark.datafiles(DATA_DIR / "report.csv")


@WITH_REPORT
def test_pandas(datafiles: Path):
    report_csv = datafiles / "report.csv"
    report = pl.read_csv(report_csv)
    processor = PandasReportProcessorForTst()
    processed = processor.process(report)
    assert "test_pd" in processed.columns
