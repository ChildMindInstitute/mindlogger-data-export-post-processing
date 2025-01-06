"""Pre-Processors for MindLogger Export."""

from __future__ import annotations

import logging
from typing import Protocol

import polars as pl
import polars.selectors as cs

LOG = logging.getLogger(__name__)


class ReportPreprocessor(Protocol):
    """Protocol for data preprocessing."""

    def run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Preprocess the report."""


class DateTimePreprocessor(ReportPreprocessor):
    """Convert timestamps to datetime."""

    def __init__(self, tz: str = "America/New_York") -> None:
        """Initialize DateTimePreprocessor object."""
        self._tz = tz

    def run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert timestamps to datetime."""
        return report.select(
            pl.all().exclude(cs.ends_with("_time")),
            cs.ends_with("_time")
            .str.to_datetime(time_zone="UTC")
            .dt.convert_time_zone(self._tz),
        )


REPORT_PREPROCESSORS = {"DateTimeConverter": DateTimePreprocessor}
