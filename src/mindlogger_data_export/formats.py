"""Output formats for MindLogger export processing package."""

from __future__ import annotations

from typing import Protocol

import polars as pl

from .processors import (
    DataDictionaryProcessor,
    DateTimeProcessor,
    OptionsUnnestingProcessor,
    ReportProcessor,
    ScoredTypedData,
    UnnestingResponseProcessor,
)


class OutputFormat(Protocol):
    """Protocol for output writers."""

    NAME: str

    FORMATS: dict[str, type[OutputFormat]] = {}

    PROCESSORS: list[ReportProcessor] = []

    def __init_subclass__(cls, **kwargs):
        """Register preprocessor subclasses."""
        super().__init_subclass__(**kwargs)
        cls.FORMATS[cls.NAME] = cls

    def produce(self, data: pl.DataFrame) -> pl.DataFrame:
        """Produce formatted data."""
        for processor in self.PROCESSORS:
            data = processor.process(data)
        return self._format(data)

    def _format(self, data: pl.DataFrame) -> pl.DataFrame:
        """Format data."""
        return data


class ConcatenatedReportFormat(OutputFormat):
    """Write concatenated report rows to CSV."""

    NAME = "concatenated"
    PROCESSORS = []


class TypedColumnsSingleValueRowsFormat(OutputFormat):
    """Write typed columns with single value rows to CSV."""

    NAME = "typed"

    PROCESSORS = [
        DateTimeProcessor(),
        UnnestingResponseProcessor(),
    ]


class DataDictionaryFormat(OutputFormat):
    """Write data dictionary to CSV."""

    NAME = "dictionary"

    PROCESSORS = [DataDictionaryProcessor()]


class OptionsFormat(OutputFormat):
    """Write options to CSV."""

    NAME = "options"

    PROCESSORS = [OptionsUnnestingProcessor()]


class ScoredResponsesFormat(OutputFormat):
    """Write scored responses to CSV."""

    NAME = "scored"

    PROCESSORS = [ScoredTypedData()]
