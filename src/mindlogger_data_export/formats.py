"""Output formats for MindLogger export processing package."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class NamedOutput:
    """Represents named output data to be written."""

    name: str
    output: pl.DataFrame


class OutputFormat(Protocol):
    """Protocol for output writers."""

    NAME: str

    FORMATS: dict[str, type[OutputFormat]] = {}

    PROCESSORS: list[type[ReportProcessor]] = []

    def __init_subclass__(cls, **kwargs):
        """Register preprocessor subclasses."""
        super().__init_subclass__(**kwargs)
        cls.FORMATS[cls.NAME] = cls

    def produce(self, data: pl.DataFrame) -> list[NamedOutput]:
        """Produce formatted data."""
        for processor in self.PROCESSORS:
            data = processor().process(data)
        return self._format(data)

    def _format(self, data: pl.DataFrame) -> list[NamedOutput]:
        """Format data to list of (name, output dataframe) outputs."""
        return [NamedOutput(self.NAME, data)]


class ConcatenatedReportFormat(OutputFormat):
    """Write concatenated report rows to CSV."""

    NAME = "concatenated"
    PROCESSORS = []


class TypedColumnsSingleValueRowsFormat(OutputFormat):
    """Write typed columns with single value rows to CSV."""

    NAME = "typed"

    PROCESSORS = [
        DateTimeProcessor,
        UnnestingResponseProcessor,
    ]


class DataDictionaryFormat(OutputFormat):
    """Write data dictionary to CSV."""

    NAME = "dictionary"

    PROCESSORS = [DataDictionaryProcessor]


class OptionsFormat(OutputFormat):
    """Write options to CSV."""

    NAME = "options"

    PROCESSORS = [OptionsUnnestingProcessor]


class ScoredResponsesFormat(OutputFormat):
    """Write scored responses to CSV."""

    NAME = "scored"

    PROCESSORS = [ScoredTypedData]
