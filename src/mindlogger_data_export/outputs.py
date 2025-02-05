"""Output formats for MindLogger export processing package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import polars as pl

from .mindlogger import MindloggerData


@dataclass
class NamedOutput:
    """Represents named output data to be written."""

    name: str
    output: pl.DataFrame


class Output(Protocol):
    """Protocol for output writers."""

    NAME: str

    TYPES: dict[str, type[Output]] = {}

    def __init_subclass__(cls, **kwargs):
        """Register preprocessor subclasses."""
        super().__init_subclass__(**kwargs)
        cls.TYPES[cls.NAME] = cls

    def produce(self, data: MindloggerData) -> list[NamedOutput]:
        """Produce formatted data."""
        return self._format(data)

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        """Format data to list of (name, output dataframe) outputs."""
        return [NamedOutput(self.NAME, data.report)]

    @classmethod
    def output_types_info(cls) -> dict[str, str]:
        """Print information about output types."""
        return {k: v.__doc__ for k, v in cls.TYPES.items() if v.__doc__}


class WideDataFormat(Output):
    """Write wide data to CSV."""

    NAME = "wide"

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        return [NamedOutput("wide_data", data.report)]


class ConcatenatedReportFormat(Output):
    """Write concatenated report rows to CSV."""

    NAME = "concatenated"


class LongDataFormat(Output):
    """Long data format with all parsed nested types unnested / exploded."""

    NAME = "long"


class DataDictionaryFormat(Output):
    """Write data dictionary to CSV."""

    NAME = "dictionary"

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        return [
            NamedOutput(
                "data_dictionary",
                data.report.select(
                    "version",
                    "activity_flow_id",
                    "activity_flow_name",
                    "activity_id",
                    "activity_name",
                    "item_id",
                    "item",
                    "prompt",
                    "options",
                ),
            )
        ]


class OptionsFormat(Output):
    """Write options."""

    NAME = "options"

    def _format(self, data):
        return [
            NamedOutput(
                "options",
                data.report.select(
                    "version",
                    "activity_flow_id",
                    "activity_flow_name",
                    "activity_id",
                    "activity_name",
                    "item_id",
                    "item",
                    "prompt",
                    "options",
                    "parsed_options",
                )
                .unique()
                .explode(pl.col("parsed_options"))
                .with_columns(
                    pl.col("parsed_options").struct.unnest().name.prefix("option_")
                )
                .unique(),
            )
        ]


class ScoredResponsesFormat(Output):
    """Write scored responses to CSV."""

    NAME = "scored"

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        return [
            NamedOutput(
                "scored_responses",
                data.long_report.filter(  # Filter out rows where option_score does not match response_value.
                    pl.col("option_value").eq_missing(pl.col("response_value"))
                ),
            )
        ]
