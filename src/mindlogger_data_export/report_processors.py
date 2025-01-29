"""Pre-Processors for MindLogger Export."""

from __future__ import annotations

import logging
from typing import Protocol

import pandas as pd
import polars as pl
import polars.selectors as cs

from .parsers import OptionsParser, ResponseParser

LOG = logging.getLogger(__name__)


class ReportProcessor(Protocol):
    """Protocol for data processing functions.

    ReportProcessor subclasses must:
        - define a NAME class attribute.
        - define necessary DEPENDENCIES class attribute as list of NAMEs.
        - implement the _run method.

    Subclasses are automatically registered in the PROCESSORS dictionary, which
    is used to process dependencies.

    ReportProcessors should only add columns or rows, never:
        - remove existing columns or rows
        - modify existing columns or rows
        - reshape DataFrame structure
    """

    NAME: str

    PROCESSORS: dict[str, type[ReportProcessor]] = {}

    DEPENDENCIES: list[str] = []

    def __init_subclass__(cls, **kwargs):
        """Register preprocessor subclasses."""
        super().__init_subclass__(**kwargs)
        cls.PROCESSORS[cls.NAME] = cls

    def process(self, report: pl.DataFrame) -> pl.DataFrame:
        """Process the report, running dependencies first."""
        for dep in self.DEPENDENCIES:
            LOG.info("Processing dependency: %s", dep)
            report = self.PROCESSORS[dep]().process(report)
        return self._run(report)

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Run process."""


class PandasReportProcessor(ReportProcessor):
    """Base class for DataFrame processors implemented in Pandas.

    Subclasses must:
        - define a NAME class attribute.
        - define necessary DEPENDENCIES class attribute as list of NAMEs.
        - implement the _run_pd method.

    The _run method converts Polars DataFrame to Pandas DataFrame and calls _run_pd.
    """

    NAME = "PandasReportProcessor"

    def __init_subclass__(cls, **kwargs):
        """Register preprocessor subclasses."""
        super().__init_subclass__(**kwargs)
        cls.PROCESSORS[cls.NAME] = cls

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert Polars DataFrame to Pandas DataFrame."""
        return pl.from_pandas(self._run_pd(report.to_pandas()))

    def _run_pd(self, report: pd.DataFrame) -> pd.DataFrame:
        """Convert Pandas DataFrame to Polars DataFrame."""


class IdentityProcessor(ReportProcessor):
    """Identity processor."""

    NAME = "Identity"

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Return the report unchanged."""
        return report


class DateTimeProcessor(ReportProcessor):
    """Convert timestamps to datetime."""

    NAME = "DateTime"
    COLUMN_SUFFIX = "_dt"

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert timestamps to datetime."""
        return report.with_columns(
            pl.from_epoch(
                (cs.ends_with("_time")).cast(pl.Int64, strict=False), time_unit="ms"
            )
            .dt.replace_time_zone(time_zone="UTC")
            .name.suffix(self.COLUMN_SUFFIX),
        )


class OptionsStructProcessor(ReportProcessor):
    """Parses options string to list of option structs.

    Options column is str typed field with comma-separated options.
    Parse to list of structs by splitting and matching on regex pattern with groups.

    Input Columns: "options"
    Output Columns: "parsed_options"

    Input:
        "<name>: <value>, ..." or "<name>: <value> (score: <score>), ..."
    Output:
        [{"name": <name (str)>, "value": <value (int)>, "score": <score (int | null)>}, ...]
    """

    NAME = "OptionsStruct"
    PARSER = OptionsParser()

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert options to strings."""
        return report.with_columns(
            parsed_options=pl.col("options")
            .str.strip_chars()
            .map_elements(
                self.PARSER.parse,
                pl.List(
                    pl.Struct({"name": pl.String, "value": pl.Int64, "score": pl.Int64})
                ),
            )
        )


class OptionsUnnestingProcessor(ReportProcessor):
    """Parses options string to list of option structs.

    Options column is str typed field with comma-separated options.
    Parse to list of structs by splitting and matching on regex pattern with groups.

    Input Columns: "parse_options"
    Output Columns: "option_name", "option_value", "option_score"

    Input:
        "<name>: <value>, ..." or "<name>: <value> (score: <score>), ..."
    Output:
        [{"name": <name (str)>, "value": <value (int)>, "score": <score (int | null)>}, ...]
    """

    NAME = "UnnestedOptions"
    PARSER = OptionsParser()
    DEPENDENCIES = ["OptionsStruct"]

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert options to strings."""
        return (
            report.select(
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
            .explode(pl.col("parsed_options"))
            .with_columns(
                pl.col("parsed_options").struct.unnest().name.prefix("option_")
            )
            .unique()
        )


class DataDictionaryProcessor(ReportProcessor):
    """Extract data dictionary columns from report and deduplicate.

    Input Columns: "version", "activity_flow_id", "activity_flow_name",
        "activity_id", "activity_name", "item_id", "item", "prompt", "options"
    Output Columns: "version", "activity_flow_id", "activity_flow_name",
        "activity_id", "activity_name", "item_id", "item", "prompt", "options"
    """

    NAME = "DataDictionary"

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        return report.select(
            "version",
            "activity_flow_id",
            "activity_flow_name",
            "activity_id",
            "activity_name",
            "item_id",
            "item",
            "prompt",
            "options",
        ).unique()


class StructResponseProcessor(ReportProcessor):
    """Convert response to struct using Lark.

    Input Columns: "response"
    Output Columns: "parsed_response"
    """

    NAME = "ResponseStruct"
    PARSER = ResponseParser()

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        return report.with_columns(
            parsed_response=pl.col("response")
            .str.strip_chars()
            .map_elements(
                self.PARSER.parse,
                self.PARSER.datatype,
            )
        )


class UnnestingResponseProcessor(ReportProcessor):
    """Convert Unnest and expand responses to have single value per cell.

    Input Columns: "parsed_response"
    Output Columns: "response_type", "response_value", "response_score", "response_text", "response_file",
        "response_date", "response_time", "response_time_range", "response_geo", "response_row_single",
        "response_row_multi", "response_row_multi_values"
    """

    NAME = "UnnestedResponse"
    DEPENDENCIES = ["ResponseStruct"]
    PARSER = ResponseParser()
    COLUMN_PREFIX = "response_"

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert response to struct using Lark parser."""
        return (
            report.with_columns(
                pl.col("parsed_response")
                .struct.unnest()
                .name.prefix(self.COLUMN_PREFIX)
            )
            # Expand value list to rows.
            .explode(f"{self.COLUMN_PREFIX}value")
            # Expand geo struct to lat/long columns.
            .with_columns(
                pl.col(f"{self.COLUMN_PREFIX}geo")
                .struct.unnest()
                .name.prefix(f"{self.COLUMN_PREFIX}geo_")
            )
            # Expand matrix list to rows.
            .explode(f"{self.COLUMN_PREFIX}matrix")
            # Unnest matrix struct to columns.
            .with_columns(
                pl.col(f"{self.COLUMN_PREFIX}matrix")
                .struct.unnest()
                .name.prefix(f"{self.COLUMN_PREFIX}matrix_")
            )
            # Expand matrix value list to rows.
            .explode(f"{self.COLUMN_PREFIX}matrix_value")
            # Exclude temporary struct columns.
            .select(
                pl.exclude(f"{self.COLUMN_PREFIX}matrix", f"{self.COLUMN_PREFIX}geo")
            )
        )


class ScoredTypedData(ReportProcessor):
    """Score typed data.

    Input Columns: "version", "activity_flow_id", "activity_flow_name",
        "activity_id", "activity_name", "item_id", "item", "prompt", "response_value"
    Output Columns: "option_name", "option_score"
    """

    NAME = "ScoredTypedData"
    DEPENDENCIES = ["UnnestedResponse"]
    OPTION_INDEX_COLUMNS = [
        "version",
        "activity_flow_id",
        "activity_flow_name",
        "activity_id",
        "activity_name",
        "item_id",
        "item",
        "prompt",
    ]

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Score typed data."""
        options = OptionsUnnestingProcessor().process(
            report.select(self.OPTION_INDEX_COLUMNS + ["options"])
        )
        return report.join(
            options,
            how="left",
            left_on=self.OPTION_INDEX_COLUMNS + ["response_value"],
            right_on=self.OPTION_INDEX_COLUMNS + ["option_value"],
        )
