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

    DEPENDENCIES: list[type[ReportProcessor]] = []

    def __init_subclass__(cls, **kwargs):
        """Register preprocessor subclasses."""
        super().__init_subclass__(**kwargs)
        cls.PROCESSORS[cls.NAME] = cls

    def process(self, report: pl.DataFrame) -> pl.DataFrame:
        """Process the report, running dependencies first."""
        for dep in self.DEPENDENCIES:
            LOG.info("Processing dependency: %s", dep.NAME)
            report = dep().process(report)
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


class ResponseStructProcessor(ReportProcessor):
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


class SubscaleProcessor(ReportProcessor):
    """Process subscale columns into response rows.

    Warning: This must be run first because selecting the subscale columns relies on selecting all unknown columns.
    """

    NAME = "Subscale"

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Process subscale columns."""
        df_cols = {
            "id",
            "activity_flow_submission_id",
            "activity_scheduled_time",
            "activity_start_time",
            "activity_end_time",
            "flag",
            "secret_user_id",
            "userId",
            "source_user_subject_id",
            "source_user_secret_id",
            "source_user_nickname",
            "source_user_relation",
            "source_user_tag",
            "target_user_subject_id",
            "target_user_secret_id",
            "target_user_nickname",
            "target_user_tag",
            "input_user_subject_id",
            "input_user_secret_id",
            "input_user_nickname",
            "activity_id",
            "activity_name",
            "activity_flow_id",
            "activity_flow_name",
            "item",
            "item_id",
            "response",
            "prompt",
            "options",
            "version",
            "rawScore",
            "reviewing_id",
            "event_id",
            "timezone_offset",
        }
        id_cols = {
            "id",
            "activity_flow_submission_id",
            "activity_scheduled_time",
            "activity_start_time",
            "activity_end_time",
            "flag",
            "secret_user_id",
            "userId",
            "source_user_subject_id",
            "source_user_secret_id",
            "source_user_nickname",
            "source_user_relation",
            "source_user_tag",
            "target_user_subject_id",
            "target_user_secret_id",
            "target_user_nickname",
            "target_user_tag",
            "input_user_subject_id",
            "input_user_secret_id",
            "input_user_nickname",
            "activity_id",
            "activity_name",
            "activity_flow_id",
            "activity_flow_name",
            "version",
            "reviewing_id",
            "event_id",
            "timezone_offset",
        }
        response_cols = df_cols - id_cols
        ssdf = (
            report.select(pl.all().exclude(response_cols))
            .rename(
                {
                    "Final SubScale Score": "subscale__final_score",
                    "Optional text for Final SubScale Score": "subscale_text__final_score",
                }
            )
            .select(
                ~cs.matches("^Optional text for .*$"),
                pl.col("^Optional text for .*$").name.map(
                    lambda n: f"subscale_text__{n[18:].replace(' ', '_')}"
                ),
            )
            .select(
                cs.by_name(id_cols),
                cs.starts_with("subscale_"),
                (~(cs.by_name(id_cols) | cs.starts_with("subscale_"))).name.map(
                    lambda ss: f"subscale__{ss.replace(" ", "_")}"
                ),
            )
            .unpivot(index=id_cols, variable_name="item", value_name="response")
            .filter(pl.col("response").is_not_null())
            .with_columns(item_id=None, prompt=None, options=None, rawScore=None)
            .select(
                pl.all().exclude("item_id", "prompt", "options", "rawScore"),
                pl.col("item_id", "prompt", "options").cast(pl.String),
                pl.col("rawScore").cast(pl.Int64),
            )
        )

        return pl.concat([report.select(df_cols), ssdf], how="align")
