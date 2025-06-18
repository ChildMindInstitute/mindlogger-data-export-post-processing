"""Pre-Processors for MindLogger Export."""

from __future__ import annotations

import logging
from typing import Protocol

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
    """Processor name."""

    PRIORITY: int = 10
    """Run order priority, lower values run first, negative values are skipped."""

    PROCESSORS: list[type[ReportProcessor]] = []
    """List of registered processors."""

    def __init_subclass__(cls, **kwargs):
        """Register preprocessor subclasses."""
        super().__init_subclass__(**kwargs)
        if cls.PRIORITY >= 0:
            cls.PROCESSORS.append(cls)

    def process(self, report: pl.DataFrame) -> pl.DataFrame:
        """Process the report, running dependencies first."""
        return self._run(report)

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Run process."""


class DropLegacyUserIdProcessor(ReportProcessor):
    """Drop legacy user ID column."""

    NAME = "DropLegacyUserId"
    PRIORITY = 0

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        return (
            report.drop("legacy_user_id")
            if "legacy_user_id" in report.columns
            else report
        )


class ColumnCastingProcessor(ReportProcessor):
    """Cast columns to expected types."""

    NAME = "ColumnCasting"
    PRIORITY = 0

    def _run(self, report) -> pl.DataFrame:
        return report.with_columns(pl.col("rawScore").cast(pl.String))


class DateTimeProcessor(ReportProcessor):
    """Convert timestamps to datetime."""

    NAME = "DateTime"
    PRIORITY = 8

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert timestamps to datetime."""
        return report.with_columns(
            pl.from_epoch(
                (cs.ends_with("_time")).cast(pl.Int64, strict=False), time_unit="ms"
            ).dt.replace_time_zone(time_zone="UTC"),
            utc_timezone_offset=pl.duration(minutes=pl.col("utc_timezone_offset")),
        )


class ResponseStructProcessor(ReportProcessor):
    """Convert response to struct using Lark.

    Input Columns: "response"
    Output Columns: "parsed_response"
    """

    NAME = "ResponseStruct"
    PARSER = ResponseParser()
    RESPONSE_SCHEMA = {
        "status": pl.String,
        "value": PARSER.datatype,
        "raw_score": pl.String,
    }

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        return report.with_columns(
            response=pl.struct(
                pl.col("item_response_status").alias("status"),
                pl.col("rawScore").alias("raw_score"),
                pl.col("item_response")
                .str.strip_chars()
                .map_elements(self.PARSER.parse, self.PARSER.datatype),
            )
        ).drop(
            "item_response_status",
            "item_response",
            "rawScore",
        )


class UserStructProcessor(ReportProcessor):
    """Convert user info to struct.

    Input Columns: "user_info"
    Output Columns: "parsed_user_info"
    """

    NAME = "UserStruct"

    USER_SCHEMA = {
        "id": pl.String,
        "secret_id": pl.String,
        "nickname": pl.String,
        "tag": pl.String,
        "relation": pl.String,
    }

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert user info to struct."""
        return report.with_columns(
            target_user=pl.struct(
                cs.starts_with("target_").name.map(lambda c: c.replace("target_", "")),
                schema=self.USER_SCHEMA,
            ),
            source_user=pl.struct(
                cs.starts_with("source_").name.map(lambda c: c.replace("source_", "")),
                schema=self.USER_SCHEMA,
            ),
            input_user=pl.struct(
                cs.starts_with("input_").name.map(lambda c: c.replace("input_", "")),
                schema=self.USER_SCHEMA,
            ),
            account_user=pl.struct(
                pl.col("userId").alias("id"),
                pl.col("secret_user_id").alias("secret_id"),
                schema=self.USER_SCHEMA,
            ),
        ).drop(
            "^target_[^u].*$",
            "^source_[^u].*$",
            "^input_[^u].*$",
            "userId",
            "secret_user_id",
        )


class ItemStructProcessor(ReportProcessor):
    """Convert item info to struct.

    Input Columns: "item_id", "item_name", "item_prompt"
    Output Columns: "item"
    """

    NAME = "ItemStruct"

    ITEM_SCHEMA = {
        "id": pl.String,
        "name": pl.String,
        "prompt": pl.String,
        "type": pl.String,
        "raw_options": pl.String,
        "response_options": pl.List(
            pl.Struct({"name": pl.String, "value": pl.Int64, "score": pl.Int64})
        ),
    }
    PARSER = OptionsParser()

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert item info to struct."""
        return report.with_columns(
            item=pl.struct(
                pl.col("item_id").alias("id"),
                pl.col("item_name").alias("name"),
                pl.col("item_prompt").alias("prompt"),
                pl.col("item_type").alias("type"),
                pl.col("item_response_options").alias("raw_options"),
                pl.col("item_response_options")
                .str.strip_chars()
                .map_elements(
                    self.PARSER.parse,
                    pl.List(
                        pl.Struct(
                            {"name": pl.String, "value": pl.Int64, "score": pl.Int64}
                        )
                    ),
                )
                .alias("response_options"),
                schema=self.ITEM_SCHEMA,
            )
        ).drop(
            "item_id", "item_name", "item_prompt", "item_type", "item_response_options"
        )


class ActivityFlowStructProcessor(ReportProcessor):
    """Convert activity flow info to struct.

    Input Columns: "activity_flow_id", "activity_flow_name"
    Output Columns: "activity_flow"
    """

    NAME = "ActivityFlowStruct"

    ACTIVITY_FLOW_SCHEMA = {
        "id": pl.String,
        "name": pl.String,
        "submission_id": pl.String,
    }

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert activity flow info to struct."""
        return report.with_columns(
            activity_flow=pl.struct(
                pl.col("activity_flow_id").alias("id"),
                pl.col("activity_flow_name").alias("name"),
                pl.col("activity_flow_submission_id").alias("submission_id"),
                schema=self.ACTIVITY_FLOW_SCHEMA,
            )
        ).drop("activity_flow_id", "activity_flow_name", "activity_flow_submission_id")


class ActivityStructProcessor(ReportProcessor):
    """Convert activity info to struct.

    Input Columns: "activity_id", "activity_name"
    Output Columns: "activity"
    """

    NAME = "ActivityStruct"

    ACTIVITY_SCHEMA = {
        "id": pl.String,
        "name": pl.String,
        "submission_id": pl.String,
        "submission_review_id": pl.String,
        "start_time": pl.Datetime(time_zone="UTC"),
        "end_time": pl.Datetime(time_zone="UTC"),
    }

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert activity info to struct."""
        return report.with_columns(
            activity=pl.struct(
                pl.col("activity_id").alias("id"),
                pl.col("activity_name").alias("name"),
                pl.col("activity_submission_id").alias("submission_id"),
                pl.col("activity_submission_review_id").alias("submission_review_id"),
                pl.col("activity_start_time").alias("start_time"),
                pl.col("activity_end_time").alias("end_time"),
                schema=self.ACTIVITY_SCHEMA,
            )
        ).drop(
            "activity_id",
            "activity_name",
            "activity_submission_id",
            "activity_submission_review_id",
            "activity_start_time",
            "activity_end_time",
        )


class ActivityScheduleStructProcessor(ReportProcessor):
    """Convert activity schedule info to struct.

    Input Columns: "activity_schedule_id", "activity_schedule_start_time"
    Output Columns: "activity_schedule"
    """

    NAME = "ActivityScheduleStruct"

    ACTIVITY_SCHEDULE_SCHEMA = {
        "id": pl.String,
        "history_id": pl.String,
        "start_time": pl.Datetime(time_zone="UTC"),
    }

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert activity schedule info to struct."""
        return report.with_columns(
            activity_schedule=pl.struct(
                pl.col("activity_schedule_id").alias("id"),
                pl.col("activity_schedule_history_id").alias("history_id"),
                pl.col("activity_schedule_start_time").alias("start_time"),
                schema=self.ACTIVITY_SCHEDULE_SCHEMA,
            )
        ).drop(
            "activity_schedule_id",
            "activity_schedule_history_id",
            "activity_schedule_start_time",
        )


class SubscaleProcessor(ReportProcessor):
    """Process subscale columns into response rows."""

    NAME = "Subscale"
    PRIORITY = 5

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Process subscale columns."""
        df_cols = {
            "target_id",
            "target_secret_id",
            "target_nickname",
            "target_tag",
            "source_id",
            "source_secret_id",
            "source_nickname",
            "source_tag",
            "source_relation",
            "input_id",
            "input_secret_id",
            "input_nickname",
            "userId",
            "secret_user_id",
            "applet_version",
            "activity_flow_id",
            "activity_flow_name",
            "activity_flow_submission_id",
            "activity_id",
            "activity_name",
            "activity_submission_id",
            "activity_schedule_history_id",
            "activity_start_time",
            "activity_end_time",
            "activity_schedule_id",
            "activity_schedule_start_time",
            "utc_timezone_offset",
            "activity_submission_review_id",
            "item_id",
            "item_name",
            "item_prompt",
            "item_response_options",
            "item_response",
            "item_response_status",
            "item_type",
            "rawScore",
        }
        response_cols = {
            "item_id",
            "item_name",
            "item_prompt",
            "item_response_options",
            "item_response",
            "item_response_status",
            "item_type",
            "rawScore",
        }
        ss_value_cs = cs.starts_with("subscale_") | cs.by_name(
            {"activity_score", "activity_score_lookup_text"}
        )

        ssdf = (
            report.select(pl.all().exclude(response_cols))
            .unpivot(
                on=ss_value_cs,
                index=list(df_cols - response_cols),
                variable_name="item_id",
                value_name="item_response",
            )
            .filter(pl.col("item_response").is_not_null())
            .with_columns(pl.lit("subscale").alias("item_type"))
            .with_columns(item_name=pl.col("item_id").str.replace("subscale_name_", ""))
            .with_columns(
                pl.lit(None).alias(c).cast(pl.String)
                for c in (
                    response_cols
                    - {"item_id", "item_name", "item_response", "item_type"}
                )
            )
            .with_columns(pl.col("item_response").cast(pl.String))
        )

        return pl.concat([report.select(~ss_value_cs), ssdf], how="align")
