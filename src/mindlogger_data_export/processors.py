"""Pre-Processors for MindLogger Export."""

from __future__ import annotations

import logging
from typing import Protocol

import polars as pl
import polars.selectors as cs

from . import schema
from .parsers import (
    FullResponseParser,
    OptionsParser,
    TypedResponseParser,
)

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

    ENABLE: bool = False

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
        raise NotImplementedError("Subclasses must implement this method.")


class DropLegacyUserIdProcessor(ReportProcessor):
    """Drop legacy user ID column."""

    NAME = "DropLegacyUserId"
    PRIORITY = 0
    ENABLE = True

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
    ENABLE = True

    def _run(self, report) -> pl.DataFrame:
        return report.with_columns(pl.col("rawScore").cast(pl.String))


class DateTimeProcessor(ReportProcessor):
    """Convert timestamps to datetime."""

    NAME = "DateTime"
    PRIORITY = 8
    ENABLE = True

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
    PARSER = FullResponseParser()
    COLUMNS = {"item_response_status", "rawScore", "item_response"}
    ENABLE = False

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        return report.with_columns(
            response=pl.struct(
                pl.col("item_response_status").alias("status"),
                pl.col("rawScore").alias("raw_score"),
                pl.col("item_response")
                .str.strip_chars()
                .map_elements(self.PARSER.parse, schema.RESPONSE_VALUE_SCHEMA)
                .alias("value"),
            )
        ).drop(
            "item_response_status",
            "item_response",
            "rawScore",
        )


class TypedResponseStructProcessor(ReportProcessor):
    """Parse response with type-specific parser."""

    NAME = "TypedResponse"
    COLUMNS = {"item_response_status", "rawScore", "item_response"}
    PARSER = TypedResponseParser()
    ENABLE = True
    PRIORITY = 40

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        return report.with_columns(
            response=pl.struct(
                status=pl.col("item_response_status"),
                raw_score=pl.col("rawScore").cast(pl.Float64).cast(pl.Int64),
                raw_response=pl.col("item_response"),
                value=pl.struct(
                    item_type=pl.col("item").struct.field("type"),
                    response=pl.col("item_response").str.strip_chars(),
                ).map_elements(
                    lambda d: self.PARSER.parse_typed(d["item_type"], d["response"]),
                    schema.RESPONSE_VALUE_SCHEMA,
                ),
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
    ENABLE = True

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert user info to struct."""
        return report.with_columns(
            target_user=pl.struct(
                cs.starts_with("target_").name.map(lambda c: c.replace("target_", "")),
                schema=schema.USER_SCHEMA,
            ),
            source_user=pl.struct(
                cs.starts_with("source_").name.map(lambda c: c.replace("source_", "")),
                schema=schema.USER_SCHEMA,
            ),
            input_user=pl.struct(
                cs.starts_with("input_").name.map(lambda c: c.replace("input_", "")),
                schema=schema.USER_SCHEMA,
            ),
            account_user=pl.struct(
                pl.col("userId").alias("id"),
                pl.col("secret_user_id").alias("secret_id"),
                schema=schema.USER_SCHEMA,
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
    ENABLE = True

    PARSER = OptionsParser()
    COLUMNS = {
        "item_id",
        "item_name",
        "item_prompt",
        "item_type",
        "item_response_options",
    }

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
                    schema.ITEM_RESPONSE_OPTIONS_SCHEMA,
                )
                .alias("response_options"),
                schema=schema.ITEM_SCHEMA,
            )
        ).drop(self.COLUMNS)


class ActivityFlowStructProcessor(ReportProcessor):
    """Convert activity flow info to struct.

    Input Columns: "activity_flow_id", "activity_flow_name"
    Output Columns: "activity_flow"
    """

    NAME = "ActivityFlowStruct"
    ENABLE = True

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert activity flow info to struct."""
        return report.with_columns(
            activity_flow=pl.struct(
                pl.col("activity_flow_id").alias("id"),
                pl.col("activity_flow_name").alias("name"),
                pl.col("activity_flow_submission_id").alias("submission_id"),
                schema=schema.ACTIVITY_FLOW_SCHEMA,
            )
        ).drop("activity_flow_id", "activity_flow_name", "activity_flow_submission_id")


class ActivityStructProcessor(ReportProcessor):
    """Convert activity info to struct.

    Input Columns:
        activity_id, activity_name,
        activity_submission_id, activity_submission_review_id, activity_start_time, activity_end_time
    Output Columns: activity, activity_submission, activity_time
    """

    NAME = "ActivityStruct"
    ENABLE = True
    COLUMNS = [
        "activity_id",
        "activity_name",
        "activity_submission_id",
        "activity_submission_review_id",
        "activity_start_time",
        "activity_end_time",
    ]

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert activity info to struct."""
        return report.with_columns(
            activity=pl.struct(
                pl.col("activity_id").alias("id"),
                pl.col("activity_name").alias("name"),
                schema=schema.ACTIVITY_SCHEMA,
            ),
            activity_submission=pl.struct(
                pl.col("activity_submission_id").alias("id"),
                pl.col("activity_submission_review_id").alias("review_id"),
                schema=schema.ACTIVITY_SUBMISSION_SCHEMA,
            ),
            activity_time=pl.struct(
                pl.col("activity_start_time").alias("start_time"),
                pl.col("activity_end_time").alias("end_time"),
                schema=schema.ACTIVITY_TIME_SCHEMA,
            ),
        ).drop(self.COLUMNS)


class ActivityScheduleStructProcessor(ReportProcessor):
    """Convert activity schedule info to struct.

    Input Columns: "activity_schedule_id", "activity_schedule_start_time"
    Output Columns: "activity_schedule"
    """

    NAME = "ActivityScheduleStruct"
    ENABLE = True

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Convert activity schedule info to struct."""
        return report.with_columns(
            activity_schedule=pl.struct(
                pl.col("activity_schedule_id").alias("id"),
                pl.col("activity_schedule_history_id").alias("history_id"),
                pl.col("activity_schedule_start_time").alias("start_time"),
                schema=schema.ACTIVITY_SCHEDULE_SCHEMA,
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
    ENABLE = True

    def _run(self, report: pl.DataFrame) -> pl.DataFrame:
        """Process subscale columns."""
        subscale_columns = cs.starts_with("subscale_") | cs.by_name(
            "activity_score", "activity_score_lookup_text", require_all=False
        )
        if len(cs.expand_selector(report, subscale_columns)) == 0:
            return report

        index_columns = ~(
            subscale_columns
            | cs.by_name(ItemStructProcessor.COLUMNS, ResponseStructProcessor.COLUMNS)
        )
        LOG.debug(
            "Subscale Columns: %s", ",\n".join(report.select(subscale_columns).columns)
        )
        LOG.debug("Index Columns: %s", ",\n".join(report.select(index_columns).columns))

        subscales = (
            report.unpivot(
                index=index_columns,
                on=subscale_columns,
                variable_name="item_id",
                value_name="item_response",
            )
            .filter(pl.col("item_response").is_not_null())
            .with_columns(
                item_type=pl.lit("subscale"),
                item_name=pl.col("item_id").str.replace("subscale_name_", ""),
                item_response_status=pl.lit("completed"),
            )
        )

        return pl.concat(
            [
                report.select(~subscale_columns),
                subscales,
            ],
            how="diagonal_relaxed",
        )
