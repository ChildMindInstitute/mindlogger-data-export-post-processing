"""Output formats for MindLogger export processing package."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import polars.selectors as cs

from .mindlogger import MindloggerData


@dataclass
class NamedOutput:
    """Represents named output data to be written."""

    name: str
    output: pl.DataFrame


class MissingExtraArgumentError(Exception):
    """Error class for output that requires an extra argument."""


class OutputGenerationError(Exception):
    """Generic error encountered in output generation."""


class Output(ABC):
    """Protocol for output writers."""

    NAME: str

    TYPES: dict[str, type[Output]] = {}

    def __init__(self, extra: dict[str, str]) -> None:
        """Initialize with dict for extra args."""
        self._extra = extra

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


class WideActivityDataFormat(Output):
    """Write wide data to CSV."""

    NAME = "wide-activity"

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
    index_columns = [
        "userId",
        "activity_submission_id",
        "source_secret_id",
        "target_secret_id",
        "input_secret_id",
        "activity_start_time_dt_utc",
        "activity_end_time_dt_utc",
        "activity_schedule_start_time_dt_utc",
        "activity_flow_id",
        "activity_flow_name",
        "activity_id",
        "activity_flow_submission_id",
        "applet_version",
    ]

    pivot_columns = [
        "activity_name",
        "item_name",
        "item_id",
        "item_response_value_index",
        # "item_response_matrix_row",
        # "item_response_matrix_value_index",
        "item_response_type",
    ]
    DROP = [
        "activity_schedule_start_time",
        "activity_start_time",
        "activity_end_time",
        "input_id",
        "input_secret_id",
        "input_nickname",
        "parsed_options",
        "parsed_response",
        "utc_timezone_offset",
        "activity_submission_id",
        "activity_submission_review_id",
        "activity_flow_submission_id",
        "item_response_file",
        "item_response_date",
        "item_response_time",
        "item_response_time_range",
        "item_response_geo_latitude",
        "item_response_geo_longitude",
        "item_response_matrix_row",
        "item_response_matrix_value",
        "item_response_matrix_value_index",
        "item_response",
        "item_prompt",
        "item_response_options",
        "item_response_status",
        "item_id",
        "item_response_type",
    ]

    ACTUAL_COLUMNS = [
        "secret_user_id",
        "source_relation",
        "source_tag",
        "source_nickname",
        "source_secret_id",
        "source_id",
        "target_secret_id",
        "target_id",
        "target_tag",
        "target_nickname",
        "userId",
        "applet_version",
        "activity_id",
        "activity_name",
        "activity_flow_id",
        "activity_flow_name",
        "activity_schedule_id",
        "activity_schedule_start_time_dt_utc",
        "activity_start_time_dt_utc",
        "activity_end_time_dt_utc",
    ]
    PIVOT_C = [
        "item_name",
        "item_type",
        "item_response_value_index",
    ]
    RESPONSE_COLUMNS = [
        "item_response_raw_value",
        "item_response_null_value",
        "item_response_value",
        "item_response_text",
        "item_response_optional_text",
        "rawScore",
    ]

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        df = data.long_response_report
        df = df.drop(self.DROP).drop(cs.duration())
        df.write_csv("wide.csv")
        df = df.with_columns((cs.by_name(self.PIVOT_C) & cs.string()).fill_null(""))
        df = df.with_columns(pl.col("item_response_value_index").fill_null(0))

        df = df.pivot(
            on=self.PIVOT_C,
            index=self.ACTUAL_COLUMNS,
            values=self.RESPONSE_COLUMNS,
            sort_columns=True,
        )
        df = df.select(
            x.name for x in filter(lambda x: x.null_count() != df.height, df)
        )
        df = df.select(
            ~cs.contains("{"),
            cs.contains("{").name.map(
                lambda x: x.replace("{", "_")
                .replace("}", "")
                .replace('"', "")
                .replace(",", "__")
            ),
        )
        return [NamedOutput("wide_data", df)]


class WideFormat(Output):
    """Wide data format with all parsed nested types unnested / exploded."""

    NAME = "wide"

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        return [NamedOutput("wide_data", data.report)]


class LongDataFormat(Output):
    """Long data format with all parsed nested types unnested / exploded."""

    NAME = "long"

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        return [NamedOutput("long_data", data.long_response_report)]


class DataDictionaryFormat(Output):
    """Write data dictionary to CSV."""

    NAME = "dictionary"

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        return [
            NamedOutput(
                "data_dictionary",
                data.report.select(
                    "applet_version",
                    pl.col("activity_flow")
                    .struct.field("id")
                    .alias("activity_flow_id"),
                    pl.col("activity_flow")
                    .struct.field("name")
                    .alias("activity_flow_name"),
                    pl.col("activity").struct.field("id").alias("activity_id"),
                    pl.col("activity").struct.field("name").alias("activity_name"),
                    pl.col("item").struct.field("id").alias("item_id"),
                    pl.col("item").struct.field("name").alias("item_name"),
                    pl.col("item").struct.field("prompt").alias("item_prompt"),
                    pl.col("item")
                    .struct.field("response_options")
                    .alias("item_response_options"),
                ).unique(),
            ),
        ]


class OptionsFormat(Output):
    """Options format represents the item options.

    Options format is similar to data dictionary format, but with one row per
    option and a separate column for the name, value and score of each option.
    """

    NAME = "options"

    def _format(self, data):
        return [
            NamedOutput(
                "options",
                data.report.select(
                    "applet_version",
                    pl.col("activity_flow")
                    .struct.field("id")
                    .alias("activity_flow_id"),
                    pl.col("activity_flow")
                    .struct.field("name")
                    .alias("activity_flow_name"),
                    pl.col("activity").struct.field("id").alias("activity_id"),
                    pl.col("activity").struct.field("name").alias("activity_name"),
                    pl.col("item").struct.field("id").alias("item_id"),
                    pl.col("item").struct.field("name").alias("item_name"),
                    pl.col("item").struct.field("prompt").alias("item_prompt"),
                    pl.col("item")
                    .struct.field("response_options")
                    .alias("item_response_options"),
                )
                .unique()
                .explode("item_response_options")
                .with_columns(
                    pl.col("item_response_options")
                    .struct.unnest()
                    .name.prefix("item_option_")
                )
                .drop("item_response_options"),
            ),
        ]


class ScoredResponsesFormat(Output):
    """Write scored responses to CSV."""

    NAME = "scored"

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        return [
            NamedOutput(
                "scored_responses",
                data.long_report.filter(  # Filter out rows where option_score does not match response_value.
                    pl.col("item_option_value")
                    .cast(pl.String)
                    .eq_missing(pl.col("item_response_value"))
                ),
            )
        ]


class YmhaAttendanceFormat(Output):
    """YMHA attendance format."""

    NAME = "ymha-attendance"
    EMA_MORNING_ITEM_COUNT = 7
    EMA_AFTERNOON_ITEM_COUNT = 5
    EMA_EVENING_ITEM_COUNT = 13

    def _participants(self) -> pl.DataFrame:
        """Load participants from file path in extra args."""
        if "ymha_participants" not in self._extra:
            raise MissingExtraArgumentError(
                "YMHA Attendance Report requires ymha_participants parameter specified in 'extra' argument."
            )
        participants_path = Path(self._extra["ymha_participants"])
        if not participants_path.is_file():
            raise FileNotFoundError("YMHA Participants file not found.")

        participants = pl.read_csv(participants_path)
        if "site" not in participants.columns:
            raise OutputGenerationError(
                "'site' column not found in YMHA participants file"
            )
        if "secretUserId" not in participants.columns:
            raise OutputGenerationError(
                "'secretUserId' column not found in YMHA participants file"
            )
        return participants.select("secretUserId", "site")

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        participants = self._participants()
        site_dfs = (
            data.report.drop(
                "activity_flow",
                "activity_schedule",
                "account_user",
                "input_user",
                "source_user",
                "response",
            )
            .group_by(["target_user", "activity"])
            .agg(item_count=pl.col("item").count())
            .with_columns(
                user_id=pl.col("target_user").struct.field("id"),
                secret_id=pl.col("target_user").struct.field("secret_id"),
                user_nickname=pl.col("target_user").struct.field("nickname"),
                activity_name=pl.col("activity").struct.field("name"),
                activity_date=pl.col("activity").struct.field("start_time").dt.date(),
            )
            .with_columns(
                activity_completed=pl.when(pl.col("activity_name") == "EMA Morning")
                .then(pl.col("item_count") == self.EMA_MORNING_ITEM_COUNT)
                .when(pl.col("activity_name") == "EMA Afternoon")
                .then(pl.col("item_count") == self.EMA_AFTERNOON_ITEM_COUNT)
                .when(pl.col("activity_name") == "EMA Evening")
                .then(pl.col("item_count") == self.EMA_EVENING_ITEM_COUNT)
                .otherwise(pl.lit(False)),  # noqa: FBT003
            )
            .drop(
                "activity",
                "target_user",
                "item_count",
            )
            .join(
                participants,
                left_on="secret_id",
                right_on="secretUserId",
                how="left",
                validate="m:1",
            )
            .partition_by("site", as_dict=True)
        )
        return [
            NamedOutput(f"ymha_attendance-site_{site[0]}", df)
            for site, df in site_dfs.items()
        ]
