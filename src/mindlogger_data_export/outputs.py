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
    ITEM_COUNTS = {
        "EMA Morning": 7,
        "EMA Afternoon": 5,
        "EMA Evening": 13,
        "Healthcare Access": 6,
        "Mental Healthcare": 5,
        "PSC": 17,
        "Mental Health Attitudes and Help-Seeking": 6,
        "Experience": 2,
        "Mentor Experience": 7,
        "Brief Version of the Big Five Personality Inventory BFI": 10,
        "MinT Mentoring Styles Questionnaire": 25,
        "MEIM-6": 7,
        "The MAP": 2,
        "Program-Developed": 3,
        "Mentor Experience (Program-Developed)": 3,
    }

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
        return participants.select(
            pl.col("secretUserId").alias("secret_id"),
            pl.col("nickname"),
            pl.col("firstName").alias("first_name"),
            pl.col("lastName").alias("last_name"),
            "site",
        )

    def _attendance(
        self, df: pl.DataFrame, participants: pl.DataFrame
    ) -> list[NamedOutput]:
        attendance = (
            df.with_columns(
                activity_date=pl.col("activity").struct.field("start_time").dt.date()
            )
            .drop("activity")
            .pivot(
                on="activity_name",
                values="activity_completed",
                sort_columns=True,
                maintain_order=True,
            )
        )
        dates = attendance.select(pl.col("activity_date").unique()).filter(
            pl.col("activity_date").is_not_null()
        )
        participant_dates = participants.join(dates, how="cross")
        all_attendance = participant_dates.join(
            attendance,
            on=["secret_id", "activity_date"],
            how="left",
        ).with_columns(pl.col("^Student Check.*$").fill_null(False))  # noqa: FBT003
        part_dfs = all_attendance.partition_by(["site", "activity_date"], as_dict=True)
        return [NamedOutput("ymha_attendance-all", all_attendance)] + [
            NamedOutput(f"ymha_attendance-site_{part[0]}-date_{part[1]}", df)
            for part, df in part_dfs.items()
        ]

    def _completion(
        self, df: pl.DataFrame, participants: pl.DataFrame
    ) -> list[NamedOutput]:
        completion = df.drop("activity").pivot(
            on="activity_name",
            values="activity_completed",
            aggregate_function=pl.element().any(),
            maintain_order=True,
            sort_columns=True,
        )
        all_completion = completion.join(
            participants, on="secret_id", how="left"
        ).select(
            "secret_id",
            "nickname",
            "first_name",
            "last_name",
            "site",
            cs.exclude(
                ["secret_id", "nickname", "first_name", "last_name", "site"]
            ).fill_null(False),  # noqa: FBT003
        )
        site_completion = all_completion.partition_by("site", as_dict=True)
        return [NamedOutput("ymha_completion-all", all_completion)] + [
            NamedOutput(f"ymha_completion-site_{part[0]}", df)
            for part, df in site_completion.items()
        ]

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        participants = self._participants()

        activities = (
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
                secret_id=pl.col("target_user").struct.field("secret_id"),
                activity_name=pl.col("activity").struct.field("name").str.strip_chars(),
            )
            .with_columns(activity_completed=pl.col("item_count").gt(0))
            .drop("target_user", "item_count")
        )

        partitioned_activities = activities.with_columns(
            is_ema=pl.col("activity_name").str.starts_with("Student Check")
        ).partition_by("is_ema", as_dict=True, include_key=False)

        return (
            self._attendance(partitioned_activities[(True,)], participants)
            if (True,) in partitioned_activities
            else []
        ) + (
            self._completion(partitioned_activities[(False,)], participants)
            if (False,) in partitioned_activities
            else []
        )
