"""Output formats for MindLogger export processing package."""

from __future__ import annotations

import logging
import re
from abc import ABC
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import polars.selectors as cs

from . import util
from .mindlogger import MindloggerData
from .schema import ItemType

LOG = logging.getLogger(__name__)


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

    DEFAULT: bool = True

    def __init__(self, extra: dict[str, str] | None = None) -> None:
        """Initialize with dict for extra args."""
        self._extra = extra if extra is not None else {}

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


class WideFormat(Output):
    """Wide data format with all parsed nested types unnested / exploded."""

    NAME = "wide"

    @staticmethod
    def _pivot_multiselect(
        df: pl.DataFrame, option_scores: pl.DataFrame
    ) -> pl.DataFrame:
        del option_scores
        return (
            df.with_columns(item_option=pl.col("item").struct.field("response_options"))
            .explode("item_option")
            # Generate value column indicating presence of response.
            .with_columns(
                response_present=pl.col("item_option")
                .struct.field("value")
                .is_in(pl.col("response_value").struct.field("value")),
                # response_index=pl.col("item_option").struct.field("value"),
                # response_name=pl.col("item_option").struct.field("name"),
            )
            .drop("response_value")
            # Generate pivot column.
            .with_columns(
                item_option_pivot=pl.concat_str(
                    pl.col("item").struct.field("name"),
                    pl.col("item_option").struct.field("value"),
                    separator="_",
                )
            )
            .drop("item_option", "item")
            .pivot(
                on=["item_option_pivot"], values="response_present", sort_columns=True
            )
        )

    @staticmethod
    def _map_response_column_names(cname: str) -> str:
        parts = cname.split("__", 1)
        return "_".join([parts[1], parts[0].removeprefix("response")])

    @staticmethod
    def _fill_item_response(*null_score_columns: str) -> Generator[pl.Expr, None, None]:
        for col in null_score_columns:
            yield pl.col(f"{col}__response").alias(col)

    @staticmethod
    def _pivot_singleselect(
        df: pl.DataFrame, option_scores: pl.DataFrame
    ) -> pl.DataFrame:
        # Rename columns in scores table.
        response_options = option_scores.rename(
            {
                "item_option_value": "response_index",
                "item_option_score": "response_score",
                "item_option_name": "response_response",
            }
        )

        df = (
            # Extract value of response.
            df.with_columns(
                response_index=pl.col("response_value").struct.field("single_value")
            )
            .drop("response_value")
            # Join to score responses.
            .join(
                response_options,
                on=[
                    "applet_version",
                    "activity_flow",
                    "activity",
                    "item",
                    "response_index",
                ],
                how="left",
                validate="m:1",
            )
            # Extract item name for pivot.
            .with_columns(item_name=pl.col("item").struct.field("name"))
            .drop("item")
            # Pivot on item_name producing 3 columns for each item.
            .pivot(on="item_name", values=cs.starts_with("response"), separator="__")
            # Rename pivoted columns to
            .with_columns(
                cs.starts_with("response").name.map(
                    WideFormat._map_response_column_names
                )
            )
            .drop(cs.starts_with("response"))
        )

        # Rename score columns to bare name of item.
        score_columns = {
            s: s.rsplit("__")[0]
            for s in cs.expand_selector(df, cs.ends_with("__score"))
        }
        # Rename <QUESTION>__score to <QUESTION>.
        df = df.rename(score_columns)
        null_score_columns = {
            col for col in score_columns.values() if df[col].is_null().all()
        }
        # Fill null <QUESTION> columns with value of <QUESTION>__response.
        return df.with_columns(WideFormat._fill_item_response(*null_score_columns))

    @staticmethod
    def _pivot_text(df: pl.DataFrame, option_scores: pl.DataFrame) -> pl.DataFrame:
        del option_scores
        return (
            df.with_columns(
                response_value=pl.col("response_value").struct.field("text"),
                item_name=pl.col("item").struct.field("name"),
            )
            .with_columns(response_response=pl.col("response_value"))
            .drop("item")
            .pivot(
                on="item_name",
                values=["response_value", "response_response"],
                separator="__",
            )
            .rename(
                lambda s: s.removesuffix("__response_response")
                if s.endswith("__response_response")
                else s.removesuffix("_value")
                if s.endswith("__response_value")
                else s
            )
        )

    @staticmethod
    def _pivot_subscale(df: pl.DataFrame, option_scores: pl.DataFrame) -> pl.DataFrame:
        del option_scores
        return (
            df.with_columns(
                response_value=pl.col("response_value").struct.field("subscale"),
                item_name=pl.col("item").struct.field("name"),
            )
            .with_columns(response_response=pl.col("response_value"))
            .drop("item")
            .pivot(
                on="item_name",
                values=["response_value", "response_response"],
                separator="__",
            )
            .rename(
                lambda s: s.removesuffix("__response_response")
                if s.endswith("__response_response")
                else s.removesuffix("_value")
                if s.endswith("__response_value")
                else s
            )
        )

    PIVOT_FNS = {
        (ItemType.MultipleSelection,): _pivot_multiselect,
        (ItemType.SingleSelection,): _pivot_singleselect,
        (ItemType.Text,): _pivot_text,
        (ItemType.Subscale,): _pivot_subscale,
    }

    def _get_pivot_fn(
        self, partition_type: tuple[ItemType]
    ) -> Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]:
        return self.PIVOT_FNS.get(partition_type, self._pivot_text)

    def _typed_pivot(
        self, df: pl.DataFrame, option_scores: pl.DataFrame
    ) -> pl.DataFrame:
        df = (
            df.with_columns(response_value=pl.col("response").struct.field("value"))
            .drop("response")
            .with_columns(
                response_value=pl.struct(
                    pl.col("response_value").struct.field(
                        "raw_value",
                        "null_value",
                        "single_value",
                        "value",
                        "text",
                        "subscale",
                        "optional_text",
                    )
                )
            )
        )

        # Partition by type. Each type should pivot independently.
        typed_partitions: dict[tuple[ItemType], pl.DataFrame] = df.with_columns(
            item_type=pl.col("item").struct.field("type")
        ).partition_by("item_type", include_key=False, as_dict=True)  # type: ignore

        # Perform pivot in function selected by type.
        pivoted_dfs = [
            self._get_pivot_fn(partition_type)(partition_df, option_scores)
            for partition_type, partition_df in typed_partitions.items()
        ]
        struct_idx_columns = [
            "target_user",
            "source_user",
            "input_user",
            "account_user",
            "activity_flow",
            "activity",
            "activity_submission",
            "activity_time",
            "activity_schedule",
        ]

        df = (
            pl.concat(pivoted_dfs, how="align")
            .with_columns(util.unnest_structs(*struct_idx_columns))
            .drop(struct_idx_columns)
        )
        idx_columns = cs.starts_with(*(["applet_version"] + struct_idx_columns))
        response_columns = cs.by_name(sorted(cs.expand_selector(df, ~idx_columns)))
        return df.select(idx_columns, response_columns)

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        ml_report = data.report.with_columns(
            utc_timezone_offset=pl.col("utc_timezone_offset").dt.to_string("iso")
        )
        if (
            "split_activities" in self._extra
            and self._extra["split_activities"].lower() == "true"
        ):
            return [
                NamedOutput(
                    f"{activity[1]}",
                    self._typed_pivot(activity_df, data.item_response_options),
                )
                for activity, activity_df in ml_report.with_columns(
                    activity_id=pl.col("activity").struct.field("id"),
                    activity_name=pl.col("activity").struct.field("name"),
                )
                .partition_by(
                    ["activity_id", "activity_name"], include_key=False, as_dict=True
                )
                .items()
            ]

        return [
            NamedOutput(
                "wide_data", self._typed_pivot(ml_report, data.item_response_options)
            )
        ]


class RedcapImportFormat(WideFormat):
    """Wide format specific for importing to REDCap."""

    NAME = "redcap"

    def __init__(self, project: str = "curious_parent_arm_1", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._project = project
        self._instrument_row_count: dict[str, int | None] = {}

    @staticmethod
    def _normalize_column_name(col: str) -> str:
        """Replace chains of underscores with single underscore."""
        return re.sub(r"_+", "_", col).lower()

    def _prepare_activity_columns(
        self, df: pl.DataFrame, activity_prefix: str
    ) -> pl.DataFrame:
        """Rename and transform columns for a single activity."""
        # Prepend activity prefix and normalize underscores
        df = df.rename(
            {
                col: self._normalize_column_name(f"{activity_prefix}_{col}")
                for col in df.columns
            }
        )

        # Clean up common suffixes
        df = df.rename({col: col.replace("_user", "") for col in df.columns}).rename(
            {
                col: col[:-5]
                for col in df.columns
                if col.endswith("_start_time", "_end_time")
            }
        )

        # Handle response columns
        # Text remains the same, selections = index + 1
        response_cols = [col for col in df.columns if col.endswith("_response")]
        index_cols = [col for col in df.columns if col.endswith("_index")]
        index_bases = {col.replace("_index", "") for col in index_cols}
        text_item_response_cols = [
            col
            for col in response_cols
            if col.replace("_response", "") not in index_bases
        ]
        df = df.select(
            [
                col
                for col in df.columns
                if not (
                    col.endswith("_response") and col not in text_item_response_cols
                )
            ]
        )
        for col in index_cols:
            response_col = col.replace("_index", "_response")
            df = df.with_columns([(pl.col(col) + 1).alias(response_col)])
        df = df.select([col for col in df.columns if not col.endswith("_index")])

        # Drop bare item columns that have a corresponding _response column
        # These are score columns that we don't need
        response_bases = {
            col.replace("_response", "")
            for col in df.columns
            if col.endswith("_response")
        }
        return df.select([col for col in df.columns if col not in response_bases])

    def _format_activity(self, df: pl.DataFrame, activity_name: str) -> pl.DataFrame:
        """Format a single activity's data for REDCap import."""
        activity_prefix = activity_name.lower()

        # Extract record_id BEFORE column transformations
        record_id = df.select("target_user_secret_id")

        df = self._prepare_activity_columns(df, activity_prefix)
        df = self._add_redcap_metadata(df, activity_prefix, record_id)

        # Track row count for this instrument
        self._instrument_row_count[activity_name] = df.shape[0]

        return df

    def _add_redcap_metadata(
        self, df: pl.DataFrame, activity_prefix: str, record_id: pl.DataFrame
    ) -> pl.DataFrame:
        """Add REDCap-required columns and form completion status."""
        # Add required REDCap columns using pre-extracted record_id
        df = df.with_columns(
            [
                record_id.to_series().alias("record_id"),
                pl.lit(self._project).alias("redcap_event_name"),
            ]
        )

        # Remove all-null columns
        df = df.select([s for s in df if s.null_count() != len(s)])

        # Reorder: required columns first, then data columns
        required_cols = ["record_id", "redcap_event_name"]
        account_cols = [
            col
            for col in df.columns
            if "account_id" in col or "account_secret_id" in col
        ]
        data_cols = [
            col
            for col in df.columns
            if col not in required_cols and col not in account_cols
        ]

        df = df.select(required_cols + data_cols)

        # Add form completion status (2 = Complete)
        return df.with_columns([pl.lit(2).alias(f"{activity_prefix}_complete")])

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        """Format data for REDCap import, split by activity."""
        # Force split_activities to be True for REDCap format
        original_extra = self._extra.copy() if hasattr(self, "_extra") else {}
        self._extra = {**original_extra, "split_activities": "true"}

        # Get wide format outputs (one per activity)
        wide_outputs = super()._format(data)

        # Restore original extra
        self._extra = original_extra

        # Format each activity for REDCap
        outputs = []
        for wide_output in wide_outputs:
            activity_name = wide_output.name
            formatted_df = self._format_activity(wide_output.output, activity_name)
            outputs.append(NamedOutput(f"{activity_name}_redcap", formatted_df))

        return outputs

    def get_instrument_row_counts(self) -> dict[str, int | None]:
        """Return the row count for each instrument processed."""
        return self._instrument_row_count.copy()


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
                    activity_flow_id=pl.col("activity_flow").struct.field("id"),
                    activity_flow_name=pl.col("activity_flow").struct.field("name"),
                    activity_id=pl.col("activity").struct.field("id"),
                    activity_name=pl.col("activity").struct.field("name"),
                    item_id=pl.col("item").struct.field("id"),
                    item_name=pl.col("item").struct.field("name"),
                    item_prompt=pl.col("item").struct.field("prompt"),
                    item_response_options=pl.col("item").struct.field(
                        "response_options"
                    ),
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
                    .struct.unnest()
                    .name.prefix("activity_flow_"),
                    pl.col("activity").struct.unnest().name.prefix("activity_"),
                    pl.col("item").struct.unnest().name.prefix("item_"),
                )
                .select(
                    "activity_flow_id",
                    "activity_flow_name",
                    "activity_id",
                    "activity_name",
                    "item_id",
                    "item_name",
                    "item_prompt",
                    "item_response_options",
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
    DEFAULT = False

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
    """YMHA attendance format.

    Outputs attendance and completion records for YMHA 2025 Mindlogger data.

    Running this output requires an additional parameter specified in the command line via the 'extra'
    keyword arguments:
    `-e ymha_participants <filepath to participants csv>`

    Participants CSV should contain the following columns at minimum:
    nickname,firstName,lastName,secretUserId,site

    Attendance records are computed on "Student Check-Ins" and "Student Check-Outs" activities.
    Attendance records are partitioned by site and date, and contain a boolean column for each activity indicating
    whether that activity was completed on that date.

    Completion records are computed on all other activities.
    Completion records are partitioned by site only, and contain boolean columns for all activities indicating whether
    that activity has been completed by the given participant.

    Activity completion is based on presence of any responses from that activity in the data.
    """

    NAME = "ymha-attendance"

    DEFAULT = False

    def _participants(self) -> pl.DataFrame:
        """Load participants from file path in extra args.

        Returns:
            pl.DataFrame with columns:
                secret_id, nickname, first_name, last_name, site, [room]
        """
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
            cs.matches("^room$"),
        )

    def _attendance(
        self, df: pl.DataFrame, participants: pl.DataFrame
    ) -> list[NamedOutput]:
        attendance = df.pivot(
            on="activity_name",
            values="activity_completed",
            sort_columns=True,
            maintain_order=True,
            aggregate_function=pl.element().any(),
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
        completion = df.drop("activity_date").pivot(
            on="activity_name",
            values="activity_completed",
            aggregate_function=pl.element().any(),
            maintain_order=True,
            sort_columns=True,
        )
        activity_col_selector = cs.exclude(
            [
                "secret_id",
                "nickname",
                "first_name",
                "last_name",
                "site",
                cs.matches("^room$"),
            ]
        )
        identifier_col_selector = cs.by_name(
            "secret_id",
            "nickname",
            "first_name",
            "last_name",
            "site",
        ) | cs.matches(r"^room$")
        all_completion = (
            participants.join(completion, on="secret_id", how="left")
            .select(
                identifier_col_selector,
                activity_col_selector.fill_null(False),  # noqa: FBT003
            )
            .with_columns(
                complete=pl.concat_list(activity_col_selector).list.all(),
            )
        )
        site_completion = all_completion.partition_by("site", as_dict=True)
        return (
            [
                NamedOutput("ymha_completion-all", all_completion),
                NamedOutput(
                    "ymha_completion_summary-all",
                    all_completion.select(identifier_col_selector, "complete"),
                ),
            ]
            + [
                NamedOutput(f"ymha_completion-site_{part[0]}", df)
                for part, df in site_completion.items()
            ]
            + [
                NamedOutput(
                    f"ymha_completion_summary-site_{part[0]}",
                    df.select(identifier_col_selector, "complete"),
                )
                for part, df in site_completion.items()
            ]
        )

    def _format(self, data: MindloggerData) -> list[NamedOutput]:
        participants = self._participants()

        activities = (
            data.report.select("target_user", "activity", "activity_time", "item")
            .group_by(["target_user", "activity", "activity_time"])
            .agg(item_count=pl.col("item").struct.field("id").count())
            .with_columns(
                secret_id=pl.col("target_user").struct.field("secret_id"),
                activity_name=pl.col("activity").struct.field("name").str.strip_chars(),
                activity_completed=pl.col("item_count").gt(0),
                activity_date=pl.col("activity_time")
                .struct.field("start_time")
                .dt.date(),
            )
            .drop("target_user", "item_count", "activity", "activity_time")
        )
        LOG.debug("Activities Columns: %s", activities.columns)

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
