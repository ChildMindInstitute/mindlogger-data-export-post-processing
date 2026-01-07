"""Data model of Mindlogger export."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path

import polars as pl

from . import util
from .models import MindloggerUser, UserType
from .processors import ReportProcessor

LOG = logging.getLogger(__name__)

MINDLOGGER_REPORT_PATTERN = "*responses*.csv"
ACTIVITY_FLOW_PATTERN = "activity_flow*.csv"
FLOW_ITEM_HISTORY_PATTERN = "flow_item_history*.csv"
SCHEDULE_HISTORY_PATTERN = "schedule_history*.csv"


class MindloggerData:
    """Data model of Mindlogger export."""

    def __init__(self, response_data: pl.DataFrame):
        """Initialize MindloggerData object."""
        LOG.debug("Response Data Columns: %s", response_data.columns)
        self._response_data = response_data

    @cached_property
    def report(self) -> pl.DataFrame:
        """Get report DataFrame."""
        return pl.DataFrame(self._response_data)

    @cached_property
    def long_options_report(self) -> pl.DataFrame:
        """Get report DataFrame with one option value per row, e.g. exploded data dictionary format."""
        return MindloggerData.expand_options(self.report)

    @cached_property
    def long_response_report(self) -> pl.DataFrame:
        """Get report DataFrame with one response value per row."""
        return MindloggerData.expand_responses(self.report)

    @cached_property
    def long_report(self) -> pl.DataFrame:
        """Get report DataFrame with one response value per row."""
        return MindloggerData.expand_options(self.long_response_report)

    @cached_property
    def input_users(self) -> list[MindloggerUser]:
        """Input users in report."""
        return self._users(UserType.INPUT)

    @cached_property
    def target_users(self) -> list[MindloggerUser]:
        """Target users in report."""
        return self._users(UserType.TARGET)

    @cached_property
    def source_users(self) -> list[MindloggerUser]:
        """Source users in report."""
        return self._users(UserType.SOURCE)

    @cached_property
    def account_users(self) -> list[MindloggerUser]:
        """Account users in report."""
        return self._users(UserType.ACCOUNT)

    @cached_property
    def users(self) -> dict[UserType, list[MindloggerUser]]:
        """Get users for specific type."""
        return {
            UserType.INPUT: self.input_users,
            UserType.TARGET: self.target_users,
            UserType.SOURCE: self.source_users,
            UserType.ACCOUNT: self.account_users,
        }

    @cached_property
    def data_dictionary(self) -> pl.DataFrame:
        """Return unique items in report."""
        return pl.DataFrame(
            self.report.select(
                "applet_version",
                *util.unnest_structs("activity_flow", "activity", "item"),
            ).unique()
        )

    @cached_property
    def item_response_options(self) -> pl.DataFrame:
        """Return options format suitable for joining to score.

        Columns:
        applet_version, activity_flow, activity, item, item_option_name, item_option_value, item_option_score
        """
        return (
            self.report.select(
                "applet_version",
                "activity_flow",
                "activity",
                "item",
                item_response_options=pl.col("item").struct.field("response_options"),
            )
            .explode("item_response_options")
            .with_columns(
                pl.col("item_response_options")
                .struct.unnest()
                .name.prefix("item_option_")
            )
            .drop("item_response_options")
            .unique()
        )

    @staticmethod
    def expand_options(df: pl.DataFrame) -> pl.DataFrame:
        """Expand options struct to columns."""
        return df.with_columns(
            item_response_options=pl.col("item").struct.field("response_options")
        ).explode("item_response_options")

    @staticmethod
    def expand_responses(df: pl.DataFrame) -> pl.DataFrame:
        """Expand responses struct to columns/rows."""
        return (
            df.with_columns(
                # Unnest response struct
                pl.col("response").struct.unnest().name.prefix("response_")
            )
            .with_columns(
                pl.col("response_value").struct.unnest().name.prefix("response_value_")
            )
            # Expand value list to rows.
            .with_columns(
                response_value_value_index=pl.int_ranges(
                    pl.col("response_value_value").list.len()
                )
            )
            .explode("response_value_value", "response_value_value_index")
            # Expand geo struct to lat/long columns.
            .with_columns(
                pl.col("response_value_geo")
                .struct.unnest()
                .name.prefix("response_value_geo_")
            )
            # Expand matrix list to rows.
            .explode("response_value_matrix")
            # Unnest matrix struct to columns.
            .with_columns(
                pl.col("response_value_matrix")
                .struct.unnest()
                .name.prefix("response_value_matrix_")
            )
            # Expand matrix value list to rows.
            .with_columns(
                response_value_matrix_value_index=pl.int_ranges(
                    pl.col("response_value_matrix_value").list.len()
                )
            )
            .explode("response_value_matrix_value", "response_value_matrix_value_index")
            # Exclude temporary struct columns.
            .select(pl.exclude("response_value_matrix", "response_value_geo"))
        )

    def _users(self, user_type: UserType) -> list[MindloggerUser]:
        """Get users for specific type."""
        return list(
            map(
                MindloggerUser.from_struct_factory(user_type),
                self.report.get_column(user_type.value).unique(),
            )
        )

    def __str__(self):
        """Return string representation of MindloggerData object reporting column names and report head."""
        return f"MindloggerData: {self._response_data.columns}\n\n{self._response_data.head()}"

    @classmethod
    def load_web_export(cls, input_path: Path) -> pl.DataFrame:
        """Read Curious web export into DataFrame."""
        LOG.debug("Reading report files from %s...", input_path)

        # Read report files.
        try:
            report = pl.concat(
                (
                    pl.read_csv(f, infer_schema_length=None)
                    for f in input_path.glob(MINDLOGGER_REPORT_PATTERN)
                ),
                how="diagonal_relaxed",
            )
            for proc in sorted(ReportProcessor.PROCESSORS, key=lambda x: x.PRIORITY):
                if not proc.ENABLE:
                    LOG.debug("Skipping disabled processor (%s)", proc.NAME)
                    continue
                LOG.debug("Running processor %s...", proc.NAME)
                report = proc().process(report)
        except pl.exceptions.ComputeError:
            LOG.exception("Error reading report files")
            raise
        return report

    @classmethod
    def load_api_export(cls, input_path: Path) -> pl.DataFrame:
        """Read Curious export from API into DataFrame."""
        LOG.debug("Reading API JSON export.")
        return pl.read_json(input_path)

    @classmethod
    def load(cls, input_path: Path) -> pl.DataFrame:
        """Read Mindlogger export into DataFrame.

        This class method reads all reports from a Mindlogger export directory,
        and returns a single DataFrame object.

        Args:
            input_path: Path to directory containing Mindlogger export.

        Returns:
            DataFrame object.

        Raises:
            FileNotFoundError: If export_dir not found, or does not contain any
                report.csv files.
            NotADirectoryError: If export_dir is not a directory.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Export directory {input_path} not found.")
        if input_path.is_dir():
            return cls.load_web_export(input_path)
        return cls.load_api_export(input_path)

    @classmethod
    def create(
        cls,
        input_path: Path,
    ) -> MindloggerData:
        """Loads Mindlogger report export and creates MindloggerData for inspection.

        Args:
            input_path: Mindlogger export directory containing report*.csv.

        Returns:
            MindloggerData object.
        """
        return cls(cls.load(input_path))
