"""Data model of Mindlogger export."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path

import pandas as pd
import polars as pl

from .models import MindloggerUser, UserType
from .processors import ReportProcessor

LOG = logging.getLogger(__name__)

MINDLOGGER_REPORT_PATTERN = "report*.csv"


class MindloggerData:
    """Data model of Mindlogger export."""

    def __init__(self, report: pl.DataFrame):
        """Initialize MindloggerData object."""
        self._report_frame = report

    @property
    def report(self) -> pl.DataFrame:
        """Get report DataFrame."""
        return pl.DataFrame(self._report_frame)

    @property
    def report_pd(self) -> pd.DataFrame:
        """Get report DataFrame in Pandas format."""
        return self._report_frame.to_pandas()

    @property
    def long_options_report(self) -> pl.DataFrame:
        """Get report DataFrame with one option value per row, e.g. exploded data dictionary format."""
        return MindloggerData.expand_options(self.report)

    @property
    def long_response_report(self) -> pl.DataFrame:
        """Get report DataFrame with one response value per row."""
        return MindloggerData.expand_responses(self.report)

    @property
    def long_report(self) -> pl.DataFrame:
        """Get report DataFrame with one response value per row."""
        return MindloggerData.expand_options(
            MindloggerData.expand_responses(self.report)
        )

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

    @property
    def data_dictionary(self) -> pl.DataFrame:
        """Return unique items in report."""
        pl.DataFrame(
            self._report_frame.select(
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
        )

    @property
    def data_dictionary_pd(self) -> pd.DataFrame:
        """Return unique items in report in Pandas format."""
        return self.data_dictionary.to_pandas()

    @staticmethod
    def expand_options(df: pl.DataFrame) -> pl.DataFrame:
        """Expand options struct to columns."""
        return (
            df.explode(pl.col("parsed_options"))
            .with_columns(
                pl.col("parsed_options").struct.unnest().name.prefix("option_")
            )
            .unique()
        )

    @staticmethod
    def expand_responses(df: pl.DataFrame) -> pl.DataFrame:
        """Expand responses struct to columns/rows."""
        return (
            df.with_columns(
                pl.col("parsed_response").struct.unnest().name.prefix("response_")
            )
            # Expand value list to rows.
            .with_columns(
                response_value_index=pl.int_ranges(pl.col("response_value").list.len())
            )
            .explode("response_value", "response_value_index")
            # Expand geo struct to lat/long columns.
            .with_columns(
                pl.col("response_geo").struct.unnest().name.prefix("response_geo_")
            )
            # Expand matrix list to rows.
            .explode("response_matrix")
            # Unnest matrix struct to columns.
            .with_columns(
                pl.col("response_matrix")
                .struct.unnest()
                .name.prefix("response_matrix_")
            )
            # Expand matrix value list to rows.
            .with_columns(
                response_matrix_value_index=pl.int_ranges(
                    pl.col("response_matrix_value").list.len()
                )
            )
            .explode("response_matrix_value", "response_matrix_value_index")
            # Exclude temporary struct columns.
            .select(pl.exclude("response_matrix", "response_geo"))
        )

    def _users(self, user_type: UserType) -> list[MindloggerUser]:
        """Get users for specific type."""
        return list(
            map(
                MindloggerUser.from_struct_factory(user_type),
                self._report_frame.select(
                    user_info=pl.struct(*UserType.columns(user_type))
                )
                .get_column("user_info")
                .unique(),
            )
        )

    def __str__(self):
        """Return string representation of MindloggerData object reporting column names and report head."""
        return f"MindloggerData: {self._report_frame.columns}\n\n{self._report_frame.head()}"

    @classmethod
    def load(cls, input_dir: Path) -> pl.DataFrame:
        """Read Mindlogger export into DataFrame.

        This class method reads all reports from a Mindlogger export directory,
        and returns a single DataFrame object.

        Args:
            input_dir: Path to directory containing Mindlogger export.

        Returns:
            DataFrame object.

        Raises:
            FileNotFoundError: If export_dir not found, or does not contain any
                report.csv files.
            NotADirectoryError: If export_dir is not a directory.
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Export directory {input_dir} not found.")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"{input_dir} is not a directory.")

        LOG.debug("Reading report files from %s...", input_dir)

        # Read report files.
        try:
            report = pl.read_csv(input_dir / MINDLOGGER_REPORT_PATTERN)
            for proc in sorted(ReportProcessor.PROCESSORS, key=lambda x: x.PRIORITY):
                LOG.debug("Running processor %s...", proc.NAME)
                report = proc().process(report)
        except pl.exceptions.ComputeError as e:
            raise FileNotFoundError(f"No report CSV files found in {input_dir}.") from e

        return report

    @classmethod
    def create(
        cls,
        input_dir: Path,
    ) -> MindloggerData:
        """Loads Mindlogger report export and creates MindloggerData for inspection.

        Args:
            input_dir: Mindlogger export directory containing report*.csv.

        Returns:
            MindloggerData object.
        """
        return cls(cls.load(input_dir))
