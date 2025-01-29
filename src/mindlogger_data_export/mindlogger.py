"""Data model of Mindlogger export."""

from __future__ import annotations

import logging
from functools import cached_property

import pandas as pd
import polars as pl

from .config import MindloggerExportConfig
from .models import MindloggerUser, UserType

LOG = logging.getLogger(__name__)

FLOW_ITEMS_FILENAME = "flow-items.csv"
FLOW_SCHEDULE_FILENAME = "user-flow-schedule.csv"
USER_ACTIVITY_SCHEDULE_FILENAME = "user-activity-schedule.csv"


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

    @cached_property
    def data_dictionary_pd(self) -> pd.DataFrame:
        """Return unique items in report in Pandas format."""
        return self.data_dictionary.to_pandas()

    def __str__(self):
        """Return string representation of MindloggerData object reporting column names and report head."""
        return f"MindloggerData: {self._report_frame.columns}\n\n{self._report_frame.head()}"

    @classmethod
    def create(
        cls,
        config: MindloggerExportConfig,
    ) -> MindloggerData:
        """Read Mindlogger export and create MindloggerData object.

        This factory method reads a full Mindlogger export directory,
        runs pre-processors, and creates a MindloggerData object to represent
        the full export.

        Args:
            config: Mindlogger export configuration.

        Returns:
            MindloggerData object.

        Raises:
            FileNotFoundError: If export_dir not found, or does not contain any
                report.csv files.
            NotADirectoryError: If export_dir is not a directory.
        """
        if not config.input_dir.exists():
            raise FileNotFoundError(f"Export directory {config.input_dir} not found.")
        if not config.input_dir.is_dir():
            raise NotADirectoryError(f"{config.input_dir} is not a directory.")

        LOG.debug("Reading report files from %s...", config.input_dir)

        # Read report files.
        try:
            report = pl.read_csv(config.input_dir / "report*.csv")
        except pl.exceptions.ComputeError:
            raise FileNotFoundError(
                f"No report CSV files found in {config.input_dir}."
            ) from None

        return cls(report)
