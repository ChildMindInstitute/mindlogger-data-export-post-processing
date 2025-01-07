"""Data model of Mindlogger export."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from .report_preprocessors import REPORT_PREPROCESSORS

LOG = logging.getLogger(__name__)

FLOW_ITEMS_FILENAME = "flow-items.csv"
FLOW_SCHEDULE_FILENAME = "user-flow-schedule.csv"
USER_ACTIVITY_SCHEDULE_FILENAME = "user-activity-schedule.csv"


class MindloggerData:
    """Data model of Mindlogger export."""

    def __init__(
        self,
        report: pl.DataFrame,
        flow: pl.DataFrame,
        history: pl.DataFrame,
        activity_history: pl.DataFrame,
    ):
        """Initialize MindloggerData object."""
        self.report_frame = report
        self.flow_frame = flow
        self.history_frame = history
        self.activity_history_frame = activity_history

    @classmethod
    def create(cls, export_dir: Path) -> MindloggerData:
        """Read Mindlogger export and create MindloggerData object.

        This factory method reads a full Mindlogger export directory,
        runs pre-processors, and creates a MindloggerData object to represent
        the full export.

        Args:
            export_dir: Path to Mindlogger export directory.

        Returns:
            MindloggerData object.

        Raises:
            FileNotFoundError: If export_dir not found, or does not contain any
                report.csv files.
            NotADirectoryError: If export_dir is not a directory.
        """
        if not export_dir.exists():
            raise FileNotFoundError(f"Export directory {export_dir} not found.")
        if not export_dir.is_dir():
            raise NotADirectoryError(f"{export_dir} is not a directory.")

        LOG.debug("Reading report files from %s...", export_dir)

        # Read report files.
        report = pl.read_csv(export_dir / "report*.csv")
        if report.is_empty():
            raise FileNotFoundError(f"No report CSV files found in {export_dir}.")

        flow = pl.read_csv(export_dir / "flow-items.csv")
        history = pl.read_csv(export_dir / "user-flow-schedule.csv")
        activity_history = (
            pl.read_csv(activity_path)
            if (activity_path := export_dir / "user-activity-schedule.csv").exists()
            else None
        )

        # TODO: Make preprocessors configurable.
        for name, report_preprocessor in REPORT_PREPROCESSORS.items():
            LOG.debug("Running Report Preprocessor: %s", name)
            report = report_preprocessor().run(report)
        return cls(report, flow, history, activity_history)
