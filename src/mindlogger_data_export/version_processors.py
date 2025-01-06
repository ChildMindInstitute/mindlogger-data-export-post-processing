"""Processors for different versions of MindLogger Export."""

from __future__ import annotations

import logging
from typing import Protocol

from packaging.version import Version

LOG = logging.getLogger(__name__)


class DataVersionProcessor(Protocol):
    """Protocol for data version processing."""

    def check_version(self, version: Version) -> bool:
        """Check if the processor can handle the given version."""

    # TODO: define input and return types.
    def process_activities(self) -> None:
        """Process a set of activities."""

    def process_report_row(self) -> None:
        """Process a row of data, adding resources to builder."""


class ExampleDataProcessor(DataVersionProcessor):
    """Processor for new MindLogger data."""

    MIN_VERSION = Version("14.6.149")

    def check_version(self, version: Version) -> bool:
        """Check if the processor can handle the given version."""
        return version > self.MIN_VERSION

    def process_activities(self) -> None:
        """Process a set of activities."""
        return

    def process_report_row(self) -> None:
        """Process a row of data to make version-specific changes."""
        return


class DefaultDataProcessor(DataVersionProcessor):
    """Default processor for MindLogger data."""

    def check_version(self, version: Version) -> bool:
        """Always return True for default processor."""
        del version
        return True

    def process_activities(self) -> None:
        """Process a set of activities."""

    def process_report_row(self) -> None:
        """Process a row of data."""
        return
