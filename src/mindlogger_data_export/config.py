"""Configuration object for Mindlogger Data Export tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Literal

from tyro.conf import EnumChoicesFromValues, UseAppendAction, arg

from .outputs import Output

MINDLOGGER_REPORT_PATTERN = "*responses*.csv"
ACTIVITY_FLOW_PATTERN = "activity_flow*.csv"
FLOW_ITEM_HISTORY_PATTERN = "flow_item_history*.csv"
SCHEDULE_HISTORY_PATTERN = "schedule_history*.csv"


class LogLevel(StrEnum):
    """Enumeration of logging levels."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class OutputTypesInfo:
    """Output information about output types and exit."""


@dataclass
class OutputConfig:
    """Run the MindLogger data export tool."""

    input: Annotated[Path, arg(aliases=["-i"])]
    f"""Path to either input directory containing Curious data export or csv file produced by Curious.

    If a directory is passed in, we look for files in the directory matching /{MINDLOGGER_REPORT_PATTERN}/.
    """

    output_dir: Annotated[Path | None, arg(aliases=["-o"])] = None
    """Path to output directory, where processed data will be written. Defaults to input_dir."""

    output_format: Annotated[
        Literal["csv", "parquet", "excel"], arg(aliases=["-f"])
    ] = "csv"

    outputs: Annotated[
        UseAppendAction[list[str]],
        EnumChoicesFromValues,
        arg(aliases=["-t"], help_behavior_hint="(default: all)"),
    ] = field(default_factory=list)
    """List of output types to generate, run tool with --output-types-info or see documentation for detailed description."""

    log_level: Annotated[LogLevel, EnumChoicesFromValues, arg(aliases=["-l"])] = (
        LogLevel.INFO
    )
    """Logging level for the tool."""

    drop_null_columns: Annotated[bool, arg(aliases=["-d"])] = False

    extra: Annotated[dict[str, str], arg(aliases=["-e"])] = field(default_factory=dict)
    """Additional parameters to be used for output-specific side-inputs, etc."""

    timezone: str = "America/New_York"
    """Timezone to which datetimes will be converted."""

    @property
    def input_dir(self) -> Path:
        """Get input directory or parent directory if input is file."""
        return self.input if self.input.is_dir() else self.input.parent

    @property
    def output_dir_or_default(self) -> Path:
        """Get output directory, defaulting to input directory."""
        return self.output_dir or (self.input_dir / "output")

    @property
    def output_types_or_all(self) -> list[str]:
        """Get output types."""
        return self.outputs or [t[0] for t in Output.TYPES.items() if t[1].DEFAULT]
