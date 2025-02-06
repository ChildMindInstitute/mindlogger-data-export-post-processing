"""Configuration object for Mindlogger Data Export tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Literal

from tyro.conf import EnumChoicesFromValues, UseAppendAction, arg

from .outputs import Output


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

    input_dir: Annotated[Path, arg(aliases=["-i"])]
    """Path to input directory, containing MindLogger data export."""

    output_dir: Annotated[Path | None, arg(aliases=["-o"])] = None
    """Path to output directory, where processed data will be written. Defaults to input_dir."""

    output_format: Annotated[Literal["csv", "parquet"], arg(aliases=["-f"])] = "csv"

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

    timezone: str = "America/New_York"
    """Timezone to which datetimes will be converted."""

    @property
    def output_dir_or_default(self) -> Path:
        """Get output directory, defaulting to input directory."""
        return self.output_dir or self.input_dir

    @property
    def output_types_or_all(self) -> list[str]:
        """Get output types."""
        return self.outputs or list(Output.TYPES.keys())
