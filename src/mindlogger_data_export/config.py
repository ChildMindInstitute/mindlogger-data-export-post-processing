"""Configuration object for Mindlogger Data Export tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Literal

from tyro.conf import EnumChoicesFromValues, Positional, UseAppendAction, arg


class LogLevel(StrEnum):
    """Enumeration of logging levels."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class OutputFormatEnum(StrEnum):
    """Enumeration of output formats."""

    CONCATENATED_REPORTS = "concatenated"
    """Concatenated report rows."""

    TYPED_COLUMNS_SINGLE_VALUE_ROWS = "typed"
    """Typed columns with single value rows."""

    DATA_DICTIONARY = "dictionary"
    """Data dictionary."""

    @classmethod
    def all(cls) -> list[OutputFormatEnum]:
        """Get all output types."""
        return list(cls)


@dataclass
class MindloggerExportConfig:
    """Configuration object for Mindlogger Data Export tool."""

    input_dir: Positional[Path]
    """Path to input directory, containing MindLogger data export."""

    output_dir: Annotated[Path | None, arg(aliases=["-o"])] = None
    """Path to output directory, where processed data will be written. Defaults to input_dir."""

    output_type: Literal["csv", "parquet"] = "csv"

    output_formats: Annotated[
        UseAppendAction[list[OutputFormatEnum]],
        EnumChoicesFromValues,
        arg(aliases=["-t"], help_behavior_hint="(default: all)"),
    ] = field(default_factory=list)
    """List of output types to generate, run tool with --output-types-info or see documentation for detailed description."""

    timezone: str = "America/New_York"
    """Timezone to which datetimes will be converted."""

    log_level: Annotated[LogLevel, EnumChoicesFromValues, arg(aliases=["-l"])] = (
        LogLevel.DEBUG
    )
    """Logging level for the tool."""

    output_types_info: bool = False
    """Output information about output types and exit."""

    @property
    def output_dir_or_default(self) -> Path:
        """Get output directory, defaulting to input directory."""
        return self.output_dir or self.input_dir

    @property
    def output_formats_or_all(self) -> list[OutputFormatEnum]:
        """Get output types."""
        return self.output_formats or OutputFormatEnum.all()
