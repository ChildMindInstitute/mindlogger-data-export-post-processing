"""Output format writers for MindLogger export processing package."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import polars as pl
import polars.selectors as cs


class OutputWriter(Protocol):
    """Protocol for output writers."""

    NAME: str

    WRITERS: dict[str, OutputWriter] = {}

    def __init_subclass__(cls, **kwargs):
        """Register preprocessor subclasses."""
        super().__init_subclass__(**kwargs)
        cls.WRITERS[cls.NAME] = cls

    def write(
        self, data: pl.DataFrame, output_path: Path, *, drop_null_columns: bool = False
    ) -> None:
        """Write data to output directory."""
        ...


class CsvWriter(OutputWriter):
    """Write data to CSV."""

    NAME = "csv"

    def write(
        self,
        data: pl.DataFrame,
        output_path: Path,
        *,
        drop_null_columns: bool = False,
    ) -> None:
        """Write data to output directory."""
        # Convert duration to milliseconds for CSV output.
        df = data.with_columns(
            cs.duration().dt.total_milliseconds().name.suffix("_ms")
        ).drop(cs.duration())

        # Drop all struct and list columns.
        df = df.drop(
            col
            for col, dtype in df.schema.items()
            if dtype.base_type() in (pl.List, pl.Struct, pl.Array, pl.Object)
        )

        # Drop columns with all null values.
        if drop_null_columns:
            df = df.select([s.name for s in df if not (s.null_count() == df.height)])

        # Write to CSV.
        df.write_csv(output_path.with_suffix(".csv"))


class ParquetWriter(OutputWriter):
    """Write data to Parquet format."""

    NAME = "parquet"

    def write(
        self,
        data: pl.DataFrame,
        output_path: Path,
        *,
        drop_null_columns: bool = False,
    ) -> None:
        """Write data to output directory."""
        del drop_null_columns
        df = data.report

        # Write to Parquet.
        df.write_parquet(output_path.with_suffix("parquet"))
