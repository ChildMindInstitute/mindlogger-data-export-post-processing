"""Utilities carried over from original scripts."""

from collections.abc import Generator

import polars as pl

from .schema import (
    ACCOUNT_USER_SCHEMA,
    INPUT_USER_SCHEMA,
    SOURCE_USER_SCHEMA,
    TARGET_USER_SCHEMA,
)


def unnest_structs(*cols: str) -> Generator[pl.Expr, None, None]:
    """Unnest struct columns and prefix resulting columns with '<column_name>_'."""
    for sc in cols:
        yield pl.col(sc).struct.unnest().name.prefix(f"{sc}_")


def unnest_user_structs() -> Generator[pl.Expr, None, None]:
    """Unnest user struct columns."""
    yield pl.col("target_user").struct.field(*TARGET_USER_SCHEMA.keys())
    yield pl.col("source_user").struct.field(*SOURCE_USER_SCHEMA.keys())
    yield pl.col("input_user").struct.field(*INPUT_USER_SCHEMA.keys())
    yield pl.col("account_user").struct.field(*ACCOUNT_USER_SCHEMA.keys())
