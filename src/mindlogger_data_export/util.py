"""Utilities carried over from original scripts."""

from collections.abc import Generator

import polars as pl


def unnest_structs(*cols: str) -> Generator[pl.Expr, None, None]:
    """Unnest struct columns and prefix resulting columns with '<column_name>_'."""
    for sc in cols:
        yield pl.col(sc).struct.unnest().name.prefix(f"{sc}_")
