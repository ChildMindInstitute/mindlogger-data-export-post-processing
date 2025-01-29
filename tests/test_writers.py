"""Tests for the writers module."""

import polars as pl
import pytest
from polars.testing.asserts import assert_frame_equal

from mindlogger_data_export.writers import OutputWriter


@pytest.fixture
def dataframe():
    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "data": ["data_1", "data_2", "data_3"],
        }
    )


@pytest.fixture
def nested_fields_df(dataframe):
    return dataframe.with_columns(
        struct_column=pl.Series([{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]),
        list_column=pl.Series([["a", "b"], ["b"], ["c"]]),
    )


@pytest.fixture
def null_columns_df(dataframe):
    return dataframe.with_columns(null_column=pl.Series([None, None, None]))


@pytest.fixture
def csv_writer():
    return OutputWriter.create("csv")


@pytest.fixture
def parquet_writer():
    return OutputWriter.create("parquet")


def test_csv_writer_removes_nested_fields(csv_writer, nested_fields_df, tmp_path):
    csv_writer.write(nested_fields_df, tmp_path / "output.csv")
    output_df = pl.read_csv(tmp_path / "output.csv")
    assert "struct_columns" not in output_df.columns
    assert "list_columns" not in output_df.columns


def test_csv_writer_removes_null_columns(csv_writer, null_columns_df, tmp_path):
    csv_writer.write(null_columns_df, tmp_path / "output.csv", drop_null_columns=True)
    output_df = pl.read_csv(tmp_path / "output.csv")
    assert "null_column" not in output_df.columns


def test_csv_writer(csv_writer, dataframe, tmp_path):
    csv_writer.write(dataframe, tmp_path / "output.csv")
    output_df = pl.read_csv(tmp_path / "output.csv")
    assert_frame_equal(dataframe, output_df)


def test_parquet_writer(parquet_writer, nested_fields_df, tmp_path):
    parquet_writer.write(nested_fields_df, tmp_path / "output.parquet")
    output_df = pl.read_parquet(tmp_path / "output.parquet")
    assert_frame_equal(nested_fields_df, output_df)
