import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import polars.selectors as cs

    return Path, cs, mo, os, pl


@app.cell(hide_code=True)
def _(cs, pl):
    _df = pl.DataFrame(
        data={
            "subscale_one": ["0", None, "f", "30"],
            "activity_score": [None, None, "99", None],
            "item_name": ["item1", "item2", "item2", "item4"],
            "item_response": ["response1", "response2", "resp3", "resp4"],
            "item_type": ["multi", "single", "single", "geo"],
            "activity_name": ["ac1", "ac2", "ac1", "ac2"],
            "id": ["id1", "id1", "id2", "id2"],
        }
    )

    _sscs = cs.by_name("subscale_one", "activity_score")

    _df = _df.with_columns(pl.struct(_sscs).alias("subscale")).drop(
        cs.by_name("subscale_one", "activity_score")
    )

    _sdf = _df.unpivot(
        index=(~cs.by_name("item_response", "item_name", "item_type", "subscale")),
        on="subscale",
        value_name="subscale_value",
        variable_name="item_type",
    )
    _sdf = (
        _sdf.with_columns(
            pl.col("subscale_value").map_elements(
                lambda e: [
                    {
                        "item_name": vn,
                        "item_response": vv,
                    }
                    for vn, vv in e.items()
                    if vv
                ],
                return_dtype=pl.List(
                    pl.Struct(
                        {
                            "item_name": pl.String,
                            "item_response": pl.String,
                        }
                    )
                ),
            )
        )
        .filter(pl.col("subscale_value").list.len() > 0)
        .explode("subscale_value")
        .unnest("subscale_value")
    )
    pl.concat([_df.drop("subscale"), _sdf], how="diagonal")


@app.cell(hide_code=True)
def _(cs, pl):
    _df = pl.DataFrame(
        data={
            "subscale_one": ["0", None, "f", "30"],
            "activity_score": [None, None, "99", None],
            "item_name": ["item1", "item2", "item2", "item4"],
            "item_response": ["response1", "response2", "resp3", "resp4"],
            "item_type": ["multi", "single", "single", "geo"],
            "activity_name": ["ac1", "ac2", "ac1", "ac2"],
            "id": ["id1", "id1", "id2", "id2"],
        }
    )

    pl.concat(
        [
            _df.drop(cs.by_name("subscale_one", "activity_score")),
            _df.unpivot(
                index=cs.by_name("id", "activity_name"),
                on=cs.by_name("subscale_one", "activity_score"),
                variable_name="item_name",
                value_name="item_response",
            )
            .filter(pl.col("item_response").is_not_null())
            .with_columns(item_type=pl.lit("subscale")),
        ],
        how="diagonal",
    )


@app.cell(hide_code=True)
def _(mo, os):
    output_dir = mo.ui.text()
    mo.vstack(
        [
            mo.md("## Output dir"),
            mo.md(f"* Current dir: {os.getcwd()}"),
            output_dir,
        ]
    )
    return (output_dir,)


@app.cell(hide_code=True)
def _(mo):
    participants_file = mo.ui.file()
    mo.vstack([mo.md("## Upload participants file"), participants_file])
    return (participants_file,)


@app.cell(hide_code=True)
def _(OutputGenerationError, cs, mo, participants_file, pl, run_button):
    def load_participants(data) -> pl.DataFrame:
        """Load participants from file path in extra args."""
        participants = pl.read_csv(data)
        if "site" not in participants.columns:
            raise OutputGenerationError(
                "'site' column not found in YMHA participants file"
            )
        if "secretUserId" not in participants.columns:
            raise OutputGenerationError(
                "'secretUserId' column not found in YMHA participants file"
            )
        return participants.select(
            pl.col("secretUserId").alias("secret_id"),
            pl.col("nickname"),
            pl.col("firstName").alias("first_name"),
            pl.col("lastName").alias("last_name"),
            "site",
            cs.matches("^room$"),
        )

    mo.stop(not run_button.value, mo.md(""))
    participants_data = load_participants(participants_file.contents())
    return (participants_data,)


@app.cell(hide_code=True)
def _(mo):
    data_file = mo.ui.file()
    mo.vstack([mo.md("## Upload data file"), data_file])
    return (data_file,)


@app.cell(hide_code=True)
def _(mo):
    run_button = mo.ui.run_button()
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def _(data_file, mo, pl, run_button):
    def load_data(mindlogger_data) -> pl.DataFrame:
        """Load data."""
        return (
            pl.read_csv(
                mindlogger_data,
                # try_parse_dates=True,
                # schema_overrides={"response_start_time": pl.Datetime()},
            )
            .select(
                pl.col("activity_name"),
                pl.col("secret_user_id").alias("secret_id"),
                pl.col("response_start_time")
                .str.to_datetime("%D %k:%M")
                .dt.date()
                .alias("activity_date"),
            )
            .unique()
            .with_columns(activity_completed=pl.lit(True))
        )

    mo.stop(not run_button.value, mo.md(""))
    data = load_data(data_file.contents())
    return (data,)


@app.cell
def _(cs, pl):
    def calc_attendance(
        df: pl.DataFrame, participants: pl.DataFrame
    ) -> list[tuple[str, pl.DataFrame]]:
        attendance = df.pivot(
            on="activity_name",
            values="activity_completed",
            sort_columns=True,
            maintain_order=True,
            aggregate_function=pl.element().any(),
        )
        dates = attendance.select(pl.col("activity_date").unique()).filter(
            pl.col("activity_date").is_not_null()
        )
        participant_dates = participants.join(dates, how="cross")
        all_attendance = participant_dates.join(
            attendance,
            on=["secret_id", "activity_date"],
            how="left",
        ).with_columns(pl.col("^Student Check.*$").fill_null(False))  # noqa: FBT003
        part_dfs = all_attendance.partition_by(["site", "activity_date"], as_dict=True)
        return [(("ymha_attendance-all",), all_attendance)] + [
            ((f"site_{part[0]}", f"date_{part[1]}", "ymha_attendance"), df)
            for part, df in part_dfs.items()
        ]

    def calc_completion(
        df: pl.DataFrame, participants: pl.DataFrame
    ) -> list[tuple[str, pl.DataFrame]]:
        completion = df.drop("activity_date").pivot(
            on="activity_name",
            values="activity_completed",
            aggregate_function=pl.element().any(),
            maintain_order=True,
            sort_columns=True,
        )
        activity_col_selector = cs.exclude(
            [
                "secret_id",
                "nickname",
                "first_name",
                "last_name",
                "site",
                cs.matches("^room$"),
            ]
        )
        identifier_col_selector = cs.by_name(
            "secret_id",
            "nickname",
            "first_name",
            "last_name",
            "site",
        ) | cs.matches(r"^room$")
        all_completion = (
            participants.join(completion, on="secret_id", how="left")
            .select(
                identifier_col_selector,
                activity_col_selector.fill_null(False),  # noqa: FBT003
            )
            .with_columns(
                complete=pl.concat_list(activity_col_selector).list.all(),
            )
        )
        site_completion = all_completion.partition_by("site", as_dict=True)
        return (
            [
                (("ymha_completion-all",), all_completion),
                (
                    ("ymha_completion_summary-all",),
                    all_completion.select(identifier_col_selector, "complete"),
                ),
            ]
            + [
                (("site_{part[0]}", "ymha_completion"), df)
                for part, df in site_completion.items()
            ]
            + [
                (
                    ("site_{part[0]}", "ymha_completion_summary"),
                    df.select(identifier_col_selector, "complete"),
                )
                for part, df in site_completion.items()
            ]
        )

    return calc_attendance, calc_completion


@app.cell
def _(
    calc_attendance,
    calc_completion,
    data,
    mo,
    participants_data,
    pl,
    run_button,
):
    mo.stop(not run_button.value, mo.md(""))
    _partitioned_activities = data.with_columns(
        is_ema=pl.col("activity_name").str.starts_with("Student Check")
    ).partition_by("is_ema", as_dict=True, include_key=False)

    outputs = (
        calc_attendance(_partitioned_activities[(True,)], participants_data)
        if (True,) in _partitioned_activities
        else []
    ) + (
        calc_completion(_partitioned_activities[(False,)], participants_data)
        if (False,) in _partitioned_activities
        else []
    )
    outputs
    return (outputs,)


@app.cell
def _(Path, cs, mo, output_dir, outputs, run_button):
    mo.stop(not run_button.value, mo.md(""))

    _output_dir = Path(output_dir.value)
    if not _output_dir.is_dir():
        disp = mo.md(
            f"OutputDir ({_output_dir}) does not exist or is not a directory. Please update output dir input above."
        )
    else:
        for _path_segments, _df in outputs:
            _df.write_excel(
                _output_dir.joinpath(*_path_segments).with_suffix(".xlsx"),
                conditional_formats={
                    cs.all(): {
                        "type": "cell",
                        "criteria": "==",
                        "value": False,
                        "format": {"bg_color": "#FFC7CE"},
                    }
                },
            )
        print(f"{len(outputs)} outputs written.")
        disp = mo.md(f"{len(outputs)} outputs written.")
    disp


if __name__ == "__main__":
    app.run()
