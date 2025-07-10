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
def _(mo, os):
    participants_file = mo.ui.file()
    data_file = mo.ui.file()
    run_button = mo.ui.run_button()
    output_dir = mo.ui.text()
    filter_to_applet_name = mo.ui.text()
    mo.vstack(
        [
            mo.hstack(
                [mo.md("### Upload participants file"), participants_file],
                justify="start",
            ),
            mo.hstack([mo.md("### Upload data file"), data_file], justify="start"),
            mo.hstack(
                [
                    mo.md("### Output dir"),
                    output_dir,
                ],
                justify="start",
            ),
            mo.hstack(
                [mo.md("### Filter data to applet_name: "), filter_to_applet_name],
                justify="start",
            ),
            mo.md(f"Current dir: {os.getcwd()}"),
            run_button,
        ]
    )
    return (
        data_file,
        filter_to_applet_name,
        output_dir,
        participants_file,
        run_button,
    )


@app.cell
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


@app.cell
def _(data_file, filter_to_applet_name, mo, pl, run_button):
    def load_data(mindlogger_data) -> pl.DataFrame:
        """Load data."""
        return (
            pl.read_csv(
                mindlogger_data,
                # try_parse_dates=True,
                # schema_overrides={"response_start_time": pl.Datetime()},
            )
            .filter(pl.col("applet_name") == filter_to_applet_name.value)
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


@app.cell(hide_code=True)
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
        part_dfs = all_attendance.partition_by(
            ["site", "activity_date"], as_dict=True
        )
        return [(("ymha_attendance-all",), all_attendance)] + [
            ((f"site_{part[0]}", f"date_{part[1]}", f"ymha_attendance"), df)
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
                    ("site_{part[0]}", f"ymha_completion_summary"),
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
            _output_path = (
                _output_dir.joinpath(*_path_segments)
                .with_suffix(".xlsx")
                .resolve()
            )
            _output_path.parent.mkdir(parents=True, exist_ok=True)
            _df.write_excel(
                _output_path,
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
    return


if __name__ == "__main__":
    app.run()
