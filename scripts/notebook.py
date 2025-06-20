import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import polars as pl

    from mindlogger_data_export import MindloggerData
    from mindlogger_data_export.outputs import OptionsFormat

    return MindloggerData, OptionsFormat, Path, pl


@app.cell
def _(MindloggerData, Path):
    md = MindloggerData.create(Path("data/greek/"))
    return (md,)


@app.cell
def _(md):
    md.report.columns


@app.cell
def _(md):
    md.report


@app.cell
def _(md, pl):
    md.report.select(pl.col("item")).unnest("item")


@app.cell
def _(OptionsFormat, md):
    OptionsFormat().produce(md)[0].output


if __name__ == "__main__":
    app.run()
