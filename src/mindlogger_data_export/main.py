"""Main module for MindLogger data export processing."""

from __future__ import annotations

import logging

from rich.console import Console
from tyro.conf import OmitArgPrefixes
from tyro.extras import SubcommandApp

from .config import OutputConfig
from .mindlogger import MindloggerData
from .outputs import Output
from .writers import OutputWriter

app = SubcommandApp()


@app.command
def output_types_info() -> None:
    """Output information about available output types."""
    console = Console()
    console.rule("Available output types:")
    console.print(
        "\n".join(
            f"\t[bold green]{kv[0]}[/bold green]: {kv[1]}"
            for kv in Output.output_types_info().items()
        )
    )


@app.command(name="run")
def main(config: OutputConfig) -> None:
    """Run data export transformations to produce outputs."""
    logging.basicConfig(level=config.log_level.upper())
    logging.debug("Starting MindLogger data export tool with config: %s.", config)

    ml_data = MindloggerData.create(config.input_dir)
    writer = OutputWriter.create(config.output_format)

    for output_types in config.output_types_or_all:
        output_producer = Output.TYPES[output_types]()
        outputs = output_producer.produce(ml_data)
        for output in outputs:
            writer.write(output, config.output_dir_or_default)


def cli() -> None:
    """Command-line interface for Graphomotor MindLogger package."""
    app.cli(config=(OmitArgPrefixes,))
