"""Main module for MindLogger data export processing."""

from __future__ import annotations

import logging

from rich.console import Console
from tyro.conf import OmitArgPrefixes
from tyro.extras import SubcommandApp

from .config import LogLevel, OutputConfig
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
    try:
        logging.basicConfig(level=config.log_level.upper())
        logging.debug("Starting MindLogger data export tool with config: %s.", config)

        ml_data = MindloggerData.create(config.input_dir)
        writer = OutputWriter.create(config.output_format)

        for output_type in config.output_types_or_all:
            if output_type not in Output.TYPES:
                raise ValueError(f"Unknown output type argument: {output_type}")  # noqa: TRY301
            logging.debug("Producing output type [%s]", output_type)
            output_producer = Output.TYPES[output_type](config.extra)
            outputs = output_producer.produce(ml_data)
            for output in outputs:
                writer.write(output, config.output_dir_or_default)
    except Exception as e:
        if config.log_level == LogLevel.DEBUG:
            raise
        logging.info(e)
        logging.info("Exiting...")


def cli() -> None:
    """Command-line interface for Graphomotor MindLogger package."""
    app.cli(config=(OmitArgPrefixes,))
