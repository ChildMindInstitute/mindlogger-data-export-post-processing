"""Main module for MindLogger data export processing."""

from __future__ import annotations

import logging

import tyro
from rich.console import Console

from .config import MindloggerDataConfig, OutputTypesInfo
from .mindlogger import MindloggerData
from .outputs import Output
from .writers import OutputWriter


def main(config: MindloggerDataConfig) -> None:
    """Main method for command-line interface."""
    console = Console()

    if isinstance(config.cmd, OutputTypesInfo):
        console.rule("Available output types:")
        console.print(
            "\n".join(
                f"\t[bold green]{kv[0]}[/bold green]: {kv[1]}"
                for kv in Output.output_types_info().items()
            )
        )
        return

    run_config = config.cmd

    logging.basicConfig(level=config.log_level.upper())
    logging.debug("Starting MindLogger data export tool with config: %s.", config)
    logging.debug("Input directory: %s", run_config.input_dir)
    logging.debug("Output directory: %s", run_config.output_dir_or_default)

    ml_data = MindloggerData.create(run_config.input_dir)
    writer = OutputWriter.create(run_config.output_format)

    for output_types in run_config.output_types_or_all:
        output_producer = Output.TYPES[output_types]()
        outputs = output_producer.produce(ml_data)
        for output in outputs:
            writer.write(output, run_config.output_dir_or_default)


def cli() -> None:
    """Command-line interface for Graphomotor MindLogger package."""
    config = tyro.cli(
        MindloggerDataConfig, config=(tyro.conf.ConsolidateSubcommandArgs,)
    )
    main(config)
