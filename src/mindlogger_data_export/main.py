"""Main module for MindLogger data export processing."""

import logging

import tyro

from .config import MindloggerExportConfig
from .formats import OutputFormat
from .mindlogger import MindloggerData
from .writers import CsvWriter


def main(config: MindloggerExportConfig) -> None:
    """Main method for command-line interface."""
    logging.basicConfig(level=config.log_level.upper())
    logging.debug("Starting MindLogger data export tool with config: %s.", config)
    logging.debug("Input directory: %s", config.input_dir)
    logging.debug("Output directory: %s", config.output_dir_or_default)

    ml_data = MindloggerData.create(config)
    # TODO: Handle parquet output.
    writer = CsvWriter()

    for output_format in config.output_formats_or_all:
        formatter = OutputFormat.FORMATS[output_format]()
        output = formatter.produce(ml_data.report)
        output_path = (
            config.output_dir_or_default
            / f"report_{formatter.NAME}.{config.output_type}"
        )
        writer.write(output, output_path)


def cli() -> None:
    """Command-line interface for Graphomotor MindLogger package."""
    config = tyro.cli(MindloggerExportConfig)
    main(config)
