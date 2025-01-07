"""Main module for MindLogger data export processing."""

import logging
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Config:
    """Configuration for MindLogger export processing."""

    input_dir: Path
    output_dir: Path


def main(config: Config) -> None:
    """Main method for command-line interface."""
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Input directory: %s", config.input_dir)
    logging.debug("Output directory: %s", config.output_dir)


def cli() -> None:
    """Command-line interface for Graphomotor MindLogger package."""
    config = tyro.cli(Config)
    main(config)
