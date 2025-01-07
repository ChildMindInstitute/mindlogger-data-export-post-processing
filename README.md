# Mindlogger Data Export Tool

## Development Instructions

1. Install [uv](https://docs.astral.sh/uv/)
2. Clone repository && `cd mindlogger-data-export`
3. `uvx pre-commit install` to install git pre-commit that runs formatting/style checks on commit.
4. `uv sync` to install dependencies

To run command line tool: `uv run mindlogger-data-export`

### Just command runner

The project uses Just to provide shortcuts for common development commands, e.g. just test just run just build-docker

Run just -l to view commands.
