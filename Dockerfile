
ARG DEBIAN_BASE_VERSION=bookworm
ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.4.5
# Python module to run in container.
ARG MAIN_MODULE=mindlogger_graphomotor

FROM python:${PYTHON_VERSION}-${DEBIAN_BASE_VERSION} AS builder

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.12 \
    UV_PROJECT_ENVIRONMENT=/app

WORKDIR /opt/run

COPY pyproject.toml /_lock/
COPY uv.lock /_lock/

RUN --mount=type=cache,target=/root/.cache <<EOT
cd /_lock
uv sync --frozen --no-dev --no-install-project
EOT

COPY . /src
RUN --mount=type=cache,target=/root/.cache \
    uv pip install --python=$UV_PROJECT_ENVIRONMENT --no-deps /src


FROM python:${PYTHON_VERSION}-slim-${DEBIAN_BASE_VERSION} AS runtime
ARG MAIN_MODULE
ENV MAIN_MODULE=${MAIN_MODULE:-mindlogger_data_export}

ENV PATH=/app/bin:$PATH

# Copy virtual environment from builder.
COPY --from=builder /app /app

WORKDIR /app

ENTRYPOINT ["mindlogger-data-export"]
