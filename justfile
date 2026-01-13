
DOCKER_TAG := "cmiresearch/mdxpp"

run +ARGS='':
    uv run mindlogger-data-export

test +ARGS='':
    uv run pytest {{ ARGS }}

build-docker:
    docker buildx build -t {{ DOCKER_TAG }} .

run-docker +ARGS='': build-docker
    docker run {{ DOCKER_TAG }} -- {{ ARGS }}

release:
    rm -f dist/*
    uv version --bump patch
    uv build
    uv publish --token "${PYPI_MINDLOGGER_DATA_EXPORT_KEY}"
