run +ARGS='':
    uv run mindlogger-data-export

test +ARGS='':
    uv run pytest {{ ARGS }}

build-docker TAG='cmiresearch/mdxpp':
    docker buildx build -t {{ TAG }} .

run-docker +ARGS='': build-docker
    docker run cmiresearch/mdxpp -- {{ ARGS }}
