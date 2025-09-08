# check=skip=FromPlatformFlagConstDisallowed

FROM --platform=linux/amd64 python:3.11

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked --mount=type=cache,target=/var/lib/apt,sharing=locked apt-get update && apt-get install -y --no-install-recommends libopenmpi-dev

RUN --mount=type=cache,target=/root/.cache/pip pip install -U pip setuptools wheel

COPY . /app

RUN --mount=type=cache,target=/root/.cache/pip pip install -e .

ENTRYPOINT ["perturbgen"]
