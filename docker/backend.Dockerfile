# syntax=docker/dockerfile:1

FROM python:3.12-slim

ARG BENTOML_VERSION=1.4.39
ARG NUMPY_VERSION=2.5.2
ARG ONNXRUNTIME_VERSION=1.29.0
ARG PYDANTIC_VERSION=2.13.4
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    BENTOML_HOME=/home/app/bentoml

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app /home/app/bentoml/models \
    && chown -R app:app /app /home/app/bentoml

RUN python -m pip install \
    "bentoml==${BENTOML_VERSION}" \
    "numpy==${NUMPY_VERSION}" \
    "onnxruntime==${ONNXRUNTIME_VERSION}" \
    "pydantic==${PYDANTIC_VERSION}"

WORKDIR /app
COPY --chown=app:app src/ /app/src/

USER app
EXPOSE 3000

CMD ["bentoml", "serve", "src/service.py:RecommendationService", "--host", "0.0.0.0", "--port", "3000", "--arg", "model_tag=deep_recsys:multvae-onnx-v1"]
