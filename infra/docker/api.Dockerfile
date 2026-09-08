FROM python:3.12-slim AS wheelhouse

WORKDIR /build
COPY pyproject.toml README.md NOTICE ./
COPY src ./src
RUN python -m pip wheel --wheel-dir /wheelhouse ".[serve]"

FROM python:3.12-slim

RUN useradd --create-home --uid 10001 app
COPY --from=wheelhouse /wheelhouse /wheelhouse
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheelhouse "recsys-platform[serve]" \
    && rm -rf /wheelhouse

USER app
WORKDIR /home/app
EXPOSE 3000
ENTRYPOINT ["recsys", "serve", "--artifact", "/artifact", "--host", "0.0.0.0", "--port", "3000"]
