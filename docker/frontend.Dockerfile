# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/scripts \
    && chown -R app:app /app

WORKDIR /app
COPY --chown=app:app api.html /app/api.html
COPY --chown=app:app scripts/serve_model_ui.py /app/scripts/serve_model_ui.py

USER app
EXPOSE 8080

CMD ["python", "scripts/serve_model_ui.py", "--host", "0.0.0.0", "--port", "8080", "--backend-url", "http://backend:3000", "--ui-file", "/app/api.html"]
