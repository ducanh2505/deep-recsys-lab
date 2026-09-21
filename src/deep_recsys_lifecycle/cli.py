from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from typing import Annotated

import typer

from .kafka import ConfluentKafkaBoundary
from .lifecycle import run_fast_lifecycle

app = typer.Typer(add_completion=False, help="Movie recommender lifecycle commands.")


def _broker_is_reachable(bootstrap_servers: str) -> bool:
    host, port_text = bootstrap_servers.rsplit(":", maxsplit=1)
    try:
        with socket.create_connection((host, int(port_text)), timeout=0.5):
            return True
    except (OSError, ValueError):
        return False


def _ensure_kafka(bootstrap_servers: str) -> None:
    if _broker_is_reachable(bootstrap_servers):
        return
    try:
        subprocess.run(["docker", "compose", "up", "-d", "kafka"], check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise typer.BadParameter(
            "Kafka is unreachable and Docker Compose could not start the official broker"
        ) from error

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if _broker_is_reachable(bootstrap_servers):
            return
        time.sleep(0.5)
    raise typer.BadParameter(f"Kafka did not become reachable at {bootstrap_servers}")


@app.command("fast")
def fast(
    output: Annotated[Path, typer.Option(help="Fresh output directory.")] = Path("artifacts/fast"),
    bootstrap_servers: Annotated[
        str, typer.Option(help="Kafka bootstrap address.")
    ] = "localhost:9092",
) -> None:
    """Run the fixture → Kafka → artifact → FastAPI-ready report path."""

    _ensure_kafka(bootstrap_servers)
    result = run_fast_lifecycle(
        output_dir=output,
        kafka=ConfluentKafkaBoundary(bootstrap_servers),
    )
    typer.echo(f"Serving Artifact: {result.artifact_path}")
    typer.echo(f"Static report: {result.report_path}")


@app.command("serve")
def serve(
    artifact: Annotated[Path, typer.Option(help="Serving Artifact directory.")],
    host: Annotated[str, typer.Option(help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
) -> None:
    """Serve a previously exported artifact through FastAPI."""

    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(artifact), host=host, port=port)


def main() -> None:
    app()
