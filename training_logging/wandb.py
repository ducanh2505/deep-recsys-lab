"""Optional W&B tracking with checkpoint identity and local replay cursors."""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
import uuid


@dataclass(frozen=True)
class TrackingOptions:
    """Keep logging settings separate from model/resume compatibility settings."""

    mode: str = "online"
    entity: str = "ducanh250599"
    project: str = "RecSys"
    name: str | None = None
    log_every: int = 100


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the same W&B flags on every standalone training command."""
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        help="W&B mode; overrides WANDB_MODE (default: online)")
    parser.add_argument("--wandb-entity", default=None,
                        help="W&B team; overrides WANDB_ENTITY (default: ducanh250599)")
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project; overrides WANDB_PROJECT (default: RecSys)")
    parser.add_argument("--wandb-name", default=None, help="Display name; default: output directory name")
    parser.add_argument("--wandb-log-every", type=int, default=100,
                        help="Batch logging interval in optimizer steps; also log first/last batches")


def options_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> TrackingOptions:
    """Resolve CLI/environment settings without reading or storing credentials."""
    mode = args.wandb_mode or os.environ.get("WANDB_MODE", "online")
    if mode not in ("online", "offline", "disabled"):
        parser.error("WANDB_MODE must be online, offline or disabled")
    if args.wandb_log_every < 1:
        parser.error("wandb-log-every must be positive")
    return TrackingOptions(mode, args.wandb_entity or os.environ.get("WANDB_ENTITY", "ducanh250599"),
                           args.wandb_project or os.environ.get("WANDB_PROJECT", "RecSys"),
                           args.wandb_name, args.wandb_log_every)


def flatten(values: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten scalar metrics into W&B namespaces, excluding non-finite numbers."""
    result = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten(value, name))
        elif isinstance(value, (str, bool, int)) or (isinstance(value, float) and math.isfinite(value)):
            result[name] = value
    return result


class TrainingTracker:
    """Track a training invocation without letting SDK failures stop training.

    Passing no options disables all tracking, including sidecar writes. This is
    used by search/benchmark callers, which are outside the current scope.
    """

    def __init__(self, directory: Path, family: str, options: TrackingOptions | None) -> None:
        self.directory = directory
        self.family = family
        self.options = options
        self.state: dict = {}
        self._sdk: Any = None
        self._run: Any = None
        self._active = False
        self._completed = False

    def __enter__(self) -> "TrainingTracker":
        return self

    def _warn(self, operation: str, error: Exception) -> None:
        message = str(error)
        secret = os.environ.get("WANDB_API_KEY")
        if secret:
            message = message.replace(secret, "[redacted]")
        print(f"W&B {operation} failed ({type(error).__name__}): {message}. "
              "Training continues with local artifacts.", file=sys.stderr, flush=True)

    def _persist(self) -> None:
        if not self.state:
            return
        try:
            path = self.directory / "wandb_state.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
        except Exception as error:
            self._warn("state save", error)

    def checkpoint_state(self) -> dict:
        """Return only serializable identity/cursors, never SDK objects or secrets."""
        return dict(self.state)

    @property
    def active(self) -> bool:
        """Whether the current SDK session is successfully logging metrics."""
        return self._active

    def start(self, config: dict, checkpoint_state: dict | None = None, resumed: bool = False) -> None:
        """Start a new logical run or reconnect to the checkpoint's online run."""
        if self.options is None:
            return
        options = self.options
        local = {}
        path = self.directory / "wandb_state.json"
        try:
            if resumed and path.exists():
                local = json.loads(path.read_text())
        except Exception as error:
            self._warn("state read", error)
        saved = dict(checkpoint_state or local) if resumed else {}
        self.state = saved or {
            "run_id": uuid.uuid4().hex[:12], "entity": options.entity, "project": options.project,
            "family": self.family, "last_batch_step": 0, "last_epoch": 0, "online_started": False,
        }
        # The sidecar can be ahead of the checkpoint when an unfinished epoch is replayed.
        if local.get("run_id") == self.state["run_id"]:
            for key in ("last_batch_step", "last_epoch"):
                self.state[key] = max(self.state.get(key, 0), local.get(key, 0))
            self.state["online_started"] = self.state.get("online_started", False) or local.get("online_started", False)
        self.state["mode"] = options.mode
        self._persist()
        if options.mode == "disabled":
            return
        try:
            if (self.state["entity"], self.state["project"]) != (options.entity, options.project):
                raise ValueError("Resume must use the checkpoint's W&B entity/project")
            if options.mode == "online" and not os.environ.get("WANDB_API_KEY"):
                raise ValueError("WANDB_API_KEY is missing; use uv run --env-file .env or export it on the GPU host")
            import wandb
            self._sdk = wandb
            session_id = uuid.uuid4().hex[:12] if options.mode == "offline" else self.state["run_id"]
            # Bound request retries on the pinned SDK; finish still drains its queue.
            request_settings = {f"x_{client}_{setting}": value
                                for client in ("file_stream", "file_transfer", "graphql")
                                for setting, value in (("retry_max", 2), ("timeout_seconds", 10),
                                                       ("retry_wait_min_seconds", 1), ("retry_wait_max_seconds", 2))}
            self._run = wandb.init(
                entity=options.entity, project=options.project, mode=options.mode, id=session_id,
                name=options.name or self.directory.name,
                group=f"{self.family}-{self.state['run_id']}", job_type="train",
                tags=[self.family, str(config.get("device_used", config.get("device", "unknown")))],
                config=config, dir=str(self.directory), save_code=False, allow_val_change=True,
                resume=("allow" if resumed or self.state.get("online_started") else "never")
                if options.mode == "online" else None,
                settings=wandb.Settings(init_timeout=15, summary_timeout=5, console="off", disable_git=True,
                                       **request_settings),
            )
            self._active = True
            self._run.define_metric("global_step")
            self._run.define_metric("epoch")
            self._run.define_metric("batch/*", step_metric="global_step")
            for namespace in ("train", "valid", "timing", "memory", "selection"):
                self._run.define_metric(f"{namespace}/*", step_metric="epoch")
            if options.mode == "online":
                self.state["online_started"] = True
                for key in ("last_batch_step", "last_epoch"):
                    self.state[key] = max(self.state.get(key, 0), int(self._run.summary.get(f"tracking/{key}", 0)))
            self.state["session_id"] = session_id
            self._run.summary["training_status"] = "running"
            self._persist()
            print(json.dumps({"event": "wandb_start", "mode": options.mode,
                              "run_id": self.state["run_id"], "session_id": session_id,
                              "url": self._run.url if options.mode == "online" else None}), flush=True)
        except Exception as error:
            self._active = False
            self._warn("initialization", error)

    def wants_batch(self, step: int, boundary: bool = False) -> bool:
        """Avoid accelerator scalar transfers except at logging boundaries."""
        return bool(self._active and step > self.state.get("last_batch_step", 0)
                    and (boundary or step % self.options.log_every == 0))

    def _log(self, values: dict, cursor: str, position: int) -> None:
        if not self._active or position <= self.state.get(cursor, 0):
            return
        try:
            self._run.log(flatten(values))
            self._run.summary[f"tracking/{cursor}"] = position
            self.state[cursor] = position
            self._persist()
        except Exception as error:
            self._active = False
            self._warn("metric logging", error)

    def log_batch(self, epoch: int, step: int, values: dict) -> None:
        """Log batch scalars against cumulative optimizer steps."""
        self._log({"epoch": epoch, "global_step": step, "batch": values}, "last_batch_step", step)

    def log_epoch(self, epoch: int, step: int, values: dict) -> None:
        """Log training/validation/timing/memory namespaces against epochs."""
        self._log({"epoch": epoch, "global_step": step, **values}, "last_epoch", epoch)

    def complete(self, metrics: dict, status: str = "completed") -> None:
        """Publish final scalars and only allowlisted small JSON artifacts."""
        self._completed = True
        if not self._active:
            return
        try:
            self._run.summary.update({"training_status": status, **flatten(metrics, "final")})
            self._run.log(flatten(metrics, "final"))
            artifact = self._sdk.Artifact(f"{self.family}-{self.state['run_id']}-metadata", type="training-metadata")
            for name in ("config.json", "history.json", "metrics.json", "training.json", "progress.json"):
                path = self.directory / name
                if path.is_file():
                    artifact.add_file(str(path), name=name)
            self._run.log_artifact(artifact)
        except Exception as error:
            self._active = False
            self._warn("final logging/upload", error)

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> bool:
        if self._run is not None:
            try:
                if exception_type is not None:
                    self._run.summary["training_status"] = "interrupted" if exception_type is KeyboardInterrupt else "failed"
                    self._run.summary["failure_type"] = exception_type.__name__
                elif not self._completed:
                    self._run.summary["training_status"] = "finished"
                self._run.finish(exit_code=1 if exception_type is not None else 0)
            except Exception as error:
                self._warn("shutdown", error)
        return False
