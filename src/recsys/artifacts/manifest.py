"""Machine-readable artifact schema."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from recsys.core.types import QueryMode


class ArtifactIntegrityError(ValueError):
    """An artifact is corrupt, incomplete, or incompatible."""


class PayloadEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[1] = 1
    artifact_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    created_at: datetime
    plugin: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    capabilities: list[QueryMode] = Field(min_length=1)
    dataset_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config: dict[str, Any]
    payloads: dict[str, PayloadEntry]
    contract_schema: dict[str, Any] = Field(alias="schema")
    catalog_size: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("capabilities")
    @classmethod
    def unique_capabilities(cls, value: list[QueryMode]) -> list[QueryMode]:
        if len(set(value)) != len(value):
            raise ValueError("artifact capabilities must be unique")
        return value
