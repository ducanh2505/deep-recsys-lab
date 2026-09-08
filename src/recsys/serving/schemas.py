"""Public recommendation API schemas."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from recsys.datasets.schema import HistoryInteraction

StrictIdentifier = StrictStr | StrictInt


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: StrictIdentifier | None = None
    interactions: list[HistoryInteraction] = Field(default_factory=list, max_length=5000)
    top_k: StrictInt = Field(default=10, ge=1, le=1000)
    exclude_seen: StrictBool = True

    @model_validator(mode="after")
    def exactly_one_query_mode(self) -> Self:
        has_user = self.user_id is not None
        has_history = bool(self.interactions)
        if has_user == has_history:
            raise ValueError("provide exactly one of user_id or non-empty interactions")
        return self


class Recommendation(BaseModel):
    item_id: StrictIdentifier
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    request_id: str
    artifact_id: str
    recommendations: list[Recommendation]


class ModelResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    contract_schema: dict[str, Any] = Field(alias="schema")
    artifact_id: str
    plugin: str
    capabilities: list[str]
    runtime: str
    catalog_size: int


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetail
