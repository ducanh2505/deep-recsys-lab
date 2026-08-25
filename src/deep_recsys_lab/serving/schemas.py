"""BentoML API request and response schemas."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator


class RecommendationRequest(BaseModel):
    movie_ids: list[StrictInt] = Field(min_length=5, max_length=500)
    top_k: StrictInt = Field(default=10, ge=1, le=100)

    @field_validator("movie_ids")
    @classmethod
    def unique_movie_ids(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("movie_ids must be unique")
        return value


class Recommendation(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float


class RecommendationResponse(BaseModel):
    request_id: str
    model_version: str
    recommendations: list[Recommendation]


class BatchRecommendationResult(BaseModel):
    """Internal per-request result that keeps domain failures batch-local."""

    model_version: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    error: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def success_or_error(self) -> Self:
        if self.error is None and not self.recommendations:
            raise ValueError("a successful batch result must contain recommendations")
        if self.error is not None and self.recommendations:
            raise ValueError("an errored batch result cannot contain recommendations")
        return self


class ModelResponse(BaseModel):
    model_version: str
    model_tag: str
    schema_version: str
    item_count: int
    mapping_hash: str
    score_note: str
    inference_backend: str
    execution_provider: str
    onnx_opset: int
