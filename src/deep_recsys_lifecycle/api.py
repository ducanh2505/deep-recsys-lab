from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from .artifact import ServingArtifact
from .query import (
    EmptyHistoryQuery,
    HistoryOnlyQuery,
    KnownUserQuery,
    Query,
    RecommendationService,
    UnknownSubjectError,
)


class RecommendationRequest(BaseModel):
    subject_id: int | None = None
    history: list[int] | None = None
    top_n: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def inputs_are_mutually_exclusive(self) -> RecommendationRequest:
        if self.subject_id is not None and self.history is not None:
            raise ValueError("subject_id and history are mutually exclusive")
        return self


def create_app(artifact: ServingArtifact | Path) -> FastAPI:
    loaded_artifact = ServingArtifact.load(artifact) if isinstance(artifact, Path) else artifact
    service = RecommendationService(loaded_artifact)
    app = FastAPI(title="Movie Recommender Lifecycle Showcase")

    @app.get("/health")
    def health() -> dict[str, object]:
        return loaded_artifact.health_metadata()

    @app.post("/recommendations")
    def recommendations(request: RecommendationRequest) -> dict[str, object]:
        query: Query
        if request.subject_id is not None:
            query = KnownUserQuery(subject_id=request.subject_id)
        elif request.history:
            query = HistoryOnlyQuery(movie_ids=tuple(request.history))
        else:
            query = EmptyHistoryQuery()

        try:
            return service.recommend(query, top_n=request.top_n).to_dict()
        except UnknownSubjectError as error:
            raise HTTPException(
                status_code=404,
                detail={"code": "unknown_subject", "subject_id": error.subject_id},
            ) from error

    return app
