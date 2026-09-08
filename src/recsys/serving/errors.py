"""Structured errors shared by engine and transport adapters."""

from __future__ import annotations


class RecommendationError(Exception):
    code = "recommendation_error"
    status_code = 422

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnknownIdentifierError(RecommendationError):
    code = "unknown_identifier"


class UnsupportedQueryModeError(RecommendationError):
    code = "unsupported_query_mode"


class CatalogExhaustedError(RecommendationError):
    code = "catalog_exhausted"
    status_code = 409
