"""Validation for the public interaction contract."""

from __future__ import annotations

import math
from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

StrictIdentifier = StrictStr | StrictInt


class Interaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: StrictIdentifier
    item_id: StrictIdentifier
    value: StrictFloat | StrictInt = 1.0
    timestamp: datetime | None = None

    @field_validator("user_id", "item_id")
    @classmethod
    def non_empty_identifier(cls, value: str | int) -> str | int:
        if isinstance(value, str) and not value:
            raise ValueError("identifiers cannot be empty")
        return value

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float | int) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("value must be finite")
        return number


class HistoryInteraction(BaseModel):
    """Serving history omits user_id because the request represents one session."""

    model_config = ConfigDict(extra="forbid")

    item_id: StrictIdentifier
    value: StrictFloat | StrictInt = 1.0
    timestamp: datetime | None = None

    @field_validator("item_id")
    @classmethod
    def non_empty_identifier(cls, value: str | int) -> str | int:
        if isinstance(value, str) and not value:
            raise ValueError("item_id cannot be empty")
        return value

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float | int) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("value must be finite")
        return number
