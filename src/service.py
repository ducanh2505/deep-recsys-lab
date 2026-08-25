"""BentoML build and serving entrypoint."""

from __future__ import annotations

import bentoml
from pydantic import BaseModel

from deep_recsys_lab.serving.service import create_bento_service


class BentoArguments(BaseModel):
    model_tag: str


arguments = bentoml.use_arguments(BentoArguments)
RecommendationService = create_bento_service(arguments.model_tag)
