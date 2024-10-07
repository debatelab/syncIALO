"""Zeroshot classifier via Huggingface's inference API."""

import aiohttp
import os

from loguru import logger
from pydantic import BaseModel
import tenacity


_API_URL = os.getenv(
    "SYNCIALO_CLASSIFIER_URL",
    "https://api-inference.huggingface.co/models/MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
)
_HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}


class ClassificationResult(BaseModel):
    sequence: str
    labels: list[str]
    scores: list[float]


async def _aquery(payload):
    async with aiohttp.ClientSession() as session:
        async with session.post(_API_URL, headers=_HEADERS, json=payload) as response:
            return await response.json()


@tenacity.retry(wait=tenacity.wait_random_exponential(multiplier=1, max=60))
async def classify(
    sequences: str | list[str],
    labels: list[str],
    hypothesis_template: str | None = None,
) -> list[ClassificationResult]:
    """Classify a text sequence with zero-shot classification."""

    if isinstance(sequences, str):
        sequences = [sequences]

    parameters = {"candidate_labels": labels}
    if hypothesis_template:
        parameters["hypothesis_template"] = hypothesis_template

    outputs = await _aquery({"inputs": sequences, "parameters": parameters})

    results: list[ClassificationResult] = []
    if "error" in outputs:
        msg = f"Error from classifier: {outputs['error']}"
        logger.warning(msg)
        raise Exception(msg)
    for output in outputs:
        results.append(ClassificationResult(**output))

    return results
