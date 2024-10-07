"""Zeroshot classifier via Huggingface's inference API."""

import aiohttp
import os

from loguru import logger
from pydantic import BaseModel


_API_URL = os.getenv(
    'SYNCIALO_CLASSIFIER_URL',
    "https://api-inference.huggingface.co/models/MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
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


async def classify(
    sequences: str | list[str],
    labels: list[str],
    hypothesis_template: str | None = None,
) -> list[ClassificationResult | dict]:
    """Classify a text sequence with zero-shot classification."""

    if isinstance(sequences, str):
        sequences = [sequences]

    parameters = {"candidate_labels": labels}
    if hypothesis_template:
        parameters["hypothesis_template"] = hypothesis_template

    outputs = await _aquery({"inputs": sequences, "parameters": parameters})

    results: list[ClassificationResult | dict] = []
    for output in outputs:
        if "error" in output:
            results.append(output)
        else:
            try:
                results.append(ClassificationResult(**output))
            except Exception as e:
                logger.error(f"Error parsing classification result: {output}")
                logger.error(e)
                results.append(
                    {
                        "error": "error parsing classification result",
                        "unexpected_output": output,
                    }
                )

    return results
