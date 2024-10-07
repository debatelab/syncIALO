"""Equivalence tests"""

from loguru import logger
from langchain_core.documents.base import Document

from syncialo.chains.argumentation import ArgumentModel, Valence
from syncialo.chains.classifier import classify, ClassificationResult


TEXT_TEMPLATE_NLI = """
CLAIM_1: {claim_1}

CLAIM_2: {claim_2}
"""
LABELS_NLI = ["entails", "contradicts", "is neutral wrt."]
HYPOTHESIS_TEMPLATE_NLI = "CLAIM_1 {} CLAIM_2"


TEXT_TEMPLATE_DIALECTICS = """In a debate about "{topic}", it is maintained:

CLAIM: {claim}

REASON: {reason}
"""
LABELS_DIALECTICS = [
    "directly confirmed by",
    "directly disconfirmed by",
    "independent of",
]
HYPOTHESIS_TEMPLATE_DIALECTICS = "CLAIM is {} REASON."


async def are_dialectically_equivalent(
    arg: ArgumentModel,
    doc: Document,
    target_reason_claim: str,
    topic: str = None,
    valence: Valence = None,
) -> bool:
    try:
        checks = await classify(
            TEXT_TEMPLATE_DIALECTICS.format(
                claim=target_reason_claim, reason=doc.page_content, topic=topic
            ),
            LABELS_DIALECTICS,
            HYPOTHESIS_TEMPLATE_DIALECTICS,
        )
    except Exception as e:
        logger.error(f"Error from classifier: {e}")
        return False
    if not isinstance(checks, list) and not isinstance(checks[0], ClassificationResult):
        logger.error(f"Unexpected output from classifier: {checks}")
        return False
    result = checks[0]
    if result.labels[0] == LABELS_DIALECTICS[0] and valence == Valence.PRO:
        return True
    if result.labels[0] == LABELS_DIALECTICS[1] and valence == Valence.CON:
        return True
    return False


async def are_semantically_equivalent(
    arg: ArgumentModel, doc: Document, topic: str = None
) -> bool:
    try:
        checks = await classify(
            [
                TEXT_TEMPLATE_NLI.format(claim_1=arg.claim, claim_2=doc.page_content),
                TEXT_TEMPLATE_NLI.format(claim_1=doc.page_content, claim_2=arg.claim),
            ],
            LABELS_NLI,
            HYPOTHESIS_TEMPLATE_NLI,
        )
    except Exception as e:
        logger.error(f"Error from classifier: {e}")
        return False
    if not all(isinstance(check, ClassificationResult) for check in checks):
        logger.error(f"Unexpected output from classifier: {checks}")
        return False

    return all(check.labels[0] == LABELS_NLI[0] for check in checks)
