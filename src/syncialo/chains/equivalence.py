"""Equivalence tests"""

from loguru import logger
import networkx as nx
from langchain_core.documents.base import Document

from syncialo.chains.argumentation import ArgumentModel, Valence
from syncialo.chains.classifier import classify, ClassificationResult


TEXT_TEMPLATE_NLI = """
CLAIM_1: {claim_1}

CLAIM_2: {claim_2}
"""
LABELS_NLI = ["entails", "contradicts"]
HYPOTHESIS_TEMPLATE_NLI = "CLAIM_1 {} CLAIM_2"


TEXT_TEMPLATE_DIALECTICS = """In a debate about "{topic}", it is maintained:

CLAIM: {claim}

REASON: {reason}
"""
LABELS_DIALECTICS = ["directly confirmed by", "directly disconfirmed by", "independent of"]
HYPOTHESIS_TEMPLATE_DIALECTICS = "CLAIM is {} REASON."


def are_dialectically_equivalent(arg: ArgumentModel, doc: Document, target_reason_claim: str, topic: str = None, valence: Valence = None) -> bool:
    check = classify(
        TEXT_TEMPLATE_DIALECTICS.format(claim=target_reason_claim, reason=doc.page_content, topic=topic),
        LABELS_DIALECTICS,
        HYPOTHESIS_TEMPLATE_DIALECTICS
    )
    if not isinstance(check, ClassificationResult):
        logger.error(f"Unexpected output from classifier: {check}")
        return False
    if check.labels[0] == "directly confirmed by" and valence == Valence.PRO:
        return True
    if check.labels[0] == "directly disconfirmed by" and valence == Valence.CON:
        return True
    return False


def are_semantically_equivalent(arg: ArgumentModel, doc: Document, topic: str = None) -> bool:
    checks = classify(
        [
            TEXT_TEMPLATE_NLI.format(claim_1=arg.claim, claim_2=doc.page_content),
            TEXT_TEMPLATE_NLI.format(claim_1=doc.page_content, claim_2=arg.claim),
        ],
        LABELS_NLI,
        HYPOTHESIS_TEMPLATE_NLI
    )
    if not all(isinstance(check, ClassificationResult) for check in checks):
        logger.error(f"Unexpected output from classifier: {checks}")
        return False

    return all(check.labels[0] == "entails" for check in checks)
