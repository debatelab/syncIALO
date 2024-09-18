"""Argumentation Chains"""

from enum import Enum
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough, chain
from langchain_core.language_models.chat_models import BaseChatModel

from .base_chain_builder import BaseChainBuilder

_SYSTEM_PROMPT = (
    "You are a helpful assistant and an expert for critical thinking and argumentation theory. "
    "Moreover, you're an experienced debater, having won major debates and served as judge "
    "in numerous debating competitions. When not serving as an assistant and advisor, you're "
    "teaching formal logic and critical thinking in Stanford's philosophy programs.\n"
    "You read instructions carefully and follow them precisely. You give concise and clear answers."
)


class Valences(Enum):
    PRO = "PRO"
    CON = "CON"


class IdentifyPremisesChain(BaseChainBuilder):

    _instruction_prompt = (
        "Task: ...\n"
    )
    _formatting_prompt = (
        'Please format your suggestions as follows:\n'
    )

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        pass


class RankPropsByPlausibilityChain(BaseChainBuilder):

    _instruction_prompt = (
        "Task: ...\n"
    )
    _formatting_prompt = (
        'Please format your suggestions as follows:\n'
    )

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        pass


class GenSupportingArgumentChain(BaseChainBuilder):

    _instruction_prompt = (
        "Task: ...\n"
    )
    _formatting_prompt = (
        'Please format your suggestions as follows:\n'
    )

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        pass


class GenAttackingArgumentChain(BaseChainBuilder):

    _instruction_prompt = (
        "Task: ...\n"
    )
    _formatting_prompt = (
        'Please format your suggestions as follows:\n'
    )

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        pass
