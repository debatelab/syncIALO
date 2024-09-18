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

_SYSTEM_PROMPT_PERSONA = (
    "You are: {persona}.\n"
    "You have been chosen to assist a collaborative debating project as an external expert, "
    "given you outstanding critical thinking and argumentation skills.\n"
    "You read instructions carefully and follow them precisely. You give concise and clear answers."
)

_ARGUMENT_BASIC_INFO = """A crucial part of critical thinking is to identify, construct, and \
evaluate arguments.

In everyday life, people often use "argument" to mean a quarrel between serveral persons. But \
in logic and critical thinking, an argument is a list of statements, one of which is the \
conclusion while the others represent the premises or assumptions of the argument.

To give an argument is to provide a set of premises as reasons for accepting the conclusion. \
So, to give an argument does not necessarily mean to attack or criticize someone. Arguments \
can also be used to support other person's viewpoints.

Here is an example of an argument:

> If you want to find a good job, you should work hard. You do want to find a good job. So you \
    should work hard.

The first two sentences here are the premises of the argument, and the last sentence is the \
conclusion. To give this argument is to offer the premises as reasons for accepting the \
conclusion."""


class Valences(Enum):
    PRO = "PRO"
    CON = "CON"


class IdentifyPremisesChain(BaseChainBuilder):

    _prompt_explicate_prems = (
        "Task: Identify premises of an argument.\n"
        "Read the following background information carefully before answering!\n"
        "/// background_information\n"
    ) + _ARGUMENT_BASIC_INFO + (
        "///\n"
        "Now, a participant has previously maintained in a debate that:\n"
        "[[A]] {argument}\n"
        "which they've advanced as a reason {valence_text}:\n"
        "[[B]] {conclusion}\n"
        "Can you please identify the premises (up to 5) "
        "of the argument [[A]]? State each premise as a single, concise sentence."
    )

    _formatting_prompt = (
        'Please format the concise premises you\'ve identified as follows:\n'
        '```json\n'
        '[\n'
        '    {{"idx": 1, "premise": "<Insert first premise here.>"}},\n'
        '    {{"idx": 2, "premise": "<Insert second premise here.>"}},\n'
        '    ...\n'
        ']\n'
        '```\n'
        'Just return the JSON code.\n'
    )

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        # step 1: identify premises
        prompt_template_explicate_prems = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("user", cls._prompt_explicate_prems)
        ])

        chain_explicate_prems = (
            RunnablePassthrough().assign(
                valence_text=(itemgetter("valence") | RunnableLambda(lambda x: str(x.value)))
            )
            | prompt_template_explicate_prems
            | llm.bind(max_tokens=512)
            | StrOutputParser()
        )

        # step 2: format premises
        prompt_template_format = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("user", "Can you please identify the premises of the previously discussed argument?"),
            ("assistant", "{premises}"),
            ("user", cls._formatting_prompt)
        ])

        compound_chain = (
            {"premises": chain_explicate_prems}
            | prompt_template_format
            | llm.bind(max_tokens=512, temperature=0)
            | SimpleJsonOutputParser()
            | RunnableLambda(lambda x: [record["premise"] for record in x if "premise" in record])
        )

        return compound_chain


class RankPropsByPlausibilityChain(BaseChainBuilder):

    _assess_prompt = (
        "Task: Rank the premises in an argument according to plausibility\n"
        "Domain: {taglist}\n"
        "Read the following background information carefully before answering!\n"
        "/// background_information\n"
        ) + _ARGUMENT_BASIC_INFO + (
        "///\n"
        "Now, an opponent has previously maintained in a debate that:\n\n"
        "{proplist}\n\n"
        "How plausible and convincing are these propositions? More specifically: "
        "Provide, for each proposition, a brief plausibility assessment (in a single "
        "sentence) and rate its plausibility on a qualitative scale from highly-plausible to "
        "most-implausible."
    )

    _rank_prompt = (
        "Thanks for this! Now, please use your assessment to compare and order the propositions "
        "in terms of plausibility. Please format your plausibility ranking, beginning with the most "
        "plausible proposition, as follows:\n"
        '```json\n'
        '[\n'
        '    {{"proposition": "<Insert most plausible proposition here.>", '
        '"prop_label": <Label of this proposition>}},\n'
        '    {{"proposition": "<Insert second most plausible proposition here.>", '
        '"prop_label": <Label of this proposition>}},\n'
        '    ...\n'
        ']\n'
        '```\n'
        'Just return the JSON code.\n'
    )

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        # step 1: assess premises
        prompt_template_assess_prems = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT_PERSONA),
            ("user", cls._assess_prompt)
        ])

        chain_assess_prems = (
            prompt_template_assess_prems
            | llm.bind(max_tokens=512)
            | StrOutputParser()
        )

        # step 2: rank premises
        prompt_template_rank = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("user", "Can you please assess the plausibility of the following propositions?\n {proplist}"),
            ("assistant", "{assessment}"),
            ("user", cls._rank_prompt)
        ])

        # step 3: postprocess
        def postprocess_ranking(ranking: list) -> list:
            labels = [record["prop_label"].strip("[]P ") for record in ranking if "prop_label" in record]
            try:
                labels = [int(label) for label in labels]
            except ValueError:
                return list(range(len(ranking)))
            return [label-1 for label in labels]

        compound_chain = (
            RunnablePassthrough()
            .assign(taglist=(itemgetter("tags") | RunnableLambda(lambda x: ' - '.join(x))))
            .assign(proplist=(itemgetter("premises") | RunnableLambda(lambda x: '\n'.join([f"Label [P{e+1}]: {p}" for e,p in enumerate(x)]))))
            | RunnablePassthrough().assign(assessment=chain_assess_prems)
            | prompt_template_rank
            | llm.bind(max_tokens=512, temperature=0)
            | SimpleJsonOutputParser()
            | RunnableLambda(postprocess_ranking)
        )

        return compound_chain


class GenSupportingArgumentChain(BaseChainBuilder):

    # TODO: generate label and gist!

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
