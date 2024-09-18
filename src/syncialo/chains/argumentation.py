"""Argumentation Chains"""

from enum import Enum
from operator import itemgetter
import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough, chain
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

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
        "\n///\n"
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
        "\n///\n"
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
            .assign(proplist=(
                itemgetter("premises")
                | RunnableLambda(lambda x: '\n'.join([f"Label [P{e+1}]: {p}" for e, p in enumerate(x)]))
            ))
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
        "Task: Provide additional supporting arguments for a given claim\n"
        "Domain tags: {taglist}\n"
        "Read the following background information carefully before answering!\n"
        "/// background_information\n"
    ) + _ARGUMENT_BASIC_INFO + (
        "\n///\n"
        "Now, a participant has previously maintained in a debate that:\n\n"
        "{premiselist}\n\n"
        "Can you provide up to {n} different and independent arguments -- each "
        "consisting in a catchy title and a single concise statement, formatted "
        "as '**title:** statement' -- that back up the {nth} proposition? Make "
        "sure your arguments argue for proposition {target_label} in specific "
        "and plausible ways without merely repeating that proposition. Be "
        "inspired by the domain tags above.\n"
        "Just provide your supporting arguments below."
    )

    _formatting_prompt = (
        'Please format your suggestions as follows:\n'
    )

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        nth = ['first','second','third','fourth','fifth'][target_idx] if target_idx<5 else f"{target_idx}th"

    
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


class GenerateProAndConChain(BaseChainBuilder):
    """
    Define a compound chain that generates one pro and one con argument,
    adopting a given assistant persona.
    """

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        rank_by_plausibility = RankPropsByPlausibilityChain.build(llm)
        gen_supporting_argument = GenSupportingArgumentChain.build(llm)
        gen_attacking_argument = GenAttackingArgumentChain.build(llm)

        def sample_tags(input_: dict) -> list:
            tags_universal = input_["tags_universal"]
            tags_per_cluster = input_["tags_per_cluster"]
            return random.sample(tags_universal, k=tags_per_cluster)

        chain_generate_pro_and_con = (
            # 1. init and pre-processing
            RunnablePassthrough()
            .assign(
                # resample tags for more diversity
                tags_pro=RunnableLambda(sample_tags),
                tags_con=RunnableLambda(sample_tags)
            )
            # 2. rank by plausibility
            | RunnablePassthrough()
            .assign(ranking=rank_by_plausibility)
            # 3. generate one pro argument
            | RunnablePassthrough()
            .assign(new_pro=gen_supporting_argument)
            # TODO: add peer-review and revise step for con
            # 4. generate one con argument
            | RunnablePassthrough()
            .assign(new_con=gen_attacking_argument)
            # TODO: add peer-review and revise step for con
        )

        return chain_generate_pro_and_con


class SelectMostSalientChain(BaseChainBuilder):

    _prompt_select_salient = (
        "Task: Identify the most salient {k} {valence_text} arguments.\n"
        "Read the following background information carefully before answering!\n"
        "/// background_information\n"
    ) + _ARGUMENT_BASIC_INFO + (
        "\n///\n"
        "Now, the participants of a debate have previously brainstormed the following arguments:\n\n"
        "{argumentlist}\n\n"
        "which they've proposed as reasons {valence_text}:\n"
        "[[B]] {conclusion}\n"
        "Can you please the {k} most salient arguments of these? Please ensure that you "
        "identify diverse and mutually independent arguments. Format your selection of the {k} "
        "arguments as follows:\n"
        '```json\n'
        '[\n'
        '    {{"idx": 1, "label": "<Insert argument label here.>", "claim": "<Insert argument gist here.>"}},\n'
        '    {{"idx": 2, "label": "<Insert argument label here.>", "claim": "<Insert argument gist here.>"}},\n'
        '    ...\n'
        ']\n'
        '```\n'
        'Just return the JSON code.\n'
    )

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        # step 1: identify premises
        prompt_template_select_salient = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("user", cls._prompt_select_salient)
        ])

        def format_args(args: list) -> str:
            formatted_args = [
                f"  \{'label': {arg['label']}, 'claim': {arg['claim']}\},"
                for arg in args
            ]
            formatted_args = '\n'.join(formatted_args)
            formatted_args = f'[\n{formatted_args}\n]'
            return formatted_args

        chain_select_salient = (
            prompt_template_select_salient
            | llm.bind(max_tokens=512, temperature=0.1)
            | SimpleJsonOutputParser()
        )

        def postprocess_salient_args(input_:dict) -> list:
            salient_args = []
            for salient_arg in input_["salient_args"]:
                orig_arg = next((oa for oa in input_["args"] if oa["label"] == salient_arg["label"]), None)
                if orig_arg:
                    salient_args.append(orig_arg)
                else:
                    logger.warning(f"Salient argument not found: {salient_arg}. Ignoring.")
            while len(salient_args) < min(input_["k"], len(input_["args"])):
                logger.info("Adding random argument to fill up salient arguments.")
                salient_args.append(random.choice([oa for oa in input_["args"] if oa not in salient_args]))
            return salient_args

        compound_chain = (
            RunnablePassthrough()
            .assign(
                valence_text=(itemgetter("valence") | RunnableLambda(lambda x: str(x.value))),
                argumentlist=(itemgetter("args") | RunnableLambda(format_args))
            )
            .assign(salient_args=chain_select_salient)
            | RunnableLambda(postprocess_salient_args)
        )

        return compound_chain
