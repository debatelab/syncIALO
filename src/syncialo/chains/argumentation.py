"""Argumentation Chains"""

from enum import Enum
from operator import itemgetter
import pydantic
import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from .base_chain_builder import BaseChainBuilder
from . import utils


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

_WRITING_ARGUMENTS_INFO = """
The following are tips for drafting and writing clear and effective arguments.

Make one point at a time:
This is a really key aspect of debating! If you make multiple points in one claim, you risk \
creating confusion and obscuring parts of your arguments. Examples are often better placed as \
their own pro claims beneath a claim, rather than integrated into it.

Keep claims short, simple and to the point:
Avoid introductory statements, restatements, and, in most cases “hedging language”. These are \
common ways that we communicate when talking out loud or writing a long-form piece – and \
absolutely have their uses – but are ill-suited to the bite-sized structure of a debate.

Keep claims directly relevant to their parent:
If a claim doesn't fit anywhere in the discussion yet, there are probably some other claims \
you'll need to create first! Claims should directly support or weaken their parent claims \
/ propositions.

Use research, evidence and facts to support your claims:
You can link directly to external sources when they back up the points you're making. But only \
do so when you're sure you know exactly what the source says and that it supports your point. \
If you're not sure, it's better to leave it out.

Use logic to support your claims:
Just throwing an idea out there is rarely persuasive! Even if you're dealing in an area where \
there's a lack of research and evidence available, you can still explain the logical links that \
lead you to draw the conclusions you're drawing. Meanwhile, keep an eye out for logical fallacies.
"""

# TODO: 🔊 Add systematic logging at info/debug levels to increase visibility


class Valence(Enum):
    PRO = "PRO"
    CON = "CON"


class ArgumentModel(pydantic.BaseModel):
    label: str
    claim: str
    target_idx: int
    valence: Valence


class IdentifyPremisesChain(BaseChainBuilder):

    # TODO: 👔 Add step that checks and discards any "global balancing" premises

    # Chat prompts

    _prompt_explicate_prems_msgs = [
            ("system", _SYSTEM_PROMPT),
            (
                "user",
                (
                    "Task: Identify premises of an argument.\n"
                    "Read the following background information carefully before answering!\n"
                    "/// background_information\n"
                ) + _ARGUMENT_BASIC_INFO + (
                    "\n///\n"
                    "Now, a participant has previously maintained in a debate that:\n"
                    "[[A]] {argument}\n"
                    "which they've advanced as a reason {valence_text}:\n"
                    "[[B]] {conclusion}\n"
                    "Can you please identify the major explicit and implicit premises (up to 5) "
                    "of the argument [[A]]? State each premise as a single, concise sentence. "
                    "Don't include any conclusions."
                )
            )
        ]

    _formatting_prompt_msgs = [
            ("system", _SYSTEM_PROMPT),
            ("user", "Can you please identify the premises of the previously discussed argument?"),
            ("assistant", "{premises}"),
            (
                "user",
                (
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
            )
        ]

    # Preprocessing methods

    pass

    # Postprocessing methods

    @staticmethod
    def postprocess_premises(input_: list) -> list:
        return [record["premise"] for record in input_ if "premise" in record]

    # Chain builder

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        chain_explicate_prems = (
            RunnablePassthrough().assign(
                valence_text=(itemgetter("valence") | RunnableLambda(lambda x: str(x.value)))
            )
            | ChatPromptTemplate.from_messages(cls._prompt_explicate_prems_msgs)
            | llm.bind(max_tokens=512, temperature=0.3)
            | StrOutputParser()
        )

        chain_format = (
            ChatPromptTemplate.from_messages(cls._formatting_prompt_msgs)
            | llm.bind(max_tokens=512, temperature=0)
            | utils.TolerantJsonOutputParser()
            | RunnableLambda(cls.postprocess_premises)
        )

        main_chain = (
            {"premises": chain_explicate_prems}
            | chain_format
        )

        return main_chain


class RankPropsByPlausibilityChain(BaseChainBuilder):

    # Chat prompts

    _assess_prompt_msgs = [
            ("system", _SYSTEM_PROMPT_PERSONA),
            (
                "user",
                (
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
            )
        ]

    _rank_prompt_msgs = [
            ("system", _SYSTEM_PROMPT),
            ("user", "Can you please assess the plausibility of the following propositions?\n {proplist}"),
            ("assistant", "{assessment}"),
            (
                "user",
                (
                    "Thanks for this! Now, please use your assessment to compare and order the propositions "
                    "in terms of plausibility. Rigourously format your plausibility ranking, beginning with the most "
                    "plausible proposition, as follows:\n"
                    '```json\n'
                    '[\n'
                    '    {{"proposition": "<Insert most plausible proposition here.>", '
                    '"prop_label": "<Label of this proposition>"}},\n'
                    '    {{"proposition": "<Insert second most plausible proposition here.>", '
                    '"prop_label": "<Label of this proposition>"}},\n'
                    '    ...\n'
                    ']\n'
                    '```\n'
                    'No comments or explanations. Just return the valid JSON code.\n'
                )
            )
        ]

    # Preprocessing methods

    @staticmethod
    def format_premises(premises: list) -> str:
        formatted_premises = '\n'.join(
            [
                f"  (P{e+1}) {premise}"
                for e, premise in enumerate(premises)
            ]
        )
        return formatted_premises

    # Postprocessing methods

    @staticmethod
    def postprocess_ranking(ranking: list) -> list:
        labels = [record["prop_label"].strip("[]P ") for record in ranking if "prop_label" in record]
        int_ranking = []
        try:
            for label in labels:
                if (int(label)-1) not in int_ranking:
                    int_ranking.append(int(label)-1)
        except ValueError:
            return list(range(len(ranking)))

        return int_ranking

    # Chain builder

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        chain_assess_prems = (
            ChatPromptTemplate.from_messages(cls._assess_prompt_msgs)
            | llm.bind(max_tokens=512, temperature=0.3)
            | StrOutputParser()
        )

        chain_rank = (
            ChatPromptTemplate.from_messages(cls._rank_prompt_msgs)
            | llm.bind(max_tokens=512, temperature=0)
            | utils.TolerantJsonOutputParser()
            | RunnableLambda(cls.postprocess_ranking)
        )

        main_chain = (
            RunnablePassthrough().assign(
                taglist=(itemgetter("tags") | RunnableLambda(lambda x: ' - '.join(x))),
                proplist=(itemgetter("premises") | RunnableLambda(cls.format_premises))
            )
            | RunnablePassthrough().assign(assessment=chain_assess_prems)
            | chain_rank
        )

        return main_chain


class AbstractGenArgumentChain(BaseChainBuilder):

    # Preprocessing methods

    @staticmethod
    def format_nth(target_idx: int) -> str:
        if target_idx < 5:
            return ['first', 'second', 'third', 'fourth', 'fifth'][target_idx]
        return f"{target_idx+1}th"

    @staticmethod
    def format_premises(premises: list) -> str:
        formatted_premises = [
            f"  (P{e+1}) {premise}"
            for e, premise in enumerate(premises)
        ]
        formatted_premises = '\n'.join(formatted_premises)
        return formatted_premises

    @staticmethod
    def format_target_label(target_idx: int) -> str:
        return f"(P{target_idx+1})"

    # Postprocessing methods

    @staticmethod
    def parse_json_arguments(input_: list[dict]) -> list[ArgumentModel]:
        arguments = []
        for record in input_["json"]:
            try:
                data = {k: v for k, v in record.items() if k in ArgumentModel.model_fields}
                data = {**data, "target_idx": input_["target_idx"], "valence": input_["valence"]}
                arguments.append(ArgumentModel(**data))
            except Exception as e:
                logger.error(f"Error parsing argument: {record}. {e}")
        return arguments

    # Routers

    pass


class GenSupportingArgumentChain(AbstractGenArgumentChain):
    """
    Input:
    - premises: list of premises, one of which is the target proposition
    - ranking: ranking of the premises by plausibility
    - tags_pro: list of keywords to steer pro argument generation
    - n: number of arguments to generate
    - persona: persona of the assistant
    """

    # Chat prompts

    _instruction_prompt_msgs = [
        ("system", _SYSTEM_PROMPT_PERSONA),
        ("user",
            (
                "Task: Provide additional supporting arguments for a given claim\n"
                "Domain tags: {taglist}\n"
                "Read the following background information carefully before answering!\n"
                "/// background_information\n"
            ) + _ARGUMENT_BASIC_INFO + "\n\n" + _WRITING_ARGUMENTS_INFO + (
                "\n///\n"
                "Now, a participant has previously maintained in a debate that:\n\n"
                "{premiselist}\n\n"
                "Can you provide up to {n} different and independent PRO arguments -- each "
                "consisting in a catchy name and a single concise statement, formatted "
                "as '**name:** statement' -- that back up the {nth} proposition? Make "
                "sure your arguments argue for proposition {target_label} in specific "
                "and plausible ways without merely repeating that proposition. Keep your "
                "arguments short and direct. Be inspired by the domain tags above.\n"
                "Just provide your supporting arguments below."
            )
         )
    ]

    _formatting_prompt_msgs = [
            ("system", _SYSTEM_PROMPT),
            ("user", "Please provide up to {n} different and independent arguments - each "
                "consisting in a catchy title and a single concise statement."),
            ("assistant", "{drafts}"),
            ("user", "Good. Now, please format these arguments as follows:\n"
                '```json\n'
                '[\n'
                '    {{"label": "<Insert title of first argument here>", '
                '"claim": "<Insert first argument here>"}},\n'
                '    {{"label": "<Insert title of first argument here>", '
                '"claim": "<Insert second argument here>"}},\n'
                '    ...\n'
                ']\n'
                '```\n'
                'Just return the JSON code.\n'
             )
    ]

    # Preprocessing methods

    @staticmethod
    def set_target_idx(ranking: list[int]) -> int:
        # TODO: ❓  Discuss whether it makes sense to target but the most plausible premise
        return ranking[0]

    # Postprocessing methods

    pass

    # Routers

    pass

    # Chain builder

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        subchain_draft = (
            ChatPromptTemplate.from_messages(cls._instruction_prompt_msgs)
            | llm.bind(max_tokens=1024, temperature=0.7)
            | StrOutputParser()
        )

        subchain_format = (
            ChatPromptTemplate.from_messages(cls._formatting_prompt_msgs)
            | llm.bind(max_tokens=1024, temperature=0)
            | utils.TolerantJsonOutputParser()
        )

        main_chain = (
            RunnablePassthrough().assign(
                target_idx=(itemgetter("ranking") | RunnableLambda(cls.set_target_idx))
            )
            | RunnablePassthrough().assign(
                valence=RunnableLambda(lambda x: Valence.PRO),
                taglist=(itemgetter("tags_pro") | RunnableLambda(lambda x: ' - '.join(x))),
                premiselist=(itemgetter("premises") | RunnableLambda(cls.format_premises)),
                nth=(itemgetter("target_idx") | RunnableLambda(cls.format_nth)),
                target_label=(itemgetter("target_idx") | RunnableLambda(cls.format_target_label))
            )
            | RunnablePassthrough().assign(
                drafts=subchain_draft
            )
            | RunnablePassthrough().assign(
                json=subchain_format
            )
            | RunnableLambda(cls.parse_json_arguments)
        )

        return main_chain


class GenAttackingArgumentChain(AbstractGenArgumentChain):
    """
    Input:
    - premises: list of premises, one of which is the target proposition
    - ranking: ranking of the premises by plausibility
    - tags_con: list of keywords to steer pro argument generation
    - n: number of arguments to generate
    - persona: persona of the assistant
    """

    # Chat prompts

    _instruction_prompt_msgs = [
        ("system", _SYSTEM_PROMPT_PERSONA),
        ("user",
            (
                "Task: Provide objections against a given claim\n"
                "Domain tags: {taglist}\n"
                "Read the following background information carefully before answering!\n"
                "/// background_information\n"
            ) + _ARGUMENT_BASIC_INFO + "\n\n" + _WRITING_ARGUMENTS_INFO + (
                "\n///\n"
                "Now, back to your task. An opponent in a debate has previously maintained that:\n\n"
                "{premiselist}\n\n"
                "Can you provide up to {n} different and independent CON arguments -- each "
                "consisting in a catchy name and a single concise statement, formatted "
                "as '**name:** statement' -- that object to the {nth} proposition? Make "
                "sure each CON argument demonstrates in a specific and plausible way why "
                "proposition {target_label} is false. Just denying the proposition won't do. "
                "However, keep your arguments short and direct (single sentenve). Be inspired "
                "by the domain tags above.\n"
                "Just provide your CON arguments below (no explanations needed)."
            )
         )
    ]

    _formatting_prompt_msgs = [
            ("system", _SYSTEM_PROMPT),
            ("user", "Please provide up to {n} different and independent arguments - each "
                "consisting in a catchy title and a single concise statement."),
            ("assistant", "{drafts}"),
            ("user", "Good. Now, please format these arguments as follows:\n"
                '```json\n'
                '[\n'
                '    {{"label": "<Insert title of first argument here>", '
                '"claim": "<Insert first argument here>"}},\n'
                '    {{"label": "<Insert title of first argument here>", '
                '"claim": "<Insert second argument here>"}},\n'
                '    ...\n'
                ']\n'
                '```\n'
                'Just return the JSON code.\n'
             )
    ]

    # Preprocessing methods

    @staticmethod
    def set_target_idx(ranking: list[int]) -> int:
        # TODO: ❓ Discuss whether it makes sense to target but the most implausible premise
        return ranking[-1]

    # Postprocessing methods

    pass

    # Routers

    pass

    # Chain builder

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        subchain_draft = (
            ChatPromptTemplate.from_messages(cls._instruction_prompt_msgs)
            | llm.bind(max_tokens=512, temperature=0.7)
            | StrOutputParser()
        )

        subchain_format = (
            ChatPromptTemplate.from_messages(cls._formatting_prompt_msgs)
            | llm.bind(max_tokens=512, temperature=0)
            | utils.TolerantJsonOutputParser()
        )

        main_chain = (
            RunnablePassthrough().assign(
                target_idx=(itemgetter("ranking") | RunnableLambda(cls.set_target_idx))
            )
            | RunnablePassthrough().assign(
                valence=RunnableLambda(lambda x: Valence.PRO),
                taglist=(itemgetter("tags_con") | RunnableLambda(lambda x: ' - '.join(x))),
                premiselist=(itemgetter("premises") | RunnableLambda(cls.format_premises)),
                nth=(itemgetter("target_idx") | RunnableLambda(cls.format_nth)),
                target_label=(itemgetter("target_idx") | RunnableLambda(cls.format_target_label))
            )
            | RunnablePassthrough().assign(
                drafts=subchain_draft
            )
            | RunnablePassthrough().assign(
                json=subchain_format
            )
            | RunnableLambda(cls.parse_json_arguments)
        )

        return main_chain


class GenerateProAndConChain(BaseChainBuilder):
    """
    Define a compound chain that generates n pro and n con arguments,
    adopting a given assistant persona.

    Input:
    - premises: list of premises, one of which is the target proposition
    - persona: persona of the assistant
    - tags: domain tags of the debate
    - tags_universal: universal tags for the assistant persona
    - tags_per_cluster: number of tags to sample per cluster
    - n: number of arguments to generate per valence
    """

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        rank_by_plausibility = RankPropsByPlausibilityChain.build(llm)
        gen_supporting_argument = GenSupportingArgumentChain.build(llm)
        gen_attacking_argument = GenAttackingArgumentChain.build(llm)

        # preprocessing methods

        def reformat_persona(persona: str) -> str:
            return persona[0].lower() + persona[1:]

        def sample_tags(input_: dict) -> list:
            tags_universal = input_["tags_universal"]
            tags_per_cluster = input_["tags_per_cluster"]
            return random.sample(tags_universal, k=tags_per_cluster)

        chain_generate_pro_and_con = (
            RunnablePassthrough().assign(
                persona=(itemgetter("persona") | RunnableLambda(reformat_persona)),
                # resample tags for more diversity
                tags_pro=RunnableLambda(sample_tags),
                tags_con=RunnableLambda(sample_tags)
            )
            | RunnablePassthrough().assign(
                ranking=rank_by_plausibility
            )
            | {
                "new_pros": gen_supporting_argument,
                "new_cons": gen_attacking_argument
            }
        )
        # TODO: add peer-review and revise step for gemerated pro and con

        return chain_generate_pro_and_con


class SelectMostSalientChain(BaseChainBuilder):
    """
    Inputs:
    - args: list of arguments (list[ArgumentModel])
    - k: number of salient arguments to select
    - conclusion: target/parent argument
    - valence: valence of the arguments relative to conclusion
    """

    # Chat prompts

    _prompt_select_salient_msgs = [
            ("system", _SYSTEM_PROMPT),
            ("user", (
                    "Task: Identify the {k} most salient {valence_text} arguments.\n"
                    "Read the following background information carefully before answering!\n"
                    "/// background_information\n"
                ) + _ARGUMENT_BASIC_INFO + (
                    "\n///\n"
                    "Now, the participants of a debate have previously brainstormed the following arguments:\n\n"
                    "{argumentlist}\n\n"
                    "which they've proposed as reasons {valence_text}:\n"
                    "[[B]] {conclusion}\n"
                    "Can you please select the {k} most salient arguments of these? Please ensure that you "
                    "identify diverse and mutually independent arguments. Format your selection of the {k} "
                    "arguments as follows:\n"
                    '```json\n'
                    '[\n'
                    '    {{"idx": 1, "label": "<Insert argument label here.>", '
                    '"claim": "<Insert argument gist here.>"}},\n'
                    '    {{"idx": 2, "label": "<Insert argument label here.>", '
                    '"claim": "<Insert argument gist here.>"}},\n'
                    '    ...\n'
                    ']\n'
                    '```\n'
                    'Just return the JSON code.\n'
                )
             )
        ]

    # Preprocessing methods

    @staticmethod
    def format_args(args: list[ArgumentModel]) -> str:
        formatted_args = '\n'.join(
            [
                f"  {{'label': {arg.label}, 'claim': {arg.claim} }},"
                for arg in args
            ]
        )
        formatted_args = f'[\n{formatted_args}\n]'
        return formatted_args

    # Postprocessing methods

    @staticmethod
    def postprocess_salient_args(input_: dict) -> list[ArgumentModel]:
        oargs: list[ArgumentModel] = input_["args"]
        salient_args: list[ArgumentModel] = []
        for salient_arg in input_["salient_args"]:
            orig_arg = next((oa for oa in oargs if oa.label == salient_arg.get("label")), None)
            if orig_arg:
                salient_args.append(orig_arg.model_copy())
            else:
                logger.warning(f"Salient argument not found: {salient_arg}. Ignoring.")
        while len(salient_args) < min(input_["k"], len(oargs)):
            logger.info("Adding random argument to fill up salient arguments.")
            remainder = [oa for oa in oargs if oa not in salient_args]
            if not remainder:
                break
            salient_args.append(random.choice(remainder).model_copy())
        return salient_args

    # Routers

    pass

    # Chain builder

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        chain_select_salient = (
            ChatPromptTemplate.from_messages(cls._prompt_select_salient_msgs)
            | llm.bind(max_tokens=512, temperature=0.1)
            | utils.TolerantJsonOutputParser()
        )

        main_chain = (
            RunnablePassthrough().assign(
                valence_text=(itemgetter("valence") | RunnableLambda(lambda x: str(x.value))),
                argumentlist=(itemgetter("args") | RunnableLambda(cls.format_args))
            )
            | RunnablePassthrough().assign(
                salient_args=chain_select_salient
            )
            | RunnableLambda(cls.postprocess_salient_args)
        )

        return main_chain
