"""Debate Design Chains"""

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough, chain
from langchain_core.language_models.chat_models import BaseChatModel

from .base_chain_builder import BaseChainBuilder

_SYSTEM_PROMPT = (
    "You are a helpful assistant and an expert for critical thinking and argumentation theory. "
    "Moreover, you're an experienced debater, having won major debates and served as judge "
    "in numerous debating competitions.\n"
    "You read instructions carefully and follow them precisely. You give concise and clear answers."
)


class SuggestTopicsChain(BaseChainBuilder):

    _instruction_prompt = (
        "Task: Suggest interesting and controversial debating topics.\n"
        "Our group is planning a debate and we're searching for novel suitable topics. "
        "The debate is supposed to be explicitly related to the following issues:\n"
        "{taglist}\n"
        "Each topic you suggest should be tailored to these issues and specifically reflect "
        "at least two of the issues (but more is better).\n\n"
        "Can you please state {n} different debating topics which touch upon the above issues "
        "and from which we choose the most suitable ones? Be creative!\n"
        "No explanation is needed, just the topics, please."
    )
    _formatting_prompt = (
        'Please format your suggestions as follows:\n'
        '```json\n'
        '[\n'
        '    {{"idx": 1, "topic": "<Insert your first topic here.>"}},\n'
        '    {{"idx": 2, "topic": "<Insert your second topic here.>"}},\n'
        '    ...\n'
        ']\n'
        '```\n'
        'Just return the JSON code.\n'
    )

    stop_words = ["</reasoning>", "\n###"]

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        # step 1: brainstorm topics
        prompt_template_draft = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("user", cls._instruction_prompt)
        ])

        chain_draft = (
            {
                "taglist": itemgetter("tags") | RunnableLambda(lambda x: ' - '.join(x)),
                "n": itemgetter("debates_per_tag_cluster"),
            }
            | prompt_template_draft
            | llm.bind(max_tokens=512)
            | StrOutputParser()
        )

        # step 2: format topics
        prompt_template_format = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("user", "Can you please suggest some debating topics?"),
            ("assistant", "{suggestions}"),
            ("user", cls._formatting_prompt)
        ])

        compound_chain = (
            {"suggestions": chain_draft}
            | prompt_template_format
            | llm.bind(max_tokens=512, temperature=0)
            | SimpleJsonOutputParser()
        )

        return compound_chain


class SuggestMotionChain(BaseChainBuilder):

    _instruction_prompt = (
        "Task: Suggest a suitable motion for our debate.\n"
        "Our group is planning a debate about the topic:\n"
        "{topic}\n"
        "The overarching issues of the debates are: {taglist}\n"
        "Can you please state a precise and very concise motion, or central claim for our debate? Hints:\n"
        "- The motion should take a clear and unequivocal stance.\n"
        "- The motion expresses the view of the 'pro'-side in the debate.\n"
        "- The motion itself does not contain any reasoning or justification.\n"
        "- The motion appeals to persons concerned about the overarching issues listed above.\n"
        "- DON'T start with \"This house ...\".\n"
        "- DON'T start with \"This debate ...\".\n"
        "No explanation is needed, just the motion, please."
    )

    _reformulation_prompt = (
        "I've noticed that the claim starts with 'This house ...' / 'This debate ...'\n"
        "Can you please simplify and reformulate the claim by dropping this phrase?"
    )

    _formatting_prompt = (
        "Please format your suggestion as follows:\n"
        "```json\n"
        "{{\n"
        "    \"motion\": \"<Insert your motion here.>\"\n"
        "}}\n"
        "```\n"
        "Just return the JSON code."
    )

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        # step 1: suggest motion
        msgs = [
            ("system", _SYSTEM_PROMPT),
            ("user", cls._instruction_prompt)
        ]
        prompt_template_draft = ChatPromptTemplate.from_messages(msgs)
        chain_draft = (
            prompt_template_draft
            | llm.bind(max_tokens=128)
            | StrOutputParser()
        )

        # optional step 2: reformulate motion
        msgs = [
            *msgs,
            ("assistant", "{motion}"),
            ("user", cls._reformulation_prompt)
        ]
        prompt_template_revise = ChatPromptTemplate.from_messages(msgs)
        chain_revise = (
            prompt_template_revise
            | llm.bind(max_tokens=128)
            | StrOutputParser()
        )

        # helper chain: format
        prompt_template_format = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("user", "Can you please suggest a motion for the debate?"),
            ("assistant", "{motion}"),
            ("user", cls._formatting_prompt)
        ])
        chain_format = (
            prompt_template_format
            | llm.bind(max_tokens=128, temperature=0)
            | SimpleJsonOutputParser()
        )

        @chain
        def revise_if_necessary(input_: dict) -> Runnable:
            motion = input_["motion"]
            if motion.startswith("This house") or motion.startswith("This debate"):
                return chain_revise | chain_format
            else:
                return RunnablePassthrough() | itemgetter("motion")

        full_chain = (
            RunnablePassthrough()
            .assign(taglist=(itemgetter("tags") | RunnableLambda(lambda x: ' - '.join(x))))
            .assign(motion=(chain_draft | chain_format))
            | revise_if_necessary
        )

        return full_chain
