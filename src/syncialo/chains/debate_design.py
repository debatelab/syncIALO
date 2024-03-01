"""Debate Design Chains"""

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models.llms import LLM

from .base_chain import BaseChain

class SuggestTopicsChain(BaseChain):

    # TODO: fixme

    _prompt_template = """### User:
Task: Suggest interesting and controversial debating topics.

Our group is planning a debate and we're searching for novel suitable topics.

The debate is supposed to be explicitly related to the following issues:

{' - '.join(tags)}\n

Each topic you suggest should be tailored to these issues and specifically reflect at least two of the issues (but more is better).

Can you please state {debates_per_tag_cluster} different debating topics which touch upon the above issues and from which we choose the most suitable ones? Be creative!\n

### Assistant:

Yes, for sure. You may pick one of the following topics:\n
"""

    stop_words = ["</reasoning>", "\n###"]

    @classmethod
    def build(cls, llm: LLM) -> Runnable:

        prompt = PromptTemplate.from_template(cls._prompt_template)
        chain = (
            prompt
            | llm.bind(stop=cls.stop_words)
            | StrOutputParser()
        )
        return chain


class SuggestMotionChain(BaseChain):
    pass