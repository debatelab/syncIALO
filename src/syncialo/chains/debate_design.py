"""Debate Design Chains"""

import pydantic

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel

from .base_chain import BaseChain


class TopicsList(pydantic.BaseModel):
    topics: list[str]


class SuggestTopicsChain(BaseChain):

    # TODO: fixme

    _instruction_prompt = """Task: Suggest interesting and controversial debating topics.

Our group is planning a debate and we're searching for novel suitable topics.

The debate is supposed to be explicitly related to the following issues:

{' - '.join(tags)}\n

Each topic you suggest should be tailored to these issues and specifically reflect at least two of the issues (but more is better).

Can you please state {debates_per_tag_cluster} different debating topics which touch upon the above issues and from which we choose the most suitable ones? Be creative!\n
"""

    _formatting_prompt = """Please format your suggestions as follows:
```json
"topics": [
    "<Insert your first topic here.>"},
    "<Insert your second topic here.>"},
    ...
]
```
Just return the JSON code.
"""

    stop_words = ["</reasoning>", "\n###"]

    @classmethod
    def build(cls, llm: BaseChatModel) -> Runnable:

        prompt_template_1 = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", cls._instruction_prompt)
        ])

        chain = (
            prompt_template_1
            | llm.bind(stop=cls.stop_words)
            | StrOutputParser()
        )

        prompt_template_2 = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "Can you please suggest some debating topics?")
            ("assistant", "{suggestions}")
            ("user", cls._formatting_prompt)
        ])

        structured_llm = llm.with_structured_output(TopicsList)

        compound_chain = (
            {"suggestions": chain}
            | prompt_template_2
            | structured_llm
        )

        return compound_chain


class SuggestMotionChain(BaseChain):
    pass
