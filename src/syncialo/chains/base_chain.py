"""Abstract Base Class for syncialo chains based on langchain"""

import abc

from langchain_core.runnables import Runnable
from langchain_core.language_models.llms import LLM

class BaseChain(abc.ABC):
    """Abstract Base Class for COT chain builders based on langchain"""

    @classmethod
    @abc.abstractmethod
    def build(cls, llm: LLM) -> Runnable:
        """Build chain

        Returns:
            Runnable: Chain
        """
        pass