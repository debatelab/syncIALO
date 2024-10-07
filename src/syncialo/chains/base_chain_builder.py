"""Abstract Base Class for syncialo chains based on langchain"""

import abc

from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel


class BaseChainBuilder(abc.ABC):
    """Abstract Base Class for chain builders based on langchain"""

    @classmethod
    @abc.abstractmethod
    def build(cls, llm: BaseChatModel, **kwargs) -> Runnable:
        """Build chain

        Returns:
            Runnable: Chain
        """
        pass
