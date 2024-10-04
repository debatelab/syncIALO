"""Utility functions for the chains module"""

from typing import Any
import re
from json import JSONDecodeError

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.json import JsonOutputParser, parse_json_markdown
from langchain_core.outputs import Generation


class TolerantJsonOutputParser(JsonOutputParser):

    def _remove_trailing_comma(self, json_string):
        """
        Remove trailing comma after last array item from invalid JSON string.
        Corrects a common mistake in generated JSON arrays
        """
        # Pattern to match a comma followed by a closing square bracket at the end of the string
        pattern = r",\s*\]\s*$"
        # Use regex to replace the matched pattern with just the closing square bracket
        return re.sub(pattern, "]", json_string)

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.
                If True, the output will be a JSON object containing
                all the keys that have been returned so far.
                If False, the output will be the full JSON object.
                Default is False.

        Returns:
            The parsed JSON object.

        Raises:
            OutputParserException: If the output is not valid JSON.
        """
        text = result[0].text
        text = text.strip()
        text = self._remove_trailing_comma(text)
        if partial:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError:
                return None
        else:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError as e:
                msg = f"Invalid json output: {text}"
                raise OutputParserException(msg, llm_output=text) from e

