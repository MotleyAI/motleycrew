import code
import re
import sys
from io import StringIO
from typing import List, Optional

from pydantic import BaseModel, Field

from motleycrew.tools import MotleyTool


class PythonREPLTool(MotleyTool):
    """Python REPL tool. Use this to execute python commands.

    Note that the tool's output is the content printed to stdout by the executed code.
    Because of this, any data you want to be in the output should be printed using `print(...)`.
    """

    def __init__(
        self, return_direct: bool = False, exceptions_to_reflect: Optional[List[Exception]] = None
    ):
        self.console = code.InteractiveConsole()
        super().__init__(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. "
            "The output will be the content printed to stdout by the executed code. "
            "The state of the REPL is preserved between calls.",
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
            args_schema=REPLToolInput,
        )

    @staticmethod
    def sanitize_input(query: str) -> str:
        """Sanitize input to the python REPL.

        Remove whitespace, backtick & python
        (if llm mistakes python console as terminal)

        Args:
            query: The query to sanitize

        Returns:
            str: The sanitized query
        """
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)
        return query

    def run(self, command: str) -> str:
        # Sanitize the input
        cleaned_command = self.sanitize_input(command)

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            self.console.push(cleaned_command)
            sys.stdout = old_stdout
            return captured_output.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            return repr(e)


class REPLToolInput(BaseModel):
    """Input for the REPL tool."""

    command: str = Field(description="code to execute")
