import functools
import logging
import re
import sys
from io import StringIO
from typing import Dict, List, Type

from pydantic import BaseModel, Field

from motleycrew.tools import MotleyTool

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def warn_once() -> None:
    """Warn once about the dangers of PythonREPL with untrusted LLMs."""
    logger.warning(
        "Python REPL can execute arbitrary code from LLMs. "
        "Only use with trusted models and in secure environments. "
        "Consider using sandboxing or code review for production systems."
    )


class MissingPrintStatementError(Exception):
    """Exception raised when a print statement is missing from the command."""

    def __init__(self, command: str):
        self.command = command
        super().__init__(
            f"Command must contain at least one print statement. Remember to print the results you want to see using print(...)."
        )


class PythonREPLTool(MotleyTool):
    """Python REPL tool. Use this to execute python commands.

    Note that the tool's output is the content printed to stdout by the executed code.
    Because of this, any data you want to be in the output should be printed using `print(...)`.
    """

    def __init__(
        self, return_direct: bool = False, handle_exceptions: bool | List[Type[Exception]] = False
    ):
        # Warn about security risks with untrusted LLMs
        warn_once()

        self.namespace: Dict = {}

        if not handle_exceptions:
            handle_exceptions = [MissingPrintStatementError]
        elif MissingPrintStatementError not in handle_exceptions:
            handle_exceptions.append(MissingPrintStatementError)

        super().__init__(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. "
            "MAKE SURE TO PRINT OUT THE RESULTS YOU CARE ABOUT USING `print(...)`. ",
            return_direct=return_direct,
            handle_exceptions=handle_exceptions,
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
        self.validate_input(cleaned_command)

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Compile and execute the command to properly catch exceptions
            compiled_code = compile(cleaned_command, "<string>", "exec")
            exec(compiled_code, self.namespace)
            sys.stdout = old_stdout
            return captured_output.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            return repr(e)

    def validate_input(self, command: str):
        if "print(" not in command:
            raise MissingPrintStatementError(command)


class REPLToolInput(BaseModel):
    """Input for the REPL tool."""

    command: str = Field(description="code to execute")
