from typing import List, Type

from langchain_core.tools import Tool
from pydantic import BaseModel, Field

try:
    from pglast import parse_sql, prettify
    from pglast.parser import ParseError
except ImportError:
    parse_sql = None
    prettify = None
    ParseError = None

from motleycrew.common.utils import ensure_module_is_installed
from motleycrew.tools import MotleyTool


class PostgreSQLLinterTool(MotleyTool):
    """PostgreSQL code verification tool."""

    def __init__(
        self,
        return_direct: bool = False,
        handle_exceptions: bool | List[Type[Exception]] = False,
    ):
        ensure_module_is_installed("pglast")

        exceptions_to_reflect = exceptions_to_reflect or [ParseError]

        langchain_tool = create_pgsql_linter_tool()
        super().__init__(
            tool=langchain_tool,
            return_direct=return_direct,
            handle_exceptions=handle_exceptions,
        )


class PostgreSQLLinterInput(BaseModel):
    """Input for the PostgreSQLLinterTool."""

    query: str = Field(description="SQL code for validation")


def create_pgsql_linter_tool() -> Tool:
    def parse_func(query: str) -> str:
        parse_sql(query)
        return prettify(query)

    return Tool.from_function(
        func=parse_func,
        name="postgresql_linter",
        description="Tool for validating PostgreSQL code",
        args_schema=PostgreSQLLinterInput,
    )
