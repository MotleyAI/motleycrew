from typing import Callable, List, Optional, Type, TypeVar

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from motleycrew.agents.langchain import ReActToolCallingMotleyAgent
from motleycrew.tools.structured_passthrough import StructuredPassthroughTool

T = TypeVar("T", bound=BaseModel)


def structured_output_with_retries(
    schema: Type[T],
    prompt: str,
    input_messages: List[HumanMessage] | dict,
    language_model: Optional[BaseLanguageModel] = None,
    post_process: Callable[[BaseModel], BaseModel] = lambda x: x,
    handle_exceptions: bool | list[Type[Exception]] = True,
) -> T:
    """
    Use MotleyCrew agent with retries to extract structured output.

    Args:
        schema: The Pydantic model to extract
        prompt: Instructions
        input_messages: List of messages containing the image and text
        language_model: The language model to use
        post_process: Optional function to post-process the output, for example for validation
        handle_exceptions: Whether to handle exceptions (True/False) or a list of specific exception types to handle

    Returns:
        An instance of the schema with extracted data
    """

    generator = ReActToolCallingMotleyAgent(
        llm=language_model,
        name="structured_output_extractor",
        tools=[
            StructuredPassthroughTool(
                schema=schema, post_process=post_process, handle_exceptions=handle_exceptions
            )
        ],
        force_output_handler=True,
        verbose=True,
        max_iterations=15,
        prompt=prompt,
        stream=False,
    )

    result = generator.invoke(input_messages)
    return result
