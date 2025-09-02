from typing import List, Optional, Type

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from motleycrew.agents.langchain import ReActToolCallingMotleyAgent
from motleycrew.tools.structured_passthrough import StructuredPassthroughTool


def structured_output_with_retries(
    schema: Type[BaseModel],
    prompt: str,
    input_messages: List[HumanMessage] | dict,
    language_model: Optional[BaseLanguageModel] = None,
) -> BaseModel:
    """
    Use MotleyCrew agent with retries to extract structured output.

    Args:
        schema: The Pydantic model to extract
        prompt: Instructions
        input_messages: List of messages containing the image and text
        language_model: The language model to use

    Returns:
        An instance of the schema with extracted data
    """

    generator = ReActToolCallingMotleyAgent(
        llm=language_model,
        name="structured_output_extractor",
        tools=[StructuredPassthroughTool(schema=schema)],
        force_output_handler=True,
        verbose=True,
        max_iterations=15,
        prompt=prompt,
        stream=False,
    )

    result = generator.invoke(input_messages)
    return result
