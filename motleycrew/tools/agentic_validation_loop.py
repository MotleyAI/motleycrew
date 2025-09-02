from typing import Any, Callable, Optional, Type

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from pydantic import BaseModel, Field, create_model

from motleycrew.agents.langchain import ReActToolCallingMotleyAgent
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm
from motleycrew.tools import MotleyTool
from motleycrew.tools.structured_passthrough import StructuredPassthroughTool


class AgenticValidationLoop(MotleyTool):

    def __init__(
        self,
        prompt: str | BasePromptTemplate,
        name: str | None = None,
        description: str | None = None,
        schema: Optional[Type[BaseModel]] = None,
        post_process: Optional[Callable] = None,
        llm: Optional[Any] = None,
        handle_exceptions: bool | list[Type[Exception]] = True,
    ):
        super().__init__(
            name=name,
            description=description,
        )
        self.llm = llm or init_llm(LLMFramework.LANGCHAIN)

        # Handle prompt template
        if not isinstance(prompt, BasePromptTemplate):
            prompt = PromptTemplate.from_template(prompt)
        self.prompt_template = prompt

        # Auto-create schema if not provided
        if schema is None and prompt.input_variables:
            fields = {
                var: (str, Field(description=f"Input {var} for validation."))
                for var in prompt.input_variables
            }
            schema = create_model("ValidationLoopInput", **fields)

        self.schema = schema
        self.post_process = post_process
        self.handle_exceptions = handle_exceptions

    def run(self, **kwargs) -> Any:
        """
        Run the tool with the provided inputs.
        """
        # Format the prompt with the provided inputs
        prompt = self.prompt_template.format(**kwargs)

        output_tool = StructuredPassthroughTool(
            schema=self.schema,
            post_process=self.post_process,
            handle_exceptions=self.handle_exceptions,
        )

        agent = ReActToolCallingMotleyAgent(
            tools=[output_tool],
            llm=self.llm,
            name=self.name + "_agent",
            force_output_handler=True,
            prompt=prompt,
        )

        # Run the agent with the prompt
        response = agent.invoke({})

        return response
