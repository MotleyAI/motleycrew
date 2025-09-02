from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union

from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.common import MotleyAgentFactory, MotleySupportedTool, logger
from motleycrew.common.exceptions import (
    AgentNotMaterialized,
    CannotModifyMaterializedAgent,
)
from motleycrew.tools import MotleyTool

if TYPE_CHECKING:
    from motleycrew import MotleyCrew


class MotleyAgentParent(MotleyAgentAbstractParent, ABC):
    """Parent class for all motleycrew agents.

    This class is abstract and should be subclassed by all agents in motleycrew.

    In most cases, it's better to use one of the specialized agent classes,
    such as LangchainMotleyAgent or LlamaIndexMotleyAgent, which provide various
    useful features, such as observability and output handling, out of the box.

    If you need to create a custom agent, subclass this class and implement the `invoke` method.
    """

    def __init__(
        self,
        prompt: str | ChatPromptTemplate | None = None,
        description: str | None = None,
        name: str | None = None,
        agent_factory: MotleyAgentFactory | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        force_output_handler: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            prompt: Prompt for the agent.

                If a string, it will be used as a prompt.
                If a string containing f-string-style placeholders, it will be used as a prompt template.
                If a ChatPromptTemplate, it will be used as a prompt template.
            description: Description of the agent.
                The description is only used for describing the agent's purpose
                when giving it as a tool to other agents.

                It is NOT included in the agent's prompt.
            name: Name of the agent.

                The name is used for identifying the agent when it is given as a tool
                to other agents, as well as for logging purposes.

                It is NOT included in the agent's prompt.
            agent_factory: Factory function to create the agent.

                The factory function should accept a dictionary of tools and return the agent.
                It is usually called right before the agent is invoked for the first time.

                See :class:`motleycrew.common.types.MotleyAgentFactory` for more details.
            tools: Tools to add to the agent.
            force_output_handler: Whether to force the agent to return through an output handler.
                If True, at least one tool must have return_direct set to True.
            verbose: Whether to log verbose output.
        """
        self.name = name or description
        self.description = description  # becomes tool description
        self.prompt = (
            ChatPromptTemplate.from_messages([("user", prompt)])
            if isinstance(prompt, str)
            else prompt
        )
        self.agent_factory = agent_factory
        self.tools: dict[str, MotleyTool] = {}
        self.force_output_handler = force_output_handler
        self.verbose = verbose
        self.crew: MotleyCrew | None = None

        self._agent = None
        self._kv_store = {}

        if tools:
            self.add_tools(tools)

        self._check_force_output_handler()

    @property
    def kv_store(self) -> dict:
        return self._kv_store

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def __str__(self):
        return self.__repr__()

    def compose_prompt(
        self, input: Union[str, dict, List[BaseMessage], None] = None, as_messages: bool = False
    ) -> Union[str, list[BaseMessage]]:
        """Compose the agent's prompt from the prompt prefix and the provided prompt.

        Args:
            input: The input to the agent.
            as_messages: Whether the prompt should be returned as a Langchain messages list instead of a single string.

        Returns:
            The composed prompt.
        """
        # If input is already a list of messages, return them directly
        if isinstance(input, list) and all(isinstance(m, BaseMessage) for m in input):
            if self.prompt is None:
                output = input
            else:
                if len(self.prompt.input_variables):
                    raise ValueError(
                        "Cannot use a prompt with variables when input is a list of messages"
                    )
                output = [HumanMessage(content=self.prompt.format())] + input

            if as_messages:
                return output
            # Convert messages to string representation (fallback)
            return "\n\n".join(
                [
                    (
                        m.content
                        if hasattr(m, "content") and isinstance(m.content, str)
                        else str(m.content)
                    )
                    for m in output
                ]
            )

        if self.prompt:
            if isinstance(input, str):
                raise ValueError(
                    "Input must be a dictionary when using a prompt template with the agent"
                )
            elif input is None:
                input = {}

            if isinstance(self.prompt, ChatPromptTemplate):
                prepared_prompt = self.prompt.invoke(input).to_messages()

            elif isinstance(self.prompt, str):
                prepared_prompt = [HumanMessage(content=self.prompt.format(**input))]

            else:
                raise ValueError("Agent prompt must be a string or a ChatPromptTemplate")

        else:
            if input is None:
                prepared_prompt = []
            elif isinstance(input, str):
                prepared_prompt = [HumanMessage(content=input)]
            # Try to find the prompt in the input dictionary
            elif "prompt" in input:
                prepared_prompt = [HumanMessage(content=input["prompt"])]
            elif "input" in input:
                prepared_prompt = [HumanMessage(content=input["input"])]
            else:
                prepared_prompt = []

        if not prepared_prompt:
            raise ValueError("No prompt provided to the agent")

        if as_messages:
            return prepared_prompt

        return "\n\n".join([m.content for m in prepared_prompt])

    @property
    def agent(self):
        """
        Getter for the inner agent that makes sure it's already materialized.
        The inner agent should always be accessed via this property method.
        """
        if not self.is_materialized:
            raise AgentNotMaterialized(agent_name=self.name)
        return self._agent

    @property
    def is_materialized(self):
        """Whether the agent is materialized."""
        return self._agent is not None

    def get_output_handlers(self):
        """Get all output handlers (tools with return_direct set to True)."""
        return [tool for tool in self.tools.values() if tool.return_direct]

    def _check_force_output_handler(self):
        """If force_output_handler is set to True, at least one tool must have return_direct set to True."""
        if self.force_output_handler and not self.get_output_handlers():
            raise ValueError(
                "force_return_through_tool is set to True, but no tools have return_direct set to True."
            )

    def materialize(self):
        """Materialize the agent by creating the agent instance using the agent factory.
        This method should be called before invoking the agent for the first time.
        """

        if self.is_materialized:
            logger.info("Agent is already materialized, skipping materialization")
            return
        assert self.agent_factory, "Cannot materialize agent without a factory provided"

        self._agent = self.agent_factory(tools=self.tools)

    def _prepare_for_invocation(
        self,
        input: Union[str, dict, List[BaseMessage], None] = None,
        prompt_as_messages: bool = False,
    ) -> Union[str, List[BaseMessage]]:
        """Prepare the agent for invocation by materializing it and composing the prompt.

        Should be called in the beginning of the agent's invoke method.

        Args:
            input: the input to the agent
            prompt_as_messages: Whether the prompt should be returned as a Langchain messages list
                instead of a single string.

        Returns:
            Union[str, List[BaseMessage]]: the composed prompt
        """
        self.materialize()

        for tool in self.tools.values():
            assert tool.agent in (None, self), "Tool already has another agent assigned to it"
            tool.agent = self
            tool.agent_input = input

        prompt = self.compose_prompt(input, as_messages=prompt_as_messages)
        return prompt

    def add_tools(self, tools: Sequence[MotleySupportedTool]):
        """Add tools to the agent.

        Args:
            tools: The tools to add to the agent.
        """
        if self.is_materialized and tools:
            raise CannotModifyMaterializedAgent(agent_name=self.name)

        for t in tools:
            motley_tool = MotleyTool.from_supported_tool(t)
            if motley_tool.name not in self.tools:
                if motley_tool.agent is not None:
                    raise ValueError(
                        f"Tool {motley_tool.name} already has an agent assigned to it, please use unique tool instances."
                    )
                motley_tool.agent = self
                self.tools[motley_tool.name] = motley_tool
            else:  # TODO: should we raise an error here?
                logger.warning(f"Tool {motley_tool.name} already added, skipping addition of {t}.")

    def as_tool(self, **kwargs) -> MotleyTool:
        """Convert the agent to a tool to be used by other agents via delegation.

        Args:
            kwargs: Additional arguments to pass to the tool.
                See :class:`motleycrew.tools.tool.MotleyTool` for more details.
        """

        if not getattr(self, "name", None) or not getattr(self, "description", None):
            raise ValueError("Agent must have a name and description to be called as a tool")

        class CallAsToolInput(BaseModel):
            input: str = Field(..., description="Input to the tool")

        def call_as_tool(*args, **kwargs):
            if args:
                return self.invoke({"prompt": args[0]})
            if len(kwargs) == 1:
                return self.invoke({"prompt": list(kwargs.values())[0]})
            return self.invoke(kwargs)

        # To be specialized if we expect structured input
        return MotleyTool.from_langchain_tool(
            StructuredTool(
                name=self.name.replace(
                    " ", "_"
                ).lower(),  # OpenAI doesn't accept spaces in function names
                description=self.description,
                func=call_as_tool,
                args_schema=CallAsToolInput,
            ),
            **kwargs,
        )

    @abstractmethod
    def invoke(
        self,
        input: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        pass
