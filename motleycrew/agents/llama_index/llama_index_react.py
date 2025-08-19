from __future__ import annotations

from typing import Sequence

try:
    from llama_index.core.agent import ReActAgent
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.llms import LLM
except ImportError:
    LLM = object

from motleycrew.agents.llama_index import LlamaIndexMotleyAgent
from motleycrew.common import LLMFramework, MotleySupportedTool
from motleycrew.common.llms import init_llm
from motleycrew.common.utils import ensure_module_is_installed
from motleycrew.tools import MotleyTool
from motleycrew.tracking import get_default_callbacks_list


class ReActLlamaIndexMotleyAgent(LlamaIndexMotleyAgent):
    """Wrapper for the LlamaIndex implementation of ReAct agent."""

    def __init__(
        self,
        prompt: str | None = None,
        description: str | None = None,
        name: str | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        force_output_handler: bool = False,
        llm: LLM | None = None,
        verbose: bool = False,
        max_iterations: int = 10,
    ):
        """
        Args:
            prompt: Prompt for the agent.
                If a string, it will be used as a prompt.
                If a string containing f-string-style placeholders, it will be used as a prompt template.

            description: Description of the agent.

                Unlike the prompt prefix, it is not included in the prompt.
                The description is only used for describing the agent's purpose
                when giving it as a tool to other agents.

            name: Name of the agent.
                The name is used for identifying the agent when it is given as a tool
                to other agents, as well as for logging purposes.

                It is not included in the agent's prompt.

            tools: Tools to add to the agent.

            force_output_handler: Whether to force the agent to return through an output handler.
                If True, at least one tool must have return_direct set to True.

            llm: LLM instance to use.

            verbose: Whether to log verbose output.

            max_iterations: Maximum number of iterations for the agent.
                Passed on to the ``max_iterations`` parameter of the ReActAgent.
        """
        ensure_module_is_installed("llama_index")
        if llm is None:
            llm = init_llm(llm_framework=LLMFramework.LLAMA_INDEX)

        def agent_factory(tools: dict[str, MotleyTool]) -> ReActAgent:
            llama_index_tools = [t.to_llama_index_tool() for t in tools.values()]
            callbacks = get_default_callbacks_list(LLMFramework.LLAMA_INDEX)
            agent = ReActAgent.from_tools(
                tools=llama_index_tools,
                llm=llm,
                verbose=verbose,
                max_iterations=max_iterations,
                callback_manager=CallbackManager(callbacks),
            )
            return agent

        super().__init__(
            prompt=prompt,
            description=description,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            force_output_handler=force_output_handler,
            verbose=verbose,
        )
