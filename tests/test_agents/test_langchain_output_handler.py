import pytest
from langchain_core.agents import AgentAction, AgentFinish

from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingMotleyAgent
from motleycrew.common import AuxPrompts
from motleycrew.common.exceptions import InvalidOutput
from motleycrew.tools import DirectOutput, MotleyTool
from tests.test_agents import MockTool

invalid_output = "Add more information about AI applications in medicine."


class ReportOutputHandler(MotleyTool):
    def __init__(self):
        super().__init__(
            name="output_handler",
            description="Output handler",
            return_direct=True,
        )

    def run(self, output: str):
        if "medical" not in output.lower():
            raise InvalidOutput(invalid_output)

        return {"checked_output": output}


def fake_agent_plan(intermediate_steps, step, **kwargs):
    return step


def fake_agent_take_next_step(
    name_to_tool_map, color_mapping, inputs, intermediate_steps, run_manager
):

    output_handler = name_to_tool_map.get("output_handler")
    result = output_handler._run(inputs, config=None)

    if isinstance(result, DirectOutput):
        raise result

    return result


@pytest.fixture
def agent():
    agent = ReActToolCallingMotleyAgent(
        tools=[MockTool(), ReportOutputHandler()],
        verbose=True,
        chat_history=True,
        force_output_handler=True,
    )
    agent.materialize()
    object.__setattr__(agent._agent, "plan", fake_agent_plan)
    object.__setattr__(agent.agent, "plan", agent.agent_plan_decorator(agent.agent.plan))

    object.__setattr__(agent._agent, "_take_next_step", fake_agent_take_next_step)
    object.__setattr__(
        agent._agent,
        "_take_next_step",
        agent.take_next_step_decorator(agent.agent._take_next_step),
    )
    return agent


@pytest.fixture
def run_kwargs(agent):
    agent_executor = agent.agent.bound.bound.last.bound.deps[0].bound

    run_kwargs = {
        "name_to_tool_map": {tool.name: tool for tool in agent_executor.tools},
        "color_mapping": {},
        "inputs": {},
        "intermediate_steps": [],
    }
    return run_kwargs


def test_agent_plan(agent):
    agent_executor = agent.agent
    agent_actions = [AgentAction("tool", "tool_input", "tool_log")]
    step = agent_executor.plan([], agent_actions)
    assert agent_actions == step

    return_values = {"output": "test_output"}
    agent_finish = AgentFinish(return_values=return_values, log="test_output")

    step = agent_executor.plan([], agent_finish)
    assert isinstance(step, AgentAction)
    assert step.tool == agent._agent_error_tool.name
    assert step.tool_input == {
        "error_message": AuxPrompts.get_direct_output_error_message(agent.get_output_handlers()),
        "message": "test_output",
    }


def test_agent_take_next_step(agent, run_kwargs):

    # test wrong output
    input_data = "Latest advancements in AI in 2024."
    run_kwargs["inputs"] = input_data
    step_result = agent.agent._take_next_step(**run_kwargs)
    assert step_result == f"{InvalidOutput.__name__}: {invalid_output}"

    # test correct output
    input_data = "Latest advancements in medical AI in 2024."
    run_kwargs["inputs"] = input_data
    step_result = agent.agent._take_next_step(**run_kwargs)
    assert isinstance(step_result, AgentFinish)
    assert isinstance(step_result.return_values, dict)
    output_result = step_result.return_values.get("output")
    assert output_result == {"checked_output": input_data}


@pytest.mark.fat
def test_anthropic_output_handler_error_recovery():
    """Test that Anthropic models handle output handler errors correctly.

    This tests the fix for the "non-consecutive system messages" error that occurs
    when using Anthropic models with force_output_handler=True. The error was caused
    by passing error feedback as system messages instead of human messages.

    See: https://github.com/anthropics/anthropic-sdk-python/issues/XXX
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        pytest.skip("langchain-anthropic not installed")

    from pydantic import BaseModel, Field

    from motleycrew.common import LLMFramework, LLMProvider
    from motleycrew.common.llms import init_llm
    from motleycrew.tools import MotleyTool

    # Define a simple output handler that requires a specific format
    class SummaryOutput(BaseModel):
        summary: str = Field(description="A brief summary")

    class SummaryOutputHandler(MotleyTool):
        def __init__(self):
            super().__init__(
                name="summary_output_handler",
                description="Use this tool to return your final summary. You MUST use this tool to provide your answer.",
                return_direct=True,
            )

        def run(self, summary: str) -> dict:
            return {"summary": summary}

    # Initialize Anthropic LLM
    llm = init_llm(
        llm_framework=LLMFramework.LANGCHAIN,
        llm_provider=LLMProvider.ANTHROPIC,
        llm_name="claude-sonnet-4-20250514",
        temperature=0,
    )

    # Create agent with force_output_handler=True
    # This will force the agent to use the output handler tool
    agent = ReActToolCallingMotleyAgent(
        llm=llm,
        tools=[SummaryOutputHandler()],
        verbose=True,
        force_output_handler=True,
    )

    # Invoke the agent with a simple task
    # The agent may initially try to return directly, triggering the error recovery path
    # If the fix works, this should not raise "non-consecutive system messages" error
    result = agent.invoke("What is 2 + 2? Provide a brief summary.")

    # Verify we got a result (the fix allows the agent to recover and use the output handler)
    assert result is not None
    assert "summary" in result or isinstance(result, dict)
