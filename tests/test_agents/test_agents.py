import os

import pytest
from langchain_core.prompts.chat import ChatPromptTemplate

from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingMotleyAgent
from motleycrew.agents.llama_index.llama_index_react import ReActLlamaIndexMotleyAgent
from motleycrew.common.exceptions import (
    AgentNotMaterialized,
    CannotModifyMaterializedAgent,
)
from tests.test_agents import MockTool

os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"


class TestAgents:
    @pytest.fixture(scope="class")
    def agent(self):
        agent = ReActToolCallingMotleyAgent(
            name="AI writer agent",
            prompt="What are the latest {topic} trends?",
            description="AI-generated content",
            tools=[MockTool()],
            verbose=True,
        )
        return agent

    def test_add_tools(self, agent):
        assert len(agent.tools) == 1
        tools = [MockTool()]
        agent.add_tools(tools)
        assert len(agent.tools) == 1

    def test_materialized(self, agent):
        with pytest.raises(AgentNotMaterialized):
            agent.agent

        assert not agent.is_materialized
        agent.materialize()
        assert agent.is_materialized

        with pytest.raises(CannotModifyMaterializedAgent):
            agent.add_tools([MockTool(name="another_tool")])

    def test_compose_prompt(self, agent):
        task_dict = {"topic": "AI"}
        prompt = agent.compose_prompt(input=task_dict)

        assert "What are the latest AI trends?" in prompt
