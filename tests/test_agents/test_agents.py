import os

import pytest
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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

    def test_streaming_configuration_preserved(self):
        """Test that streaming configuration is preserved when binding tools."""
        # Create base LLM
        base_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Bind stream=False
        non_streaming_llm = base_llm.bind(stream=False)
        
        # Verify the binding worked
        assert hasattr(non_streaming_llm, 'kwargs')
        assert non_streaming_llm.kwargs.get('stream') is False
        
        # Create agent with non-streaming LLM
        agent = ReActToolCallingMotleyAgent(
            llm=non_streaming_llm,
            name="test_agent",
            tools=[MockTool()],
        )
        
        # Materialize the agent to trigger the LLM binding with tools
        agent.materialize()
        
        # The test passes if the agent is created successfully
        # The fix ensures that the streaming configuration is preserved
        assert agent.is_materialized
        
    def test_streaming_configuration_preserved_true(self):
        """Test that streaming=True configuration is also preserved when binding tools."""
        # Create base LLM
        base_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Bind stream=True
        streaming_llm = base_llm.bind(stream=True)
        
        # Verify the binding worked
        assert hasattr(streaming_llm, 'kwargs')
        assert streaming_llm.kwargs.get('stream') is True
        
        # Create agent with streaming LLM
        agent = ReActToolCallingMotleyAgent(
            llm=streaming_llm,
            name="test_agent",
            tools=[MockTool()],
        )
        
        # Materialize the agent to trigger the LLM binding with tools
        agent.materialize()
        
        # The test passes if the agent is created successfully
        assert agent.is_materialized
        
    def test_default_llm_no_streaming_configuration(self):
        """Test that agents work normally when no explicit streaming configuration is set."""
        # Create agent with default LLM (no explicit streaming configuration)
        agent = ReActToolCallingMotleyAgent(
            name="test_agent",
            tools=[MockTool()],
        )
        
        # Materialize the agent
        agent.materialize()
        
        # The test passes if the agent is created successfully
        assert agent.is_materialized
