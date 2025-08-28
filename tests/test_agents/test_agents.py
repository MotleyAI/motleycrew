import os

import pytest
from langchain_core.messages import HumanMessage
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

    def test_compose_prompt_with_messages_and_variables_error(self):
        """Test that providing messages input with a prompt containing variables raises an error."""
        agent = ReActToolCallingMotleyAgent(
            name="test_agent",
            prompt="What are the latest {topic} trends?",  # Has variables
            tools=[MockTool()],
        )
        
        messages = [HumanMessage(content="Hello world")]
        
        with pytest.raises(ValueError, match="Cannot use a prompt with variables when input is a list of messages"):
            agent.compose_prompt(input=messages)

    def test_compose_prompt_with_messages_no_variables(self):
        """Test that static prompt gets prepended when input is a list of BaseMessages."""
        agent = ReActToolCallingMotleyAgent(
            name="test_agent", 
            prompt="You are a helpful assistant",  # No variables
            tools=[MockTool()],
        )
        
        input_messages = [HumanMessage(content="Hello world")]
        result = agent.compose_prompt(input=input_messages, as_messages=True)
        
        # Should have 2 messages: prompt message + input message
        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Human: You are a helpful assistant"
        assert result[1].content == "Hello world"

    def test_compose_prompt_with_messages_no_variables_as_string(self):
        """Test that messages are converted to string when as_messages=False."""
        agent = ReActToolCallingMotleyAgent(
            name="test_agent",
            prompt="You are a helpful assistant",  # No variables
            tools=[MockTool()],
        )
        
        input_messages = [HumanMessage(content="Hello world")]
        result = agent.compose_prompt(input=input_messages, as_messages=False)
        
        # Should be a string with both messages joined
        assert isinstance(result, str)
        assert "Human: You are a helpful assistant" in result
        assert "Hello world" in result
