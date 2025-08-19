from typing import List

import pytest

from motleycrew.agents.langchain import ReActToolCallingMotleyAgent
from motleycrew.storage.kv_store_domain import SimpleRetrievableObject
from motleycrew.tools import MotleyTool


class TestObjectInsertionTool(MotleyTool):
    def run(self, query: str) -> List[str]:
        test_obj = SimpleRetrievableObject(
            id="test123", name="test_object", description="test description", payload={"value": 42}
        )
        self.agent.kv_store[test_obj.id] = test_obj
        return [test_obj.summary]


class TestObjectFetcherTool(MotleyTool):
    def run(self, object_id: str) -> str:
        result = self.agent.kv_store[object_id]
        return str(result.payload)


@pytest.fixture
def kv_store_agent():
    agent = ReActToolCallingMotleyAgent(
        tools=[
            TestObjectInsertionTool(
                name="test_insertion_tool",
                description="Test tool for inserting objects into kv store",
            ),
            TestObjectFetcherTool(
                name="test_fetcher_tool", description="Test tool for fetching objects from kv store"
            ),
        ],
        description="Test KV store agent",
        name="Test KV store agent",
    )
    return agent


def test_kv_store_insertion(kv_store_agent):
    insertion_tool = kv_store_agent.tools["test_insertion_tool"]
    summaries = insertion_tool.run(query="any")

    assert len(summaries) == 1
    assert "test123" in kv_store_agent.kv_store
    stored_obj = kv_store_agent.kv_store["test123"]
    assert stored_obj.name == "test_object"
    assert stored_obj.description == "test description"
    assert stored_obj.payload == {"value": 42}


def test_kv_store_retrieval(kv_store_agent):
    # First insert an object
    insertion_tool = kv_store_agent.tools["test_insertion_tool"]
    insertion_tool.run(query="any")

    # Then retrieve it
    fetcher_tool = kv_store_agent.tools["test_fetcher_tool"]
    result = fetcher_tool.run(object_id="test123")

    assert result == "{'value': 42}"


def test_kv_store_nonexistent_key(kv_store_agent):
    fetcher_tool = kv_store_agent.tools["test_fetcher_tool"]

    with pytest.raises(KeyError):
        fetcher_tool.run(object_id="nonexistent")
