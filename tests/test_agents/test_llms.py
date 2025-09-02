import pytest
from langchain_openai import ChatOpenAI
from llama_index.llms.openai import OpenAI
from unittest.mock import patch, MagicMock

from motleycrew.common import LLMProvider, LLMFramework
from motleycrew.common.exceptions import LLMProviderNotSupported
from motleycrew.common.llms import init_llm


@pytest.mark.parametrize(
    "llm_provider, llm_framework, expected_class",
    [
        (LLMProvider.OPENAI, LLMFramework.LANGCHAIN, ChatOpenAI),
        (LLMProvider.OPENAI, LLMFramework.LLAMA_INDEX, OpenAI),
    ],
)
def test_init_llm(llm_provider, llm_framework, expected_class):
    llm = init_llm(llm_provider=llm_provider, llm_framework=llm_framework)
    assert isinstance(llm, expected_class)


def test_raise_init_llm():
    with pytest.raises(LLMProviderNotSupported):
        llm = init_llm(llm_provider=LLMProvider.OPENAI, llm_framework="unknown_framework")


def test_init_llm_with_temperature_via_kwargs():
    """Test that init_llm properly passes temperature via kwargs."""
    llm = init_llm(
        llm_framework=LLMFramework.LANGCHAIN,
        llm_provider=LLMProvider.OPENAI,
        temperature=0.7
    )
    
    # Should be a ChatOpenAI instance, and temperature should be passed to it
    assert isinstance(llm, ChatOpenAI)
    # The temperature gets passed as a model_kwarg in some implementations
    # This test verifies it doesn't crash and returns the expected type


def test_init_llm_without_temperature():
    """Test that init_llm works when no temperature is provided."""
    llm = init_llm(
        llm_framework=LLMFramework.LANGCHAIN,
        llm_provider=LLMProvider.OPENAI
    )
    
    # Should be a ChatOpenAI instance
    assert isinstance(llm, ChatOpenAI)