import pytest
from langchain_openai import ChatOpenAI
from llama_index.llms.openai import OpenAI
from unittest.mock import patch, MagicMock

from motleycrew.common import LLMProvider, LLMFramework
from motleycrew.common.exceptions import LLMProviderNotSupported
from motleycrew.common.llms import (
    init_llm,
    langchain_openai_llm,
    llama_index_openai_llm,
    langchain_anthropic_llm,
    llama_index_anthropic_llm,
    langchain_replicate_llm,
    llama_index_replicate_llm,
)


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


@patch('motleycrew.common.llms.ChatOpenAI')
def test_langchain_openai_llm_with_temperature(mock_chat_openai):
    """Test that temperature is passed when explicitly set."""
    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance
    
    result = langchain_openai_llm(llm_temperature=0.7)
    
    mock_chat_openai.assert_called_once_with(model='gpt-4.1', temperature=0.7)
    assert result == mock_instance


@patch('motleycrew.common.llms.ChatOpenAI')
def test_langchain_openai_llm_without_temperature(mock_chat_openai):
    """Test that temperature is omitted when None."""
    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance
    
    result = langchain_openai_llm(llm_temperature=None)
    
    mock_chat_openai.assert_called_once_with(model='gpt-4.1')
    assert result == mock_instance


@patch('motleycrew.common.llms.ensure_module_is_installed')
@patch('motleycrew.common.llms.OpenAI')
def test_llama_index_openai_llm_with_temperature(mock_openai, mock_ensure):
    """Test that temperature is passed when explicitly set for LlamaIndex."""
    mock_instance = MagicMock()
    mock_openai.return_value = mock_instance
    
    result = llama_index_openai_llm(llm_temperature=0.7)
    
    mock_ensure.assert_called_once_with('llama_index')
    mock_openai.assert_called_once_with(model='gpt-4.1', temperature=0.7)
    assert result == mock_instance


@patch('motleycrew.common.llms.ensure_module_is_installed')
@patch('motleycrew.common.llms.OpenAI')
def test_llama_index_openai_llm_without_temperature(mock_openai, mock_ensure):
    """Test that temperature is omitted when None for LlamaIndex."""
    mock_instance = MagicMock()
    mock_openai.return_value = mock_instance
    
    result = llama_index_openai_llm(llm_temperature=None)
    
    mock_ensure.assert_called_once_with('llama_index')
    mock_openai.assert_called_once_with(model='gpt-4.1')
    assert result == mock_instance


@patch('motleycrew.common.llms.ChatAnthropic')
def test_langchain_anthropic_llm_with_temperature(mock_chat_anthropic):
    """Test that temperature is passed when explicitly set for Anthropic."""
    mock_instance = MagicMock()
    mock_chat_anthropic.return_value = mock_instance
    
    result = langchain_anthropic_llm(llm_temperature=0.7)
    
    mock_chat_anthropic.assert_called_once_with(model='gpt-4.1', temperature=0.7)
    assert result == mock_instance


@patch('motleycrew.common.llms.ChatAnthropic')
def test_langchain_anthropic_llm_without_temperature(mock_chat_anthropic):
    """Test that temperature is omitted when None for Anthropic."""
    mock_instance = MagicMock()
    mock_chat_anthropic.return_value = mock_instance
    
    result = langchain_anthropic_llm(llm_temperature=None)
    
    mock_chat_anthropic.assert_called_once_with(model='gpt-4.1')
    assert result == mock_instance


@patch('motleycrew.common.llms.ensure_module_is_installed')
@patch('motleycrew.common.llms.Anthropic')
def test_llama_index_anthropic_llm_with_temperature(mock_anthropic, mock_ensure):
    """Test that temperature is passed when explicitly set for LlamaIndex Anthropic."""
    mock_instance = MagicMock()
    mock_anthropic.return_value = mock_instance
    
    result = llama_index_anthropic_llm(llm_temperature=0.7)
    
    mock_ensure.assert_called_once_with('llama_index')
    mock_anthropic.assert_called_once_with(model='gpt-4.1', temperature=0.7)
    assert result == mock_instance


@patch('motleycrew.common.llms.ensure_module_is_installed')
@patch('motleycrew.common.llms.Anthropic')
def test_llama_index_anthropic_llm_without_temperature(mock_anthropic, mock_ensure):
    """Test that temperature is omitted when None for LlamaIndex Anthropic."""
    mock_instance = MagicMock()
    mock_anthropic.return_value = mock_instance
    
    result = llama_index_anthropic_llm(llm_temperature=None)
    
    mock_ensure.assert_called_once_with('llama_index')
    mock_anthropic.assert_called_once_with(model='gpt-4.1')
    assert result == mock_instance


@patch('motleycrew.common.llms.Replicate')
def test_langchain_replicate_llm_with_temperature(mock_replicate):
    """Test that temperature is added to model_kwargs when explicitly set for Replicate."""
    mock_instance = MagicMock()
    mock_replicate.return_value = mock_instance
    
    result = langchain_replicate_llm(llm_temperature=0.7)
    
    expected_model_kwargs = {'temperature': 0.7}
    mock_replicate.assert_called_once_with(model='gpt-4.1', model_kwargs=expected_model_kwargs)
    assert result == mock_instance


@patch('motleycrew.common.llms.Replicate')
def test_langchain_replicate_llm_without_temperature(mock_replicate):
    """Test that temperature is omitted from model_kwargs when None for Replicate."""
    mock_instance = MagicMock()
    mock_replicate.return_value = mock_instance
    
    result = langchain_replicate_llm(llm_temperature=None)
    
    expected_model_kwargs = {}
    mock_replicate.assert_called_once_with(model='gpt-4.1', model_kwargs=expected_model_kwargs)
    assert result == mock_instance


@patch('motleycrew.common.llms.ensure_module_is_installed')
@patch('motleycrew.common.llms.Replicate')
def test_llama_index_replicate_llm_with_temperature(mock_replicate, mock_ensure):
    """Test that temperature is passed when explicitly set for LlamaIndex Replicate."""
    mock_instance = MagicMock()
    mock_replicate.return_value = mock_instance
    
    result = llama_index_replicate_llm(llm_temperature=0.7)
    
    mock_ensure.assert_called_once_with('llama_index')
    mock_replicate.assert_called_once_with(model='gpt-4.1', temperature=0.7)
    assert result == mock_instance


@patch('motleycrew.common.llms.ensure_module_is_installed')
@patch('motleycrew.common.llms.Replicate')
def test_llama_index_replicate_llm_without_temperature(mock_replicate, mock_ensure):
    """Test that temperature is omitted when None for LlamaIndex Replicate."""
    mock_instance = MagicMock()
    mock_replicate.return_value = mock_instance
    
    result = llama_index_replicate_llm(llm_temperature=None)
    
    mock_ensure.assert_called_once_with('llama_index')
    mock_replicate.assert_called_once_with(model='gpt-4.1')
    assert result == mock_instance


def test_init_llm_with_temperature():
    """Test that init_llm properly handles explicit temperature values."""
    with patch('motleycrew.common.llms.langchain_openai_llm') as mock_func:
        mock_instance = MagicMock()
        mock_func.return_value = mock_instance
        
        result = init_llm(
            llm_framework=LLMFramework.LANGCHAIN,
            llm_provider=LLMProvider.OPENAI,
            llm_temperature=0.7
        )
        
        mock_func.assert_called_once_with(
            llm_name='gpt-4.1',
            llm_temperature=0.7
        )
        assert result == mock_instance


def test_init_llm_without_temperature():
    """Test that init_llm properly handles None temperature values."""
    with patch('motleycrew.common.llms.langchain_openai_llm') as mock_func:
        mock_instance = MagicMock()
        mock_func.return_value = mock_instance
        
        result = init_llm(
            llm_framework=LLMFramework.LANGCHAIN,
            llm_provider=LLMProvider.OPENAI,
            llm_temperature=None
        )
        
        mock_func.assert_called_once_with(
            llm_name='gpt-4.1',
            llm_temperature=None
        )
        assert result == mock_instance
