import base64
import tempfile
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from motleycrew.utils.image_utils import (
    human_message_from_image_bytes,
    image_file_to_bytes_and_mime_type,
    is_this_a_chart,
)


def test_human_message_from_image_bytes():
    """Test creating HumanMessage from image bytes."""
    # Simple 1x1 pixel PNG data
    image_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==")
    mime_type = "image/png"

    message = human_message_from_image_bytes(image_bytes, mime_type)

    assert isinstance(message, HumanMessage)
    assert len(message.content) == 1
    assert message.content[0]["type"] == "image_url"
    assert message.content[0]["image_url"]["url"].startswith(f"data:{mime_type};base64,")


def test_image_file_to_bytes_and_mime_type():
    """Test extracting bytes and mime type from image file."""
    # Create a temporary image file
    image_content = b"fake image data"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(image_content)
        tmp_file.flush()

        image_bytes, mime_type = image_file_to_bytes_and_mime_type(tmp_file.name)

        assert image_bytes == image_content
        assert mime_type == "image/jpeg"


def test_image_file_to_bytes_and_mime_type_unknown_type():
    """Test image_file_to_bytes_and_mime_type with unknown file extension."""
    image_content = b"fake image data"

    with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as tmp_file:
        tmp_file.write(image_content)
        tmp_file.flush()

        image_bytes, mime_type = image_file_to_bytes_and_mime_type(tmp_file.name)

        assert image_bytes == image_content
        # Should default to image/jpeg
        assert mime_type == "image/jpeg"


def test_is_this_a_chart():
    """Test chart detection functionality."""
    # Simple 1x1 pixel PNG data
    image_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==")
    mime_type = "image/png"

    # Mock LLM
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.is_chart = True

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value.bind.return_value = mock_structured_llm

    result = is_this_a_chart(image_bytes, mime_type, mock_llm)

    assert result is True
    mock_llm.with_structured_output.assert_called_once()
    mock_structured_llm.invoke.assert_called_once()

    # Verify the invoke was called with correct messages
    invoke_args = mock_structured_llm.invoke.call_args[0][0]
    assert len(invoke_args) == 2  # prompt message + image message
    assert "Classify this image as a chart or not" in invoke_args[0].content


def test_is_this_a_chart_not_chart():
    """Test chart detection returns False for non-charts."""
    image_bytes = b"fake image data"
    mime_type = "image/jpeg"

    # Mock LLM
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.is_chart = False

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value.bind.return_value = mock_structured_llm

    result = is_this_a_chart(image_bytes, mime_type, mock_llm)

    assert result is False
