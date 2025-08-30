import pytest
from langchain_core.messages import HumanMessage
from unittest.mock import MagicMock

from motleycrew.tools.image import image_file_to_human_message, is_this_a_chart


def test_image_file_to_human_message():
    """Test that image_file_to_human_message creates a proper HumanMessage."""
    image_path = "examples/images/girl.png"
    
    result = image_file_to_human_message(image_path)
    
    assert isinstance(result, HumanMessage)
    assert len(result.content) == 1
    assert result.content[0]["type"] == "image_url"
    assert "data:image/png;base64," in result.content[0]["image_url"]["url"]


def test_image_file_to_human_message_jpg():
    """Test with a different image format."""
    # Create a simple test image file temporarily
    import tempfile
    import base64
    
    # Simple 1x1 pixel JPEG data
    jpeg_data = base64.b64decode('/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/wA==')
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        f.write(jpeg_data)
        temp_path = f.name
    
    try:
        result = image_file_to_human_message(temp_path)
        assert isinstance(result, HumanMessage)
        assert "data:image/jpeg;base64," in result.content[0]["image_url"]["url"]
    finally:
        import os
        os.unlink(temp_path)


def test_is_this_a_chart_with_chart_image():
    """Test that is_this_a_chart correctly identifies a chart image."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.is_chart = True
    
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value.bind.return_value = mock_structured_llm
    
    result = is_this_a_chart("examples/images/chart.png", mock_llm)
    
    assert result is True
    mock_llm.with_structured_output.assert_called_once()
    mock_structured_llm.invoke.assert_called_once()


def test_is_this_a_chart_with_non_chart_image():
    """Test that is_this_a_chart correctly identifies a non-chart image."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.is_chart = False
    
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value.bind.return_value = mock_structured_llm
    
    result = is_this_a_chart("examples/images/girl.png", mock_llm)
    
    assert result is False
    mock_llm.with_structured_output.assert_called_once()
    mock_structured_llm.invoke.assert_called_once()