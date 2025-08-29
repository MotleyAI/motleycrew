import pytest
from langchain_core.messages import HumanMessage

from motleycrew.tools.image import image_file_to_human_message


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