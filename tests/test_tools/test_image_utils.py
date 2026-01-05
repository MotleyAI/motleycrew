import os
import pytest
from langchain_core.messages import HumanMessage
from unittest.mock import MagicMock

from motleycrew.utils.image_utils import (
    convert_image_to_png,
    human_message_from_image_bytes,
    image_file_to_bytes_and_mime_type,
    is_this_a_chart,
    SUPPORTED_MIME_TYPES,
    LIBREOFFICE_FORMATS,
)


@pytest.fixture
def image_path_girl():
    """Fixture providing absolute path to girl.png test image."""
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "..", "examples", "images", "girl.png"))


@pytest.fixture
def image_path_chart():
    """Fixture providing absolute path to chart.png test image."""
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "..", "examples", "images", "chart.png"))


def test_human_message_from_image_bytes(image_path_girl):
    """Test that human_message_from_image_bytes creates a proper HumanMessage."""
    image_bytes, mime_type = image_file_to_bytes_and_mime_type(image_path_girl)
    result = human_message_from_image_bytes(image_bytes, mime_type)

    assert isinstance(result, HumanMessage)
    assert len(result.content) == 1
    assert result.content[0]["type"] == "image_url"
    assert f"data:{mime_type};base64," in result.content[0]["image_url"]["url"]


def test_image_file_to_bytes_and_mime_type_jpg():
    """Test extracting bytes and mime type from different image format."""
    # Create a simple test image file temporarily
    import tempfile
    import base64

    # Simple 1x1 pixel JPEG data
    jpeg_data = base64.b64decode(
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/wA=="
    )

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(jpeg_data)
        temp_path = f.name

    try:
        image_bytes, mime_type = image_file_to_bytes_and_mime_type(temp_path)
        assert image_bytes == jpeg_data
        assert mime_type == "image/jpeg"
        
        # Test creating message from bytes
        result = human_message_from_image_bytes(image_bytes, mime_type)
        assert isinstance(result, HumanMessage)
        assert "data:image/jpeg;base64," in result.content[0]["image_url"]["url"]
    finally:
        import os

        os.unlink(temp_path)


def test_is_this_a_chart_with_chart_image(image_path_chart):
    """Test that is_this_a_chart correctly identifies a chart image."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.is_chart = True

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value.bind.return_value = mock_structured_llm

    # Get bytes and mime type from the image file
    image_bytes, mime_type = image_file_to_bytes_and_mime_type(image_path_chart)
    result = is_this_a_chart(image_bytes, mime_type, mock_llm)

    assert result is True
    mock_llm.with_structured_output.assert_called_once()
    mock_structured_llm.invoke.assert_called_once()


def test_is_this_a_chart_with_non_chart_image(image_path_girl):
    """Test that is_this_a_chart correctly identifies a non-chart image."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.is_chart = False

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value.bind.return_value = mock_structured_llm

    # Get bytes and mime type from the image file
    image_bytes, mime_type = image_file_to_bytes_and_mime_type(image_path_girl)
    result = is_this_a_chart(image_bytes, mime_type, mock_llm)

    assert result is False
    mock_llm.with_structured_output.assert_called_once()
    mock_structured_llm.invoke.assert_called_once()


def test_convert_image_to_png_supported_format_unchanged(image_path_girl):
    """Test that supported formats are returned unchanged."""
    image_bytes, mime_type = image_file_to_bytes_and_mime_type(image_path_girl)

    # PNG should be returned unchanged
    result_bytes, result_mime = convert_image_to_png(
        image_bytes=image_bytes, source_mime_type=mime_type
    )

    assert result_bytes == image_bytes
    assert result_mime == mime_type


def test_convert_image_to_png_all_supported_types():
    """Test that all supported MIME types are returned unchanged."""
    dummy_bytes = b"fake image data"

    for mime_type in SUPPORTED_MIME_TYPES:
        result_bytes, result_mime = convert_image_to_png(
            image_bytes=dummy_bytes, source_mime_type=mime_type
        )
        assert result_bytes == dummy_bytes
        assert result_mime == mime_type


def test_libreoffice_formats_constant():
    """Test that LIBREOFFICE_FORMATS contains expected EMF/WMF types."""
    assert "image/x-emf" in LIBREOFFICE_FORMATS
    assert "image/x-wmf" in LIBREOFFICE_FORMATS
    assert "image/emf" in LIBREOFFICE_FORMATS
    assert "image/wmf" in LIBREOFFICE_FORMATS


def test_supported_mime_types_constant():
    """Test that SUPPORTED_MIME_TYPES contains expected types."""
    assert "image/jpeg" in SUPPORTED_MIME_TYPES
    assert "image/png" in SUPPORTED_MIME_TYPES
    assert "image/gif" in SUPPORTED_MIME_TYPES
    assert "image/webp" in SUPPORTED_MIME_TYPES
