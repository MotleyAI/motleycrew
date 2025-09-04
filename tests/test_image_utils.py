import base64
import tempfile
import pytest

from motleycrew.utils.image_utils import (
    _create_human_message_from_base64,
    image_file_to_human_message,
    image_data_to_human_message,
    image_to_human_message,
    _GSLIDES_AVAILABLE,
)
from langchain_core.messages import HumanMessage


def test_create_human_message_from_base64():
    """Test creating HumanMessage from base64 data."""
    base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    mime_type = "image/png"

    message = _create_human_message_from_base64(base64_data, mime_type)

    assert isinstance(message, HumanMessage)
    assert len(message.content) == 1
    assert message.content[0]["type"] == "image_url"
    assert message.content[0]["image_url"]["url"] == f"data:{mime_type};base64,{base64_data}"


def test_image_file_to_human_message():
    """Test creating HumanMessage from image file."""
    # Create a temporary image file
    image_content = b"fake image data"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(image_content)
        tmp_file.flush()

        message = image_file_to_human_message(tmp_file.name)

        assert isinstance(message, HumanMessage)
        assert len(message.content) == 1
        assert message.content[0]["type"] == "image_url"

        # Decode the base64 to verify content
        url = message.content[0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        base64_part = url.split("base64,")[1]
        decoded = base64.b64decode(base64_part)
        assert decoded == image_content


def test_image_file_to_human_message_unknown_type():
    """Test image_file_to_human_message with unknown file extension."""
    image_content = b"fake image data"

    with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as tmp_file:
        tmp_file.write(image_content)
        tmp_file.flush()

        message = image_file_to_human_message(tmp_file.name)

        # Should default to image/jpeg
        url = message.content[0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")


@pytest.mark.skipif(not _GSLIDES_AVAILABLE, reason="gslides-api not available")
def test_image_data_to_human_message():
    """Test creating HumanMessage from ImageData."""
    from gslides_api.domain import ImageData

    image_content = b"fake image data"
    image_data = ImageData(content=image_content, mime_type="image/png", filename="test.png")

    message = image_data_to_human_message(image_data)

    assert isinstance(message, HumanMessage)
    assert len(message.content) == 1
    assert message.content[0]["type"] == "image_url"

    # Verify the content
    url = message.content[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    base64_part = url.split("base64,")[1]
    decoded = base64.b64decode(base64_part)
    assert decoded == image_content


@pytest.mark.skipif(_GSLIDES_AVAILABLE, reason="Testing without gslides-api")
def test_image_data_to_human_message_no_gslides():
    """Test image_data_to_human_message raises ImportError when gslides-api unavailable."""
    with pytest.raises(ImportError, match="gslides-api package is required"):
        image_data_to_human_message("fake_data")


@pytest.mark.skipif(not _GSLIDES_AVAILABLE, reason="gslides-api not available")
def test_image_data_to_human_message_wrong_type():
    """Test image_data_to_human_message raises TypeError for wrong type."""
    with pytest.raises(TypeError, match="Expected ImageData object"):
        image_data_to_human_message("not_image_data")


def test_image_to_human_message_with_file_path():
    """Test image_to_human_message with file path."""
    image_content = b"fake image data"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(image_content)
        tmp_file.flush()

        message = image_to_human_message(tmp_file.name)

        assert isinstance(message, HumanMessage)
        assert len(message.content) == 1
        assert message.content[0]["type"] == "image_url"


@pytest.mark.skipif(not _GSLIDES_AVAILABLE, reason="gslides-api not available")
def test_image_to_human_message_with_image_data():
    """Test image_to_human_message with ImageData."""
    from gslides_api.domain import ImageData

    image_content = b"fake image data"
    image_data = ImageData(content=image_content, mime_type="image/png", filename="test.png")

    message = image_to_human_message(image_data)

    assert isinstance(message, HumanMessage)
    assert len(message.content) == 1
    assert message.content[0]["type"] == "image_url"


def test_image_to_human_message_wrong_type():
    """Test image_to_human_message raises TypeError for wrong type."""
    with pytest.raises(TypeError, match="Expected str or ImageData"):
        image_to_human_message(123)


def test_image_to_human_message_preserves_backward_compatibility():
    """Test that existing code using image_file_to_human_message still works."""
    image_content = b"fake image data"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(image_content)
        tmp_file.flush()

        # Both should produce the same result
        old_message = image_file_to_human_message(tmp_file.name)
        new_message = image_to_human_message(tmp_file.name)

        assert old_message.content == new_message.content
