"""Utilities for working with images in LLM contexts."""

import base64
import mimetypes

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# Import ImageData with fallback for when gslides-api is not available
try:
    from gslides_api.domain import ImageData

    _GSLIDES_AVAILABLE = True
except ImportError:
    ImageData = None
    _GSLIDES_AVAILABLE = False


def _create_human_message_from_base64(base64_data: str, mime_type: str) -> HumanMessage:
    """Create a HumanMessage from base64 image data.

    Args:
        base64_data: Base64 encoded image data
        mime_type: MIME type of the image

    Returns:
        HumanMessage containing the image content
    """
    human_content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}},
    ]
    return HumanMessage(content=human_content)


def image_file_to_human_message(image_path: str) -> HumanMessage:
    """Create a HumanMessage containing an image from a local file.

    Args:
        image_path: Path to the local image file

    Returns:
        HumanMessage containing the image content
    """
    # Determine the MIME type from file extension
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None or not mime_type.startswith("image/"):
        # Default to jpeg if we can't determine the type
        mime_type = "image/jpeg"

    # Read and encode the image as base64
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    return _create_human_message_from_base64(base64_data, mime_type)


def image_data_to_human_message(image_data) -> HumanMessage:
    """Create a HumanMessage containing an image from ImageData.

    Args:
        image_data: ImageData object from gslides-api containing image bytes and metadata

    Returns:
        HumanMessage containing the image content

    Raises:
        ImportError: If gslides-api is not available
        TypeError: If image_data is not an ImageData instance
    """
    if not _GSLIDES_AVAILABLE:
        raise ImportError("gslides-api package is required to use ImageData objects")

    if not isinstance(image_data, ImageData):
        raise TypeError(f"Expected ImageData object, got {type(image_data)}")

    base64_data = base64.b64encode(image_data.content).decode("utf-8")
    return _create_human_message_from_base64(base64_data, image_data.mime_type)


def image_to_human_message(source) -> HumanMessage:
    """Create a HumanMessage containing an image from either a file path or ImageData.

    Args:
        source: Either a file path string or an ImageData object

    Returns:
        HumanMessage containing the image content

    Raises:
        ImportError: If gslides-api is not available when using ImageData
        TypeError: If source is neither string nor ImageData
        ValueError: If the string path doesn't exist or ImageData is invalid
    """
    if isinstance(source, str):
        return image_file_to_human_message(source)
    elif _GSLIDES_AVAILABLE and isinstance(source, ImageData):
        return image_data_to_human_message(source)
    else:
        raise TypeError(f"Expected str or ImageData, got {type(source)}")


def is_this_a_chart(image, llm: BaseLanguageModel) -> bool:
    prompt = """Classify this image as a chart or not. 
              By chart here is meant an image that contains data that can be extracted into a table, 
              create with the intent of displaying said data to the user, such as could be
              produced by matplotlib, plotly, or similar software. 
              If this image is more of a decorative kind, return False, even if it contains a chart as
              part of the imagery. 
              Only return True if it's a genuine chart meant for data display 
              of some sort, for example using lines, bars, funnels, pies, etc., shown without distortion and 
              only shown using elements that could have been produced by charting software such 
              as matplotlib or plotly.
              Glyphs without axes are NOT charts.
              """
    human_msg = HumanMessage(content=prompt)

    class Response(BaseModel):
        is_chart: bool = Field(
            description="True if the image contains a chart with data, False otherwise"
        )

    image_msg = image_to_human_message(image)

    structured_llm = llm.with_structured_output(Response).bind(stream=False)
    result = structured_llm.invoke([human_msg, image_msg])
    return result.is_chart
