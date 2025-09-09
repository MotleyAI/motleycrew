"""Utilities for working with images in LLM contexts."""

import base64
import mimetypes
from typing import Tuple

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


def human_message_from_image_bytes(image_bytes: bytes, mime_type: str) -> HumanMessage:
    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    human_content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}},
    ]
    return HumanMessage(content=human_content)


def image_file_to_bytes_and_mime_type(image_path: str) -> Tuple[bytes, str]:
    """Create a HumanMessage containing an image represented as bytes

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

    return image_bytes, mime_type


def is_this_a_chart(image_bytes: bytes, mime_type: str, llm: BaseLanguageModel) -> bool:
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

    image_msg = human_message_from_image_bytes(image_bytes, mime_type)

    structured_llm = llm.with_structured_output(Response).bind(stream=False)
    result = structured_llm.invoke([human_msg, image_msg])
    return result.is_chart
