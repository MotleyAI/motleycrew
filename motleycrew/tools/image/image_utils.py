"""Utilities for working with images in LLM contexts."""
import base64
import mimetypes
from langchain_core.messages import HumanMessage


def image_file_to_human_message(image_path: str) -> HumanMessage:
    """Create a HumanMessage containing an image.

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
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the HumanMessage content with image
    human_content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}},
    ]

    return HumanMessage(content=human_content)