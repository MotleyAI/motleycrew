"""Utilities for working with images in LLM contexts."""

import base64
import logging
import mimetypes
import os
import shutil
import subprocess
import tempfile
from typing import Tuple

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Supported image formats for LLM APIs (Claude, OpenAI, etc.)
SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Formats that require LibreOffice conversion (EMF/WMF not supported by ImageMagick on Linux)
LIBREOFFICE_FORMATS = {"image/x-emf", "image/x-wmf", "image/emf", "image/wmf"}


def _convert_with_libreoffice(image_bytes: bytes, source_mime_type: str) -> Tuple[bytes, str]:
    """Convert image using LibreOffice (for EMF/WMF on Linux).

    Args:
        image_bytes: Raw image bytes
        source_mime_type: MIME type of the source image

    Returns:
        Tuple of (converted_bytes, mime_type) or original if conversion fails.
    """
    soffice_path = shutil.which("soffice")
    if not soffice_path:
        logger.warning(
            f"LibreOffice (soffice) not available to convert {source_mime_type} to PNG."
        )
        return image_bytes, source_mime_type

    # Determine file extension from mime type
    ext = "emf" if "emf" in source_mime_type.lower() else "wmf"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, f"input.{ext}")
            with open(input_path, "wb") as f:
                f.write(image_bytes)

            # Run LibreOffice conversion
            result = subprocess.run(
                [
                    soffice_path,
                    "--headless",
                    "--convert-to",
                    "png",
                    "--outdir",
                    tmpdir,
                    input_path,
                ],
                capture_output=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.warning(
                    f"LibreOffice conversion failed: {result.stderr.decode()}"
                )
                return image_bytes, source_mime_type

            output_path = os.path.join(tmpdir, "input.png")
            if not os.path.exists(output_path):
                logger.warning("LibreOffice conversion produced no output file")
                return image_bytes, source_mime_type

            with open(output_path, "rb") as f:
                png_bytes = f.read()

            logger.info(f"Converted image from {source_mime_type} to image/png using LibreOffice")
            return png_bytes, "image/png"

    except subprocess.TimeoutExpired:
        logger.warning("LibreOffice conversion timed out")
        return image_bytes, source_mime_type
    except Exception as e:
        logger.warning(f"LibreOffice conversion failed: {e}")
        return image_bytes, source_mime_type


def _convert_with_wand(image_bytes: bytes, source_mime_type: str) -> Tuple[bytes, str]:
    """Convert image using Wand/ImageMagick.

    Args:
        image_bytes: Raw image bytes
        source_mime_type: MIME type of the source image

    Returns:
        Tuple of (converted_bytes, mime_type) or original if conversion fails.
    """
    try:
        from wand.image import Image as WandImage

        with WandImage(blob=image_bytes) as img:
            img.format = "png"
            png_bytes = img.make_blob()

        logger.info(f"Converted image from {source_mime_type} to image/png using Wand")
        return png_bytes, "image/png"
    except ImportError:
        logger.warning(
            f"Wand/ImageMagick not available to convert {source_mime_type} to PNG."
        )
        return image_bytes, source_mime_type
    except Exception as e:
        logger.warning(f"Wand conversion failed for {source_mime_type}: {e}")
        return image_bytes, source_mime_type


def convert_image_to_png(image_bytes: bytes, source_mime_type: str) -> Tuple[bytes, str]:
    """Convert image bytes to PNG format if the source format is unsupported.

    Uses LibreOffice for EMF/WMF (which ImageMagick can't handle on Linux),
    and Wand/ImageMagick for other formats.
    For supported formats (JPEG, PNG, GIF, WebP), returns original bytes unchanged.

    Args:
        image_bytes: Raw image bytes
        source_mime_type: MIME type of the source image

    Returns:
        Tuple of (converted_bytes, mime_type). Returns original if already supported.
    """
    if source_mime_type in SUPPORTED_MIME_TYPES:
        return image_bytes, source_mime_type

    # Use LibreOffice for EMF/WMF (not supported by ImageMagick on Linux)
    if source_mime_type.lower() in LIBREOFFICE_FORMATS:
        return _convert_with_libreoffice(image_bytes=image_bytes, source_mime_type=source_mime_type)

    # Try Wand/ImageMagick for other formats
    return _convert_with_wand(image_bytes=image_bytes, source_mime_type=source_mime_type)


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
    # Convert unsupported formats (EMF, WMF, etc.) to PNG before sending to LLM
    image_bytes, mime_type = convert_image_to_png(
        image_bytes=image_bytes, source_mime_type=mime_type
    )

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

    image_msg = human_message_from_image_bytes(image_bytes=image_bytes, mime_type=mime_type)

    structured_llm = llm.with_structured_output(Response).bind(stream=False)
    result = structured_llm.invoke([human_msg, image_msg])
    return result.is_chart
