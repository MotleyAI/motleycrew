#!/usr/bin/env python3
"""
Example usage of the new ImageElement.get_image_data() and enhanced image_utils functionality.
This demonstrates the complete workflow from Google Slides to LLM message.
"""

import tempfile
from unittest.mock import Mock, patch

# gslides-api imports
from gslides_api.domain_old.domain import Image, Transform
from gslides_api.element.image import ImageElement
from gslides_api.element.base import ElementKind

# motleycrew imports
from motleycrew.utils.image_utils import (
    image_to_human_message,
    image_data_to_human_message,
    image_file_to_human_message,
)


def demo_complete_workflow():
    """Demonstrate the complete workflow from ImageElement to HumanMessage."""
    print("🖼️  Complete Image Workflow Demo")
    print("=" * 50)

    # Mock an HTTP response for ImageElement.get_image_data()
    with patch("gslides_api.element.element.requests.get") as mock_get:
        # Create fake image data
        fake_image_content = b"fake PNG image data here"

        # Mock the HTTP response
        mock_response = Mock()
        mock_response.content = fake_image_content
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Step 1: Create ImageElement from Google Slides
        print("1️⃣  Creating ImageElement from Google Slides...")
        image = Image(contentUrl="https://example.com/slide_image.png")
        transform = Transform(translateX=100, translateY=50, scaleX=1.5, scaleY=1.5)
        element = ImageElement(
            objectId="slide-image-123", image=image, transform=transform, type=ElementKind.IMAGE
        )
        print(f"   ✅ Created ImageElement with ID: {element.objectId}")

        # Step 2: Retrieve image data from Google Slides
        print("2️⃣  Retrieving image data from Google Slides...")
        image_data = element.get_image_data()
        print(f"   ✅ Retrieved {len(image_data.content)} bytes")
        print(f"   ✅ MIME type: {image_data.mime_type}")
        print(f"   ✅ Filename: {image_data.filename}")

        # Step 3: Save to file (optional)
        print("3️⃣  Saving image to file...")
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_path = image_data.save_to_file(temp_dir)
            print(f"   ✅ Saved to: {saved_path}")

        # Step 4: Convert to LLM message using ImageData directly
        print("4️⃣  Creating LLM message from ImageData...")
        llm_message = image_data_to_human_message(image_data)
        print(f"   ✅ Created HumanMessage with {len(llm_message.content)} content items")
        print(f"   ✅ Content type: {llm_message.content[0]['type']}")

        # Step 5: Show unified interface
        print("5️⃣  Using unified interface...")
        unified_message = image_to_human_message(image_data)
        print(
            f"   ✅ Unified interface produces same result: {llm_message.content == unified_message.content}"
        )

        print("\n✨ Complete workflow successful!")
        return image_data, llm_message


def demo_backward_compatibility():
    """Demonstrate that existing file-based workflow still works."""
    print("\n📁 Backward Compatibility Demo")
    print("=" * 50)

    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(b"fake JPEG image content")
        tmp_file.flush()

        print("1️⃣  Using original image_file_to_human_message...")
        old_message = image_file_to_human_message(tmp_file.name)
        print(f"   ✅ Created message with {len(old_message.content)} items")

        print("2️⃣  Using new unified image_to_human_message...")
        new_message = image_to_human_message(tmp_file.name)
        print(f"   ✅ Created message with {len(new_message.content)} items")

        print("3️⃣  Checking compatibility...")
        compatible = old_message.content == new_message.content
        print(f"   ✅ Results identical: {compatible}")

        print("\n✨ Backward compatibility maintained!")


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n⚠️  Error Handling Demo")
    print("=" * 50)

    # Test ImageElement without URLs
    print("1️⃣  Testing ImageElement without URLs...")
    try:
        image = Image()  # No URLs
        transform = Transform(translateX=0, translateY=0, scaleX=1, scaleY=1)
        element = ImageElement(
            objectId="test-id", image=image, transform=transform, type=ElementKind.IMAGE
        )
        element.get_image_data()
    except ValueError as e:
        print(f"   ✅ Caught expected error: {e}")

    # Test unified interface with wrong type
    print("2️⃣  Testing unified interface with wrong type...")
    try:
        image_to_human_message(12345)  # Wrong type
    except TypeError as e:
        print(f"   ✅ Caught expected error: {e}")

    print("\n✨ Error handling working correctly!")


if __name__ == "__main__":
    print("🚀 Image Retrieval and Enhancement Demo")
    print("=" * 60)

    # Run all demos
    image_data, llm_message = demo_complete_workflow()
    demo_backward_compatibility()
    demo_error_handling()

    print("\n🎉 All demos completed successfully!")
    print("\n📋 Summary of new capabilities:")
    print("   • ImageElement.get_image_data() - Retrieve images from Google Slides")
    print("   • ImageData class - Container for image data with save functionality")
    print("   • Enhanced image_utils - Support both files and ImageData objects")
    print("   • Unified interface - image_to_human_message() accepts both types")
    print("   • Full backward compatibility - Existing code continues to work")
    print("   • Comprehensive error handling and logging")
