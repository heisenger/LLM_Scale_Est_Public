import os
import base64
from typing import Optional
from io import BytesIO
from PIL import Image
import re


def get_image_base64_data_uri(image_path: str) -> Optional[str]:
    """
    Reads an image file from the given path, Base64 encodes it,
    and returns it as a data URI string.

    Args:
        image_path (str): The full path to the image file (e.g., "path/to/my_image.png").

    Returns:
        Optional[str]: A data URI string (e.g., "data:image/png;base64,...")
                       if successful, otherwise None.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return None

    try:
        # Determine the MIME type based on the file extension
        # This is important for the data URI format
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension == ".png":
            mime_type = "image/png"
        elif file_extension in (".jpg", ".jpeg"):
            mime_type = "image/jpeg"
        elif file_extension == ".gif":
            mime_type = "image/gif"
        elif file_extension == ".webp":
            mime_type = "image/webp"
        elif file_extension == ".svg":
            mime_type = "image/svg+xml"
        else:
            print(
                f"Warning: Unknown image file type for '{file_extension}'. Defaulting to 'application/octet-stream'."
            )
            mime_type = "application/octet-stream"  # Generic binary type

        with open(image_path, "rb") as image_file:
            # Read the file in binary mode
            binary_data = image_file.read()
            # Base64 encode the binary data
            encoded_data = base64.b64encode(binary_data).decode("utf-8")

            # Construct the data URI
            return f"data:{mime_type};base64,{encoded_data}"

    except Exception as e:
        print(f"Error encoding image '{image_path}': {e}")
        return None


def decode_base64_data_uri_to_image(
    data_uri: str, save_path: Optional[str] = None, show_image: bool = False
) -> Optional[Image.Image]:
    """
    Decodes a base64 image data URI string back to a PIL Image object.
    Optionally saves the image to disk and/or displays it.

    Args:
        data_uri (str): The data URI string (e.g., "data:image/png;base64,...").
        save_path (Optional[str]): If provided, saves the image to this path.
        show_image (bool): If True, displays the image using PIL.Image.show().

    Returns:
        Optional[Image.Image]: The decoded PIL Image object, or None on failure.
    """
    try:
        # Extract base64 part from data URI
        match = re.match(r"^data:(.*?);base64,(.*)$", data_uri)
        if not match:
            print("Error: Invalid data URI format.")
            return None
        base64_str = match.group(2)
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))

        if save_path:
            image.save(save_path)
        if show_image:
            try:
                # Try to display inline in Jupyter
                from IPython.display import display

                display(image)
            except ImportError:
                # Fallback to default show
                image.show()
        return image
    except Exception as e:
        print(f"Error decoding base64 data URI: {e}")
        return None
