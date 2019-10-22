import os
from enum import Enum
from typing import Tuple, Optional


class ImageFormat(Enum):
    """
    Class enumerating the types of images we can work with.
    """
    JPG = {"jpg", "JPG", "jpeg", "JPEG"}
    PNG = {"png", "PNG"}

    @classmethod
    def get_associated_image(cls, filename: str) -> Tuple[Optional[str], Optional["ImageFormat"]]:
        """
        Gets an image associated with the given filename, by replacing its
        extension with one of the valid image formats.

        :param filename:    The filename to find an image for.
        :return:            The filename of the image and the image format,
                            or None, None if not found.
        """
        # Remove the extension from the filename
        filename = os.path.splitext(filename)[0]

        # Check each variation of the extension for each format
        for image_format in ImageFormat:
            for extension in image_format.value:
                # Get the hypothetical filename for an image of this format
                image_filename: str = f"{filename}.{extension}"

                # If an image of this format exists, return it
                if os.path.exists(image_filename):
                    return image_filename, image_format

        # No image found
        return None, None
