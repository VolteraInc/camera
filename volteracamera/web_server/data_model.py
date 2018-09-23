"""
This class handles storing image data, thumbnails and associated analysis data for display in the web browser.
"""

import numpy as np
from PIL import Image

THUMBNAIL_SIZE=(128, 128)

class ImageData:
    """
    Class for saving image data and associate other data.
    """

    def __init__(self, image_array->np.ndarray)->None:
        self.image_array = image_array
        self.image = PIL.Image.fromarray(self.image_array, "RGB")
        self.thumbnail = image.thumbnail (THUMBNAIL_SIZE)


