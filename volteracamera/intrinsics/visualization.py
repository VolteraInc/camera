"""
Tools for visualizing calibration images.
"""
import cv2
import numpy as np

def show_image(image, timeout=10000):
    """
    This function simplifies scaling and viewing an image using the opencv functions. Just don't close the window directly.
    """
    new_image = cv2.resize(image, (750, 750))
    cv2.imshow("Image", new_image)
    if timeout is not None:
        cv2.waitKey(timeout)
    else:
        cv2.waitKey()
    cv2.destroyWindow("Image")

def merge_images (first, second):
    new_image = np.copy(first)
    new_image[:, :, 1] = second[:, :, 1]
    new_image[:, :, 2] = 0
    return new_image

