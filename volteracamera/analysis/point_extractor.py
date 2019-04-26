"""
This module is used to extract the position of features from images.
"""
import numpy as np
from pathlib import Path
import matplotlib.pylab as pl
import cv2
import sys

def extract_feature_position (image: np.ndarray):
    """
    Extract a feature from an image (usually a hole or a dot).

    Convert to Grayscale, invert the colours, smooth and return the background, threshold and them average the blob point positions.
    """
    grey = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(grey)
    
    inverted = cv2.medianBlur(inverted, 15)

    kernel_size=1000
    background= cv2.boxFilter(inverted, -1, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
    indexes = background > inverted
    background[indexes] = inverted[indexes]
    inverted = np.abs(inverted-background)
    max_value = np.max(inverted)
    inverted = cv2.threshold(inverted, 0.75*max_value, 255, cv2.THRESH_BINARY)[1]

    indexes = np.transpose(np.nonzero(inverted))
    mean_value = np.mean(indexes, axis=0)
    
    return [mean_value[1], mean_value[0]] #to correct for mismatch between numpy and opencv

def preview_image(image: np.ndarray, position: list, timeout = 0):
    cv2.circle(image, (int(position[1]), int(position[0])), 25, (0, 0, 255), 5)
    output = cv2.resize(image, (800, 800))
    cv2.imshow ("Test Image", output)
    cv2.waitKey (timeout)
    cv2.destroyAllWindows()

if __name__=="__main__":
    import glob, re

    #filepath = "/home/rwicks/Voltera/GridImages/vacuumbed/*.jpg"
    filepath = "/home/rwicks/Voltera/GridImages/paper/*.jpg"
    #filepath = "/home/rwicks/Voltera/GridImages/phone/*.jpg"

    files = glob.glob(filepath)
    files.sort()

    for file in files:
        file_path = Path(file)
        if not file_path.exists():
            print ("{} does not exist.".format (file))
            continue
        image = cv2.imread(file)

        mean_value = extract_feature_position (image)
        preview_image(image, mean_value, 500)

        file_stem = file_path.stem

        pos = re.findall(r'(\d+(?:\.\d+)?)', file_stem)
        print ("{}, {}, {}, {}, {}".format(pos[1], pos[2], pos[3], mean_value[0], mean_value[1])) #Is this the problem, flipped intrinsics?



    