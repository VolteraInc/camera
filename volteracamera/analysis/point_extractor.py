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
    """
    grey = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(grey)
    #inverted = grey

    inverted = cv2.medianBlur(inverted, 15)

    kernel_size=1000
    #kernel = np.ones ((kernel_size, kernel_size), np.uint8)
    background= cv2.boxFilter(inverted, -1, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
    indexes = background > inverted
    background[indexes] = inverted[indexes]
    inverted = np.abs(inverted-background)
    max_value = np.max(inverted)
    inverted = cv2.threshold(inverted, 0.75*max_value, 255, cv2.THRESH_BINARY)[1]

    indexes = np.transpose(np.nonzero(inverted))
    mean_value = np.mean(indexes, axis=0)

    # Setup SimpleBlobDetector parameters.
    #params = cv2.SimpleBlobDetector_Params()
    # Filter by Circularity
    #params.filterByCircularity = True
    #params.minCircularity = 0.1
    #params.filterByArea = True
    #params.minArea = 10
    #params.minThreshold = 254
    #params.maxThreshold=255
    
    #detector = cv2.SimpleBlobDetector_create(params)

    #keypoints = detector.detect(inverted)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    #inverted = cv2.drawKeypoints(inverted, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    #morph_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    #inverted = cv2.morphologyEx( inverted, cv2.MORPH_OPEN, morph_element)

    #rows = inverted.shape[0]
    #circles = cv2.HoughCircles(inverted, cv2.HOUGH_GRADIENT, 1, rows / 8,
    #                           param1=100, param2=30,
    #                           minRadius=1, maxRadius=-1)

    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         center = (i[0], i[1])
    #         # circle center
    #         cv2.circle(image, center, 1, (0, 100, 100), 3)
    #         # circle outline
    #         radius = i[2]
    #         cv2.circle(image, center, radius, (255, 0, 255), 3)

    

    
    #preview_image(inverted)

    return mean_value

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

    files = glob.glob(filepath)
    files.sort()
    for file in files:
        file_path = Path(file)
        if not file_path.exists():
            print ("{} does not exist.".format (file))
            continue
        image = cv2.imread(file)

        mean_value = extract_feature_position (image)
        preview_image(image, mean_value)

        file_stem = file_path.stem

        pos = re.findall(r'\d+', file_stem)
        print ("{}, {}, {}, {}, {}".format(pos[1], pos[2], pos[3], mean_value[0], mean_value[1]))



    