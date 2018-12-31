"""
Tools for projecting images into different orientations.
"""
from .screen_based_calibration import create_projected_image

import random
import transforms3d as tfd
import numpy as np
import cv2

MAX_ANGLE=12*np.pi/180
LATERAL=0
Z_MIN = 1
Z_RANGE=1.5

def save_scaled_images(images: list, max_image_size: int):
    """
    Save a set of images scales to a new size (default, not scaling, otherwise tuple)
    """
    for num, image in enumerate (images):
        if max_image_size is not None:
            max_dimension = max(image.shape)
            scale = max_image_size/max_dimension
            image_size = [x for x in image.shape]
            image_size[0] = int(image_size[0]*scale)
            image_size[1] = int(image_size[1]*scale)
            image = cv2.resize(image, (image_size[1], image_size[0]))
        cv2.imwrite(str(num) + ".png", image)

def generate_random_rvec_tvec ():
    """
    Generate a random rvec and tvec, returned as a tuple.
    """
    roll = random.uniform(-MAX_ANGLE, MAX_ANGLE)
    pitch = random.uniform(-MAX_ANGLE, MAX_ANGLE)
    yaw = random.uniform(-MAX_ANGLE/10, MAX_ANGLE/10)
    dx = random.uniform(-LATERAL, LATERAL)
    dy = random.uniform(-LATERAL, LATERAL)
    dz = random.uniform(Z_MIN, Z_RANGE)
    rvec, angle = tfd.axangles.mat2axangle(
           np.dot ( np.dot( tfd.axangles.axangle2mat([1, 0, 0], roll),
           tfd.axangles.axangle2mat([0, 1, 0], pitch)),
           tfd.axangles.axangle2mat([0, 0, 1], yaw)))
    rvec = rvec*angle
    tvec = np.array([dx, dy, dz])
    return (rvec, tvec)

def generate_random_cal_images(input_image: np.ndarray, number_of_images: int, rvecs: list = None, tvecs: list = None)->list: 
    """
    Generate a set of random calibration images
    """
    images = []
    random.seed(1)
    if rvecs is None and tvecs is None:
        for _ in range(number_of_images):
            rvec, tvec = generate_random_rvec_tvec()
            try:
                image = create_projected_image(input_image, rvec, tvec)
            except:
                print ("Failed to project image.")
                continue
            images.append(image)
    else:
        for rvec, tvec in zip (rvecs, tvecs):
            try:
                image = create_projected_image(input_image, rvec, tvec)
            except:
                print ("Failed to project image.")
                continue
            images.append(image)
    return images


