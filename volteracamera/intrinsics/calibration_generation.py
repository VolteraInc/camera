"""
Tools for projecting images into different orientations.
"""
from .screen_based_calibration import create_projected_image

import random
import transforms3d as tfd
import numpy as np
import cv2

MAX_ANGLE= np.radians(12)
LATERAL=0
Z_MIN = 500
Z_RANGE=1

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
    dz = random.uniform(Z_MIN, Z_MIN+Z_RANGE)
    rvec, angle = tfd.axangles.mat2axangle(
           np.dot ( np.dot( tfd.axangles.axangle2mat([1, 0, 0], roll),
           tfd.axangles.axangle2mat([0, 1, 0], pitch)),
           tfd.axangles.axangle2mat([0, 0, 1], yaw)))
    rvec = rvec*angle
    tvec = np.array([dx, dy, dz])
    return (rvec, tvec)

def generate_random_cal_images(input_image: np.ndarray, number_of_images: int, image_size: tuple, rvecs: list = None, tvecs: list = None)->list: 
    """
    Generate a set of random calibration images
    """
    images = []
    ranges = []
    random.seed(1)
    if rvecs is None and tvecs is None:
        rvecs = []
        tvecs = []
        for _ in range(number_of_images):
            rvec, tvec = generate_random_rvec_tvec()
            rvecs.append(rvec)
            tvecs.append(tvec)

    for rvec, tvec in zip (rvecs, tvecs):
        try:
            image, a_range = create_projected_image(input_image, rvec, tvec, image_size)
        except:
            print ("Failed to project image.")
            continue
        images.append(image)
        ranges.append(a_range)

    return images, ranges

def crop_images_to_range(images: list, ranges: list, image_size:tuple)->list:
    """
    Take a set of images and associated ranges and crop them all around a center point to the size image_size.
    """
    mean_range = np.array([ np.array([ar[0], ar[1], ar[2], ar[3]]) for ar in ranges])
    mean_range = mean_range.mean(axis=0)

    center = ((mean_range[1]+mean_range[0])/2, (mean_range[3]+mean_range[2])/2)
    
    crop_range = (int(round(center[0] - image_size[0]/2)), int(round(center[0] + image_size[0]/2)), int(round(center[1] - image_size[1]/2)), int(round(center[1] + image_size[1]/2)))
    print("image size = {}, {}".format(crop_range[1]-crop_range[0], crop_range[3] - crop_range[2]))
    print("Requested size: " + str(image_size))
    print ("image_size: " + str(images[0].shape))
    print("Crop ranges: " + str(crop_range))
    out_images = []
    for image in images:
        try:
            out_images.append(image[crop_range[0]:crop_range[1], crop_range[2]:crop_range[3]])
        except:
            print("Skipping image, could not crop to requested image size.")
            print("Requested size: " + str(image_size))
            print("Input ranges:" + str(mean_range))
            print("Crop ranges: " + str(crop_range))
            continue
    return out_images

