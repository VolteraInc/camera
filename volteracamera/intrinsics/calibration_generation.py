"""
Tools for projecting images into different orientations.
"""

import random
import transforms3d as tfd
import numpy as np
import cv2

MAX_ANGLE=12*np.pi/180
LATERAL=0
Z_MIN = 100
Z_RANGE=150

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

def create_projected_image(image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray)->np.ndarray:
    """
    Create a new image from a given input image that shows the input projected in space by the 
    homography given by rvec and tvec. rvec is in angle axis form.
    """
    norm = np.sqrt(np.dot (rvec, rvec))
    axis = rvec/norm
    rot_matrix = tfd.axangles.axangle2mat(axis, norm)
    width, height, _ = image.shape
    center = np.array([width/2, height/2, 0])
    f = 250
    cam_matrix = np.array([[f, 0, center[0]],
                           [0, f, center[1]],
                           [0, 0, 1]])
    points = np.array([[0, 0, f],
              [width, height, f],
              [0, height, f],
              [width, 0, f]]) - center
    offset = np.array([0, 0, f])
    transformed_points = [np.dot(rot_matrix, point-offset)+offset + tvec  for point in points]
    projected_points = np.array([np.dot(cam_matrix, point) for point in transformed_points], dtype="float32")
    projected_points = np.array([[point[0]/point[2], point[1]/point[2]] for point in projected_points], dtype="float32")
    #projected_points = np.array([[point[0], point[1]] for point in points], dtype="float32")
    object_points = np.array([np.dot(cam_matrix, point) for point in points], dtype="float32")
    object_points = np.array([[point[0]/point[2], point[1]/point[2]] for point in object_points], dtype="float32")
    #object_points = np.array([[point[0], point[1]] for point in transformed_points], dtype="float32")
    
    H = cv2.getPerspectiveTransform(object_points, projected_points)
    warped_image = cv2.warpPerspective(image, H, (width, height), borderValue=(255, 255, 255))
    return warped_image

def generate_random_cal_images(input_image: np.ndarray, number_of_images: int)->list: 
    """
    Generate a set of random calibration images
    """
    images = []
    random.seed(1)
    for _ in range(number_of_images):
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
        image = create_projected_image(input_image, rvec, tvec)
        images.append(image)
    return images


