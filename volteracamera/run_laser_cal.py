"""
Run the laser calibration.
"""
import argparse
import glob
import cv2
import sys
import numpy as np
from pathlib import Path
from volteracamera.analysis.undistort import Undistort
from volteracamera.analysis.laser_fitter import LaserFitter, LAYER_THICKNESS
from volteracamera.analysis.laser_line_finder import LaserLineFinder
from volteracamera.analysis.laser_fitter import LaserFitter

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input camera calibration file.")
    parser.add_argument("image_directory", type=str, help="director of images, sorted by name from closest to furthest.")
    parser.add_argument("output_file", type=str, help="output laser plane filename.")

    args = parser.parse_args()

    input_dir = Path(args.image_directory)

    image_files = glob.glob(str(input_dir / "*"))
    image_files.sort()

    data_points_list = []
    distance_list = []
    with LaserLineFinder() as finder:
        for count, an_image in enumerate(image_files):
            image = cv2.imread(an_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print ("Loaded " + an_image)
            data_points_list.append(finder.process(image))
            distance_list.append(count * LAYER_THICKNESS)

    undistort = Undistort.read_json_file(args.input_file)
    if not undistort:
        print ("Could not load camera parameters from " + args.input_file+". Exiting...")
        sys.exit()

    fitter = LaserFitter(undistort, data_points_list, distance_list)
    res, laser_plane = fitter.process()

    print (res)
    print (laser_plane)

    laser_plane.write_file(args.output_file)

    

