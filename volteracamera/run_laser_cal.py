"""
Run the laser calibration.
"""
import argparse
import glob
import cv2
import sys
import re
import numpy as np
from pathlib import Path
from volteracamera.analysis.undistort import Undistort
from volteracamera.analysis.laser_fitter import LaserFitter
from volteracamera.analysis.laser_line_finder import LaserLineFinder, point_overlay, reject_outlier
from volteracamera.analysis.laser_fitter import LaserFitter


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_intrinsics_file", type=str, help="Input camera calibration file.")
    parser.add_argument("image_directory", type=str, help="directory of images")
    parser.add_argument("output_file", type=str, help="output laser plane filename.")
    parser.add_argument('--preview', '-p', dest='preview', action='store_true')
    parser.add_argument('--no-preview', dest='preview', action='store_false')
    parser.set_defaults(preview=True)

    args = parser.parse_args()

    input_dir = Path(args.image_directory)

    image_files = glob.glob(str(input_dir / "*"))
    image_files.sort()

    data_points_list = []
    distance_list = []
    with LaserLineFinder() as finder:
        for count, an_image in enumerate(image_files):
            file_stem = Path(an_image).stem
            pos = float(re.findall(r'(\d+(?:\.\d+)?)', file_stem)[1])
            image = cv2.imread(an_image)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print ("Loaded {} at height {}".format(an_image, pos))
            current_points_total = finder.process(image_gray)
            current_points_filtered = reject_outlier(np.asarray(current_points_total))
            if args.preview:
                overlayed_image = point_overlay(image, current_points_filtered)
                resized_image = cv2.resize(overlayed_image, (750, 750))
                cv2.imshow('image', resized_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            data_points_list.append(current_points_filtered)
            distance_list.append(pos/1000.0) #Convert mm to m.

    undistort = Undistort.read_json_file(args.input_intrinsics_file)
    if not undistort:
        print ("Could not load camera parameters from " + args.input_intrinsics_file+". Exiting...")
        sys.exit()

    fitter = LaserFitter(undistort, data_points_list, distance_list)
    res, laser_plane = fitter.process()

    print (res)
    print (laser_plane)

    laser_plane.write_file(args.output_file)

    

