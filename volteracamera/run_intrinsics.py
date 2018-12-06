"""
Program to run the intrinsics given a directory of images
"""
import argparse
import glob
import cv2
import numpy as np
from pathlib import Path
from volteracamera.analysis.undistort import Undistort
from volteracamera.intrinsics.circles_calibration import run_calibration

FILTER_WIDTH=1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_directory", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("-d", "--display", action="store_true")

    args = parser.parse_args()

    input_dir = Path(args.image_directory)

    image_files = glob.glob(str(input_dir / "*"))

    image_list = []
    for an_image in image_files:

        try:
            image = cv2.imread(an_image)
            blurred = cv2.blur(image, (FILTER_WIDTH, FILTER_WIDTH), 0)
            image_list.append(blurred)
            print ("Loaded " + an_image)
        except:
            print ("Failed to load " + an_image)

    out_cal = run_calibration (image_list, args.display)

    print (out_cal)

    print ("Saving Calibration to " + args.output_file)

    out_cal.write_file(args.output_file)
