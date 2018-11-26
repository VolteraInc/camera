"""
Run the laser capture program (assuming the camera/laser are already running on the web sever or with the run_camera script.
"""
import argparse
import sys
from volteracamera.analysis.undistort import Undistort
from volteracamera.analysis.plane import Plane
from volteracamera.analysis.laser_line_finder import LaserProcessingServer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_parameters", type=str, help="json file containing undistortion class.")
    parser.add_argument("laser_plane", type=str, help="json file containing laser plane class.")
    parser.add_argument("-o", "--output_file", type=str, help="optional output file for the laser data, saved in csv format.")

    args = parser.parse_args()

    #load in the calibration files
    cam_params = Undistort.read_json_file(args.camera_parameters)
    laser_plane = Plane.read_json_file(args.laser_plane)

    output_file = args.output_file

    if cam_params is None:
        print ("Failed to load camera parameters. Exiting...")
        sys.exit()
    if laser_plane is None:
        print ("Failed to load laser plane parameters. Exiting...")
        sys.exit()

    laser_processor = LaserProcessingServer (cam_params, laser_plane)

    if output_file:
        laser_processor.save_data (output_file)
    laser_processor.start()