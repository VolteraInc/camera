"""
Run the laser capture program (assuming the camera/laser are already running on the web sever or with the run_camera script.
"""
import argparse
import sys
import cv2
import zmq
from multiprocessing import Process
from volteracamera.analysis.undistort import Undistort
from volteracamera.analysis.plane import Plane
from volteracamera.analysis.point_projector import PointProjector 
from volteracamera.analysis.laser_line_finder import LaserLineFinder
from volteracamera.control.camera import Camera, CameraReader

PUBLISHING_PROTOCOL = "tcp"
PUBLISHING_PORT = "2223"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_parameters", type=str, help="json file containing undistortion class.")
    parser.add_argument("laser_plane", type=str, help="json file containing laser plane class.")
    parser.add_argument("-o", "--output_file", type=str, help="optional output file for the laser data, saved in csv format.")

    args = parser.parse_args()

    #load in the calibration files
    cam_params = Undistort.read_json_file(args.camera_parameters)
    laser_plane = Plane.read_json_file(args.laser_plane)

    if cam_params is None:
        print ("Failed to load camera parameters. Exiting...")
        sys.exit()
    if laser_plane is None:
        print ("Failed to load laser plane parameters. Exiting...")
        sys.exit()
    point_projector = PointProjector(cam_params, laser_plane)

    print ("Starting the Camera Server")
    #cam = Camera()
    #cam.open()
    #p = Process (target=cam.run)
    #p.start()
    print ("Camera Server started.")

    print ("Starting Camera Reader.")
    cam_reader = CameraReader()
    print ("Camera reader started.")

    print ("Setting up the data publisher")
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket_address ="{}://*:{}".format(PUBLISHING_PROTOCOL, PUBLISHING_PORT) 
    socket.bind(socket_address.encode("utf-8"))
    print ("Data publisher running at {}".format(socket_address))

    #conditional setup for displaying data.
    with LaserLineFinder() as finder:
        image_count = 0
        while (True):
            image = cam_reader.capture()
            print ("Captured Image : {}".format(image_count))
            image_points =finder.process(image[:,:,2]) 
            image_points_full = [[[i, j]] for i, j in enumerate(image_points)]
            data_points = point_projector.project (image_points_full)
    
            for point in data_points:
            #for point in image_points_full:
                socket.send_string("{}, {}, {}, {}".format(image_count, point[0], point[1], point[2]));        
                #socket.send_string("{}, {}, {}, {}".format(image_count, point[0][0], point[0][1], 0));        
            socket.send_string("")

            if args.output_file:
                with open(args.output_file, "a") as fid:
                    for point in data_points:
                        fid.write("{}, {}, {}, {}\n".format(image_count, point[0], point[1], point[2]))
            
            image_count += 1

    p.join()
