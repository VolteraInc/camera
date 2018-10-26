"""
Run the laser capture program (assuming the camera/laser are already running on the web sever or with the run_camera script.
"""
import argparse
import sys
from volteracamera.analysis.undistort import Undistort
from volteracamera.analysis.plane import Plane
from volteracamera.analysis.point_projector import PointProjector 
from volteracamera.analysis.laser_line_finder import LaserLineFinder
from volteracamera.control.camera import CameraReader


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

DATA_BUFFER_SIZE = 100
SWEEP_DISTANCE = 0.001

class Plot3dScatterClass( object ):

    def __init__( self ):
        """
        Set up the initial plot.
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111, projection='3d' )
        self.marker = "o"
        self.color = "r" 
        self.plot.scatter ([0], [0], [0])
        plt.draw()

    def draw_swept(self, data_list):
        """
        take the data as input and put it into the correct format, and sweep it across to aid in visualization.
        """
        xs = []
        yx = []
        zs = []
        
        for count, a_list in enumerate(data_list):
            for point in a_list:
                xs.append(point[0])
                ys.append(point[1] + SWEEP_DISTANCE*0.001)
                zs.append(point[2])
        
        self.draw_now(xs, ys, zs)

    def draw_now( self, xs, ys, zs ):
        """
        Update thee plot given properly formatted x, y, z lists.
        """
        self.plot.remove()
        self.plot = ax.scatter(xs, ys, zs, c=self.color, marker=self.marker)
        plt.draw()                      # redraw the canvas

matplotlib.interactive(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_parameters", type=str, help="json file containing undistortion class.")
    parser.add_argument("laser_plane", type=str, help="json file containing laser plane class.")
    parser.add_argument("-o", "--output_file", type=str, help="optional output file for the laser data, saved in csv format.")
    parser.add_argument("-d", "--display", action="store_true")

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
    point_projector = PointProjector(cam_parame, laser_plane)

    cam_reader = CamReader()

    #conditional setup for displaying data.
    if args.display:
        data = []
        plotter = Plot3dScatterClass()

    image_count = 0
    while (True):
        image = cam_reader.capture()
        
        image_points = LaserLineFinder(image).process()

        data_points = [point_projector.project (point) for point in image_points]

        if args.display:
            data.append(data_points)
            if len(data) > DATA_BUFFER_SIZE:
                data = data[(DATA_BUFFER_SIZE-len(data)):] #much more efficient than pop.
            plotter.draw_swept(data)

        if args.output_file:
            with open(args.output_file, "a") as fid:
                for point in data_points:
                    fid.write("{}, {}, {}, {}\n".format(image_count, point[0], point[1], point[2]))
        image_count += 1
