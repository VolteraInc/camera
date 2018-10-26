"""
Run the laser capture program (assuming the camera/laser are already running on the web sever or with the run_camera script.
"""
import argparse

from .analysis.undistort import Undistort
from .analysis.plane import Plane
from .analysis.point_projector import PointProjector 
from .analysis.laser_line_finder import LaserLineFinder
from .control.camera import CameraReader


#modified from https://stackoverflow.com/questions/5179589/continuous-3d-plotting-i-e-figure-update-using-python-matplotlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib

DATA_BUFFER_SIZE = 100

class plot3dScatterClass( object ):

    def __init__( self ):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111, projection='3d' )

        self.ax.w_zaxis.set_major_locator( LinearLocator( 10 ) )
        self.ax.w_zaxis.set_major_formatter( FormatStrFormatter( '%.03f' ) )
        self.marker = "o"
        self.color = "r" 
        self.plot.scatter ([0], [0], [0])
        plt.draw()

    def draw_now( self, xs, ys, zs ):

        self.plot.remove()
        self.plot = ax.scatter(xs, ys, zs, c=self.color, marker=self.marker)
        
        #self.surf = self.ax.plot_surface( 
        #    self.X, self.Y, heightR, rstride=1, cstride=1, 
        #    cmap=cm.jet, linewidth=0, antialiased=False )
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

    data_buffer = []
    while (True):
        image = cam_reader.capture()
        
        image_points = LaserLineFinder(image).process()

        data_points = [point_projector.project (point) for point in image_points]

