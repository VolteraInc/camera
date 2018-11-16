"""
Plot data produced by the laser scanner.
"""
import argparse
import zmq
import numpy as np
from volteracamera.capture_laser_data import (PUBLISHING_PROTOCOL, PUBLISHING_PORT)

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MAX_BUFFER_SIZE=1
SWEPT_DISTANCE=0.0005


class Plot2dClass( object ):

    def __init__( self ):
        """
        Set up the initial plot.
        """
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111 )
        
        self.marker = "."
        self.color = "r" 
        self.line = ""
        self.plot = self.ax.plot ([0], [0])
        self.fig.show()
        self.fig.canvas.draw()
        plt.cla()

    def draw_swept(self, data_list):
        """
        take the data as input and put it into the correct format, and sweep it across to aid in visualization.
        """
        xs = []
        ys = []
        
        for count, a_list in enumerate(data_list):
            for point in a_list:
                xs.append(point[0])
                ys.append(point[2])
        
        self.draw_now(xs, ys)

    def draw_now( self, x, y ):
        """
        Update thee plot given properly formatted x, y lists.
        """
        self.plot = self.ax.plot(x, y, c=self.color, marker=self.marker, linestyle=self.line)
        self.fig.canvas.draw()                      # redraw the canvas
        del self.ax.lines[0]
        



class Plot3dScatterClass( object ):

    def __init__( self ):
        """
        Set up the initial plot.
        """
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111, projection='3d' )
        
        self.marker = "o"
        self.color = "r" 
        self.plot = self.ax.scatter ([0], [0], [0])
        self.fig.show()
        self.fig.canvas.draw()

    def draw_swept(self, data_list):
        """
        take the data as input and put it into the correct format, and sweep it across to aid in visualization.
        """
        xs = []
        ys = []
        zs = []
        
        for count, a_list in enumerate(data_list):
            for point in a_list:
                xs.append(point[0])
                ys.append(point[1] + count*SWEPT_DISTANCE)
                zs.append(point[2])
        
        self.draw_now(xs, ys, zs)

    def draw_now( self, xs, ys, zs ):
        """
        Update thee plot given properly formatted x, y, z lists.
        """
        self.plot.remove()
        self.plot = self.ax.scatter(xs, ys, zs, c=self.color, marker=self.marker)
        self.fig.canvas.draw()                      # redraw the canvas

matplotlib.interactive(True)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ip_address", type=str, help="IP address of the laser device..")

    args = parser.parse_args()

    print ("Setting up data subsriber port.")
    socket_address ="{}://{}:{}".format(PUBLISHING_PROTOCOL, args.ip_address, PUBLISHING_PORT)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect (socket_address.encode("utf-8"))
    print ("Listening on {}".format(socket_address))

    socket.setsockopt(zmq.SUBSCRIBE, "".encode("utf-8")) #no filtering of incoming data.

    plot = Plot2dClass()

    full_data = []
    data = []
    count = 0
    while (True):
        in_string = socket.recv_string()
        if in_string == "": #empty line denotes end of one dataset.
            full_data.append(data)
            if len(full_data) > MAX_BUFFER_SIZE:
                full_data = full_data[1:]
            print ("Plotting Image {}".format(count))
            plot.draw_swept(full_data)
            data = []
            count += 1
            continue
            
        _, x, y, z = in_string.split(',')
        data.append([float(x), float(y), float(z)])

        
  
