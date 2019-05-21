"""
This class is used to analyze incoming images for laser lines. It returns a set 
of sensor pixel positions. As input, it takes a greyscale image in a numpy array format.
"""
import numpy as np
import time
import cv2
import zmq
import multiprocessing
import threading
from multiprocessing import Process, Pool
from volteracamera.analysis.undistort import Undistort
from volteracamera.analysis.plane import Plane
from volteracamera.analysis.point_projector import PointProjector 
from volteracamera.control.camera import CameraReader, Camera
import logging

DATA_INTERFACE="ipc:///tmp/data_thread"
PROFILE_HIGH_WATER_MARK = 10000

KERNEL=51
FILTER_SMOOTH = (np.zeros(KERNEL) + 1.0)/KERNEL
FILTER_SHARPEN = (np.zeros(KERNEL) - 1.0)
FILTER_SHARPEN[int(KERNEL/2)] = KERNEL 
INTERVAL = 1
PROFILE_OFFSET = 0.0001 #move each profile over by 0.1 mm in the y direction.


def reject_outlier (data, m = 2, image_edge_threshold=200):
    """
    Reject points further than m standard deviations from the median value, and away from the bottom of the sensor.
    """
    data_return = np.zeros (len(data))
    indexes_away_from_bottom = data > image_edge_threshold
    indexes = abs(data - np.median(data[indexes_away_from_bottom])) < m * np.std(data[indexes_away_from_bottom])
    data_return[indexes] = data[indexes]
    return data_return

class LaserLineFinder (object):
    """
    A class for finding laser lines in images. This class is designed to work the 
    image arrangement of the voltera laser setup (the laser line in line with the
    bottom of the image)
    """

    def __init__(self):
        """
        Take in a numpy array representing the grayscale image
        """
        self.pool = None
        
    def __enter__(self):
        """
        """
        self.pool = Pool()    
        return self
 
    def __exit__(self, exc_type, exc_value, traceback):
        """ 
        """
        self.pool.close()
        self.pool.join()
 
    def process(self, image):
        """
        Analyze the image. Do it in parallel.
        """ 
        logging.debug ("Laser line finder started.")
        if not self.pool:
            raise RuntimeError ("Use process inside with statement.")
        if (len(image.shape) != 2):
            raise RuntimeError ("Grayscale images only.")
        _, width = image.shape

        cols = np.arange(0,width)
    
        #col_values = [self.image[:,int(col-INTERVAL/2):int(col+INTERVAL/2)].mean(axis=1) for col in cols]
        col_values = [image[:,col] for col in cols]
    
        laser_points = self.pool.map(LaserLineFinder._find_center_point, col_values)
        logging.debug ("Laser line finder finished, finding {} points".format(len(laser_points)))
        return laser_points

    @staticmethod
    def _find_center_point(col_vals):
        """
        Run filter with a gaussian to smooth and highlight the peak and then find the max.
        """
        filtered = np.convolve(col_vals, FILTER_SMOOTH, 'same')
        #filtered = np.convolve(filtered, FILTER_SHARPEN, 'same')
        peak = filtered.argmax()
        return peak
        
    @staticmethod
    def _find_center_point_old(colVals):
        """
        Find the center point of the column of data.
        """
        windowSize = 30
        
        filter = np.zeros(len(colVals), np.uint8)
        values = []
    
        window = []
        windowLastIndex = []
        windowLastIndex.append(0)
        maxWindowTotal = 0

        threshold = max(colVals)*0.75
            
        for u in range(len(colVals)):
            if len(window) >= windowSize:
                if colVals[u] > threshold:
                    window.append(threshold)
                    filter[u] = threshold
                else:
                    window.append(0.0)
                    filter[u] = 0                        
            
                window.pop(0)
                    
                if sum(window) > maxWindowTotal:
                    windowLastIndex = []
                    windowLastIndex.append(u)
                    maxWindowTotal = sum(window)
                elif sum(window) == maxWindowTotal:
                    windowLastIndex.append(u) 
            else:
                if colVals[u] > threshold:
                    window.append(threshold)
                    filter[u] = threshold
                else:
                    window.append(0.0)
                    filter[u] = 0
                        
                maxWindowTotal = sum(window)
                windowLastIndex[0] = u
            
        avgWindowLastIndex = sum(windowLastIndex)/len(windowLastIndex)
            
        if maxWindowTotal > 0:
            retVal = avgWindowLastIndex-((windowSize-1)/2)
  ##        values.append([v, avgWindowLastIndex-((windowSize-1)/2)])
        else:
            retVal = -1

        return retVal

def point_overlay (image_array, point_list):
    """
    Draw the laser line over top of the image.
    """ 
    if (len(image_array.shape) != 3):
        raise RuntimeError ("Need an RGB numpy array to draw on.")

    for (x, y) in enumerate(point_list):
        cv2.circle(image_array, (int(x), int(y)), 1, (255, 0, 0))
   
    return image_array

def preview_image():
    import sys
    import timeit

    if len(sys.argv) != 2:
        print ("USAGE: FindLaser <filename>")
        sys.exit()
    filename = sys.argv[1]

    num_analysis = 1
    image = cv2.imread(filename, 1)
    with LaserLineFinder() as finder:
        start = timeit.default_timer()
        for _ in range (num_analysis):
            points = finder.process(image[:, :, 2])
    diff = timeit.default_timer() - start
    print ("Analyzed " + str(num_analysis) + " in "+ str(diff) + "s." )
    image = point_overlay (image, points)
    image = cv2.resize(image, (1000, 1000))
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite("out.jpg", image)

class LaserProcessingServer (threading.Thread):
    """
    Server for analyzing data captured by the camera server.
    """

    def __init__(self, camera_parameters, laser_plane, camera):
        """
        Set up the laser analysis server.
        """
        super().__init__()
        self.point_projector = PointProjector(camera_parameters, laser_plane)
        self.camera = camera
        self.camera_reader = CameraReader()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        #socket_address ="{}://*:{}".format(PUBLISHING_PROTOCOL, PUBLISHING_PORT) 
        self.socket.set_hwm(PROFILE_HIGH_WATER_MARK)
        self.socket.bind(DATA_INTERFACE.encode("utf-8"))
        self.stop_capture =  0
        self.save_data_flag = 0
        self.output_file = ""

    def run (self):
        self.stop_capture = 0
        self.camera.stop()
        logging.debug("Laser processing started.")
        with LaserLineFinder() as finder:
            logging.debug("Starting processing loop.")
            while (True):
                image_count = 0
                if (self.stop_capture == 0):
                    while ( self.stop_capture == 0):
                        image = self.camera.capture_single()
                        logging.debug("Capturing image {}".format(image_count))
                        logging.debug ("Captured Image : {}".format(image_count))
                        image_points = finder.process(image[:,:,0]) 

                        #REMOVE THIS j MATH WHEN CAMERA FIXED
                        ###############################################
                        image_points_full = [[[i, (j -2464/2) * 2 + 2464/2  ]] for i, j in enumerate(image_points)]
                        ##########################################

                        intensities = [ image[int(ind[0][1]), int(ind[0][0]), 2] if ind[0][1] >= 0 and ind[0][1] < 2464 else 0 for ind in image_points_full ]
                        data_points = self.point_projector.project (image_points_full)

                        logging.debug("{} were found and are being transferred for display.".format (len(data_points)))
                        buffer = ""
                        for point, intensity in zip(data_points, intensities):
                            buffer += "{}, {}, {}, {}, {}\n".format(image_count, point[0], point[1], point[2], intensity) 
                        self.socket.send_string(buffer)

                        if self.save_data_flag != 0:
                            with open(self.output_file, "a") as fid:
                                fid.write(buffer)

                        image_count += 1
                else:
                    time.sleep(1)

    def stop (self):
        """
        Stop the process in another thread.
        """
        self.stop_capture = 1
        logging.debug("Processor stopped.")

    def restart(self):
        self.stop_capture = 0
        logging.debug("Processor restarted.")

    def save_data (self, output_file):
        """
        Set up the data saving.
        """
        logging.debug ("Save data requested in file {}".format(output_file))
        self.output_file.value = output_file
        self.save_data_flag = 1

class LaserProcessingClient(object):
    """
    Recieve laser data.
    """ 

    def __init__(self, ip_address="localhost") -> None:
        """
        Set up a reader for the processed laser data.
        """
        #socket_address ="{}://{}:{}".format(PUBLISHING_PROTOCOL, ip_address, PUBLISHING_PORT)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect (DATA_INTERFACE.encode("utf-8"))
        self.socket.set_hwm(PROFILE_HIGH_WATER_MARK)
        self.socket.setsockopt(zmq.SUBSCRIBE, "".encode("utf-8")) #no filtering of incoming data.
        logging.debug("Laser processing client set up.")

    def get_data(self)->list:
        """
        Grab the queued data.
        """
        in_string = self.socket.recv_string()
        lines = [line.split(",")for line in in_string.split("\n") ][0:-1]
        if len(lines) == 0:
            return []
        points = [ {"x": float(point[1]), "y":float(point[2]) + float(point[0])*PROFILE_OFFSET, "z":float(point[3]), "i":point[4] } for point in lines ]
        logging.debug("Laser processing client recieved {} points".format(len(points)))
        return points


