"""
This class is used to analyze incoming images for laser lines. It returns a set 
of sensor pixel positions. As input, it takes a greyscale image in a numpy array format.
"""
import numpy as np
import cv2
from multiprocessing import Pool

KERNEL=51
FILTER_SMOOTH = (np.zeros(KERNEL) + 1.0)/KERNEL
FILTER_SHARPEN = (np.zeros(KERNEL) - 1.0)
FILTER_SHARPEN[int(KERNEL/2)] = KERNEL 
INTERVAL = 1

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
        if not self.pool:
            raise RuntimeError ("Use process inside with statement.")
        if (len(image.shape) != 2):
            raise RuntimeError ("Grayscale images only.")
        height, width = image.shape

        cols = np.arange(0,width)
    
        #col_values = [self.image[:,int(col-INTERVAL/2):int(col+INTERVAL/2)].mean(axis=1) for col in cols]
        col_values = [image[:,col] for col in cols]
    
        laser_points = self.pool.map(LaserLineFinder._find_center_point, col_values)
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
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("out.jpg", image)
