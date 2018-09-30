"""
Class for interfacing with the camera.
"""
import time
from io import BytesIO
from PIL import Image
import numpy as np
import zmq
from threading import Thread

import importlib
picam_spec = importlib.util.find_spec("picamera")
picam_found = picam_spec is not None
if picam_found:
    from picamera import PiCamera
    from picamera.array import PiRGBArray

CAMERA_INTERFACE="ipc:///tmp/camera_thread"

RESOLUTION = (1280,720)
ZOOM = (320/1920, 180/1080, 1280/1920, 720/1080)
AWB_MODE = "off"
AWB_GAINS = 1.6
FRAMERATE = 30
SHUTTER_SPEED = 5000
EXPOSURE_MODE = "off"

class Camera(object):
    """
    Class that reads from the camera.
    """

    def __init__(self):
        """
        Initialization of the camera.
        """
        if picam_found:
            print ("Starting Camera")
            self.camera = PiCamera(sensor_mode = 1)
            self.camera.resolution = RESOLUTION
            self.camera.zoom = ZOOM
            self.camera.awb_mode = AWB_MODE
            self.camera.awb_gains = AWB_GAINS
            self.camera.framerate = FRAMERATE
            #time.sleep(3)
            self.camera.shutter_speed = SHUTTER_SPEED
            self.camera.exposure_mode = EXPOSURE_MODE
            
            #set up zmq context and publishing port. Only works on Unix like systems.
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind (CAMERA_INTERFACE)

            #set up the capture process
            self.stop_capture = False
            self.capture_process = Thread (target = self._capture_continuous)

    def open(self):
        """
        Start the camera preview
        """
        #self.camera.start_preview()
        #time.sleep(2)

    @property
    def exposure(self):
        return self.camera.shutter_speed / 1E3

    @exposure.setter
    def exposure(self, exposure):
        """
        Set the exposure value (in ms)
        """
        if exposure == 0:
            self.camera.exposure_mode="auto"
        else:
            self.camera.exposure_mode="off"
        self.camera.shutter_speed = int(exposure*1E3)
     
    def __enter__(self):
        """
        Context manager enter
        """
        self.open()
        return self

    def close(self):
        """
        Called to clean up camera context.
        """
        self.stop_capture = True
        self.capture_process.join()
        if picam_found:
            self.camera.close()

    def __exit__(self, *args):
        """
        Context manager exit
        """
        self.close()

    @staticmethod
    def _send_array(socket, A, flags=0, copy=True, track=False):
        """
        send a numpy array with metadata
        """
        md = dict(
            dtype = str(A.dtype),
            shape = A.shape,
        )
        socket.send_json(md, flags|zmq.SNDMORE)
        return socket.send(A, flags, copy=copy, track=track)

    def _capture_continuous (self):
        """
        Capture images continuously from the camera.
        """
        raw_capture = PiRGBArray(self.camera, size=(RESOLUTION[1], RESOLUTION[0]))
        for frame in self.camera.capture_continuous(raw_capture, format="rgb", use_video_port=True):
            Camera._send_array (self.socket, frame.array)
            raw_capture.truncate(0)
            if self.stop_capture: 
                break

    def run(self):
        """
        This method starts the camera running in continuous mode, and publishes images over an IPC socket.
        """
        self.stop_capture = False
        self.capture_process.start()

class CameraReader():
    """
    This class reads single images from the free running camera and returns them in the requested format.
    """
    def __init__(self):
        """
        Initialize the zmq socket communication.
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(CAMERA_INTERFACE)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"") #subscribe to all sensors. 

    @staticmethod
    def _recv_array(socket, flags=0, copy=True, track=False):
        """
        recv a numpy array
        """
        md = socket.recv_json(flags=flags)
        msg = socket.recv(flags=flags, copy=copy, track=track)
        A = np.frombuffer(msg, dtype=md['dtype'])
        return A.reshape(md['shape'])

    @staticmethod
    def array_to_image( input_array: np.ndarray)->Image:
        """
        Turn an image array into a PILLOW image.
        """
        return Image.fromarray(input_array.astype('uint8'), 'RGB') 


    def capture(self)->np.ndarray:
        """
        Return a 3 channel np.array (RGB)
        """
        output = CameraReader._recv_array(self.socket)
        return output


def preview_camera():
    """
    Method to preview the camera (requires a GUI).
    """
    import timeit
    num_images=10
    with Camera() as cam:
        cam.run()
        start = timeit.default_timer()
        capture = CameraReader()
        for i in range(num_images):
            print("Capturing raw image " + str(i) )
            _ = capture.capture()
        diff = timeit.default_timer() - start
        print ("Captured " + str(num_images) + " raw images in " + str(diff) + "s.")
        start = timeit.default_timer()
        for i in range(num_images):
            print("Capturing PIL image " + str(i) )
            image = capture.capture()
            _ = CameraReader.array_to_image(image)
        diff = timeit.default_timer() - start
        print ("Captured " + str(num_images) + " jpeg images in " + str(diff) + "s.")


