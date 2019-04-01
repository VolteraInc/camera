"""
Class for interfacing with the camera.
"""
import time
from io import BytesIO
from PIL import Image
import numpy as np
import zmq
import threading
import logging
import importlib
from importlib import util
picam_spec = util.find_spec("picamera")
picam_found = picam_spec is not None
if picam_found:
    logging.info ("Camera found, running on raspbery pi.")
    from picamera import PiCamera
    from picamera.array import PiRGBArray

CAMERA_INTERFACE="ipc:///tmp/camera_thread"
CAMERA_HIGH_WATER_MARK=10

#RESOLUTION = (2592, 1944)
RESOLUTION = (3280, 2464)
#RESOLUTION = (1280,720)
ZOOM = (320/1920, 180/1080, 1280/1920, 720/1080)
AWB_MODE = "off"
AWB_GAINS = 1.6
FRAMERATE = 10
SHUTTER_SPEED = 5000
EXPOSURE_MODE = "off"

class Camera(threading.Thread):
    """
    Class that reads from the camera.
    """

    def __init__(self):
        """
        Initialization of the camera.
        """
        super().__init__()
        if picam_found:
            logging.info ("Starting Camera")
            self.camera = PiCamera()
            self.camera.resolution = RESOLUTION
            #self.camera.zoom = ZOOM
            self.camera.awb_mode = AWB_MODE
            self.camera.awb_gains = AWB_GAINS
            self.camera.framerate = FRAMERATE
            #time.sleep(3)
            #self.camera.shutter_speed = SHUTTER_SPEED
            #self.camera.exposure_mode = EXPOSURE_MODE
            self.raw_capture = PiRGBArray(self.camera, size=(RESOLUTION[0], RESOLUTION[1])) 
            self.frame = None
            self.frame_mutex = threading.Lock()
            
        #set up zmq context and publishing port. Only works on Unix like systems.
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.set_hwm(CAMERA_HIGH_WATER_MARK)
        self.socket.bind (CAMERA_INTERFACE)

        #set up the capture process
        self.stop_capture = False
        logging.debug("Camera initialized.")

    def open(self):
        """
        Start the camera preview
        """
        #self.camera.start_preview()
        #time.sleep(2)

    @property
    def exposure(self):
        """
        Return the exposure time in ms
        """
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
        self.stop()
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
        while (True):
            if not self.stop_capture:
                if picam_found:
                    for frame in self.camera.capture_continuous(self.raw_capture, format="rgb", use_video_port=True):
                        Camera._send_array (self.socket, frame.array)
                        logging.debug("Sent real image.")
                        self.raw_capture.truncate(0)
                        with self.frame_mutex:
                            self.frame = frame.array.copy()
                        if self.stop_capture: 
                            logging.debug("Capture stopped.")
                            break
                else:
                    while (True):
                        frame = Image.new(mode="RGB", size=RESOLUTION)
                        imarr = np.asarray(frame)
                        imarr.flags.writeable = True
                        imarr[400, :, 2] = 128
                        Camera._send_array (self.socket, np.asarray(frame))
                        logging.debug("Sent simulated image.")
                        time.sleep(0.5) # slow down the capture to about 10fps
                        if self.stop_capture:
                            logging.debug("Capture stopped.")
                            break
            else:
                time.sleep(1)

    def capture_single (self):
        """
        Capture a single image from the camera.
        """
        if picam_found:
            with self.frame_mutex:
                frame = self.frame.copy()
            return frame
        else:
            frame = Image.new(mode="RGB", size=(RESOLUTION))
            imarr = np.asarray(frame)
            imarr.flags.writeable = True
            imarr[:, 400, 0] = 128
            logging.debug("Sent simulated image.")
            return frame

    def run(self):
        """
        This method starts the camera running in continuous mode, and publishes images over an IPC socket.
        """
        logging.debug("Camera capture started.")
        self.stop_capture = False
        self._capture_continuous()

    def stop(self):
        """
        stop capturing
        """
        #self.stop_capture = True

    def restart(self):
        self.stop_capture = False

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
        self.socket.set_hwm(CAMERA_HIGH_WATER_MARK)
        self.socket.connect(CAMERA_INTERFACE)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"") #subscribe to all sensors. 
        logging.debug ( "Camera reader started." )

    @staticmethod
    def _recv_array(socket, flags=0, copy=True, track=False):
        """
        recv a numpy array
        """
        md = socket.recv_json(flags=flags)
        msg = socket.recv(flags=flags, copy=copy, track=track)
        A = np.frombuffer(msg, dtype=md['dtype'])
        try:
            A = A.reshape(md['shape'])
            return A
        except:
            logging.warning("Image data was sent incomplete over the wire, replacing with black image.")
            return None

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
        logging.debug ("Image captured by camera reader.")
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


