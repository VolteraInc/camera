import os

from functools import partial

from flask import Flask
from ..control.camera import Camera
from ..control.laser import Laser
from ..analysis.laser_line_finder import LaserProcessingServer, Undistort
from ..analysis.plane import Plane

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)


DEFAULT_LASER_PLANE_FILE="laser.json"
DEFAULT_CAMERA_FILE="camera.json"

#initialize global camera and laser objects.
cam = None
laser = None
processor = None
data = {
        "images": [],
    }

def get_cam():
    """
    access the global camera object 
    """
    global cam
    if cam == None:
        cam = Camera()
        cam.open()
    return cam

def get_laser():
    """
    access the global laser object.
    """
    global laser
    if laser == None:
        laser = Laser()
    return laser

def get_processor():
    """
    Global Accessor for the processor.

    This is created during initialization.
    """
    global processor
    return processor

def get_data_store():
    """
    Access the global data storage.
    """
    global data
    return data

def initialize( file_path ):
    """
    Called on first web server request. This starts all the different global processes.
    """
    global processor
    logging.debug("Initializing global structures.")
    #Instantiate and start the camera
    cam = get_cam()
    cam.start()
    #Intantiate and start the laser
    _ = get_laser()
    #load in the calibration files
    print ("Loading Cals from {}".format(file_path))
    logging.debug("Loading {}".format(os.path.join( file_path, DEFAULT_CAMERA_FILE)))
    cam_params = Undistort.read_json_file(os.path.join( file_path, DEFAULT_CAMERA_FILE))
    print (cam_params)
    logging.debug("Loading {}".format(os.path.join( file_path, DEFAULT_LASER_PLANE_FILE)))
    laser_plane = Plane.read_json_file(os.path.join(file_path, DEFAULT_LASER_PLANE_FILE))   
    print (laser_plane)
    #initialize the laser processor
    processor = LaserProcessingServer (cam_params, laser_plane, cam)

# prevent cached responses
def add_header(r):
    """
    Add headers to both disable page caching.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

def create_app(test_config=None):
    """
    create and configure the app
    """
    application = Flask(__name__, instance_relative_config=True)
    application.config.from_mapping(
        SECRET_KEY='dev',
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        application.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        application.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(application.instance_path)
    except OSError:
        pass

    from . import app
    from . import controls
    from . import calibration
    from . import viewer
    from . import api
    application.register_blueprint(app.bp) 
    application.register_blueprint(controls.bp)
    application.register_blueprint(calibration.bp)
    application.register_blueprint(viewer.bp)
    application.register_blueprint(api.bp)
    
    application.before_first_request (partial (initialize, application.instance_path))
    application.after_request (add_header)
    return application
