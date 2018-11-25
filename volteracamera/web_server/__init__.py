import os

from flask import Flask
from ..control.camera import Camera
from ..control.laser import Laser

#initialize global camera and laser objects.
cam = None
laser = None
data = {
        "images": [],
        "intrinsics": None,
        "distortion": None,
        "laser_plane":None
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

def get_data_store():
    """
    Access the global data storage.
    """
    global data
    return data

def initialize():
    #Instantiate and start the camera
    cam = get_cam()
    cam.run()
    #Intantiate and start the laser
    laser = get_laser()

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
    application.register_blueprint(app.bp) 
    application.register_blueprint(controls.bp)
    application.register_blueprint(calibration.bp)
    application.register_blueprint(viewer.bp)

    application.before_first_request (initialize)
    application.after_request (add_header)
    return application
