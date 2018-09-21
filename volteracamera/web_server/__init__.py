import os

from flask import Flask
from ..control.camera import Camera
from ..control.laser import Laser

#initialize global camera and laser objects.
cam = None
laser = None

#access the global camera object 
def get_cam():
    global cam
    if cam == None:
        cam = Camera()
        cam.open()
    return cam

#access the global laser object.
def get_laser():
    global laser
    if laser == None:
        laser = Laser()
    return laser



def create_app(test_config=None):
    # create and configure the app
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
    application.register_blueprint(app.bp) 
    application.register_blueprint(controls.bp)
    application.register_blueprint(calibration.bp)

    return application
