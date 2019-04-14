from flask import Blueprint

bp = Blueprint ("api", __name__,  url_prefix="/api")

from . import laser_control, sensor_control, calibration
