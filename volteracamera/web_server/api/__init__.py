from flask import Blueprint

bp = Blueprint ('api', __name__)

from . import laser_control, sensor_control, calibration
