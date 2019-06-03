from flask import Blueprint
from flask.json import JSONEncoder

from volteracamera.analysis.plane import encode_plane_settings
from volteracamera.analysis.undistort import encode_undistort_settings

bp = Blueprint ("api", __name__,  url_prefix="/api")

class CustomCalEncoder (JSONEncoder):
    """
    Class used to encode and decode plane and undistortion objects.
    """
    def default (self, obj):
        try:
            return encode_plane_settings(obj)
        except TypeError:
            pass
        try:
            return encode_undistort_settings(obj)
        except TypeError:
            pass
        return JSONEncoder.default(self, obj)
        
bp.json_encoder = CustomCalEncoder

from . import camera_processing, laser_processing