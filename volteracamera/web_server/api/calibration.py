from flask import jsonify, request
from . import bp
from volteracamera.analysis.plane import Plane, decode_plane_settings
from volteracamera.analysis.undistort import Undistort, decode_undistort_settings
import json

LASER_CAL_FILE = "~/laser.xml"
SENSOR_CAL_FILE = "~/sensor.xml"

@bp.route("laser/calibration", methods=["GET", "POST"])
def laser_calibration():
    """
    Method for getting and setting the laser cal file.
    """
    success = True
    message = ""
    if request.method == "POST":
        cal_file = request.json
        try:
            laser_cal = json.loads(cal_file, JSONDecoder=decode_plane_settings)
            laser_cal.write_file(LASER_CAL_FILE)
        except:
            success = False
            message = "Failed to parse passed laser calibration."
        return jsonify ({"success": success,
                         "message": message})
    else:
        try:
            plane = Plane.read_json_file(LASER_CAL_FILE)
        except:
            plane = None
            success = False
            message = "Failed to load laser xml file."
        return jsonify ({"laser_cal_file": plane,
                         "success": success,
                         "message": message})

@bp.route("sensor/calibration", methods=["GET", "POST"])
def sensor_calibration():
    """
    Method for getting and setting the sensor cal file.
    """
    success = True
    message = ""
    if request.method == "POST":
        cal_file = request.json
        try:
            sensor_cal = json.loads(cal_file, JSONDecoder=decode_sensor_settings)
            sensor_cal.write_file(SENSOR_CAL_FILE)
        except:
            success = False
            message = "Failed to parse passed sensor calibration."
        return jsonify ({"success": success,
                         "message": message})
    else:
        try:
            sensor_cal = Undistort.read_json_file(SENSOR_CAL_FILE)
        except:
            sensor_cal = None
            success = False
            message = "Failed to load sensor xml file."
        return jsonify ({"sensor_cal_file": sensor_cal,
                         "success": success,
                         "message": message})

