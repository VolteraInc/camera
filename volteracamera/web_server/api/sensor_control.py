from flask import jsonify, url_for, send_file
from collections import OrderedDict
import random 
import string 
import datetime
from . import bp
from .errors import bad_request
from .. import get_cam
from volteracamera.control.camera import CameraReader
from io import BytesIO

@bp.route('/sensor/exposure')
def get_sensor_exposure():
    """
    Return the sensor exposure.
    """
    cam = get_cam()
    return jsonify ({"exposure": cam.exposure,
                     "unit": "ms"})

@bp.route('/sensor/exposure/<int:exposure>')
def set_sensor_exposure(exposure):
    """
    Set the camera exposure.
    """
    cam = get_cam()
    success = True
    message = ""
    if exposure < 0:
        success = False
        message = "Sensor exposure must be positive."
    cam.exposure = exposure
    return jsonify ({"exposure": cam.exposure,
                     "unit": "ms",
                     "success": success,
                     "message": message})
  
# defining function for random 
# string id with parameter 
def ran_gen(size, chars=string.ascii_uppercase + string.digits): 
    return ''.join(random.choice(chars) for x in range(size)) 

MAX_IMAGE_STORAGE_DEPTH=10
image_list = OrderedDict()

@bp.route('/sensor/capture')
def start_capture():
    """
    Capture an image and return a json message with the url pointing to the address to download it.
    """
    cam = CameraReader()
    key = ran_gen(12)
    success = True
    message = ""
    try:
        image_list[key] = cam.capture()
        if len(image_list) > MAX_IMAGE_STORAGE_DEPTH:
            _ = image_list.popitem()
    except:
        success = False
        message = "Failed to capture image."
        key = ""
        
    return jsonify({"url": url_for("get_image", id=key),
                    "time": (datetime.datetime.now() - datetime.datetime(1970, 1, 1)).total_seconds(),
                    "success": success,
                    "message": message})

@bp.route('/sensor/capture/<str:id>')
def get_image(id):
    """
    Return the image requested by id. The start_capture must be called first. After the file is retrieved, it is cleared immediately.
    """
    global image_list
    try:
        image_array = image_list.pop(id)
    except:
        return bad_request("Image {} does not exist.".format(id))
    image = CameraReader.array_to_image(image_array)
    image_stream = BytesIO()
    image.save (image_stream, format="jpeg")
    image_stream.seek(0)
    return send_file(image_stream, mimetype='image/jpeg')
    
