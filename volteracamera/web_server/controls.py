from flask import Blueprint, render_template, send_file, request, jsonify

from . import get_cam, get_laser, get_data_store
from .data_model import ImageData

bp = Blueprint("controls", __name__, url_prefix="/controls")

#laser and camera state checks (prevent multiple initialization)
#routes
@bp.route('/preview')
def preview():
    """
    Render the preview page.
    """
    return render_template("controls/preview.html")

@bp.route('/controls')
def controls():
    """
    Render the controls page
    """
    return render_template("controls/controls.html")

#Requests for preview image data.
@bp.route('/preview_cam_image')
def preview_cam_image():
    cam = get_cam()
    image_stream = cam.capture_stream()
    return send_file(image_stream, mimetype='image/jpeg')

#Capture an image and store it in the request structure
@bp.route('/capture_proper_image')
def capture_proper_image():
    """
    Capture a full size image and store it to the local store.
    """
    cam = get_cam()
    image_array = cam.capture_array()
    data = get_data_store()
    data["images"].append(ImageData(image_array))
    return "OK"

#Requests for laser state. No data means query for state.
@bp.route('/laser_state', methods={"POST"})
def laser_state():
    """
    If data is recieved, set the laser values and return the new state.
    If no data is recieved, return the current state.
    state variables are "on_off" for the state (true or false) and "intensity" as a percent
    """
    laser = get_laser()
    req_data = request.get_json()
    if req_data != None:
        on_off_state = bool(req_data["on_off"])
        intensity = int(req_data["intensity"])
        if on_off_state: #if value is set to on, intensity jumps to 100%
            laser.on()
        try:
            print ("laser to " + str (intensity))
            laser.power = intensity
        except ValueError:
            print ("Bad laser power received.")
        if not on_off_state: #state switches to on when value changed.
            laser.off()
    lstate = {"on_off": laser.state(), "intensity": laser.power }
    return jsonify(lstate)

#Requests for sensor state. No data means query for state.
@bp.route('sensor_state', methods=["POST"])
def sensor_state():
    """
    Set the exposure time of the camera in ms.
    If no data sent, return current value.
    If data sent, update, and then send back updated value.
    """
    cam = get_cam()
    req_data = request.get_json()
    if req_data != None:
        print (req_data['exposure'])
        cam.exposure = int(req_data['exposure'])
    sstate = {'exposure': cam.exposure}
    return jsonify(sstate)
