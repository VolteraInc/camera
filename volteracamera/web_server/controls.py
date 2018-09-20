from flask import Blueprint, render_template, send_file, current_app, request, jsonify

from ..control.camera import Camera
from ..control.laser import Laser

bp = Blueprint("controls", __name__, url_prefix="/controls")

cam = None
laser = None

#laser and camera state checks (prevent multiple initialization)
def check_cam():
    global cam
    if cam == None:
        cam = Camera()
        cam.open()

def check_laser():
    global laser
    if laser == None:
        laser = Laser()


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

#Requests for image data.
@bp.route('/cam_image')
def cam_image():
    global cam
    check_cam()
    image_stream = cam.capture_stream()
    return send_file(image_stream, mimetype='image/png')

#Requests for laser state. No data means query for state.
@bp.route('/laser_state', methods={"POST"})
def laser_state():
    """
    If data is recieved, set the laser values and return the new state.
    If no data is recieved, return the current state.
    state variables are "on_off" for the state (true or false) and "intensity" as a percent
    """
    global laser
    check_laser()
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
    check_sensor()
    global cam
    pass
