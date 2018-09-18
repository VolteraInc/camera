from flask import Blueprint, render_template, send_file, current_app
from ..control.camera import Camera

bp = Blueprint("controls", __name__, url_prefix="/controls")

cam = None

#routes
@bp.route('/preview')
def preview():
    return render_template("controls/preview.html")

@bp.route('/controls')
def controls():
    return render_template("controls/controls.html")

#Requests for data.
@bp.route('/cam_image')
def cam_image():
    global cam
    if cam == None:
        cam = Camera()
        cam.open()
    image_stream = cam.capture_stream()
    return send_file(image_stream, mimetype='image/png')
