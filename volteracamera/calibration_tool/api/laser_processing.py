from flask import jsonify, session, request

from . import bp
from volteracamera.analysis.plane import Plane

@bp.route('/laser_calibration', methods=["GET", "POST"])
def laser_calibration():
    """
    return a plane calibration file in json.
    """
    if request.method == "GET":
        if 'laser_plane' not in session:
            session['laser_plane'] = Plane()
        return jsonify (session['laser_plane'])
    else:
        json_object = request.get_json()
        if "__plane__" in json_object and json_object['__plane__']:
            session['laser_plane'] = json_object
        return jsonify ({"success": True})

@bp.route('/find_laser_points', methods=["POST"])
def find_laser_points():
    pass

@bp.route('/parse_laser_height', methods=["POST"])
def find_laser_position ():
    """
    Parse the image filename and return the position.
    """
    json_object = request.get_json()
    if "filename" not in json_object:
        return jsonify ({"success": False, "message": "No filename with height sent.", "position":[]})

    file_path = Path(json_object["filename"])
    file_stem = file_path.stem

    pos = float(re.findall(r'(\d+(?:\.\d+)?)', file_stem)[1])

    return jsonify ({"success": True, "message": "", "height":pos})


