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




