from flask import jsonify
from . import bp
from .. import get_laser

@bp.route('/laser/power/<int:power>')
def set_laser_power(power):
    """
    Handle setting laser power.
    """
    error_message = ""
    success = True
    laser = get_laser()
    try:
        laser.power = power
    except ValueError:
        success = False
        error_message = "Laser power must be between 0 and 100."
    return jsonify({"power": laser.power, 
                    "is_on": laser.state(), 
                    "success": success,
                    "message":error_message})

@bp.route('/laser/power')
def get_laser_power():
    """
    Return the laser power.
    """
    laser = get_laser()
    return jsonify({"power": laser.power, 
                    "is_on": laser.state()})

@bp.route('/laser/power/on')
def power_laser_on():
    """
    Turn on the laser
    """
    laser = get_laser()
    laser.on()
    return jsonify({"power": laser.power, 
                    "is_on": laser.state(), 
                    "success": True,
                    "message": ""})

@bp.route('/laser/power/off')
def power_laser_off():
    """
    Turn off the laser
    """
    laser = get_laser()
    laser.off()
    return jsonify({"power": laser.power, 
                    "is_on": laser.state(), 
                    "success": True,
                    "message": ""})

        