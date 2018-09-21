from flask import Blueprint, render_template

bp = Blueprint ("calibration", __name__, url_prefix="/calibration")

#main route for calibration.
@bp.route('index')
def index():
    return render_template("calibration/index.html")

@bp.route('intrinsics_generation')
def intrinsics_generate(methods=["POST","GET"]):
    return "Not Implemented."
    
@bp.route('intrinsics_analyze')
def intrinsics_analyze():
    return "Not Implemented."

@bp.route('laser_analyze')
def laser_analyze():
    return "Not Implented."
