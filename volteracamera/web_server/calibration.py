"""
Methods for serving the calibration pages.
"""
from flask import ( Blueprint, 
                    render_template,
                    send_file,
                    make_response,
                    url_for )
from io import BytesIO
from PIL import Image
from ..intrinsics.calibration_generation import generate_random_cal_images

bp = Blueprint ("calibration", __name__, url_prefix="/calibration")

cal_images=None

def get_cal_images():
    """
    return the global cal images
    """
    global cal_images
    if not cal_images:
        from ..intrinsics.circles_calibration import IMG
        input_cal_image = IMG
        cal_images = generate_random_cal_images (input_cal_image, 25)
    return cal_images

#-----------------------------------
#Routes
#-----------------------------------

#main route for calibration.
@bp.route('index')
def index():
    return render_template("calibration/index.html")

@bp.route('intrinsics_generation/<num>')
def intrinsics_pages(num):
    """
    Display the next intrinsics images.
    """
    cal_images = get_cal_images()
    if (int(num) >= len(cal_images)):
        return make_response ("Not enough calibration images", 404)
    return render_template("calibration/intrinsics.html", 
                            image_url=url_for('calibration.intrinsics_images', num=num), 
                            previous_url = None if int(num)==0 else url_for('calibration.intrinsics_pages', num=int(num)-1), 
                            next_url = None if int(num) >= len(cal_images)-1 else url_for('calibration.intrinsics_pages', num=int(num)+1) )

@bp.route('intrinsics_generation/image/<num>')
def intrinsics_images(num):
    """
    Return the intrinsics image requested.
    """
    cal_images = get_cal_images()
    if (int(num) >= len(cal_images)):
        return make_response ("Not enough calibration images", 404)
    image_stream = BytesIO()
    print (cal_images[int(num)])  
    image = Image.fromarray(cal_images[int(num)].astype('uint8'), "L")
    image.save(image_stream, format='jpeg')
    image_stream.seek(0)
    return send_file (image_stream, mimetype="image/jpeg")

    
@bp.route('intrinsics_analyze')
def intrinsics_analyze():
    return "Not Implemented."

@bp.route('laser_analyze')
def laser_analyze():
    return "Not Implented."
