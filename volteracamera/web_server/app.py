from flask import Blueprint, render_template, session, jsonify, url_for
from io import BytesIO

from . import get_data_store

bp = Blueprint("main", __name__, "/")

# a simple page that says hello
@bp.route('/')
def index():
    return render_template("index.html")

@bp.route('/load_sidebar')
def load_sidebar():
    """
    Method that loads the sidebar and return a json bobject with the locations of images and thumbnails
    """
    image_data = get_data_store()["images"]
    json_data = {"images": []}
    for i, _ in enumerate (image_data):
        json_data["images"].append({
                            "thumbnail": url_for('.capture_image_thumb', num=str(i)),
                            "image": url_for('.capture_image_full', num=str(i))
                        })
    print (json_data)
    return jsonify (json_data)

@bp.route("/capture_image_thumb/<num>")
def capture_image_thumb(num):
    """
    return the capture image thumbnail at location num
    """
    image_data = get_data_store()["images"]
    if len(image_data) >= int(num):
        return make_respoonse ("Not enough images available", 404)

    image_stream = BytesIO
    image_data[int(num)].thumbnail.save(image_stream, format='png')       
    return send_file(image_stream, mimetype='image/png')

@bp.route("/capture_image_full/<num>")
def capture_image_full(num):
    """
    return the full size captured image at location num
    """
    image_data = get_data_store()["images"] 
    if len(image_data) >= int(num):
        return make_respoonse ("Not enough images available", 404)

    image_stream = BytesIO
    image_data[int(num)].image.save(image_stream, format='png')       
    return send_file(image_stream, mimetype='image/png')


