from flask import Blueprint, render_template

bp = Blueprint("main", __name__, "/")

# a simple page that says hello
@bp.route('/')
def index():
    return render_template("index.html")


