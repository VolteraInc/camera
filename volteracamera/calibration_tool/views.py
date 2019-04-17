from flask import render_template

from . import app

@app.route ('/')
@app.route ('/index')
@app.route ('/index.html')
def index ():
    return render_template ('/index.html')