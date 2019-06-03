from flask import render_template, request

from . import app


@app.route('/')
@app.route('/index')
@app.route('/index.html')
def index():
    return render_template('/index.html')


@app.route('/shutdown')
def shutdown():
    """
    Shut down the server.
    """
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return "The server is shutting down."
