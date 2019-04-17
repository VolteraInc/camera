from flask import Flask

def create_app (test_config=None):
    """
    Create the app.
    """
    application = Flask(__name__, instance_relative_config=True)
    application.config.from_mapping(
        SECRET_KEY="dev",
        CAMERA_CAL_FILE="camera.json",
        LASER_CAL_FILE="laser.json",
    )

    from . import api
    application.register_blueprint(api.bp)

    return application


app = create_app()

from . import views