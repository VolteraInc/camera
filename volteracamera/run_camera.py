"""
Run the camera/laser combo from the command line (rather than the web server.
"""
import argparse

from .control.camera import Camera
from .control.laser import Laser


laser = Laser()
cam = Camera()
cam.open()
cam.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--laserpower", help = "Laser power as a percentage", type=int, default=100)
    parser.add_argument("-s", "--sensorexposure", help = "Sensor exposure time in ms", type=int, default = 0)
    args = parser.parse_args()

    laser.power(args.laserpower)
    sensor.exposure(args.sensorexposure)

    raw_input("Press any key to exit.")

