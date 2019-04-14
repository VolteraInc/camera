"""
Run the camera/laser combo from the command line (rather than the web server.
"""
import argparse

from volteracamera.control.camera import Camera
from volteracamera.control.laser import Laser

if __name__ == "__main__":
    laser = Laser()
    cam = Camera()
    cam.open()
    cam.run()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--laserpower", help = "Laser power as a percentage", type=int, default=100)
    parser.add_argument("-s", "--sensorexposure", help = "Sensor exposure time in ms", type=int, default = 0)
    args = parser.parse_args()

    laser.power = args.laserpower
    cam.exposure = args.sensorexposure

    input("Press any key to exit.")

