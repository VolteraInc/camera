"""
Class for controlling the laser power.
"""
from gpiozero import PWMLED

GPIO_PIN=4

class Laser(object):
    """
    Class that uses PWM to control the laser.
    """

    def __init__(self)->None:
        """
        Initialization
        """
        self.laser = PWMLED(GPIO_PIN)
        self.current_power = 1

    def on(self)->None:
        """
        Turn the laser on.
        """
        self.laser.on()

    def off(self)->None:
        """
        Turn the laser on.
        """
        self.laser.off()

    def power(self, percent_power: int)->None:
        """
        Set the power as a percentage. Raises a value error is the power
        is outside the expected range of 0-100%
        """
        if percent_power > 100:
            raise ValueError("Laser Power cannot be greater than 100%")
        if percent_power < 0:
            raise ValueError("Laser Power cannot be less than 0%")

        self.current_power = percent_power/100
        self.laser.value = self.current_power
