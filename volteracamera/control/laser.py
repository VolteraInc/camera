"""
Class for controlling the laser power.
"""
import importlib
gpio_spec = importlib.util.find_spec("gpiozero")
gpio_found = gpio_spec is not None
if gpio_found:
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
        if gpio_found:
            self.laser = PWMLED(GPIO_PIN)
        self._current_power = 0;
        self.power = 100

    def state(self)->bool:
        """
        Query the laser state
        """
        if gpio_found:
            return self.laser.is_lit
        else:
            return False

    def on(self)->None:
        """
        Turn the laser on.
        """
        if gpio_found:
            self.laser.on()

    def off(self)->None:
        """
        Turn the laser on.
        """
        if gpio_found:
            self.laser.off()

    @property
    def power(self)->int:
        """
        Query the laser power %
        """
        return int(self._current_power * 100)
    
    @power.setter
    def power(self, percent_power: int)->None:
        """
        Set the power as a percentage. Raises a value error is the power
        is outside the expected range of 0-100%
        """
        if percent_power > 100:
            raise ValueError("Laser Power cannot be greater than 100%")
        if percent_power < 0:
            raise ValueError("Laser Power cannot be less than 0%")
        self._current_power = percent_power/100
        if gpio_found:
            self.laser.value = self._current_power

    
