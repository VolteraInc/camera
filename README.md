# Voltera Camera Python Module

This project contains python code associated with calibrating and using the Voltera Camera module.

## Setup the virtual environment
To set up the virtual environment to run this library in:

``` bash
python3 -m venv venv
source venv/bin/activate
pip install -r Requirements.txt
```

This will fail on non Raspberry Pi machines, but you can still test the software (uninstall the gpiozero library first, however, pip uninstall gpiozero)

## Running the Web Server

To run the web interface, the software must be transferred to a Raspberry Pi. 

Transfer this project over the the pi:

``` bash
export VOLTERA_CAMERA_PI_IP=<address of pi on network>
make copy-to-server
```

ssh onto the Pi, and set the virtual environment using commands in the first section. The software was copied into the /home/pi/camera directory.

To run the server, type 

``` bash
make run
```

Point your browser to <the Raspberry Pi's ip address>:5000.

