*********
REST API
*********

This page outlines the details of the REST API that allows Voltera technicians to access and control the camera and laser directly.

Overview
========

The Raspberry Pi server by default starts a limited web server that listens for HTTP requests and serves those requests. Through 
this interface, the user can:

* capture individual images 
* capture individual profiles of XYZ data
* set or query the laser power
* set or query the camera exposure time
* upload calibrtion files
* download calibration files

All responses from the server return JSON responses, with the exception of retrieving the image captured by the camera (the details of
which are described below.

Commands
========

Get Laser Power
---------------

.. list-table:: Get Laser Power
   :widths: 15 5 70 10
   :header-rows: 1

   * - Resource URL
     - Method
     - Response
     - Code
   * - /api/laser/power
     - GET 
     - | {
       |  "power": <int>,
       |  "is_on": <bool>
       | }
     - 200

This command reads the current laser power. It returns an object containing the current power in percent and state of the laser (on True).

Set Laser Power
---------------

.. list-table:: Set Laser Power
   :widths:  15 5 70 10
   :header-rows: 1

   * - Resource URL
     - Method
     - Response
     - Code
   * - /api/laser/power/<power>
     - GET 
     - | {
       |  "power": <int>, 
       |  "is_on": <bool>,
       |  "success": <bool>,
       |  "message": <string>
       | }
     - 200
   
This command sets the laser power to the integer value passed in the url as a percentage. In all cases, the current laser power and state are
returned. In the case of the set being successful, the success flag is true and the message field is empty. If the power set was not 
successful, the success field if false and the error message is returned in the message field.

Turn Laser On
---------------

.. list-table:: Turn Laser On
   :widths: 15 5 70 10
   :header-rows: 1

   * - Resource URL
     - Method
     - Response
     - Code
   * - /api/laser/on
     - GET 
     - | {
       |  "power": <int>, 
       |  "is_on": <bool>,
       |  "success": <bool>,
       |  "message": <string>
       | }
     - 200

This command turns the laser on. In all cases the power and current laser state are returned. If the laser was not turned on, the success
field is false and an appropriate message is returned.

Turn Laser Off
---------------

.. list-table:: Turn Laser Off
   :widths: 15 5 70 10
   :header-rows: 1

   * - Resource URL
     - Method
     - Response
     - Code
   * - /api/laser/off
     - GET 
     - | {
       |  "power": <int>, 
       |  "is_on": <bool>,
       |  "success": <bool>,
       |  "message": <string>
       | }
     - 200

This command turns the laser off. In all cases the power and current laser state are returned. If the laser was not turned on, the success
field is false and an appropriate message is returned.

Get Sensor Exposure
-------------------

.. list-table:: Get Sensor Exposure
   :widths: 15 5 70 10
   :header-rows: 1

   * - Resource URL
     - Method
     - Response
     - Code
   * - /api/sensor/exposure
     - GET 
     - | {
       |  "exposure": <int>,
       |  "unit": <string>
       | }
     - 200

This command reads the current sensor exposure time. It returns an object containing the current exposure time and unit (normally ms).

Set Sensor Exposure
-------------------

.. list-table:: Set Sensor Exposure
   :widths: 15 5 70 10
   :header-rows: 1

   * - Resource URL
     - Method
     - Response
     - Code
   * - /api/sensor/exposure/<time in ms>
     - GET 
     - | {
       |  "exposure": <int>,
       |  "unit": <string>,
       |  "success": <bool>,
       |  "message": <string>
       | }
     - 200

This command sets the current sensor exposure time. It returns an object containing the current exposure time and unit (normally ms) and a success and 
message field. If success is true, the exposure time is the value that was requested and if it is false, the message field details the error.

Capture Image
-------------

.. list-table:: Capture Image
   :widths: 15 5 70 10
   :header-rows: 1

   * - Resource URL
     - Method
     - Response
     - Code
   * - /api/sensor/capture
     - GET 
     - | {
       |  "url": <string>,
       |  "time": <double>,
       |  "success": <bool>,
       |  "message": <string>
       | }
     - 200

This command triggers the server to capture and hold the next captured image in memory so that it can be retrieved from the url 
given in the url field. The time field contains the capture time in the servers clock in seconds from the unix epoch. If the 
capture was unsuccesful, the success field is false and the message details the reasons for the failure.

This message will block until the image is ready to retrieve from the given url.

Capture Profile
---------------

.. list-table:: Capture Profile
   :widths: 15 5 70 10
   :header-rows: 1

   * - Resource URL
     - Method
     - Response
     - Code
   * - /api/profile
     - GET 
     - | {
       |  "points": [ ...,
       |            {x: <double>, y: <double>, z: <double>, intensity: int},
       |            ... ],
       |  "time": <double>,
       |  "success": <bool>,
       |  "message": <string>
       | }
     - 200

This command triggers the server to capture and return an array of points captured from the sensor. The time field contains the capture time in the servers clock in seconds from the unix epoch. If the 
capture was unsuccesful, the success field is false and the message details the reasons for the failure.

Set Laser Calibration 
---------------------

.. list-table:: Set Laser Calibration
   :widths: 15 5 70 10
   :header-rows: 1

   * - Resource URL
     - Method
     - Response
     - Code
   * - /api/laser/calibration
     - POST 
     - | {
       |  "success": <bool>
       |  "message": <string>
       | }
     - 201

This method is used to load a laser calibration file onto the scanner. The contents of the laser calibration file are loaded onto the 
server as a JSON post. The server responds with the success or failure of the load. The uploaded file is persistently stored on the
hard drive of the system.

Here is an example laser calibration payload as json (generated by the Plane class.:

.. code-block:: json

  {
    "__plane__": true,
    "normal": [
        0.873959239096664,
        -0.48476359921105544,
        -0.03463381696439748
    ],
    "point": [
        0.0,
        0.0,
        -0.06892279036809151
    ]
  }

Get Laser Calibration 
---------------------

.. list-table:: Get Laser Calibration
   :widths: 15 5 70 10
   :header-rows: 1

   * - Resource URL
     - Method
     - Response
     - Code
   * - /api/laser/calibration
     - GET 
     - | {
       |  "__plane__": true,
       |  "normal": <3 point array of doubles>,
       |  "point": <3 point array of doubles>
       | }
     - 200

This method returns the currently loaded laser calibration (which can be made directly into a Plane object).
     
Set Sensor Calibration 
----------------------
 
.. list-table:: Set Sensor Calibration
   :widths: 15 5 70 10
   :header-rows: 1

   * - Resource URL
     - Method
     - Response
     - Code
   * - /api/sensor/calibration
     - POST 
     - | {
       |  "success": <bool>
       |  "message": <string>
       | }
     - 201

This method is used to load a sensor calibration file onto the scanner. The contents of the sensor calibration file are loaded onto the 
server as a JSON post. The server responds with the success or failure of the load. The uploaded file is persistently stored on the
hard drive of the system.

Here is an example of a sensor json payload (generated by the Undistort class):

.. code-block:: json

  {
    "__undistort__": true,
    "camera_matrix": [
        [
            343.48482932426873,
            0.0,
            605.3938703519042
        ],
        [
            0.0,
            345.2203242582044,
            444.44739246975746
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ],
    "distortion": [
        -0.003609073951994043,
        -0.002719750867048638,
        -0.0034879480586716876,
        0.002790665977824752,
        0.0007516470987650315
    ]
  }

Get Sensor Calibration 
----------------------
 
.. list-table:: Get Sensor Calibration
   :widths: 15 5 70 10
   :header-rows: 1

   * - Resource URL
     - Method
     - Response
     - Code
   * - /api/sensor/calibration
     - GET 
     - | {
       |  "__undistort__": True,
       |  "camera_matrix": <3x3 array of doubles>,
       |  "distortion": <5x1 array of doubles>
       | }
     - 200

This method returns a undistortion object representing the currently loaded sensor calibration.

Errors
======

In the event of a request being made to a resource that doesn't exists, the server will return a 404 status with the details described in a 
json message containing a single message field.