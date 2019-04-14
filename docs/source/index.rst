
Welcome to Voltera Laser Camera Documentation documentation!
============================================================

This document covers the details of how to use the Voltera Laser Camera python tools and libraries. There
are a variety of different components to this software package, including:

* A device server that runs on a Raspberry Pi compute module and provides both a system level SPI based communication
  channel for accessing xyz data from the server during normal use and a REST API for accessing full images and XYZ data from the 
  system during calibration.
* A set of classes and tools for accessing the server from a remote computer through the SPI (Serial Peripheral Interface) bus.
* A suite of browser based tools for calibrating the camera laser system.

.. toctree::
   :maxdepth: 2

   install
   rest_api
   spi
   modules 

   
..   :caption: Contents: 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
