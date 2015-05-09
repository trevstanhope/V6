#!/bin/sh
# Initialize the AgGPS Trimble 542 Rover on /dev/ttyUSB0

stty speed 38400 </dev/ttyUSB0
gpsd -nN /dev/ttyUSB0
