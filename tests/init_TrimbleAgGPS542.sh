#!/bin/sh
# Initialize the AgGPS Trimble 542 Rover on /dev/ttyUSB0

stty -F /dev/ttyS0 ispeed 38400
gpsd -nN /dev/ttyS0
