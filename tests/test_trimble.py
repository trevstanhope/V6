"""
First, execute the following commands to initialize GPSD:
    sudo stty speed 38400 </dev/ttyUSB0
    sudo gpsd -nN /dev/ttyUSB0
"""
import gps as gpsd
import time

trimble = gpsd.gps()
trimble.stream(gpsd.WATCH_ENABLE)
for report in trimble:
    print report
    lon = trimble.fix.longitude
    lat = trimble.fix.latitude
    print lat, lon
    time.sleep(0.1)
