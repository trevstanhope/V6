"""
Capture video stream and generate text log of exact frame time
Trevor Stanhope
"""

import cv, cv2
import numpy as np
import time
from datetime import datetime
import os
import serial
import sys
import thread

class VideoRTK:
    def __init__(self):

        # Constants
        index = 0
        terrain = raw_input("Enter terrain: ")
        file_dir = "data"
        vid_type = ".avi"
        vid_date = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
        vid_file = "%s_%s%s" % (vid_date, terrain, vid_type)
        vid_path = os.path.join(file_dir, vid_file)
        log_type = '.csv'
        log_file = "%s %s%s" % (vid_date, terrain, log_type)
        log_path = os.path.join(file_dir, log_file)
        fps = 25.0
        frame_size = (640, 480) 

        gps_device="/dev/ttyS0"
        gps_baud=38400

        # Initialize Video Capture
        try:
            self.camera = cv2.VideoCapture(index)
            print 'cam initialized'
        except:
            print 'Cam failed'

        # Initialize Video Writer
        try:
            self.vid_writer = cv2.VideoWriter(vid_path, cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, frame_size, True)
            print "video initialized"
        except:
            print 'Video failed'

        # Initialize CSV file
        try:
            self.csvfile = open(log_path, 'w')
        except:
            print 'CSV failed'

        # Initialize GPS
        try:
            self.lat = 0.0
            self.lon = 0.0
            self.alt = 0.0
            self.speed = 0.0    
            self.gps = serial.Serial(gps_device, gps_baud)
            thread.start_new_thread(self.update_gps, ())
            print 'gps initialized'
        except Exception as e:
            print 'GPS failed %s' % str(e)

    # Loop until keyboard interrupt
    def run(self):
        a = time.time()
        try:
            while self.vid_writer.isOpened():
                (s, bgr) = self.camera.read()
                if s:
                    t = time.time()
                    print self.speed, t - a 
                    dt = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S.%f")
                    newline = ','.join([dt, str(t), str(self.lat), str(self.lon), str(self.alt), str(self.speed), '\n'])
                    self.csvfile.write(newline)
                    self.vid_writer.write(bgr)
        except KeyboardInterrupt:
            self.camera.release()
            self.vid_writer.release()
        except Exception as e:
            print str(e)

    def update_gps(self, verbose=False):
        while True:
            try:
                sentence = self.gps.readline()
                #print sentence
                sentence_parsed = sentence.rsplit(',')
                nmea_type = sentence_parsed[0]
                if verbose: print sentence
                if nmea_type == '$GPVTG':
                    self.speed = float(sentence_parsed[7])
                elif nmea_type == '$GPGGA':
                    self.lat = float(sentence_parsed[2])
                    self.lon = float(sentence_parsed[4])
                    self.alt = float(sentence_parsed[9])
            except Exception as e:
                self.lat = 0.0
                self.lon = 0.0
                self.alt = 0.0
                self.speed = 0.0
                if verbose: pretty_print("GPS", str(e))

if __name__ == '__main__':
    app = VideoRTK()
    app.run()
