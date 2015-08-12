"""
V6 - Vision Speed Inference Extension

Runs as a ZMQ client.

The main function, run(), takes into account that the algorithm is slower than the 
camera framerate.
"""

__author__ = 'Trevor Stanhope'
__version__ = '0.1'

import scipy.cluster.hierarchy as hcluster
import pango
import shutil
import cv2, cv
import numpy as np
import time
import sys
from matplotlib import pyplot as plt
from itertools import cycle
import zmq
import json
from datetime import datetime
import gps as gpsd
import os
import thread
import serial
import pygtk
pygtk.require('2.0')
import gtk

# Useful Functions 
def pretty_print(task, msg):
    date = datetime.strftime(datetime.now(), '[%d/%b/%Y:%H:%M:%S]')
    print("%s %s %s" % (date, task, msg))

class V6:

    """
    Initialize
    Optional Arguments:
        capture :
        fov : in degrees
        d : depth from surface (meters)
        roll : 
        pitch :
        yaw : 
        hessian : iterations of hessian filter
        frame
    """
    def __init__(self, 
        capture=0,
        fov=0.75,
        f=6,
        cropfactor=1.125,
        d=1000,
        roll=0,
        pitch=0,
        yaw=0,
        hessian=1000,
        w=640,
        h=480,
        neighbors=2,
        factor=0.75
    ):
        
        # Things which should be set once
        pretty_print("CV6", "Initializing capture ...")
        try:
            if type(capture) is int:
                pass
            elif capture.isdigit():
                capture = int(capture)
        except Exception as e:
            print str(e)
            raise Exception
        self.camera = cv2.VideoCapture(capture)
        pretty_print("CV6", "Stream setup successful")
        
        # Things which can be changed at any time
        try:
            self.set_matchfactor(factor)
            self.set_resolution(w, h)
            self.set_fov(fov) # set the field of view (horizontal)
            self.set_cropfactor(cropfactor)
            self.set_focal_length(f)
            self.set_pitch(pitch) # 0 rad
            self.set_roll(roll) # 0 rad
            self.set_yaw(yaw) # 0 rad
            self.set_depth(d) # camera distance at center
            self.set_neighbors(neighbors)
            self.set_matcher(hessian)
        except Exception as e:
            print str(e)
        pretty_print("CV6", "Initialization successful")
        
    """
    Set the keypoint matcher configuration, supports BF or FLANN
    """
    def set_matcher(self, hessian, use_flann=False):
        try:
            self.keypoint_filter = cv2.SURF(hessian, nOctaves=3, nOctaveLayers=2, extended=1, upright=1)
            # Use the FLANN matcher
            if use_flann:
                self.FLANN_INDEX_KDTREE = 1
                self.FLANN_TREES = 5
                self.FLANN_CHECKS = 50
                self.INDEX_PARAMS = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=self.FLANN_TREES)
                self.SEARCH_PARAMS = dict(checks=self.FLANN_CHECKS) # or pass empty dictionary
                self.matcher = cv2.FlannBasedMatcher(self.INDEX_PARAMS, self.SEARCH_PARAMS)
                pretty_print("CV6", "Using FLANN matcher")
            # Use the Brute Force matcher
            else:
                pretty_print("CV6", "Using BruteForce matcher")
                self.matcher = cv2.BFMatcher()
        except Exception as e:
            raise Exception("Failed to generate a matcher")
    
    """
    Close
    """  
    def close(self):
        try:
            self.camera.release()
        except:
            raise Exception("Camera failed to close properly")
    
    """
    Set Match Factor
    """
    def set_matchfactor(self, factor):
        if factor < 0:
            raise Exception("Cannot have match less than 1")
        else:
            self.factor = factor
            pretty_print("CV6", "Match factor of %f" % self.factor)
            
    """
    Set Neighbors
    """
    def set_neighbors(self, neighbors):
        if neighbors < 2:
            raise Exception("Cannot have neighbors less than 2")
        else:
            self.neighbors = neighbors
            pretty_print("CV6", "Using %d neighbors" % self.neighbors)
            
    """
    Set Resolution
    w : image width [px]
    h : image height [px]
    """
    def set_resolution(self, w, h):
        if (w <= 0) or (h <= 0):
            raise Exception("Improper frame size")
        else:
            self.w = w
            self.h = h
            self.camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, w)
            self.camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, h)
            pretty_print("CV6", "Camera dimensions of %dpx by %dpx" % (self.w, self.h))
    
    """
    Set distance at center of frame
    d : depth of view [m]
    """
    def set_depth(self, d):
        if d <= 0:
            raise Exception("Improper distance")
        else:
            self.d = d
            pretty_print("CV6", "Camera depth of %dmm" % (self.d))

    """
    Set Pitch
    pitch [rad]
    0 is orthogonal to surface 
    """
    def set_pitch(self, pitch):
        if pitch < 0:
            raise Exception("Cannot have negative inclination")
        elif pitch > np.pi/2.0:
            raise Exception("Cannot have inclination parallel to surface")
        else:
            self.pitch = pitch
            pretty_print("CV6", "Camera Pitch of %f radians" % (self.pitch))

    """
    Set Roll
    roll [rad]
    0 is parallel to surface 
    """
    def set_roll(self, roll):
        self.roll = roll
        pretty_print("CV6", "Camera Roll of %f radians" % (self.roll))

    """
    Set Yaw (relative to direction of travel)
    yaw [rad]
    0 is parallel to direction of travel 
    """
    def set_yaw(self, yaw):
        self.yaw = yaw
        pretty_print("CV6", "Camera Yaw of %f radians" % (self.yaw))
    
    """
    Set Field-of-View
    fov [rad]
    """
    def set_fov(self, fov):
        if fov <= 0:
            raise Exception("Cannot have negative FOV")
        if fov >= np.pi:
            raise Exception("Cannot have FOV greater than Pi radians")
        else:
            self.fov = fov
            pretty_print("CV6", "Field-of-View of %f radians" % (self.fov))

    """
    Set cropfactor Ratio
    cropfactor [constant]
    """
    def set_cropfactor(self, cropfactor):
        if cropfactor <= 0:
            raise Exception("Cannot have negative cropfactor ratio")
        else:
            self.cropfactor = cropfactor
            pretty_print("CV6", "cropfactor ratio of %f" % (self.cropfactor))

    """
    Set Focal Length
    f [mm]
    """
    def set_focal_length(self, f):
        if f <= 0:
            raise Exception("Cannot have negative focal length")
        else:
            self.f = f
            pretty_print("CV6", "Focal length of %d mm" % (self.f))
                
    """
    Match Images
    Find (good) pairs of matching points between two images
    Returns: [ (pt1, pt2), ... ]
    """
    def match_images(self, bgr1, bgr2, target_pairs=64, upper_lim=10000, lower_lim=100, gain=10):
        gray1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
        (pts1, desc1) = self.keypoint_filter.detectAndCompute(gray1, None)
        (pts2, desc2) = self.keypoint_filter.detectAndCompute(gray2, None)
        
        matching_pairs = []
        if pts1 and pts2:
            all_matches = self.matcher.knnMatch(desc1, desc2, k=self.neighbors)
            for (m,n) in all_matches:
                try:
                    if m.distance < self.factor * n.distance:
                        pt1 = pts1[m.queryIdx]
                        pt2 = pts2[m.trainIdx]
                        xy1 = (pt1.pt[0], pt1.pt[1])
                        xy2 = (pt2.pt[0], pt2.pt[1])
                        matching_pairs.append((xy1, xy2))
                except Exception as e:
                    pretty_print("CV6", "Missing match")
            e = len(matching_pairs) - target_pairs
            self.keypoint_filter.hessianThreshold += gain * e
            if self.keypoint_filter.hessianThreshold < lower_lim: self.keypoint_filter.hessianThreshold = lower_lim
            if self.keypoint_filter.hessianThreshold > upper_lim: self.keypoint_filter.hessianThreshold = upper_lim
            return matching_pairs

    """
    Cartesian to Polar
    """
    def cart2pol(self, x, y):
        """ Cartesian to Polar """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
        
    # Distance
    def vector(self, pt1, pt2, project=False):
        """
        Distance between two keypoints, where keypoints are in units of pixels
        Arguments: pt1 : (int x1, int y1), pt2 : (int x2, int y2)
        Returns: distance : float
        """
        (x1, y1) = pt1
        (x2, y2) = pt2
        if project:
            (x1, y1) = self.project(x1, y1)
            (x2, y2) = self.project(x2, y2)
        dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
        theta = np.arctan(float(y2 - y1) / float(x2 - x1))
        return dist, theta
        
    """
    Project points from pixels to real units
    Required arguments:
        x : 
        y : 
    Optional arguments:
        h : height of camera from surface (any units)
        fov : horizontal field of view
        w : width of image (in pixels)
        pitch : angle of inclination (in radians)
        f : focal length
    Returns:
        (X, Y): point location
    """
    def project(self, x, y, rotated=True):
        f = 0.69 #(2.0 / self.cropfactor) * np.tan(self.fov / (2.0))
        if rotated:
            l = self.w / f
            theta = np.arctan(y / l)
            Y = self.d / np.tan( (np.pi / 2.0 - self.pitch) - theta)
            X = x * np.sqrt( (self.d**2 + Y**2) / (l**2 + y**2) )
        else:
            l = self.w / f
            theta = np.arctan(y / l)
            Y = self.d / np.tan( (np.pi / 2.0 - self.pitch) - theta)
            X = x * np.sqrt( (self.d**2 + Y**2) / (l**2 + y**2) )
        return (X, Y)
    
    """
    Optional:
        dt : time differential between bgr1 and bgr2
        p_min : minimum percentile for change in distance
        p_max : maximum percentile for change in distance
        output_units: either 'mph', 
    Returns:
        v : the estimated speed of travel
        t : the estimated angle moved between two keypoints
        pairs : matching pairs between bgr1 and bgr2
        bgr1 : the first image
        bgr2 : the second image
        
    """
    def estimate_vector(
        self,
        bgr1, 
        bgr2,
        diff,
        fps=None,
        output_units="kilometers",
        dilation = 1.0
    ):

        # Read first
        gps_data = [str(g) for g in [self.lat, self.lon, self.alt, self.speed]]
        
        # If no fps specificed:
        if not fps:
            self.dt.reverse()
            self.dt.pop()
            self.dt.reverse()
            self.dt.append(diff)
            dt = np.mean(self.dt) * dilation
            if dt<0:
                raise Exception("Negative time differential!")
        else:
            dt = 1/float(fps)
        
        # Match keypoint pairs
        try:
            pairs = self.match_images(bgr1, bgr2)
        except Exception as e:
            raise e

        # Convert units
        try:
            vectors = [self.vector(pt1, pt2, project=True) for (pt1, pt2) in pairs]
            [dists, thetas] = zip(*vectors)
            dists = np.array(dists)
            thetas = np.array(thetas)
            if output_units=="kilometers":
                v_all=(3.6 / 1000.0) * (dists / dt) # convert from m/s to km/hr
            elif output_units=="miles":
                v_all=(2.2369356 / 1000.0) * (dists / dt) # convert from m/s to miles/hr
            elif output_units=="meters":
                v_all=(0.001) * (dists / dt)
            else:
                v_all = (dists / dt)
            t_all = thetas * 180 / np.pi
        except Exception as e:
            raise e
                
        return (v_all, t_all, pairs, bgr1, bgr2, 1/dt, gps_data)
    
    """
    Run GUI
    This function and its handlers (start_stop and start_gravel, etc) are used for running the GUI
    """
    def start_stop(self, widget):
        if self.start_stop_command:
            self.start_stop_command = False
            pretty_print("CV6", "Stopping trial")
        else:
            self.start_stop_command = True
            pretty_print("CV6", "Starting trial")
    
    def mark_important(self, widget):
        shutil.copy2(self.log_path, self.log_path[:-len(self.log_ext)] + ' IMPORTANT' + self.log_ext)

    def start_grass_short(self, widget):
        self.terrain = "Short Grass"
        self.close_log()
        self.create_log()
        pretty_print("CV6", "Setting mode to Short Grass")
        
    def start_grass_tall(self, widget):
        self.terrain = "Tall Grass"
        self.close_log()
        self.create_log()
        pretty_print("CV6", "Setting mode to Tall Grass")
        
    def start_gravel(self, widget):
        self.terrain = "Gravel"
        self.close_log()
        self.create_log()
        pretty_print("CV6", "Setting mode to Gravel")

    def start_soy_tall(self, widget):
        self.terrain = "Tall Soy"
        self.close_log()
        self.create_log()
        pretty_print("CV6", "Setting mode to Tall Soy")

    def start_soy_short(self, widget):
        self.terrain = "Short Soy"
        self.close_log()
        self.create_log()
        pretty_print("CV6", "Setting mode to Short Soy")

    def start_sand(self, widget):
        self.terrain = "Sand"
        self.close_log()
        self.create_log()
        pretty_print("CV6", "Setting mode to Sand")

    def start_soil(self, widget):
        self.terrain = "Soil"
        self.close_log()
        self.create_log()
        pretty_print("CV6", "Setting mode to Soil")
    
    def create_log(self, log_dir='logs', date_format="%Y-%m-%d %H:%M:%S", log_ext='.csv'):
        pretty_print("CV6", "Creating log file for %s" % self.terrain)
        try:
            self.log_ext = log_ext
            self.log_dir = log_dir
            self.log_name = datetime.strftime(datetime.now(), date_format) + ' ' + self.terrain + log_ext
            self.log_path = os.path.join(log_dir, self.log_name)            
            self.log_file = open(self.log_path, 'w')
        except Exception as e:
            pretty_print("CV6", str(e))

    def close_log(self):
        pretty_print("CV6", "Attemping to close log for %s" % self.terrain)
        try:
            self.log_file.close()
        except:
            pretty_print("CV6", "Failed to close log, maybe it did not exist?")

    def update_gui(self):
        self.label_speed.set_markup(self.label_speed_format % self.display_speed)
        self.label_msg.set_markup(self.label_msg_format % self.log_name)
        self.label_fps.set_markup(self.label_fps_format % self.display_fps)
        self.label_gps.set_markup(self.label_gps_format % (self.speed, self.lat, self.lon))
        self.label_status.set_markup(self.label_status_format % str(self.start_stop_command))
        self.label_terrain.set_markup(self.label_terrain_format % str(self.terrain))
        self.label_matches.set_markup(self.label_matches_format % self.num_matches)
        while gtk.events_pending():
            gtk.main_iteration_do(False)
    
    def update_gps(self, verbose=False):
        while True:
            try:
                sentence = self.gps.readline()
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

    def update_video(self, N=5, verbose=False):
        self.diff = 0
        while True:
            time_deltas = []
            while len(time_deltas) < N:
                t1a = time.time()
                s1, bgr1 = self.camera.read()            
                t1b = time.time()
                t2a = time.time()            
                s2, bgr2 = self.camera.read()
                t2b = time.time()
                diff = (t2b+t2a)/2.0 - (t1b+t1a)/2.0
                if (s1 and s2) and (diff > 0.01):
                    time_deltas.append(diff)
                    self.bgr1 = bgr1
                    self.bgr2 = bgr2
            self.diff = np.median(time_deltas)

    def close_gui(self, widget, event, data=None):
        try:
            self.window.destroy()
            self.run_while = False
        except:
            raise Exception("Window failed to close properly")
            
    def run_gui(
        self,
        gps=True,
        gps_device="/dev/ttyS0",
        gps_baud=38400,
        date_format="%Y-%m-%d %H:%M:%S",
        log_path='logs',
        num_fps_hist=3,
        fps=None
    ):
        
        # Imports
        import pygtk
        pygtk.require('2.0')
        import gtk        
        
        # Initialize Terrain, etc. at Default Values
        self.terrain = None
        self.start_stop_command = False
        self.display_speed = 0.0
        self.display_fps = 0.0
        self.display_msg = ''
        self.log_name = ''
        self.run_while = True
        self.lon = 0
        self.lat = 0
        self.speed = 0
        self.alt = 0
        self.num_matches = 0
        self.dt = [0] * num_fps_hist 
        self.bgr1 = np.zeros((640, 480, 3), np.uint8)
        self.bgr2 = np.zeros((640, 480, 3), np.uint8)
        self.bgr = np.vstack([self.bgr1, self.bgr2])
        self.pix = gtk.gdk.pixbuf_new_from_array(self.bgr, gtk.gdk.COLORSPACE_RGB, 8)

        # Format Strings
        self.label_msg_format = '<span size="20000">Output File: %s</span>'
        self.label_speed_format = '<span size="20000">CV Speed: %6.2f km/h</span>'
        self.label_fps_format = '<span size="20000">FPS: %f Hz</span>'
        self.label_gps_format = '<span size="20000">RTK: %6.2f km/h at (%f N, %f E)</span>'
        self.label_status_format = '<span size="20000">Logging: %s</span>'
        self.label_terrain_format = '<span size="20000">Terrain: %s</span>'
        self.label_matches_format = '<span size="20000">Matches: %d</span>'

        # GPS
        if gps:
            try:
                self.gps = serial.Serial(gps_device, gps_baud)
                thread.start_new_thread(self.update_gps, ())
                pretty_print("GPS", "GPS connected")
            except Exception as e:
                pretty_print("GPS", str(e))                
                pretty_print("GPS", "GPS failed to connect!")
                self.gps = None
        else:
            pretty_print("CV6", "GPS disabled")
            self.gps = gps
            
        # Create Window
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.set_title("V6 Trial")
        self.window.set_size_request(700, 300)
        self.window.connect("delete_event", self.close_gui)
        self.window.set_border_width(10)
        self.window.maximize()
        self.window.show()
        
        # Horizontal Box
        self.hbox = gtk.HBox(False, 0)
        self.hbox.show()
        self.window.add(self.hbox)
        
        # Images VBox
        self.vbox1 = gtk.VBox(False, 0)
        self.image = gtk.Image()
        self.image.set_from_pixbuf(self.pix)
        self.image.show()
        self.vbox1.add(self.image)
        self.vbox1.show()
        self.hbox.add(self.vbox1)

        # Output Table with Labels
        self.table_layout = gtk.Table(rows=2, columns=1, homogeneous=True)
        ## Message Label
        self.label_msg = gtk.Label(self.label_msg_format % self.display_msg)
        self.label_msg.show()
        self.label_msg.set_use_markup(True)
        self.table_layout.attach(self.label_msg, 0, 1, 0, 1)
        self.hbox.add(self.table_layout)
        ## FPS Label
        self.label_fps = gtk.Label(self.label_fps_format % self.display_fps)
        self.label_fps.show()
        self.table_layout.attach(self.label_fps, 0, 1, 0, 2)
        self.table_layout.show()
        ## Velocity Label
        self.label_speed = gtk.Label(self.label_speed_format % self.display_speed)
        self.label_speed.set_use_markup(True)
        self.label_speed.show()
        self.table_layout.attach(self.label_speed, 0, 1, 0, 3)
        ## GPS Label
        self.label_gps = gtk.Label(self.label_gps_format % (self.lat, self.lon, self.speed))
        self.label_gps.show()
        self.table_layout.attach(self.label_gps, 0, 1, 0, 4)
        self.table_layout.show()
        ## Logging Label
        self.label_status = gtk.Label(self.label_status_format % str(self.start_stop_command))
        self.label_status.show()
        self.table_layout.attach(self.label_status, 0, 1, 0, 5)
        self.table_layout.show()
        ## Terrain Label
        self.label_terrain = gtk.Label(self.label_terrain_format % str(self.terrain))
        self.label_terrain.show()
        self.table_layout.attach(self.label_terrain, 0, 1, 0, 6)
        self.table_layout.show() 
        ## Matches Label
        self.label_matches = gtk.Label(self.label_matches_format % self.num_matches)
        self.label_matches.show()
        self.table_layout.attach(self.label_matches, 0, 1, 0, 7)
        self.table_layout.show() 
       
        # Create Vboz
        self.vbox_app = gtk.VBox(False, 0)
        self.hbox.add(self.vbox_app)
        self.vbox_app.show()
            
        # Tall Grass Button
        self.button_grass_tall = gtk.Button("Tall Grass")
        self.button_grass_tall.child.modify_font(pango.FontDescription("sans 30"))
        self.button_grass_tall.connect("clicked", self.start_grass_tall)
        self.vbox_app.pack_start(self.button_grass_tall, True, True, 0)
        self.button_grass_tall.show()
        
        # Tall Grass Button
        self.button_grass_short = gtk.Button("Short Grass")
        self.button_grass_short.child.modify_font(pango.FontDescription("sans 30"))
        self.button_grass_short.connect("clicked", self.start_grass_short)
        self.vbox_app.pack_start(self.button_grass_short, True, True, 0)
        self.button_grass_short.show()
        
        # Gravel Button
        self.button_gravel = gtk.Button("Gravel")        
        self.button_gravel.child.modify_font(pango.FontDescription("sans 30"))
        self.button_gravel.connect("clicked", self.start_gravel)
        self.vbox_app.pack_start(self.button_gravel, True, True, 0)
        self.button_gravel.show()
        
        # Soil Button
        self.button_soil = gtk.Button("Soil")
        self.button_soil.child.modify_font(pango.FontDescription("sans 30"))
        self.button_soil.connect("clicked", self.start_soil)
        self.vbox_app.pack_start(self.button_soil, True, True, 0)
        self.button_soil.show()

        # Short Soy Button
        self.button_short_soy = gtk.Button("Short Soy")
        self.button_short_soy.connect("clicked", self.start_soy_short)
        self.button_short_soy.child.modify_font(pango.FontDescription("sans 30"))
        self.vbox_app.pack_start(self.button_short_soy, True, True, 0)
        self.button_short_soy.show()

        # Tall Soy Button
        self.button_tall_soy = gtk.Button("Tall Soy")
        self.button_tall_soy.child.modify_font(pango.FontDescription("sans 30"))
        self.button_tall_soy.connect("clicked", self.start_soy_tall)
        self.vbox_app.pack_start(self.button_tall_soy, True, True, 0)
        self.button_tall_soy.show()

        # Important Button
        self.button_mark_important = gtk.Button("Mark Important")
        self.button_mark_important.child.modify_font(pango.FontDescription("sans 30"))
        self.button_mark_important.connect("clicked", self.mark_important)
        self.vbox_app.pack_start(self.button_mark_important, True, True, 0)
        self.button_mark_important.show()
        
        # Stop Button
        self.button_start_stop = gtk.Button("Start/Stop")
        self.button_start_stop.child.modify_font(pango.FontDescription("sans 30"))
        self.button_start_stop.connect("clicked", self.start_stop)
        self.vbox_app.pack_start(self.button_start_stop, True, True, 0)
        self.button_start_stop.show()

        # Wait for terrain mode
        pretty_print("CV6", "Waiting for Terrain mode to be set by user")
        while not self.terrain and self.run_while:
            self.update_gui()
            if self.start_stop_command:
                break
                
        # Run Loop
        while self.run_while:
            self.update_gui()
            try:
                if True: #self.start_stop_command:
                    (v_all, t_all, pairs, bgr1, bgr2, fps_avg, gps_data) = self.estimate_vector(fps=fps)
                    self.update_gui()                    
                    bgr = np.vstack([bgr1, bgr2])
                    (h,w,d) = bgr1.shape
                                        
                    for ((x1,y1), (x2,y2)) in pairs:
                        pt1 = (int(x1), int(y1))
                        pt2 = (int(x2), int(y2+h))
                        cv2.circle(bgr, pt1, 5, (0,255,0), 1)
                        cv2.circle(bgr, pt2, 5, (255,0,0), 1)
                    pix = gtk.gdk.pixbuf_new_from_array(np.array(bgr), gtk.gdk.COLORSPACE_RGB, 8) 
                    self.image.set_from_pixbuf(pix)
                    
                    # Filter for best
                    # 1. round t_all to 1 deg
                    # 2. find the most common direction of travel
                    # 3. find the most common speed of travel (rounded to 0.01 km/h)
                    # 4. find the thetas near the mode within a tolerance of 1 deg
                    # 5. find the associated velocities
                    pretty_print("FLT", "Running filter")
                    t_rounded = np.abs(np.around(t_all, 0).astype(np.int32))
                    t_counts = np.bincount(t_rounded)
                    print t_counts
                    t_mode = np.argmax(t_counts)
                    print t_mode
                    t_best = np.isclose(t_rounded, t_mode, atol=1)
                    print t_best                    
                    v_best = v_all[t_best]
                    print v_best
                    self.display_speed = np.mean(v_best)
                    self.display_fps = fps_avg
                    self.num_matches = len(pairs)
                    pretty_print("CV6", "Vector Degree:\t%f" % np.mean(t_best))
                    pretty_print("CV6", "Ground Speed:\t%f km/hr" % self.display_speed)
                    pretty_print("CV6", "Frames per Second:\t%f Hz" % self.display_fps)
                    
                    # Format to CSV
                    self.update_gui()
                    try:
                        date_time = [datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S.%f")]
                        v_best = [str(v) for v in v_best.tolist()]
                        newline1 = date_time + [str(fps_avg)] + gps_data + v_best + ['\n']
                        if self.start_stop_command:
                            self.log_file.write(','.join(newline1))
                    except Exception as e:
                        pretty_print("CV6", str(e))
            except Exception as e:
                pretty_print("CV6", str(e))
                
    def run_cli(
        self,
        terrain='NONE',
        gps=True,
        gps_device="/dev/ttyS0",
        gps_baud=38400,
        date_format="%Y-%m-%d %H:%M:%S",
        log_path='logs',
        num_fps_hist=3,
        fps=None
    ):  
        
        # Initialize Terrain, etc. at Default Values
        self.terrain = terrain
        self.display_speed = 0.0
        self.display_fps = 0.0
        self.display_msg = ''
        self.log_name = ''
        self.run_while = True
        self.lon = 0
        self.lat = 0
        self.speed = 0
        self.alt = 0
        self.num_matches = 0
        self.dt = [0] * num_fps_hist 
        self.create_log()
        self.bgr1 = np.zeros((640, 480, 3), np.uint8)
        self.bgr2 = np.zeros((640, 480, 3), np.uint8)

        # GPS
        if gps:
            try:
                self.gps = serial.Serial(gps_device, gps_baud)
                thread.start_new_thread(self.update_gps, ())
                pretty_print("GPS", "GPS connected")
            except Exception as e:
                pretty_print("GPS", str(e))                
                pretty_print("GPS", "GPS failed to connect!")
                self.gps = None
        else:
            pretty_print("CV6", "GPS disabled")
            self.gps = gps

        # Video
        thread.start_new_thread(self.update_video, ())
        pretty_print("CAM", "Videofeed started")
                
        # Run Loop
        a = time.time()
        b = time.time()
        hz = 0    
        while True:
            try:
                if True:
                    print "------------------------------------------"
                    a = time.time()
                    bgr1 = self.bgr1
                    bgr2 = self.bgr2
                    diff = self.diff                    
                    (v_all, t_all, pairs, bgr1, bgr2, fps_avg, gps_data) = self.estimate_vector(bgr1, bgr2, diff, fps=fps)
                   
                    # Filter for best
                    # 1. round t_all to 1 deg
                    # 2. find the most common direction of travel
                    # 3. find the most common speed of travel (rounded to 0.01 km/h)
                    # 4. find the thetas near the mode within a tolerance of 1 deg
                    # 5. find the associated velocities
                    t_rounded = np.abs(np.around(t_all, 0).astype(np.int32))
                    t_counts = np.bincount(t_rounded)
                    t_mode = np.argmax(t_counts)
                    t_best = np.isclose(t_rounded, t_mode, atol=1)                 
                    v_best = v_all[t_best]
                    self.display_speed = np.mean(v_best)
                    self.display_fps = fps_avg
                    # Format to CSV
                    date_time = [datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S.%f")]
                    v_best = [str(np.median(v_best))] #[str(v) for v in v_best.tolist()]
                    matches = [str(len(pairs))]
                    hessian = [str(self.keypoint_filter.hessianThreshold)]
                    newline1 = date_time + [str(fps_avg)] + gps_data + v_best + matches + hessian + [str(hz)] + ['\n']
                    self.log_file.write(','.join(newline1))
                    b = time.time()
                    hz = 1 / (b - a)
                    pretty_print("CV6", "Hz:\t%f" % (1 / (b - a)))
                    pretty_print("CV6", "Vector Degree:\t%f" % np.mean(t_best))
                    pretty_print("CV6", "CV Speed:\t%f km/hr" % self.display_speed)
                    pretty_print("CV6", "RTK Speed:\t%s km/hr" % gps_data[3])
                    pretty_print("CV6", "Hessian:\t%s km/hr" % hessian[0])
                    pretty_print("CV6", "Matches:\t%s km/hr" % matches[0])
                    pretty_print("CV6", "FPS:\t%f Hz" % self.display_fps)
                                        
            except KeyboardInterrupt:
                self.log_file.close()
                break
            except Exception as e:
                pretty_print("CV6", str(e))
               
