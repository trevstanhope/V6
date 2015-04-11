"""
V6 - Vision Speed Inference Extension

Needs to have run() which takes into account that the algorithm is slower than the 
camera framerate.
"""

__author__ = 'Trevor Stanhope'
__version__ = '0.1'

import cv2, cv
import numpy as np
import time
import sys
from matplotlib import pyplot as plt
from itertools import cycle

class V6:
    
    """
    Initialize
    Optional Arguments:
        capture :
        fov : 
        d : depth from surface (meters)
        roll : 
        pitch :
        yaw : 
        hessian : iterations of hessian filter
        frame
    """
    def __init__(self, capture=0, fov=0.50, d=1.0, roll=0, pitch=0, yaw=0, hessian=1000, w=640, h=480, neighbors=2, factor=0.5):
        self.camera = cv2.VideoCapture(capture)
        self.set_matchfactor(factor)
        self.set_resolution(w, h)
        self.set_fov(fov) # set the field of view (horizontal)
        self.set_pitch(pitch) # 0 rad
        self.set_roll(roll) # 0 rad
        self.set_yaw(yaw) # 0 rad
        self.set_depth(d) # camera distance at center
        self.set_neighbors(neighbors)
        self.set_matcher(hessian)
    
    """
    Set the keypoint matcher configuration, supports BF or FLANN
    """
    def set_matcher(self, hessian, use_flann=False):
        try:
            self.surf = cv2.SURF(hessian)
            if use_flann:
                self.FLANN_INDEX_KDTREE = 0
                self.FLANN_TREES = 10
                self.FLANN_CHECKS = 1000
                self.INDEX_PARAMS = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=self.FLANN_TREES)
                self.SEARCH_PARAMS = dict(checks=self.FLANN_CHECKS) # or pass empty dictionary
                self.matcher = cv2.FlannBasedMatcher(self.INDEX_PARAMS, self.SEARCH_PARAMS)
            else:
                self.matcher = cv2.BFMatcher()
        except Exception as e:
            print str(e)
            raise Exception("Failed to generate a matcher")
    """
    Close
    """  
    def close(self):
        self.camera.release()
        
    def set_matchfactor(self, factor):
        if factor < 0:
            raise Exception("Cannot have match less than 1")
        else:
            self.factor = factor
            
    """
    Set Neighbors
    """
    def set_neighbors(self, neighbors):
        if neighbors < 2:
            raise Exception("Cannot have neighbors less than 2")
        else:
            self.neighbors = neighbors
            
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
    
    """
    Set distance at center of frame
    d : depth of view [m]
    """
    def set_depth(self, d):
        if d <= 0:
            raise Exception("Improper distance")
        else:
            self.d = d
    
    """
    Set Inclination
    incl [rad]
    0 is orthogonal to surface 
    """
    def set_pitch(self, pitch):
        if pitch < 0:
            raise Exception("Cannot have negative inclination")
        elif pitch > np.pi/2.0:
            raise Exception("Cannot have inclination parallel to surface")
        else:
            self.pitch = pitch

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

    """
    Set Roll
    roll [rad]
    0 is parallel to surface 
    """
    def set_roll(self, roll):
        self.roll = roll

    """
    Set Yaw (relative to direction of travel)
    yaw [rad]
    0 is parallel to direction of travel 
    """
    def set_yaw(self, yaw):
        self.yaw = yaw
    
    """
    Set Field-of-View
    fov [rad]
    """
    def set_fov(self, fov):
        if fov <= 0:
            raise Exception("Cannot have negative FOV")
        else:
            self.fov = fov
    
    """ 
    Flush Buffer
    To get the most recent images, flush the buffer of older frames
    """
    def flush(self, frames=2):
        for i in range(frames):
            (s, bgr) = self.camera.read()
            
    """
    Match Images
    Find (good) pairs of matching points between two images
    Returns: [ (pt1, pt2), ... ]
    """
    def match_images(self, bgr1, bgr2):
        if self.matcher is not None:
            gray1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
            (pts1, desc1) = self.surf.detectAndCompute(gray1, None)
            (pts2, desc2) = self.surf.detectAndCompute(gray2, None)
            matching_pairs = []
            if pts1 and pts2:
                all_matches = self.matcher.knnMatch(desc1, desc2, k=self.neighbors)
                try:
                    for m,n in all_matches:
                        if m.distance < self.factor * n.distance:
                            pt1 = pts1[m.queryIdx]
                            pt2 = pts2[m.trainIdx]
                            pt1 = (pt1.pt[0], pt1.pt[1])
                            pt2 = (pt2.pt[0], pt2.pt[1])
                            matching_pairs.append((pt1, pt2))
                except Exception as e:
                    print str(e)
            return matching_pairs
        else:
            raise Exception("No matcher exists!")
    
    """
    Distance between two keypoints, where keypoints are in units of pixels
    Arguments:
        pt1 : (int x1, int y1) 
        pt2 : (int x2, int y2)
    Returns:
        distance : float
    """
    def distance(self, pt1, pt2, project=False):
        (x1, y1) = pt1
        (x2, y2) = pt2
        if project:
            (x1, y1) = self.project(x1, y1)
            (x2, y2) = self.project(x2, y2)
        distance = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
        return distance
    
    """
    Find the direction of travel
    Arguments
        pt1 : (int x1, int y1)
        pt2 : (int x2, int y2)
    Returns:
        theta : float
    """
    def direction(self, pt1, pt2, project=False):
        (x1, y1) = pt1
        (x2, y2) = pt2
        if project:
            (x1, y1) = self.project(x1, y1)
            (x2, y2) = self.project(x2, y2)
        theta = np.tan( (x2 - x1) / (y2 - y1) )
        return theta
    
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
    def project(self, x, y, d=None, fov=None, w=None, pitch=None, f=None):
        f = self.w / (2 * np.tan(self.fov / 2.0))
        theta = np.arctan(y / f)
        Y = self.d / np.tan( (np.pi/2.0 - self.pitch) - theta)
        X = x * np.sqrt( (self.d**2 + Y**2) / (f**2 + y**2) )
        return (X, Y)
    
    """
    Test the matching algorithm on a video file with a fixed frame rate
    Optional Arguments:
        dt : the time between each frame
    """
    def test_algorithm(self, dt=0.03):
        results = []
        while True:
            try:
                a = time.time()
                (s1, bgr1) = self.camera.read()
                (s2, bgr2) = self.camera.read()
                pairs = self.match_images(bgr1, bgr2)
                dists = [self.distance(pt1, pt2, project=True) for (pt1, pt2) in pairs]
                dists = np.array(dists)
                v = 3.6 * dists / dt # convert from m/s to km/hr
                v_nonzero = v[v > 1] # eliminate non-moving matches (e.g. shadows)
                v_out = np.mean(v_nonzero)
                b = time.time()
                print v_out, (1 / (b - a))
            except Exception as e:
                print str(e)
                break
    """
    Run algorithm with buffer flushing
    This compensates for the relatively slow pace of the algorithm
    WARNING: this function is meant to be used with a LIVE VIDEO STREAM ONLY
    """
    def run(self, n=10):
        window = [0] * n
        for i in cycle(range(n)):
            self.flush()
            (s1, bgr1) = self.camera.read()
            t1 = time.time()
            (s2, bgr2) = self.camera.read()
            t2 = time.time()
            pairs = self.match_images(bgr1, bgr2)
            dists = [self.distance(pt1, pt2, project=True) for (pt1, pt2) in pairs]
            dists = np.array(dists)
            v = 3.6 * dists / (t2 - t1) # convert from m/s to km/hr
            v_nonzero = v[v > 1] # eliminate non-moving matches (e.g. shadows)
            v_out = np.mean(v_nonzero)
            window[i] = v_out
            v_avg = np.mean(window)
            print v_avg
        
if __name__ == '__main__':
    source = sys.argv[1]
    ext = V6(capture=source)
    try:
        #ext.test_algorithm()
        ext.run()
    except KeyboardInterrupt:
        ext.close()
