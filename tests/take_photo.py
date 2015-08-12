"""
Capture video stream and generate text log of exact frame time
Trevor Stanhope
"""

import cv, cv2
import numpy as np
import time
from datetime import datetime
import os
import sys
import uuid

# Constants
index = 0
filename= str(uuid.uuid4()) + '.jpg'
flush = 30

# Initialize Video Capture
camera = cv2.VideoCapture(index)

# Loop until keyboard interrupt
try:
    for i in range(flush):
        (s, bgr) = camera.read()
    cv2.imwrite(filename, bgr)    
    cv2.imshow('', bgr)    
    cv2.waitKey(0)
except Exception as e:
    camera.release()
    print str(e)
