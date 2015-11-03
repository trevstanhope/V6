import cv, cv2
from matplotlib import pyplot as plot
import numpy as np

CAMERA_INDEX = 0
PIXEL_WIDTH = 640
PIXEL_HEIGHT = 480
camera = cv2.VideoCapture(CAMERA_INDEX)
camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, PIXEL_WIDTH)
camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, PIXEL_HEIGHT)
camera.set(cv.CV_CAP_PROP_SATURATION, 1.0)
camera.set(cv.CV_CAP_PROP_BRIGHTNESS, 0.5)
camera.set(cv.CV_CAP_PROP_CONTRAST, 0.7)
while True:
    try:
        (s, bgr) = camera.read()
        if s:
	    bgr[:,320,:] = 255
	    bgr[240,:,:] = 255
            cv2.imshow('', bgr)
            if cv2.waitKey(5) == 27:
                pass
    except Exception as error:
        print('ERROR: %s' % str(error))
        break
