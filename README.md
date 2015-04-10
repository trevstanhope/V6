# V6 (Vision Speed Inference eXtension)
A robust and fast travel speed inference system based on SURF keypoint matching

## Importance of timestamps
The SURF keypoint algorithm is extremely robust, however it is essential that the accuracy of 
logging the time of capture for each image is to within the microsecond.

For example, at speeds of 10 km/h and a 0.03 s/frame capture rate, the distance
between two points in two consecutive images is 0.0877 meters. If the estimation of the time
differential is off by 0.01 seconds, this causes the predicted velocity to change from
10.0 km/h to 7.893 km/h.
