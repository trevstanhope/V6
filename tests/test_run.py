# Test script for the run() function
# Run the V6 algorithm on a video source (.avi or USB camera)

from os import sys, path
import V6

try:
    source = sys.argv[1]
    if not source.isdigit():
        fps = float(sys.argv[2])
        dt = 1 / fps
    else:
        raise Exception("WARNING: No FPS specified for video file")
except Exception as e:
    source = 0
    dt = None
    print 'WARNING: No source specified, default to 0'

# Excecute run()
try:
    ext = V6.V6(capture=source)
    ext.run(dt=dt, plot=True, display=True)
except Exception as error:
    print str(error)
except KeyboardInterrupt:
    ext.close()
