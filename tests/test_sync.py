# Test script for the run() function
# Run the V6 algorithm on a video source (.avi or USB camera)

from os import sys, path
import V6

try:
    source = sys.argv[1]
except Exception:
    source = 0
    print 'WARNING: No source specified, default to 0'

# Excecute run()
try:
    ext = V6.V6(capture=source)
    dt = 0.04
    display = True
    ext.run(dt=dt, display=display)
except Exception as error:
    print str(error)
except KeyboardInterrupt:
    ext.close()
