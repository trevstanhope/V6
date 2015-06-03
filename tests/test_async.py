# Test script for the async() function
# Run the V6 algorithm on a video source (.avi or USB camera)

from os import sys, path
import V6

try:
    source = sys.argv[1]
except Exception:
    source = 0
    print 'WARNING: No source specified, default to 0'
    
# ASYNC RUN
try:
    ext = V6.V6(capture=source)
    ext.run_async(dt=1/25.0, method="mean")
except KeyboardInterrupt:
    ext.close()
except Exception as e:
    print str(e)
    ext.close()
