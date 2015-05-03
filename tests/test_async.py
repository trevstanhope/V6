from os import sys, path
import V6
try:
    source = sys.argv[1]
except Exception:
    print 'no source specified'
    
# ASYNC RUN
try:
    ext = V6.V6(capture=source)
    ext.run_async(dt=0.03)
except KeyboardInterrupt:
    ext.close()
