from os import sys, path
import V6
try:
    source = sys.argv[1]
except Exception:
    print 'no source specified'
    
# RUN
try:
    ext = V6.V6(capture=source)
    ext.run(dt=0.04, display=True)
except KeyboardInterrupt:
    ext.close()
