#
from os import sys, path
import V6

TRIAL_DURATION = 60 # seconds

# Excecute run()
try:
    source = 0
    ext = V6.V6(capture=source)
    ext.run(logging=True, display=True, gps=True)
except Exception as error:
    print str(error)
except KeyboardInterrupt:
    ext.close()
