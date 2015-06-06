#

import Tkinter as tk
from os import sys, path
import V6
import sys

#
try:
    d = int(sys.argv[1])
except:
    d = 1000
try:
    ext = V6.V6(d=d)
    ext.run_gui(gps=True)
except Exception as error:
    print str(error)
except KeyboardInterrupt:
    ext.close()
