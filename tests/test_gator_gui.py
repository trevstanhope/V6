#

import Tkinter as tk
from os import sys, path
import V6

#
try:
    source = 0
    ext = V6.V6(capture=source, d=300)
    ext.run_gui(gps=True)
except Exception as error:
    print str(error)
except KeyboardInterrupt:
    ext.close()
