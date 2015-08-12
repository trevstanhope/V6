# 
from os import sys, path
import V6
import sys

terrain = raw_input("Enter terrain type: ")
d = int(raw_input("Enter distance: "))

try:
    ext = V6.V6(d=d)
    ext.run_cli(terrain=terrain, gps=True)
except KeyboardInterrupt:
    ext.close()
