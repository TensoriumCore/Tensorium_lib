import sys, os

# 1) Figure out the absolute path to the directory where THIS file lives...
HERE = os.path.abspath(os.path.dirname(__file__))

# 2) Build the path to your .so
BUILD_DIR = os.path.join(HERE, "pybuild")

# 3) Stick that at the front of sys.path
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

# 4) Now Python will find tensorium.cpython-311-…so in pybuild/
from tensorium import *

print("✅ imported tensorium!")
