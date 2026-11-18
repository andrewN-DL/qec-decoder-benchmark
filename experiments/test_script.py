import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
PROJECT_ROOT = os.path.abspath('stabilizer_sim')
sys.path.append(PROJECT_ROOT)

from stabilizer_sim.codes.surface import SurfaceCode

dist = 11
code = SurfaceCode(dist=dist)
error_rate= code.run(
    shots=1000,
    rounds=dist,
    px=0.04,
    py=0.04,
    pz=0.04,
    pm=0.00002,
    show_errors=False
)

print(error_rate)