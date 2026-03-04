import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json
import sys

path = sys.argv[1]

lx = ly = 12
mz_matrix = np.zeros((lx, ly))
# folder = "./data"
# name = os.listdir(folder)[0]
# path = folder + "/" + name + "/"
with open(os.path.join(path, "Sz.json"), "r") as f:
    Sz = json.load(f)
k = np.array([np.pi, np.pi])
Sq = 0.0
for x in range(lx):
    for y in range(ly):
        if (f"mz_{x}_{y}") in Sz:
            Sq = Sq + ((-1) ** (x + y)) * Sz[f"mz_{x}_{y}"]
Sq = Sq / (lx * ly)
with open(os.path.join(path, "Staggered_Sz.txt"), "w") as f:
    f.write("staggered_Sz=" + str(Sq))
