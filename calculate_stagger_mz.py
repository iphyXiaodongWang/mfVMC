import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json
import sys
import argparse

def parse_arguments():
    """用途: 解析命令行参数. 参数: 无. 返回: argparse.Namespace, 包含path和L."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="数据目录路径, 例如 logs/target_sz_0")
    parser.add_argument(
        "--L",
        type=int,
        default=12,
        help="系统线性尺寸L, 默认12, 程序按Lx=Ly=L处理",
    )
    args = parser.parse_args()
    if args.L <= 0:
        raise ValueError("--L must be a positive integer.")
    return args

args = parse_arguments()
path = args.path

lx = ly = args.L
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
