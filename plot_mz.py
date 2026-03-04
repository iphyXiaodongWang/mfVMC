import pickle
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import re
import os
import json
import sys


def plot_spin_lattice(mz_matrix, path, scale=1.0, show_value=False):
    """绘制自旋格点图, 可选标注sz数值并绘制箭头. 参数: mz_matrix为二维numpy数组, path为保存路径字符串, scale为箭头缩放系数float, show_value为bool表示是否显示数值. 返回: None."""
    lx, ly = mz_matrix.shape
    plt.figure(figsize=(6, 6))
    for x in range(lx):
        for y in range(ly):
            plt.scatter(x, y, s=10, c="black")
            if show_value:
                plt.text(
                    x,
                    y,
                    f"{mz_matrix[x, y]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
            if mz_matrix[x, y] != 0:
                plt.arrow(
                    x,
                    y - 0.5 * mz_matrix[x, y] * scale,
                    0,
                    mz_matrix[x, y] * scale,
                    width=0.07,
                    head_width=0.2,
                    head_length=0.1,
                    length_includes_head=False,
                    color="red" if mz_matrix[x, y] >= 0 else "blue",
                )
            else:
                circle = plt.Circle(
                    (x, y), radius=0.5, fill=False, edgecolor="black", linewidth=1
                )
                plt.gca().add_artist(circle)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks(range(lx))
    plt.yticks(range(ly))
    plt.title("Spin configuration (z-direction)")
    plt.savefig(os.path.join(path, "spin_lattice.png"))


path = sys.argv[1]
lx = ly = 12
mz_matrix = np.zeros((lx, ly))
folder = "./data/"
""" for name in os.listdir(folder):
    path = folder + name + "/"
    with open(path + "mz.pkl", "rb") as f:
        data = pickle.load(f)
    # print(data)
    mz_re = re.compile(r"mz_(\d+)_(\d+)")
    for k, v in data.items():
        m = mz_re.search(k)
        i, j = int(m.group(1)), int(m.group(2))
        mz_matrix[i, j] = v

    with open(path + "mx.pkl", "rb") as f:
        data = pickle.load(f)
    # print(data)
    mx_re = re.compile(r"mx_(\d+)_(\d+)")
    for k, v in data.items():
        m = mx_re.search(k)
        i, j = int(m.group(1)), int(m.group(2))
        mx_matrix[i, j] = v

    # print(mz_matrix)
    plot_spin_lattice(mz_matrix, mx_matrix, path)
    mz_matrix = np.zeros((lx, ly))
    mx_matrix = np.zeros((lx, ly)) """
""" name = os.listdir(folder)[0]
path = folder + name + "/" """
""" with open(path + "mz.pkl", "rb") as f:
    data = pickle.load(f)
    # print(data)
    mz_re = re.compile(r"mz_(\d+)_(\d+)")
    for k, v in data.items():
    m = mz_re.search(k)
    i, j = int(m.group(1)), int(m.group(2))
    mz_matrix[i, j] = v """
# print(mz_matrix)
# 物理量绘图
with open(os.path.join(path, "Sz.json"), "r") as f:
    Sz = json.load(f)
for x in range(lx):
    for y in range(ly):
        if (f"mz_{x}_{y}") in Sz:
            mz_matrix[x, y] = Sz[f"mz_{x}_{y}"]
plot_spin_lattice(mz_matrix, path, show_value=True)
