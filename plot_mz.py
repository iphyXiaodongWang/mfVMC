import pickle
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import re
import os
import json
import sys
import argparse


def plot_spin_lattice(mz_matrix, path, scale=1.0, show_value=False):
    """绘制自旋格点图, 仅把缺失点视为 defect. 参数: mz_matrix为二维numpy数组, path为保存路径字符串, scale为箭头缩放系数float, show_value为bool表示是否显示数值. 返回: None."""
    lx, ly = mz_matrix.shape
    plt.figure(figsize=(6, 6))
    for x in range(lx):
        for y in range(ly):
            if np.isnan(mz_matrix[x, y]):
                circle = plt.Circle(
                    (x, y), radius=0.5, fill=False, edgecolor="black", linewidth=1
                )
                plt.gca().add_artist(circle)
                continue

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
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks(range(lx))
    plt.yticks(range(ly))
    plt.title("Spin configuration (z-direction)")
    plt.savefig(os.path.join(path, "spin_lattice.png"))


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
mz_matrix = np.full((lx, ly), np.nan)
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
        key = f"mz_{x}_{y}"
        if key in Sz:
            mz_matrix[x, y] = Sz[key]
plot_spin_lattice(mz_matrix, path, show_value=True)
