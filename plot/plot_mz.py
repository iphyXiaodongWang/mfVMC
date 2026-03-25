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


def plot_spin_lattice(
    mz_matrix,
    path,
    scale=1.0,
    show_value=False,
    title="Spin configuration (z-direction)",
    output="spin_lattice.png",
):
    """用途: 绘制自旋格点图, 仅把缺失点视为 defect.

    参数:
    - mz_matrix: 2D numpy.ndarray, 形状为 `(Lx, Ly)`, 非 defect 位置存放 `<S_i^z>`, defect 位置为 `np.nan`.
    - path: str, 输出目录路径.
    - scale: float, 箭头长度缩放系数.
    - show_value: bool, 是否在每个非 defect 格点上显示数值.
    - title: str, 图片标题.
    - output: str, 输出图片文件名.

    返回:
    - None.

    说明:
    - 当 `show_value=True` 时, 根据系统尺寸自动放大画布并缩放字号, 以减轻大尺寸系统中数值标签重叠的问题.
    """
    lx, ly = mz_matrix.shape
    lattice_size = max(lx, ly)
    if show_value:
        figure_size = max(9.0, 0.7 * lattice_size)
        font_size = max(4.0, min(8.0, 120.0 / lattice_size))
        value_format = "{:.2f}"
    else:
        figure_size = max(6.0, 0.45 * lattice_size)
        font_size = 8.0
        value_format = "{:.3f}"

    plt.figure(figsize=(figure_size, figure_size))
    axis = plt.gca()
    for x in range(lx):
        for y in range(ly):
            if np.isnan(mz_matrix[x, y]):
                circle = plt.Circle(
                    (x, y), radius=0.5, fill=False, edgecolor="black", linewidth=1
                )
                axis.add_artist(circle)
                continue

            plt.scatter(x, y, s=10, c="black")
            if show_value:
                plt.text(
                    x,
                    y,
                    value_format.format(mz_matrix[x, y]),
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="black",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.75,
                        "pad": 0.05,
                    },
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
    axis.set_aspect("equal", adjustable="box")
    plt.xticks(range(lx))
    plt.yticks(range(ly))
    plt.title(title)
    plt.savefig(os.path.join(path, output), dpi=200, bbox_inches="tight")
    plt.close()


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
inverse_mz_matrix = np.full((lx, ly), np.nan)
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
            inverse_mz_matrix[x, y] = Sz[key] * ((-1) ** (x + y))
plot_spin_lattice(mz_matrix, path, show_value=True)
plot_spin_lattice(
    inverse_mz_matrix,
    path,
    show_value=False,
    title="Domain Configuration",
    output="domain.png",
)
