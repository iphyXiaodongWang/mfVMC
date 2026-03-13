import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json
import sys
import argparse


def k_grid_2d(Lx, Ly, a=1.0, center=True):
    kx = 2 * np.pi / (Lx * a) * np.arange(Lx)
    ky = 2 * np.pi / (Ly * a) * np.arange(Ly)

    if center:
        kx = (kx + np.pi / a) % (2 * np.pi / a) - np.pi / a
        ky = (ky + np.pi / a) % (2 * np.pi / a) - np.pi / a

    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return KX, KY


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
mx_matrix = np.zeros((lx, ly))
# folder = "./data"
# name = os.listdir(folder)[0]
# path = folder + "/" + name + "/"
with open(os.path.join(path, "SS_all.json"), "r") as f:
    SS_all = json.load(f)
# 生成傅里叶变换的k点
KX, KY = k_grid_2d(lx, ly, center=False)
Z = np.zeros_like(KX)
# k = np.array([np.pi, np.pi])
Nsite = 0
for ix in range(lx):
    for iy in range(ly):
        k = np.array([KX[ix, iy], KY[ix, iy]])
        Sq = 0.0
        site_count = np.zeros((lx, ly))
        for x0 in range(lx):
            for y0 in range(ly):
                for x1 in range(lx):
                    for y1 in range(ly):
                        if (f"SS_{x0}_{y0}_{x1}_{y1}") in SS_all:
                            Sq = (
                                Sq
                                + 2
                                * np.real(
                                    np.exp(1j * np.dot(k, np.array([x0 - x1, y0 - y1])))
                                )
                                * SS_all[f"SS_{x0}_{y0}_{x1}_{y1}"]
                            )
                            if site_count[x0, y0] == 0:
                                site_count[x0, y0] = 1
                            if site_count[x1, y1] == 0:
                                site_count[x1, y1] = 1
        Nsite = np.sum(site_count)
        Sq = (Sq + Nsite * 0.75) / ((lx * ly) ** 2)
        Z[ix, iy] = Sq

""" Sq = Sq + Nsite * 0.75 * 2
Sq = np.real(Sq) / ((lx * ly) ** 2)
Sq = np.sqrt(Sq) """
print(f"Nsite={Nsite}")
""" fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(KX, KY, Z)

ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_zlabel(r"$E(k)$")
plt.show() """
plt.figure(figsize=(5, 4))
plt.imshow(
    Z,
    extent=[KX.min(), KX.max(), KY.min(), KY.max()],
    origin="lower",
    cmap="inferno",
)
plt.colorbar(label=r"$S(\mathbf{k})$")
plt.scatter(np.pi, np.pi, c="cyan", marker="x", s=80)  # 标出 (π,π)
plt.xlabel(r"$k_x$")
plt.ylabel(r"$k_y$")
plt.title(r"$S(k_x,k_y)$")
# plt.show()
plt.savefig(os.path.join(path, "Sq.png"))

""" kx = KX[:, 0]
ky = KY[0, :]
ix_pi = int(np.argmin(np.abs(kx - np.pi)))
iy_pi = int(np.argmin(np.abs(ky - np.pi)))
Sq_pipi = Z[ix_pi, iy_pi]
with open(os.path.join(path, "Sq.txt"), "w") as f:
    f.write("S(pi,pi)=" + str(Sq_pipi)) """
