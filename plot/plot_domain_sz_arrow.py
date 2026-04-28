#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用途: 根据 Sz.json 绘制 domain+Sz 箭头图.

说明:
- 箭头方向和长度表示 <Sz>.
- 箭头颜色表示 domain(x, y) = (-1)^(x+y) * <Sz(x, y)>.
- 输入参数接口参考 plot/plot_mz.py, 只接受 path 和 --L.
- 默认只输出一张 domain_sz_arrow.png, 不添加标题.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


OUTPUT_FILENAME = "domain_sz_arrow.png"
COLORBAR_LABEL = "domain = (-1)^(x+y) * <Sz>"
KEY_PATTERN = re.compile(r"^mz_(\d+)_(\d+)$")
FIGURE_WIDTH = 6.8
FIGURE_HEIGHT = 5.8
VALID_SITE_MARKER_SIZE = 10
VALID_SITE_MARKER_ALPHA = 0.35
VALID_SITE_MARKER_FACE_COLOR = "black"
VALID_SITE_MARKER_EDGE_COLOR = "black"
VALID_SITE_MARKER_LINEWIDTH = 0.7
DEFECT_MARKER_SIZE = 60
DEFECT_MARKER_FACE_COLOR = "black"
DEFECT_MARKER_COLOR = "black"
DEFECT_MARKER_ALPHA = 1.0
DEFECT_MARKER_LINEWIDTH = 0.8
COLORBAR_FRACTION = 0.045
COLORBAR_PAD = 0.03
DEFAULT_QUIVER_SCALE = 0.55
DEFAULT_QUIVER_WIDTH = 0.018
DEFAULT_QUIVER_ALPHA = 1.0
DEFAULT_QUIVER_ANGLES = "xy"
DEFAULT_QUIVER_SCALE_UNITS = "xy"
DEFAULT_QUIVER_HEADWIDTH = 6.2
DEFAULT_QUIVER_HEADLENGTH = 7.0
DEFAULT_QUIVER_HEADAXISLENGTH = 6.3
DEFAULT_QUIVER_MINLENGTH = 0.02
DEFAULT_COLORMAP_NAME = "coolwarm"
DEFAULT_COLORMAP_NEUTRAL_COLOR = "#F7F7F7"
DEFAULT_GRID_COLOR = "black"
DEFAULT_GRID_ALPHA = 0.18
DEFAULT_GRID_LINEWIDTH = 0.65
DEFAULT_BOND_ZORDER = 0.5
DEFAULT_SHOW_PERIODIC_BOUNDARY_BONDS = False
DEFAULT_PERIODIC_BOUNDARY_STUB_LENGTH = 0.25


def parse_arguments() -> argparse.Namespace:
    """用途: 解析命令行参数.

    参数:
    - 无.

    返回:
    - argparse.Namespace, 包含:
      - path: str, Sz.json 所在目录.
      - L: int, 系统线性尺寸, 默认按 Lx=Ly=L 处理.
    """
    parser = argparse.ArgumentParser(description="根据 Sz.json 绘制 domain+Sz 箭头图.")
    parser.add_argument("path", type=str, help="数据目录路径, 例如 logs/target_sz_0")
    parser.add_argument(
        "--L",
        type=int,
        default=12,
        help="系统线性尺寸 L, 默认 12, 程序按 Lx=Ly=L 处理",
    )
    arguments = parser.parse_args()
    if arguments.L <= 0:
        raise ValueError("--L must be a positive integer.")
    return arguments


def load_sz_matrix(data_dir: Path, lattice_size: int) -> np.ndarray:
    """用途: 从 Sz.json 读取 <Sz> 并构造二维矩阵.

    参数:
    - data_dir: Path, 含有 Sz.json 的目录.
    - lattice_size: int, 系统线性尺寸 L, 返回矩阵形状为 (L, L).

    返回:
    - np.ndarray, 形状为 (L, L), 非 defect 位置存放 <Sz>, defect 位置为 np.nan.
    """
    sz_path = data_dir / "Sz.json"
    if not sz_path.is_file():
        raise FileNotFoundError(f"缺少文件: {sz_path}")

    with sz_path.open("r", encoding="utf-8") as file_obj:
        sz_data = json.load(file_obj)

    sz_matrix = np.full((lattice_size, lattice_size), np.nan, dtype=float)
    valid_count = 0

    for key, value in sz_data.items():
        matched = KEY_PATTERN.match(key)
        if matched is None:
            continue
        x_coord = int(matched.group(1))
        y_coord = int(matched.group(2))
        if not (0 <= x_coord < lattice_size and 0 <= y_coord < lattice_size):
            raise ValueError(
                f"Sz.json 中坐标超出给定 L={lattice_size}: key={key}, path={sz_path}"
            )
        sz_matrix[x_coord, y_coord] = float(value)
        valid_count += 1

    if valid_count == 0:
        raise ValueError(f"Sz.json 中未发现有效的 mz_x_y 键: {sz_path}")

    return sz_matrix


def load_dmrg_txt_matrix(
    txt_path: Path,
    lattice_size_x: int,
    lattice_size_y: int,
    defect_locations: list[tuple[int, int]],
) -> np.ndarray:
    """用途: 从 DMRG 一维 txt 数据恢复二维 <Sz> 矩阵.

    参数:
    - txt_path: Path, DMRG 输出 txt 文件路径, 文件内容为一列 <Sz>.
    - lattice_size_x: int, x 方向尺寸 Lx.
    - lattice_size_y: int, y 方向尺寸 Ly.
    - defect_locations: list[tuple[int, int]], defect 坐标列表, 使用 1-based index,
      每个元素格式为 (x, y).

    返回:
    - np.ndarray, 形状为 (Lx, Ly), 非 defect 位置存放 <Sz>, defect 位置为 np.nan.

    说明:
    - 数据映射顺序与 results/benchmark_domain/read_m_domain_new2.py 保持一致.
    - 扫描顺序为 y 从 1 到 Ly, x 从 1 到 Lx, 遇到 defect 位置时跳过.
    """
    if lattice_size_x <= 0 or lattice_size_y <= 0:
        raise ValueError("lattice_size_x 和 lattice_size_y 必须为正整数.")
    if not txt_path.is_file():
        raise FileNotFoundError(f"缺少 DMRG txt 文件: {txt_path}")

    raw_data = np.loadtxt(txt_path, dtype=float)
    flat_sz_values = np.atleast_1d(raw_data).astype(float).reshape(-1)
    defect_location_set = set(defect_locations)

    valid_coords: list[tuple[int, int]] = []
    for y_coord in range(1, lattice_size_y + 1):
        for x_coord in range(1, lattice_size_x + 1):
            if (x_coord, y_coord) in defect_location_set:
                continue
            valid_coords.append((x_coord, y_coord))

    if flat_sz_values.size != len(valid_coords):
        raise ValueError(
            "DMRG txt 数据长度与非 defect 格点数不一致: "
            f"len(data)={flat_sz_values.size}, "
            f"valid_site_count={len(valid_coords)}, "
            f"path={txt_path}"
        )

    sz_matrix = np.full((lattice_size_x, lattice_size_y), np.nan, dtype=float)
    for one_sz_value, (x_coord, y_coord) in zip(flat_sz_values, valid_coords):
        sz_matrix[x_coord - 1, y_coord - 1] = float(one_sz_value)
    return sz_matrix


def build_domain_matrix(sz_matrix: np.ndarray) -> np.ndarray:
    """用途: 根据 <Sz> 构造 staggered domain 矩阵.

    参数:
    - sz_matrix: np.ndarray, 形状为 (L, L), 非 defect 位置存放 <Sz>, defect 位置为 np.nan.

    返回:
    - np.ndarray, 形状为 (L, L), 满足
      domain(x, y) = (-1)^(x+y) * <Sz(x, y)>.
      defect 位置保持为 np.nan.
    """
    lattice_size_x, lattice_size_y = sz_matrix.shape
    x_grid, y_grid = np.meshgrid(
        np.arange(lattice_size_x),
        np.arange(lattice_size_y),
        indexing="ij",
    )
    return ((-1.0) ** (x_grid + y_grid)) * sz_matrix


def compute_domain_abs_max(sz_matrix: np.ndarray) -> float:
    """用途: 计算单张 domain 图的颜色归一化尺度上界.

    参数:
    - sz_matrix: np.ndarray, 形状为 (Lx, Ly), 非 defect 位置存放 <Sz>, defect 位置为 np.nan.

    返回:
    - float, 颜色归一化所需的 max_abs_domain, 即
      max_abs_domain = max(|domain(x, y)|).
    """
    domain_matrix = build_domain_matrix(sz_matrix)
    valid_mask = np.isfinite(domain_matrix)
    if not np.any(valid_mask):
        raise RuntimeError("domain 数据为空, 无法计算颜色尺度.")

    max_abs_domain = float(np.max(np.abs(domain_matrix[valid_mask])))
    if np.isclose(max_abs_domain, 0.0):
        max_abs_domain = 1.0
    return max_abs_domain


def build_shared_domain_norm(sz_matrices: list[np.ndarray]) -> mcolors.TwoSlopeNorm:
    """用途: 为多张 domain 图构建共享的颜色归一化.

    参数:
    - sz_matrices: list[np.ndarray], 多张 <Sz> 矩阵列表.

    返回:
    - matplotlib.colors.TwoSlopeNorm, 共享的对称归一化对象, 满足
      vmin = -max_abs_domain, vcenter = 0, vmax = max_abs_domain.
    """
    if len(sz_matrices) == 0:
        raise ValueError("sz_matrices 不能为空.")

    shared_abs_max = max(
        compute_domain_abs_max(one_sz_matrix) for one_sz_matrix in sz_matrices
    )
    return mcolors.TwoSlopeNorm(
        vmin=-shared_abs_max,
        vcenter=0.0,
        vmax=shared_abs_max,
    )


def build_diverging_colormap(
    negative_color: str,
    positive_color: str,
    neutral_color: str = DEFAULT_COLORMAP_NEUTRAL_COLOR,
    colormap_name: str = "custom_diverging_colormap",
) -> mcolors.LinearSegmentedColormap:
    """用途: 根据两端颜色和中心颜色构造自定义 diverging colormap.

    参数:
    - negative_color: str, 对应负值一侧的颜色.
    - positive_color: str, 对应正值一侧的颜色.
    - neutral_color: str, 对应零点附近的中心颜色.
    - colormap_name: str, 返回 colormap 的名称.

    返回:
    - matplotlib.colors.LinearSegmentedColormap, 线性插值的自定义 colormap.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        colormap_name,
        [
            mcolors.to_rgba(negative_color),
            mcolors.to_rgba(neutral_color),
            mcolors.to_rgba(positive_color),
        ],
        N=256,
    )


def build_lattice_bond_segments(
    sz_matrix: np.ndarray,
    show_periodic_boundary_bonds: bool = DEFAULT_SHOW_PERIODIC_BOUNDARY_BONDS,
    periodic_boundary_stub_length: float = DEFAULT_PERIODIC_BOUNDARY_STUB_LENGTH,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """用途: 根据 <Sz> 矩阵生成最近邻 lattice bond 线段, 并跳过 defect 相邻 bond.

    参数:
    - sz_matrix: np.ndarray, 形状为 (Lx, Ly), 非 defect 位置存放 <Sz>, defect 位置为 np.nan.
    - show_periodic_boundary_bonds: bool, 是否额外绘制 PBC 边界短半键.
    - periodic_boundary_stub_length: float, PBC 边界短半键向图外延伸的长度.

    返回:
    - list[tuple[tuple[float, float], tuple[float, float]]], 每个元素表示一条 bond
      的两个端点坐标, 格式为 ((x0, y0), (x1, y1)).

    说明:
    - 只生成 x 和 y 方向的最近邻 bond.
    - 若一条 bond 的任一端点是 defect, 则该 bond 不绘制.
    - 若启用 PBC, 则对跨越左右/上下边界的周期键使用“短半键”方式绘制.
    """
    lattice_size_x, lattice_size_y = sz_matrix.shape
    bond_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if periodic_boundary_stub_length < 0.0:
        raise ValueError("periodic_boundary_stub_length 必须为非负数.")

    for x_coord in range(lattice_size_x):
        for y_coord in range(lattice_size_y):
            if not np.isfinite(sz_matrix[x_coord, y_coord]):
                continue

            if x_coord + 1 < lattice_size_x and np.isfinite(
                sz_matrix[x_coord + 1, y_coord]
            ):
                bond_segments.append(
                    (
                        (float(x_coord), float(y_coord)),
                        (float(x_coord + 1), float(y_coord)),
                    )
                )

            if y_coord + 1 < lattice_size_y and np.isfinite(
                sz_matrix[x_coord, y_coord + 1]
            ):
                bond_segments.append(
                    (
                        (float(x_coord), float(y_coord)),
                        (float(x_coord), float(y_coord + 1)),
                    )
                )

            if (
                show_periodic_boundary_bonds
                and lattice_size_x > 1
                and x_coord == lattice_size_x - 1
                and np.isfinite(sz_matrix[0, y_coord])
            ):
                bond_segments.append(
                    (
                        (float(x_coord), float(y_coord)),
                        (
                            float(x_coord) + float(periodic_boundary_stub_length),
                            float(y_coord),
                        ),
                    )
                )
                bond_segments.append(
                    (
                        (0.0, float(y_coord)),
                        (-float(periodic_boundary_stub_length), float(y_coord)),
                    )
                )

            if (
                show_periodic_boundary_bonds
                and lattice_size_y > 1
                and y_coord == lattice_size_y - 1
                and np.isfinite(sz_matrix[x_coord, 0])
            ):
                bond_segments.append(
                    (
                        (float(x_coord), float(y_coord)),
                        (
                            float(x_coord),
                            float(y_coord) + float(periodic_boundary_stub_length),
                        ),
                    )
                )
                bond_segments.append(
                    (
                        (float(x_coord), 0.0),
                        (float(x_coord), -float(periodic_boundary_stub_length)),
                    )
                )

    return bond_segments


def draw_lattice_bonds(
    axis,
    bond_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    bond_color: str,
    bond_alpha: float,
    bond_linewidth: float,
) -> LineCollection:
    """用途: 在坐标轴上绘制 lattice bond 线段集合.

    参数:
    - axis: matplotlib.axes.Axes, 目标坐标轴对象.
    - bond_segments: list[tuple[tuple[float, float], tuple[float, float]]], bond 线段列表.
    - bond_color: str, bond 颜色.
    - bond_alpha: float, bond 透明度.
    - bond_linewidth: float, bond 线宽.

    返回:
    - matplotlib.collections.LineCollection, 已添加到坐标轴中的 bond 集合对象.
    """
    bond_rgba = mcolors.to_rgba(bond_color, alpha=bond_alpha)
    bond_collection = LineCollection(
        bond_segments,
        colors=[bond_rgba],
        linewidths=bond_linewidth,
        zorder=DEFAULT_BOND_ZORDER,
    )
    axis.add_collection(bond_collection)
    return bond_collection


def draw_domain_sz_panel(
    axis,
    sz_matrix: np.ndarray,
    norm: mcolors.Normalize | None = None,
    colormap: mcolors.Colormap | None = None,
    site_marker_size: float = VALID_SITE_MARKER_SIZE,
    site_marker_face_color: str = VALID_SITE_MARKER_FACE_COLOR,
    site_marker_edge_color: str = VALID_SITE_MARKER_EDGE_COLOR,
    site_marker_alpha: float = VALID_SITE_MARKER_ALPHA,
    site_marker_linewidth: float = VALID_SITE_MARKER_LINEWIDTH,
    defect_marker_size: float = DEFECT_MARKER_SIZE,
    defect_marker_face_color: str = DEFECT_MARKER_FACE_COLOR,
    defect_marker_edge_color: str = DEFECT_MARKER_COLOR,
    defect_marker_alpha: float = DEFECT_MARKER_ALPHA,
    defect_marker_linewidth: float = DEFECT_MARKER_LINEWIDTH,
    grid_color: str = DEFAULT_GRID_COLOR,
    grid_alpha: float = DEFAULT_GRID_ALPHA,
    grid_linewidth: float = DEFAULT_GRID_LINEWIDTH,
    quiver_scale: float = DEFAULT_QUIVER_SCALE,
    quiver_width: float = DEFAULT_QUIVER_WIDTH,
    quiver_alpha: float = DEFAULT_QUIVER_ALPHA,
    quiver_angles: str = DEFAULT_QUIVER_ANGLES,
    quiver_scale_units: str | None = DEFAULT_QUIVER_SCALE_UNITS,
    quiver_headwidth: float = DEFAULT_QUIVER_HEADWIDTH,
    quiver_headlength: float = DEFAULT_QUIVER_HEADLENGTH,
    quiver_headaxislength: float = DEFAULT_QUIVER_HEADAXISLENGTH,
    quiver_minlength: float = DEFAULT_QUIVER_MINLENGTH,
    colormap_name: str = DEFAULT_COLORMAP_NAME,
    show_periodic_boundary_bonds: bool = DEFAULT_SHOW_PERIODIC_BOUNDARY_BONDS,
    periodic_boundary_stub_length: float = DEFAULT_PERIODIC_BOUNDARY_STUB_LENGTH,
):
    """用途: 在已有坐标轴上绘制一张 domain+Sz 箭头图.

    参数:
    - axis: matplotlib.axes.Axes, 目标坐标轴对象.
    - sz_matrix: np.ndarray, 形状为 (Lx, Ly), 非 defect 位置存放 <Sz>, defect 位置为 np.nan.
    - norm: matplotlib.colors.Normalize | None, 颜色归一化对象.
      若为 None, 则对当前 sz_matrix 单独计算归一化范围.
    - colormap: matplotlib.colors.Colormap | None, 箭头颜色所使用的 colormap 对象.
      若为 None, 则回退到 `colormap_name`.
    - site_marker_size: float, 非 defect site 参考散点面积, 对应 scatter 的 s 参数.
    - site_marker_face_color: str, 非 defect 参考点填充颜色, 传入 "none" 表示空心点.
    - site_marker_edge_color: str, 非 defect 空心点边框颜色.
    - site_marker_alpha: float, 非 defect 空心点边框透明度.
    - site_marker_linewidth: float, 非 defect 空心点边框线宽.
    - defect_marker_size: float, defect 空心点面积, 对应 scatter 的 s 参数.
    - defect_marker_face_color: str, defect 点填充颜色, 传入 "none" 表示空心点.
    - defect_marker_edge_color: str, defect 空心点边框颜色.
    - defect_marker_alpha: float, defect 空心点边框透明度.
    - defect_marker_linewidth: float, defect 空心点边框线宽.
    - grid_color: str, lattice 网格线颜色.
    - grid_alpha: float, lattice 网格线透明度.
    - grid_linewidth: float, lattice 网格线线宽.
    - quiver_scale: float, matplotlib quiver 的 scale 参数, 控制箭头整体长度.
    - quiver_width: float, matplotlib quiver 的 width 参数, 控制箭杆粗细.
    - quiver_alpha: float, matplotlib quiver 的 alpha 参数, 控制箭头透明度.
    - quiver_angles: str, matplotlib quiver 的 angles 参数, 控制箭头方向解释方式.
    - quiver_scale_units: str | None, matplotlib quiver 的 scale_units 参数, 控制箭头尺度单位.
    - quiver_headwidth: float, matplotlib quiver 的 headwidth 参数, 控制箭头头部宽度.
    - quiver_headlength: float, matplotlib quiver 的 headlength 参数, 控制箭头头部长度.
    - quiver_headaxislength: float, matplotlib quiver 的 headaxislength 参数, 控制箭头头部轴向长度.
    - quiver_minlength: float, matplotlib quiver 的 minlength 参数, 控制最短箭头显示长度.
    - colormap_name: str, 箭头颜色所使用的 matplotlib colormap 名称.
    - show_periodic_boundary_bonds: bool, 是否绘制跨边界 PBC 短半键.
    - periodic_boundary_stub_length: float, PBC 短半键向图外延伸的长度.

    返回:
    - dict[str, object], 关键 artist 字典, 包含:
      - bond_collection: lattice bond 线段集合.
      - site_scatter: 非 defect 参考散点.
      - defect_scatter: defect 黑点散点.
      - quiver: 箭头对象.
      - norm: 实际使用的颜色归一化对象.
    """
    lattice_size_x, lattice_size_y = sz_matrix.shape
    domain_matrix = build_domain_matrix(sz_matrix)
    valid_mask = np.isfinite(domain_matrix)
    if not np.any(valid_mask):
        raise RuntimeError("domain 数据为空, 无法绘图.")

    if norm is None:
        norm = build_shared_domain_norm([sz_matrix])
    color_map = colormap if colormap is not None else plt.get_cmap(colormap_name)
    site_marker_edge_rgba = mcolors.to_rgba(site_marker_edge_color, alpha=site_marker_alpha)
    defect_marker_edge_rgba = mcolors.to_rgba(
        defect_marker_edge_color,
        alpha=defect_marker_alpha,
    )
    site_marker_face_rgba = (
        "none"
        if site_marker_face_color == "none"
        else mcolors.to_rgba(site_marker_face_color, alpha=site_marker_alpha)
    )
    defect_marker_face_rgba = (
        "none"
        if defect_marker_face_color == "none"
        else mcolors.to_rgba(defect_marker_face_color, alpha=defect_marker_alpha)
    )
    bond_segments = build_lattice_bond_segments(
        sz_matrix,
        show_periodic_boundary_bonds=show_periodic_boundary_bonds,
        periodic_boundary_stub_length=periodic_boundary_stub_length,
    )

    x_coords = []
    y_coords = []
    u_components = []
    v_components = []
    color_components = []
    defect_x_coords = []
    defect_y_coords = []

    for x_coord in range(lattice_size_x):
        for y_coord in range(lattice_size_y):
            sz_value = sz_matrix[x_coord, y_coord]
            if not np.isfinite(sz_value):
                defect_x_coords.append(float(x_coord))
                defect_y_coords.append(float(y_coord))
                continue

            x_coords.append(float(x_coord))
            y_coords.append(float(y_coord))
            u_components.append(0.0)
            v_components.append(float(sz_value))
            color_components.append(float(domain_matrix[x_coord, y_coord]))

    bond_collection = draw_lattice_bonds(
        axis=axis,
        bond_segments=bond_segments,
        bond_color=grid_color,
        bond_alpha=grid_alpha,
        bond_linewidth=grid_linewidth,
    )
    site_scatter = axis.scatter(
        x_coords,
        y_coords,
        s=site_marker_size,
        facecolors=site_marker_face_rgba,
        edgecolors=[site_marker_edge_rgba],
        linewidths=site_marker_linewidth,
    )
    defect_scatter = axis.scatter(
        defect_x_coords,
        defect_y_coords,
        s=defect_marker_size,
        facecolors=defect_marker_face_rgba,
        edgecolors=[defect_marker_edge_rgba],
        linewidths=defect_marker_linewidth,
        marker="o",
        zorder=4,
    )
    quiver_object = axis.quiver(
        np.array(x_coords),
        np.array(y_coords),
        np.array(u_components),
        np.array(v_components),
        np.array(color_components),
        cmap=color_map,
        norm=norm,
        alpha=quiver_alpha,
        angles=quiver_angles,
        scale_units=quiver_scale_units,
        scale=quiver_scale,
        width=quiver_width,
        headwidth=quiver_headwidth,
        headlength=quiver_headlength,
        headaxislength=quiver_headaxislength,
        minlength=quiver_minlength,
        pivot="middle",
    )

    axis.set_xlim(-0.6, lattice_size_x - 0.4)
    axis.set_ylim(-0.6, lattice_size_y - 0.4)
    axis.set_xticks(range(lattice_size_x))
    axis.set_yticks(range(lattice_size_y))
    axis.set_aspect("equal", adjustable="box")
    return {
        "bond_collection": bond_collection,
        "site_scatter": site_scatter,
        "defect_scatter": defect_scatter,
        "quiver": quiver_object,
        "norm": norm,
    }


def build_domain_sz_figure(
    sz_matrix: np.ndarray,
    show_colorbar: bool = True,
):
    """用途: 构建 domain+Sz 箭头图对应的 figure 和 artist.

    参数:
    - sz_matrix: np.ndarray, 形状为 (Lx, Ly), 非 defect 位置存放 <Sz>, defect 位置为 np.nan.
    - show_colorbar: bool, 是否绘制右侧 colorbar.

    返回:
    - tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, dict[str, object]]
      - figure: matplotlib.figure.Figure, 图像对象.
      - axis: matplotlib.axes.Axes, 主坐标轴对象.
      - artist_dict: dict[str, object], 关键 artist 字典, 包含:
        - site_scatter: 非 defect 参考散点.
        - defect_scatter: defect 黑点散点.
        - quiver: 箭头对象.
        - colorbar: colorbar 对象, 若 show_colorbar=False 则为 None.
    """
    figure, axis = plt.subplots(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), constrained_layout=True
    )
    artist_dict = draw_domain_sz_panel(axis, sz_matrix)

    color_bar = None
    if show_colorbar:
        color_bar = figure.colorbar(
            artist_dict["quiver"],
            ax=axis,
            fraction=COLORBAR_FRACTION,
            pad=COLORBAR_PAD,
        )
        color_bar.set_label(COLORBAR_LABEL, rotation=90)
    artist_dict["colorbar"] = color_bar
    return figure, axis, artist_dict


def plot_domain_sz_arrow(
    sz_matrix: np.ndarray,
    output_path: Path,
    show_colorbar: bool = True,
) -> None:
    """用途: 绘制 domain+Sz 箭头图并保存.

    参数:
    - sz_matrix: np.ndarray, 形状为 (L, L), 非 defect 位置存放 <Sz>, defect 位置为 np.nan.
    - output_path: Path, 输出图片路径, 文件名通常为 domain_sz_arrow.png.
    - show_colorbar: bool, 是否绘制右侧 colorbar.

    返回:
    - None.
    """
    figure, _, _ = build_domain_sz_figure(sz_matrix, show_colorbar=show_colorbar)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """用途: 主入口, 读取 Sz.json 并输出 domain_sz_arrow.png.

    参数:
    - 无.

    返回:
    - None.
    """
    arguments = parse_arguments()
    data_dir = Path(arguments.path).resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"数据目录不存在: {data_dir}")

    sz_matrix = load_sz_matrix(data_dir, arguments.L)
    output_path = data_dir / OUTPUT_FILENAME
    plot_domain_sz_arrow(sz_matrix, output_path)
    print(f"[OK] output: {output_path}")


if __name__ == "__main__":
    main()
