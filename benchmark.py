#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用途: 绘制 benchmark_domain 的 VMC/DMRG 对比图.

图像包含四个横向排列的 panel:
- (a) hole 下 VMC 与 DMRG 的 energy per site 随 doping 的变化.
- (b) hole/electron 下 VMC 与 DMRG 的 S(pi,pi) 随 doping 的变化.
- (c) hole, Ndefect=15 下 VMC 计算得到的 <Sz> 空间分布.
- (d) hole, Ndefect=15 下 DMRG 计算得到的 <Sz> 空间分布.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

from plot.plot_domain_sz_arrow import (
    build_diverging_colormap,
    build_shared_domain_norm,
    draw_domain_sz_panel,
    load_dmrg_txt_matrix,
    load_sz_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "results" / "benchmark_domain"
LATTICE_SIZE_X = 12
LATTICE_SIZE_Y = 12
TOTAL_SITE_COUNT = LATTICE_SIZE_X * LATTICE_SIZE_Y
OUTPUT_PNG_PATH = DATA_ROOT / "benchmark.png"
OUTPUT_PDF_PATH = DATA_ROOT / "benchmark.pdf"

VMC_HOLE_CSV_PATH = DATA_ROOT / "auto_submit" / "best_sector_observables_vs_doping.csv"
VMC_ELECTRON_CSV_PATH = (
    DATA_ROOT / "auto_submit_electron" / "best_sector_observables_vs_doping.csv"
)
DMRG_HOLE_SPIPI_PATH = DATA_ROOT / "DMRG" / "data" / "Spipi.txt"
DMRG_ELECTRON_SPIPI_PATH = DATA_ROOT / "DMRG" / "data_electron" / "Spipi.txt"
VMC_HOLE_ENERGY_PATH = DATA_ROOT / "auto_submit" / "best_target_sz_summary.txt"
VMC_ELECTRON_ENERGY_PATH = (
    DATA_ROOT / "auto_submit_electron" / "best_target_sz_summary.txt"
)
DMRG_HOLE_ENERGY_PATH = DATA_ROOT / "DMRG_energy.txt"
VMC_HOLE_NDEFECT15_SZ_DIR = (
    DATA_ROOT / "auto_submit" / "Ndefect15" / "logs" / "target_sz_1"
)
DMRG_HOLE_NDEFECT15_SZ_PATH = DATA_ROOT / (
    "M D=10000_MPSdefect_Heisenberg_OBC_12_12_15series_S1"
    "[12.0, 12.0, 1.0, 1.0, 0.5, 1.25, 0.3].txt"
)
DMRG_HOLE_NDEFECT15_DEFECT_LOCATIONS = [
    (4, 2),
    (10, 7),
    (7, 10),
    (5, 8),
    (2, 11),
    (12, 6),
    (11, 1),
    (4, 12),
    (9, 2),
    (2, 8),
    (10, 10),
    (9, 5),
    (2, 4),
    (5, 4),
    (5, 10),
]

APS_SERIF_FONT_FAMILY = [
    "Times New Roman",
    "Times",
    "Nimbus Roman No9 L",
    "DejaVu Serif",
]
PANEL_LABEL_FONTSIZE = 11.0
AXIS_LABEL_FONTSIZE = 10.0
TICK_LABEL_FONTSIZE = 8.5
LEGEND_FONTSIZE = 7.2
CURVE_LINEWIDTH = 1.25
CURVE_MARKER_SIZE = 4.0
CURVE_CAPSIZE = 2.5
DOMAIN_ARROW_SCALE = 7.0
DOMAIN_ARROW_WIDTH = 0.0080
DOMAIN_ARROW_ALPHA = 0.88
DOMAIN_ARROW_ANGLES = "uv"
DOMAIN_ARROW_SCALE_UNITS = None
DOMAIN_ARROW_HEADWIDTH = 6.0
DOMAIN_ARROW_HEADLENGTH = 4.2
DOMAIN_ARROW_HEADAXISLENGTH = 3.8
DOMAIN_ARROW_MINLENGTH = 1.0
DOMAIN_NEGATIVE_COLOR = "#F2B134"
DOMAIN_ZERO_COLOR = "#F7F7F7"
DOMAIN_POSITIVE_COLOR = "#2B7BFF"
DOMAIN_SITE_MARKER_SIZE = 7.0
DOMAIN_SITE_MARKER_FACE_COLOR = "none"
DOMAIN_SITE_MARKER_EDGE_COLOR = "black"
DOMAIN_SITE_MARKER_ALPHA = 0.8
DOMAIN_SITE_MARKER_LINEWIDTH = 0.9
DOMAIN_DEFECT_MARKER_SIZE = DOMAIN_SITE_MARKER_SIZE + 12.0
DOMAIN_DEFECT_MARKER_FACE_COLOR = "none"
DOMAIN_DEFECT_MARKER_EDGE_COLOR = "#ff0000"
DOMAIN_DEFECT_MARKER_ALPHA = 0.7
DOMAIN_DEFECT_MARKER_LINEWIDTH = 1.0
DOMAIN_GRID_COLOR = "black"
DOMAIN_GRID_ALPHA = 0.5
DOMAIN_GRID_LINEWIDTH = 0.85
DOMAIN_SHOW_PERIODIC_BOUNDARY_BONDS = False
DOMAIN_PERIODIC_BOUNDARY_STUB_LENGTH = 0.75
OBSERVABLE_X_MAJOR_TICK_VALUES = [-0.10, 0.00, 0.10]
OBSERVABLE_X_MINOR_TICK_SPACING = 0.02
OBSERVABLE_Y_MAJOR_TICK_COUNT = 4


def configure_prb_style() -> None:
    """用途: 配置接近 PRB/APS 的 matplotlib 全局绘图风格.

    参数:
    - 无.

    返回:
    - None.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": APS_SERIF_FONT_FAMILY,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "xtick.labelsize": TICK_LABEL_FONTSIZE,
            "ytick.labelsize": TICK_LABEL_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_vmc_spipi_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """用途: 从 VMC 汇总 csv 中读取 doping 与 S(pi,pi).

    参数:
    - csv_path: Path, best_sector_observables_vs_doping.csv 的路径.

    返回:
    - tuple[np.ndarray, np.ndarray], 第一个数组为 doping, 第二个数组为 S(pi,pi).

    公式:
    - y = S(pi,pi).
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"缺少 VMC 汇总文件: {csv_path}")

    doping_values: list[float] = []
    spipi_values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            if row.get("doping", "") == "" or row.get("S_pi_pi", "") == "":
                continue
            spipi_value = float(row["S_pi_pi"])
            if spipi_value < 0.0:
                raise ValueError(f"S(pi,pi) 不能为负数: {spipi_value}, path={csv_path}")
            doping_values.append(float(row["doping"]))
            spipi_values.append(float(spipi_value))

    if not doping_values:
        raise ValueError(f"未从 VMC 汇总文件读到有效数据: {csv_path}")

    return sort_curve_by_x(doping_values, spipi_values)


def load_dmrg_spipi_curve(txt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """用途: 从 DMRG Spipi.txt 中读取 doping 与 S(pi,pi).

    参数:
    - txt_path: Path, DMRG Spipi.txt 的路径, 需要包含 Ndefect 和 S(pi,pi) 列.

    返回:
    - tuple[np.ndarray, np.ndarray], 第一个数组为 doping, 第二个数组为 S(pi,pi).

    公式:
    - doping = Ndefect / (Lx * Ly).
    - y = S(pi,pi).
    """
    if not txt_path.is_file():
        raise FileNotFoundError(f"缺少 DMRG S(pi,pi) 文件: {txt_path}")

    doping_values: list[float] = []
    spipi_values: list[float] = []
    with txt_path.open("r", encoding="utf-8") as file_obj:
        header = file_obj.readline()
        if "Ndefect" not in header or "S(pi,pi)" not in header:
            raise ValueError(f"DMRG Spipi.txt 表头不符合预期: {txt_path}")
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 3:
                continue
            n_defect = int(float(tokens[0]))
            spipi_value = float(tokens[2])
            if spipi_value < 0.0:
                raise ValueError(f"S(pi,pi) 不能为负数: {spipi_value}, path={txt_path}")
            doping_values.append(float(n_defect) / float(TOTAL_SITE_COUNT))
            spipi_values.append(float(spipi_value))

    if not doping_values:
        raise ValueError(f"未从 DMRG S(pi,pi) 文件读到有效数据: {txt_path}")

    return sort_curve_by_x(doping_values, spipi_values)


def load_vmc_energy_per_site_curve(summary_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """用途: 从 VMC best sector 汇总文件读取 energy per site 曲线.

    参数:
    - summary_path: Path, best_target_sz_summary.txt 的路径, 需要包含 Ndefect 和 min_energy.

    返回:
    - tuple[np.ndarray, np.ndarray], 第一个数组为 doping, 第二个数组为 energy per site.

    公式:
    - doping = Ndefect / (Lx * Ly).
    - energy_per_site = min_energy / (Lx * Ly - Ndefect).
    """
    if not summary_path.is_file():
        raise FileNotFoundError(f"缺少 VMC 能量汇总文件: {summary_path}")

    doping_values: list[float] = []
    energy_per_site_values: list[float] = []
    with summary_path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 3:
                continue
            n_defect = int(float(tokens[0]))
            total_energy = float(tokens[2])
            effective_site_count = calculate_effective_site_count(n_defect)
            doping_values.append(float(n_defect) / float(TOTAL_SITE_COUNT))
            energy_per_site_values.append(total_energy / float(effective_site_count))

    if not doping_values:
        raise ValueError(f"未从 VMC 能量汇总文件读到有效数据: {summary_path}")

    return sort_curve_by_x(doping_values, energy_per_site_values)


def load_dmrg_energy_per_site_curve(
    energy_path: Path,
    doping_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """用途: 从 DMRG_energy.txt 读取 energy per site 曲线.

    参数:
    - energy_path: Path, DMRG_energy.txt 的路径, 每行格式如 `Ndefect: E_tot = - 91.671`.
    - doping_type: str, 读取的数据类型, 可选 "hole" 或 "electron".

    返回:
    - tuple[np.ndarray, np.ndarray], 第一个数组为 doping, 第二个数组为 energy per site.

    公式:
    - doping = Ndefect / (Lx * Ly).
    - energy_per_site = E_tot / (Lx * Ly - Ndefect).
    """
    if not energy_path.is_file():
        raise FileNotFoundError(f"缺少 DMRG 能量文件: {energy_path}")
    if doping_type not in {"hole", "electron"}:
        raise ValueError('doping_type 必须为 "hole" 或 "electron".')

    hole_line_pattern = re.compile(
        r"^\s*(?P<ndefect>\d+)\s*:\s*E_tot\s*=\s*(?P<energy>[+-]?\s*\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*$"
    )
    electron_line_pattern = re.compile(
        r"^\s*(?P<ndefect>\d+)\s+defect\s+S\s*=\s*[+-]?\d+\s+(?P<energy>[+-]?\s*\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*$"
    )
    doping_values: list[float] = []
    energy_per_site_values: list[float] = []
    current_section = "hole"
    with energy_path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().rstrip(":") == "electron":
                current_section = "electron"
                continue

            matched = (
                hole_line_pattern.match(line)
                if current_section == "hole"
                else electron_line_pattern.match(line)
            )
            if matched is None:
                raise ValueError(f"DMRG 能量行格式不符合预期: {line}, path={energy_path}")
            if current_section != doping_type:
                continue
            n_defect = int(matched.group("ndefect"))
            total_energy = float(matched.group("energy").replace(" ", ""))
            effective_site_count = calculate_effective_site_count(n_defect)
            doping_values.append(float(n_defect) / float(TOTAL_SITE_COUNT))
            energy_per_site_values.append(total_energy / float(effective_site_count))

    if not doping_values:
        raise ValueError(f"未从 DMRG 能量文件读到有效数据: {energy_path}")

    return sort_curve_by_x(doping_values, energy_per_site_values)


def combine_electron_hole_curve(
    electron_curve: tuple[np.ndarray, np.ndarray],
    hole_curve: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """用途: 将 electron 与 hole 数据合并为以 doping 正负号区分的一条曲线.

    参数:
    - electron_curve: tuple[np.ndarray, np.ndarray], electron 侧的 (doping, observable).
    - hole_curve: tuple[np.ndarray, np.ndarray], hole 侧的 (doping, observable).

    返回:
    - tuple[np.ndarray, np.ndarray], 合并后的 (signed_doping, observable).

    公式:
    - signed_doping = -doping for electron.
    - signed_doping = +doping for hole.
    """
    electron_doping_values, electron_observable_values = electron_curve
    hole_doping_values, hole_observable_values = hole_curve
    nonzero_electron_mask = ~np.isclose(electron_doping_values, 0.0)
    signed_doping_values = np.concatenate(
        [
            -electron_doping_values[nonzero_electron_mask],
            hole_doping_values,
        ]
    )
    observable_values = np.concatenate(
        [
            electron_observable_values[nonzero_electron_mask],
            hole_observable_values,
        ]
    )
    return sort_curve_by_x(signed_doping_values, observable_values)


def sort_curve_by_x(
    x_values: Iterable[float],
    y_values: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """用途: 按横坐标从小到大排序一条曲线.

    参数:
    - x_values: Iterable[float], 横坐标数据.
    - y_values: Iterable[float], 纵坐标数据.

    返回:
    - tuple[np.ndarray, np.ndarray], 排序后的 x 和 y 数组.
    """
    x_array = np.array(list(x_values), dtype=float)
    y_array = np.array(list(y_values), dtype=float)
    if x_array.size != y_array.size:
        raise ValueError("x_values 与 y_values 长度必须相同.")
    order = np.argsort(x_array)
    return x_array[order], y_array[order]


def calculate_effective_site_count(n_defect: int) -> int:
    """用途: 计算扣除 defect 后的有效 site 数.

    参数:
    - n_defect: int, defect 的个数.

    返回:
    - int, 有效 site 数.

    公式:
    - effective_site_count = Lx * Ly - Ndefect.
    """
    effective_site_count = TOTAL_SITE_COUNT - n_defect
    if effective_site_count <= 0:
        raise ValueError(f"有效 site 数必须为正数: Ndefect={n_defect}")
    return effective_site_count


def configure_observable_axis_ticks(axis) -> None:
    """用途: 配置曲线 panel 的主刻度和次刻度, 减少 tick label 数量.

    参数:
    - axis: matplotlib.axes.Axes, 需要调整刻度的曲线坐标轴.

    返回:
    - None.
    """
    axis.set_xticks(OBSERVABLE_X_MAJOR_TICK_VALUES)
    axis.xaxis.set_minor_locator(MultipleLocator(OBSERVABLE_X_MINOR_TICK_SPACING))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=OBSERVABLE_Y_MAJOR_TICK_COUNT))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.tick_params(which="minor", length=1.6, width=0.45)


def plot_energy_panel(axis) -> None:
    """用途: 绘制 energy per site 随 signed doping 变化的 VMC/DMRG 对比 panel.

    参数:
    - axis: matplotlib.axes.Axes, 目标坐标轴对象.

    返回:
    - None.

    公式:
    - energy per site = E_tot / (Lx * Ly).
    - signed doping 中 electron 为负半轴, hole 为正半轴.
    """
    vmc_doping_values, vmc_energy_values = combine_electron_hole_curve(
        electron_curve=load_vmc_energy_per_site_curve(VMC_ELECTRON_ENERGY_PATH),
        hole_curve=load_vmc_energy_per_site_curve(VMC_HOLE_ENERGY_PATH),
    )
    dmrg_doping_values, dmrg_energy_values = combine_electron_hole_curve(
        electron_curve=load_dmrg_energy_per_site_curve(
            DMRG_HOLE_ENERGY_PATH,
            doping_type="electron",
        ),
        hole_curve=load_dmrg_energy_per_site_curve(
            DMRG_HOLE_ENERGY_PATH,
            doping_type="hole",
        ),
    )
    curve_specs = [
        (
            vmc_doping_values,
            vmc_energy_values,
            "VMC",
            "P",
            "#4C78A8",
            "-",
        ),
        (
            dmrg_doping_values,
            dmrg_energy_values,
            "DMRG",
            "X",
            "#222222",
            (0, (3.0, 1.8)),
        ),
    ]

    for doping_values, energy_values, label, marker, color, line_style in curve_specs:
        axis.plot(
            doping_values,
            energy_values,
            marker=marker,
            markersize=CURVE_MARKER_SIZE,
            linewidth=CURVE_LINEWIDTH,
            linestyle=line_style,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            label=label,
        )

    axis.set_xlabel(r"doping $\delta$($+$:hole, $-$:electron)")
    axis.set_ylabel(r"$E/N$")
    axis.yaxis.set_label_coords(-0.18, 0.5)
    axis.set_xlim(-0.112, 0.112)
    configure_observable_axis_ticks(axis)
    axis.axvline(0.0, color="black", linewidth=0.6, alpha=0.45)
    axis.grid(False)
    axis.legend(
        frameon=True,
        loc="lower right",
        handlelength=1.5,
        labelspacing=0.28,
        borderpad=0.1,
        handletextpad=0.45,
        facecolor=(1.0, 1.0, 1.0, 0.82),
        edgecolor="none",
        framealpha=0.82,
    )
    axis.tick_params(top=False, right=False)


def plot_spipi_panel(axis) -> None:
    """用途: 绘制 S(pi,pi) 随 doping 变化的对比 panel.

    参数:
    - axis: matplotlib.axes.Axes, 目标坐标轴对象.

    返回:
    - None.
    """
    vmc_doping_values, vmc_spipi_values = combine_electron_hole_curve(
        electron_curve=load_vmc_spipi_curve(VMC_ELECTRON_CSV_PATH),
        hole_curve=load_vmc_spipi_curve(VMC_HOLE_CSV_PATH),
    )
    dmrg_doping_values, dmrg_spipi_values = combine_electron_hole_curve(
        electron_curve=load_dmrg_spipi_curve(DMRG_ELECTRON_SPIPI_PATH),
        hole_curve=load_dmrg_spipi_curve(DMRG_HOLE_SPIPI_PATH),
    )
    curve_specs = [
        (
            vmc_doping_values,
            vmc_spipi_values,
            "VMC",
            "P",
            "#4C78A8",
            "-",
        ),
        (
            dmrg_doping_values,
            dmrg_spipi_values,
            "DMRG",
            "X",
            "#222222",
            (0, (3.0, 1.8)),
        ),
    ]

    for doping_values, spipi_values, label, marker, color, line_style in curve_specs:
        axis.plot(
            doping_values,
            spipi_values,
            marker=marker,
            markersize=CURVE_MARKER_SIZE,
            linewidth=CURVE_LINEWIDTH,
            linestyle=line_style,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            label=label,
        )

    axis.set_xlabel(r"doping $\delta$($+$:hole, $-$:electron)")
    axis.set_ylabel(r"$S(\pi,\pi)$")
    axis.yaxis.set_label_coords(-0.18, 0.5)
    axis.set_xlim(-0.112, 0.112)
    configure_observable_axis_ticks(axis)
    axis.axvline(0.0, color="black", linewidth=0.6, alpha=0.45)
    axis.grid(False)
    axis.tick_params(top=False, right=False)


def load_hole_ndefect15_sz_matrices() -> tuple[np.ndarray, np.ndarray]:
    """用途: 读取 hole, Ndefect=15 的 VMC 与 DMRG <Sz> 矩阵.

    参数:
    - 无.

    返回:
    - tuple[np.ndarray, np.ndarray], 第一个矩阵为 VMC <Sz>, 第二个矩阵为 DMRG <Sz>.
      两个矩阵形状均为 (12, 12), defect 位置为 np.nan.
    """
    vmc_sz_matrix = load_sz_matrix(VMC_HOLE_NDEFECT15_SZ_DIR, LATTICE_SIZE_X)
    dmrg_sz_matrix = load_dmrg_txt_matrix(
        DMRG_HOLE_NDEFECT15_SZ_PATH,
        LATTICE_SIZE_X,
        LATTICE_SIZE_Y,
        DMRG_HOLE_NDEFECT15_DEFECT_LOCATIONS,
    )
    return vmc_sz_matrix, dmrg_sz_matrix


def plot_domain_panel(
    axis,
    sz_matrix: np.ndarray,
    norm,
    panel_title: str,
) -> None:
    """用途: 绘制单个 <Sz> 空间分布 panel.

    参数:
    - axis: matplotlib.axes.Axes, 目标坐标轴对象.
    - sz_matrix: np.ndarray, 形状为 (12, 12), 非 defect 位置为 <Sz>, defect 为 np.nan.
    - norm: matplotlib.colors.Normalize, VMC/DMRG 共用的 domain 颜色归一化对象.
    - panel_title: str, panel 上方短标题.

    返回:
    - None.

    公式:
    - domain(x,y) = (-1)^(x+y) * <Sz(x,y)>.
    """
    colormap = build_diverging_colormap(
        negative_color=DOMAIN_NEGATIVE_COLOR,
        positive_color=DOMAIN_POSITIVE_COLOR,
        neutral_color=DOMAIN_ZERO_COLOR,
    )
    draw_domain_sz_panel(
        axis=axis,
        sz_matrix=sz_matrix,
        norm=norm,
        colormap=colormap,
        site_marker_size=DOMAIN_SITE_MARKER_SIZE,
        site_marker_face_color=DOMAIN_SITE_MARKER_FACE_COLOR,
        site_marker_edge_color=DOMAIN_SITE_MARKER_EDGE_COLOR,
        site_marker_alpha=DOMAIN_SITE_MARKER_ALPHA,
        site_marker_linewidth=DOMAIN_SITE_MARKER_LINEWIDTH,
        defect_marker_size=DOMAIN_DEFECT_MARKER_SIZE,
        defect_marker_face_color=DOMAIN_DEFECT_MARKER_FACE_COLOR,
        defect_marker_edge_color=DOMAIN_DEFECT_MARKER_EDGE_COLOR,
        defect_marker_alpha=DOMAIN_DEFECT_MARKER_ALPHA,
        defect_marker_linewidth=DOMAIN_DEFECT_MARKER_LINEWIDTH,
        grid_color=DOMAIN_GRID_COLOR,
        grid_alpha=DOMAIN_GRID_ALPHA,
        grid_linewidth=DOMAIN_GRID_LINEWIDTH,
        quiver_scale=DOMAIN_ARROW_SCALE,
        quiver_width=DOMAIN_ARROW_WIDTH,
        quiver_alpha=DOMAIN_ARROW_ALPHA,
        quiver_angles=DOMAIN_ARROW_ANGLES,
        quiver_scale_units=DOMAIN_ARROW_SCALE_UNITS,
        quiver_headwidth=DOMAIN_ARROW_HEADWIDTH,
        quiver_headlength=DOMAIN_ARROW_HEADLENGTH,
        quiver_headaxislength=DOMAIN_ARROW_HEADAXISLENGTH,
        quiver_minlength=DOMAIN_ARROW_MINLENGTH,
        show_periodic_boundary_bonds=DOMAIN_SHOW_PERIODIC_BOUNDARY_BONDS,
        periodic_boundary_stub_length=DOMAIN_PERIODIC_BOUNDARY_STUB_LENGTH,
    )
    axis.set_title(panel_title, fontsize=AXIS_LABEL_FONTSIZE, pad=2.0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.tick_params(
        bottom=False,
        left=False,
        top=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )


def add_panel_label(axis, panel_label: str) -> None:
    """用途: 给 panel 添加 PRB 风格的左上角标签.

    参数:
    - axis: matplotlib.axes.Axes, 目标坐标轴对象.
    - panel_label: str, panel 标签, 例如 "(a)".

    返回:
    - None.
    """
    axis.text(
        -0.12,
        1.08,
        panel_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        color="black",
    )


def build_benchmark_figure() -> plt.Figure:
    """用途: 构建 benchmark_domain 的四 panel 横向对比图.

    参数:
    - 无.

    返回:
    - matplotlib.figure.Figure, 已完成绘制的图像对象.
    """
    configure_prb_style()
    vmc_sz_matrix, dmrg_sz_matrix = load_hole_ndefect15_sz_matrices()
    shared_domain_norm = build_shared_domain_norm([vmc_sz_matrix, dmrg_sz_matrix])

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(7.95, 2.34),
        gridspec_kw={"width_ratios": [0.82, 0.92, 1.0, 1.0], "wspace": 0.32},
    )
    plot_energy_panel(axes[0])
    plot_spipi_panel(axes[1])
    plot_domain_panel(axes[2], vmc_sz_matrix, shared_domain_norm, "VMC")
    plot_domain_panel(axes[3], dmrg_sz_matrix, shared_domain_norm, "DMRG")

    for axis, panel_label in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
        add_panel_label(axis, panel_label)

    figure.subplots_adjust(left=0.060, right=0.995, bottom=0.18, top=0.90)
    return figure


def main() -> None:
    """用途: 主入口, 生成 benchmark.png 和 benchmark.pdf.

    参数:
    - 无.

    返回:
    - None.
    """
    figure = build_benchmark_figure()
    figure.savefig(OUTPUT_PNG_PATH, dpi=450)
    figure.savefig(OUTPUT_PDF_PATH)
    plt.close(figure)
    print(f"[OK] output: {OUTPUT_PNG_PATH}")
    print(f"[OK] output: {OUTPUT_PDF_PATH}")


if __name__ == "__main__":
    main()
