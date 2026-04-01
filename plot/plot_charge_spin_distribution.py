#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用途: 从 block_binning_mean.json 读取格点测量值并绘制 Charge/Spin 分布图.

输入文件示例键:
- n_x_y: 站点电荷密度 <n_i>.
- Sz_x_y: 站点自旋密度 <S_i^z>.

输出:
- charge_distribution.png
- spin_distribution.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


KEY_PATTERN = re.compile(r"^(n|Sz)_(\d+)_(\d+)$")


def parse_arguments() -> argparse.Namespace:
    """用途: 解析命令行参数.

    参数:
    - 无.

    返回:
    - argparse.Namespace, 包含:
      - input: str, 输入 json 路径.
      - output_dir: str, 输出目录.
      - lx: int|None, 可选晶格尺寸 Lx.
      - ly: int|None, 可选晶格尺寸 Ly.
    """
    parser = argparse.ArgumentParser(
        description="读取 block_binning_mean.json 并绘制 charge/spin 分布图"
    )
    parser.add_argument(
        "input",
        type=str,
        help="输入文件路径, 例如 logs/block_binning_mean.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="输出目录, 默认与输入文件同目录",
    )
    parser.add_argument(
        "--lx",
        type=int,
        default=None,
        help="可选, 手动指定 Lx",
    )
    parser.add_argument(
        "--ly",
        type=int,
        default=None,
        help="可选, 手动指定 Ly",
    )
    args = parser.parse_args()

    if args.lx is not None and args.lx <= 0:
        raise ValueError("--lx must be positive")
    if args.ly is not None and args.ly <= 0:
        raise ValueError("--ly must be positive")
    return args


def load_measurement_json(input_path: Path) -> Dict[str, float]:
    """用途: 读取测量 json 数据.

    参数:
    - input_path: Path, 输入文件路径.

    返回:
    - Dict[str, float], 键值对测量数据.
    """
    with input_path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return data


def collect_site_values(data: Dict[str, float], prefix: str) -> Dict[Tuple[int, int], float]:
    """用途: 从 json 键中提取站点数据.

    参数:
    - data: Dict[str, float], 原始数据.
    - prefix: str, "n" 或 "Sz".

    返回:
    - Dict[(x, y), value], 站点值映射.
    """
    site_map: Dict[Tuple[int, int], float] = {}
    for key, value in data.items():
        match = KEY_PATTERN.match(key)
        if match is None:
            continue
        found_prefix = match.group(1)
        if found_prefix != prefix:
            continue
        x_coord = int(match.group(2))
        y_coord = int(match.group(3))
        site_map[(x_coord, y_coord)] = float(value)
    return site_map


def infer_lattice_shape(
    n_map: Dict[Tuple[int, int], float],
    sz_map: Dict[Tuple[int, int], float],
    lx_user: int | None,
    ly_user: int | None,
) -> Tuple[int, int, int, int]:
    """用途: 推断晶格尺寸并返回索引偏移.

    参数:
    - n_map: Charge 站点映射.
    - sz_map: Spin 站点映射.
    - lx_user: 用户指定 Lx.
    - ly_user: 用户指定 Ly.

    返回:
    - (lx, ly, x_offset, y_offset):
      - lx, ly: 最终用于绘图的尺寸.
      - x_offset, y_offset: 原始索引到数组索引的偏移, 如 1-based 数据偏移为 1.
    """
    all_sites = set(n_map.keys()) | set(sz_map.keys())
    if not all_sites:
        raise ValueError("输入 json 中未找到 n_x_y 或 Sz_x_y 键")

    x_values = [site[0] for site in all_sites]
    y_values = [site[1] for site in all_sites]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    lx_infer = x_max - x_min + 1
    ly_infer = y_max - y_min + 1

    lx = lx_user if lx_user is not None else lx_infer
    ly = ly_user if ly_user is not None else ly_infer

    if lx < lx_infer or ly < ly_infer:
        raise ValueError(
            f"用户指定尺寸过小, 推断至少需要 ({lx_infer}, {ly_infer}), 实际给定 ({lx}, {ly})"
        )

    return lx, ly, x_min, y_min


def build_matrix(
    site_map: Dict[Tuple[int, int], float],
    lx: int,
    ly: int,
    x_offset: int,
    y_offset: int,
) -> np.ndarray:
    """用途: 将站点映射转为二维矩阵.

    参数:
    - site_map: Dict[(x, y), value], 站点数据.
    - lx, ly: 绘图尺寸.
    - x_offset, y_offset: 索引偏移.

    返回:
    - np.ndarray, 形状为 (lx, ly), 缺失位置为 np.nan.
    """
    matrix = np.full((lx, ly), np.nan, dtype=float)
    for (x_raw, y_raw), value in site_map.items():
        x_idx = x_raw - x_offset
        y_idx = y_raw - y_offset
        if 0 <= x_idx < lx and 0 <= y_idx < ly:
            matrix[x_idx, y_idx] = value
    return matrix


def configure_matplotlib_font() -> None:
    """用途: 配置 matplotlib 中文字体回退.

    参数:
    - 无.

    返回:
    - None.
    """
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def compute_circle_radius(
    value_abs: float,
    value_abs_max: float,
    min_radius: float = 0.06,
    max_radius: float = 0.42,
) -> float:
    """用途: 将数值绝对值线性映射到圆半径.

    参数:
    - value_abs: float, 当前格点的绝对值.
    - value_abs_max: float, 全局最大绝对值.
    - min_radius: float, 最小半径.
    - max_radius: float, 最大半径.

    返回:
    - float, 映射后的圆半径.
    """
    if value_abs_max <= 1e-12:
        return 0.5 * (min_radius + max_radius)
    scale = max(0.0, min(1.0, value_abs / value_abs_max))
    return min_radius + scale * (max_radius - min_radius)


def setup_lattice_axis(axis: plt.Axes, lx: int, ly: int, title_text: str) -> None:
    """用途: 配置格点绘图坐标轴.

    参数:
    - axis: plt.Axes, 目标坐标轴.
    - lx, ly: Int, 晶格尺寸.
    - title_text: str, 标题.

    返回:
    - None.
    """
    axis.set_xlim(-0.5, lx - 0.5)
    axis.set_ylim(-0.5, ly - 0.5)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title_text)
    axis.set_xticks(np.arange(lx))
    axis.set_yticks(np.arange(ly))
    axis.grid(color="#bbbbbb", linewidth=0.6, alpha=0.6)


def draw_charge_circles(
    axis: plt.Axes,
    charge_matrix: np.ndarray,
    facecolor: str = "#4C78A8",
) -> None:
    """用途: 在格点上绘制 charge 密度圆圈图.

    参数:
    - axis: plt.Axes, 目标坐标轴.
    - charge_matrix: np.ndarray, charge 数据矩阵.
    - facecolor: str, 圆填充颜色.

    返回:
    - None.
    """
    valid_values = charge_matrix[np.isfinite(charge_matrix)]
    if valid_values.size == 0:
        return
    vmax_abs = float(np.max(np.abs(valid_values)))

    lx, ly = charge_matrix.shape
    for x_coord in range(lx):
        for y_coord in range(ly):
            value = charge_matrix[x_coord, y_coord]
            if not np.isfinite(value):
                continue
            radius = compute_circle_radius(abs(float(value)), vmax_abs)
            circle = Circle(
                (x_coord, y_coord),
                radius=radius,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.75,
            )
            axis.add_patch(circle)


def draw_spin_circles(axis: plt.Axes, spin_matrix: np.ndarray) -> None:
    """用途: 在格点上绘制 spin 密度圆圈图.

    参数:
    - axis: plt.Axes, 目标坐标轴.
    - spin_matrix: np.ndarray, spin 数据矩阵.

    返回:
    - None.

    说明:
    - 圆半径表示 |Sz|.
    - 圆颜色表示符号, 正值红色, 负值蓝色, 零值灰色.
    """
    valid_values = spin_matrix[np.isfinite(spin_matrix)]
    if valid_values.size == 0:
        return
    vmax_abs = float(np.max(np.abs(valid_values)))

    lx, ly = spin_matrix.shape
    for x_coord in range(lx):
        for y_coord in range(ly):
            value = spin_matrix[x_coord, y_coord]
            if not np.isfinite(value):
                continue
            value_float = float(value)
            radius = compute_circle_radius(abs(value_float), vmax_abs)
            if value_float > 0:
                facecolor = "#D62728"
            elif value_float < 0:
                facecolor = "#1F77B4"
            else:
                facecolor = "#9E9E9E"
            circle = Circle(
                (x_coord, y_coord),
                radius=radius,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.80,
            )
            axis.add_patch(circle)


def annotate_site_values(
    axis: plt.Axes,
    matrix: np.ndarray,
    fmt: str = "{:.3f}",
    text_color: str = "black",
) -> None:
    """用途: 在每个格点附近标注数值.

    参数:
    - axis: plt.Axes, 目标坐标轴.
    - matrix: np.ndarray, 待标注数据矩阵.
    - fmt: str, 数值格式化模板.
    - text_color: str, 文字颜色.

    返回:
    - None.
    """
    lx, ly = matrix.shape
    font_size = 9 if max(lx, ly) <= 10 else 7
    for x_coord in range(lx):
        for y_coord in range(ly):
            value = matrix[x_coord, y_coord]
            if not np.isfinite(value):
                continue
            axis.text(
                x_coord + 0.12,
                y_coord + 0.12,
                fmt.format(float(value)),
                fontsize=font_size,
                color=text_color,
                ha="left",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 0.2},
            )


def plot_charge_map(
    charge_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """用途: 绘制 charge 密度分布图(圆半径表示大小).

    参数:
    - charge_matrix: np.ndarray, charge 数据矩阵.
    - output_path: Path, 输出图片路径.

    返回:
    - None.
    """
    figure, axis = plt.subplots(figsize=(6.2, 5.6))
    setup_lattice_axis(axis, charge_matrix.shape[0], charge_matrix.shape[1], "Charge distribution <n_i>")
    draw_charge_circles(axis, charge_matrix)
    annotate_site_values(axis, charge_matrix, fmt="{:.3f}", text_color="#111111")

    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_spin_map(
    spin_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """用途: 绘制 spin 密度分布图(颜色表示正负, 半径表示大小).

    参数:
    - spin_matrix: np.ndarray, spin 数据矩阵.
    - output_path: Path, 输出图片路径.

    返回:
    - None.
    """
    figure, axis = plt.subplots(figsize=(6.2, 5.6))
    setup_lattice_axis(axis, spin_matrix.shape[0], spin_matrix.shape[1], "Spin distribution <S_i^z>")
    draw_spin_circles(axis, spin_matrix)
    annotate_site_values(axis, spin_matrix, fmt="{:+.3f}", text_color="#111111")

    legend_handles = [
        Circle((0, 0), radius=0.2, facecolor="#D62728", edgecolor="black", label="Sz > 0"),
        Circle((0, 0), radius=0.2, facecolor="#1F77B4", edgecolor="black", label="Sz < 0"),
    ]
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )

    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """用途: 脚本主入口.

    参数:
    - 无.

    返回:
    - None.
    """
    args = parse_arguments()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib_font()

    data = load_measurement_json(input_path)
    charge_map = collect_site_values(data, prefix="n")
    spin_map = collect_site_values(data, prefix="Sz")

    if not charge_map:
        raise ValueError("未找到 charge 键, 需要 n_x_y 格式")
    if not spin_map:
        raise ValueError("未找到 spin 键, 需要 Sz_x_y 格式")

    lx, ly, x_offset, y_offset = infer_lattice_shape(
        n_map=charge_map,
        sz_map=spin_map,
        lx_user=args.lx,
        ly_user=args.ly,
    )

    charge_matrix = build_matrix(charge_map, lx, ly, x_offset, y_offset)
    spin_matrix = build_matrix(spin_map, lx, ly, x_offset, y_offset)

    charge_path = output_dir / "charge_distribution.png"
    spin_path = output_dir / "spin_distribution.png"

    plot_charge_map(charge_matrix, charge_path)
    plot_spin_map(spin_matrix, spin_path)

    print(f"[OK] input: {input_path}")
    print(f"[OK] lattice: Lx={lx}, Ly={ly}, index_offset=({x_offset},{y_offset})")
    print(f"[OK] output: {charge_path}")
    print(f"[OK] output: {spin_path}")


if __name__ == "__main__":
    main()
