#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用途: 从 block_binning_mean.json 读取格点测量值并绘制 Charge/Spin 分布图.

输入文件示例键:
- n_x_y: 站点电荷密度 <n_i>.
- Sz_x_y: 站点自旋密度 <S_i^z>.

输出:
- charge_distribution.png
- spin_distribution.png
- charge_spin_distribution.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
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


def plot_single_map(
    matrix: np.ndarray,
    title_text: str,
    colorbar_label: str,
    colormap_name: str,
    output_path: Path,
    symmetric_color_scale: bool = False,
) -> None:
    """用途: 绘制单个二维分布图.

    参数:
    - matrix: np.ndarray, 数据矩阵.
    - title_text: str, 图标题.
    - colorbar_label: str, 色条标签.
    - colormap_name: str, 颜色映射名.
    - output_path: Path, 输出图片路径.
    - symmetric_color_scale: bool, 是否使用关于0对称色标.

    返回:
    - None.
    """
    figure, axis = plt.subplots(figsize=(6.2, 5.6))
    valid_values = matrix[np.isfinite(matrix)]

    if valid_values.size == 0:
        vmin = 0.0
        vmax = 1.0
    elif symmetric_color_scale:
        vmax_abs = float(np.max(np.abs(valid_values)))
        vmax_abs = max(vmax_abs, 1e-12)
        vmin, vmax = -vmax_abs, vmax_abs
    else:
        vmin = float(np.min(valid_values))
        vmax = float(np.max(valid_values))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12

    image = axis.imshow(
        matrix.T,
        origin="lower",
        cmap=colormap_name,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title_text)
    axis.set_xticks(np.arange(matrix.shape[0]))
    axis.set_yticks(np.arange(matrix.shape[1]))
    axis.grid(color="white", linewidth=0.45, alpha=0.35)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)

    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_combined_maps(
    charge_matrix: np.ndarray,
    spin_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """用途: 在同一画布展示 Charge 与 Spin 分布.

    参数:
    - charge_matrix: np.ndarray, Charge 数据矩阵.
    - spin_matrix: np.ndarray, Spin 数据矩阵.
    - output_path: Path, 输出图片路径.

    返回:
    - None.
    """
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)

    charge_valid = charge_matrix[np.isfinite(charge_matrix)]
    charge_vmin = float(np.min(charge_valid)) if charge_valid.size > 0 else 0.0
    charge_vmax = float(np.max(charge_valid)) if charge_valid.size > 0 else 1.0
    if abs(charge_vmax - charge_vmin) < 1e-12:
        charge_vmax = charge_vmin + 1e-12

    spin_valid = spin_matrix[np.isfinite(spin_matrix)]
    spin_vmax_abs = float(np.max(np.abs(spin_valid))) if spin_valid.size > 0 else 1.0
    spin_vmax_abs = max(spin_vmax_abs, 1e-12)

    image0 = axes[0].imshow(
        charge_matrix.T,
        origin="lower",
        cmap="viridis",
        vmin=charge_vmin,
        vmax=charge_vmax,
        interpolation="nearest",
    )
    axes[0].set_title("Charge distribution <n_i>")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_xticks(np.arange(charge_matrix.shape[0]))
    axes[0].set_yticks(np.arange(charge_matrix.shape[1]))
    axes[0].grid(color="white", linewidth=0.45, alpha=0.35)
    colorbar0 = figure.colorbar(image0, ax=axes[0])
    colorbar0.set_label("<n_i>")

    image1 = axes[1].imshow(
        spin_matrix.T,
        origin="lower",
        cmap="coolwarm",
        vmin=-spin_vmax_abs,
        vmax=spin_vmax_abs,
        interpolation="nearest",
    )
    axes[1].set_title("Spin distribution <S_i^z>")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_xticks(np.arange(spin_matrix.shape[0]))
    axes[1].set_yticks(np.arange(spin_matrix.shape[1]))
    axes[1].grid(color="white", linewidth=0.45, alpha=0.35)
    colorbar1 = figure.colorbar(image1, ax=axes[1])
    colorbar1.set_label("<S_i^z>")

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
    combined_path = output_dir / "charge_spin_distribution.png"

    plot_single_map(
        matrix=charge_matrix,
        title_text="Charge distribution <n_i>",
        colorbar_label="<n_i>",
        colormap_name="viridis",
        output_path=charge_path,
        symmetric_color_scale=False,
    )
    plot_single_map(
        matrix=spin_matrix,
        title_text="Spin distribution <S_i^z>",
        colorbar_label="<S_i^z>",
        colormap_name="coolwarm",
        output_path=spin_path,
        symmetric_color_scale=True,
    )
    plot_combined_maps(charge_matrix, spin_matrix, combined_path)

    print(f"[OK] input: {input_path}")
    print(f"[OK] lattice: Lx={lx}, Ly={ly}, index_offset=({x_offset},{y_offset})")
    print(f"[OK] output: {charge_path}")
    print(f"[OK] output: {spin_path}")
    print(f"[OK] output: {combined_path}")


if __name__ == "__main__":
    main()
