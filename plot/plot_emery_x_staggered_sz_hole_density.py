#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用途: 绘制 Emery 结果的 x 方向 staggered Cu spin 和 O hole density 分布.

输入文件:
- Emery `measure` 输出的 `block_binning_mean.json`.

输出文件:
- 默认保存为输入 JSON 同目录下的 `emery_x_staggered_sz_hole_density.png`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EMERY_KEY_PATTERN = re.compile(r"^(n|Sz)_(d|px|py)_(-?\d+)_(\d+)$")


class EmeryObservableValue(NamedTuple):
    """用途: 保存单个 Emery density/spin observable 的解析结果."""

    observable: str
    orbital: str
    x_cell: int
    y_cell: int
    value: float


def parse_arguments() -> argparse.Namespace:
    """用途: 解析命令行参数.

    参数:
    - 无.

    返回:
    - argparse.Namespace, 包含输入 JSON, 可选输出路径和晶格尺寸.
    """
    parser = argparse.ArgumentParser(
        description="读取 Emery block_binning_mean.json, 绘制 x 方向 staggered Cu Sz 和 O hole density."
    )
    parser.add_argument(
        "input",
        type=str,
        help="输入 JSON 路径, 例如 logs/block_binning_mean.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出图片路径. 默认保存到输入文件同目录.",
    )
    parser.add_argument(
        "--lx",
        type=int,
        default=None,
        help="Cu cell 的 x 方向长度. 未给出时从 key 自动推断.",
    )
    parser.add_argument(
        "--ly",
        type=int,
        default=None,
        help="Cu cell 的 y 方向长度. 未给出时从 key 自动推断.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="输出图片 dpi.",
    )
    args = parser.parse_args()
    if args.lx is not None and args.lx <= 0:
        raise ValueError("--lx must be positive.")
    if args.ly is not None and args.ly <= 0:
        raise ValueError("--ly must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    return args


def configure_matplotlib_font() -> None:
    """用途: 配置 matplotlib 字体, 保证中文标题或负号可正常显示.

    参数:
    - 无.

    返回:
    - None.
    """
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_measurement_json(input_path: Path) -> Dict[str, Any]:
    """用途: 读取 Emery measure 输出的 JSON 文件.

    参数:
    - `input_path`: Path, `block_binning_mean.json` 路径.

    返回:
    - `Dict[str, Any]`: JSON 中的 observable 均值字典.
    """
    with input_path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(f"输入 JSON 顶层必须是 object: {input_path}")
    return data


def parse_emery_observable_values(data: Dict[str, Any]) -> List[EmeryObservableValue]:
    """用途: 解析 JSON 中所有 Emery density/spin observable.

    参数:
    - `data`: Dict[str, Any], `block_binning_mean.json` 内容.

    返回:
    - `List[EmeryObservableValue]`: 已解析的 observable 列表.
    """
    values: List[EmeryObservableValue] = []
    for key, raw_value in data.items():
        match = EMERY_KEY_PATTERN.match(key)
        if match is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 的值不能转换为 float: {raw_value!r}") from exc
        values.append(
            EmeryObservableValue(
                observable=match.group(1),
                orbital=match.group(2),
                x_cell=int(match.group(3)),
                y_cell=int(match.group(4)),
                value=value,
            )
        )
    if not values:
        raise ValueError("输入 JSON 中没有找到 Emery density/spin observable.")
    return values


def infer_lattice_shape(
    values: Iterable[EmeryObservableValue],
    lx: Optional[int],
    ly: Optional[int],
) -> Tuple[int, int]:
    """用途: 从 observable key 推断或校验 Emery Cu cell 尺寸.

    参数:
    - `values`: Iterable[EmeryObservableValue], 已解析 observable.
    - `lx`: Optional[int], 用户指定的 x 方向 Cu cell 长度.
    - `ly`: Optional[int], 用户指定的 y 方向 Cu cell 长度.

    返回:
    - `Tuple[int, int]`: `(lx, ly)`.
    """
    value_list = list(values)
    d_x_values = [item.x_cell for item in value_list if item.orbital == "d"]
    y_values = [item.y_cell for item in value_list]
    if not d_x_values or not y_values:
        raise ValueError("无法从 JSON 推断 Emery lattice shape: 缺少 d 轨道或 y 坐标.")
    inferred_lx = max(d_x_values)
    inferred_ly = max(y_values)
    final_lx = inferred_lx if lx is None else lx
    final_ly = inferred_ly if ly is None else ly
    if final_lx < inferred_lx:
        raise ValueError(f"--lx={final_lx} 小于 JSON 中最大 Cu x={inferred_lx}.")
    if final_ly < inferred_ly:
        raise ValueError(f"--ly={final_ly} 小于 JSON 中最大 y={inferred_ly}.")
    return final_lx, final_ly


def build_value_lookup(values: Iterable[EmeryObservableValue]) -> Dict[Tuple[str, str, int, int], float]:
    """用途: 将 Emery observable 列表转换为按 key 查询的字典.

    参数:
    - `values`: Iterable[EmeryObservableValue], 已解析 observable.

    返回:
    - `Dict[Tuple[str, str, int, int], float]`: key 为 `(observable, orbital, x, y)`.
    """
    return {
        (item.observable, item.orbital, item.x_cell, item.y_cell): item.value
        for item in values
    }


def compute_staggered_cu_sz_profile(
    lookup: Dict[Tuple[str, str, int, int], float],
    lx: int,
    ly: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """用途: 计算 x 方向的 staggered Cu spin profile.

    数学公式:
    - `m_stag(x) = (1 / Ly) * sum_y (-1)^(x+y) * Sz_d(x,y)`.

    参数:
    - `lookup`: Dict, Emery observable 查询表.
    - `lx, ly`: int, Cu cell 尺寸.

    返回:
    - `Tuple[np.ndarray, np.ndarray]`: `(x_positions, staggered_sz_values)`.
    """
    x_positions = np.arange(1, lx + 1, dtype=float)
    staggered_values = np.zeros(lx, dtype=float)
    for x in range(1, lx + 1):
        column_values = []
        for y in range(1, ly + 1):
            key = ("Sz", "d", x, y)
            if key not in lookup:
                raise ValueError(f"缺少 observable: Sz_d_{x}_{y}")
            stagger_sign = -1.0 if (x + y) % 2 else 1.0
            column_values.append(stagger_sign * lookup[key])
        staggered_values[x - 1] = float(np.mean(column_values))
    return x_positions, staggered_values


def compute_o_hole_density_profiles(
    lookup: Dict[Tuple[str, str, int, int], float],
    lx: int,
    ly: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用途: 计算 x 方向 O hole density profile.

    数学公式:
    - `n_px(x+1/2) = (1 / Ly) * sum_y n_px(x,y)`, 其中 `x=0..Lx`.
    - `n_py(x) = (1 / Ly) * sum_y n_py(x,y)`, 其中 `x=1..Lx`.

    参数:
    - `lookup`: Dict, Emery observable 查询表.
    - `lx, ly`: int, Cu cell 尺寸.

    返回:
    - `Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`:
      `(px_x_positions, px_density, py_x_positions, py_density)`.
    """
    px_x_positions = np.asarray([x + 0.5 for x in range(0, lx + 1)], dtype=float)
    px_density = np.zeros(lx + 1, dtype=float)
    for x in range(0, lx + 1):
        column_values = []
        for y in range(1, ly + 1):
            key = ("n", "px", x, y)
            if key not in lookup:
                raise ValueError(f"缺少 observable: n_px_{x}_{y}")
            column_values.append(lookup[key])
        px_density[x] = float(np.mean(column_values))

    py_x_positions = np.arange(1, lx + 1, dtype=float)
    py_density = np.zeros(lx, dtype=float)
    for x in range(1, lx + 1):
        column_values = []
        for y in range(1, ly + 1):
            key = ("n", "py", x, y)
            if key not in lookup:
                raise ValueError(f"缺少 observable: n_py_{x}_{y}")
            column_values.append(lookup[key])
        py_density[x - 1] = float(np.mean(column_values))

    return px_x_positions, px_density, py_x_positions, py_density


def plot_x_profiles(
    cu_x: np.ndarray,
    staggered_sz: np.ndarray,
    px_x: np.ndarray,
    px_density: np.ndarray,
    py_x: np.ndarray,
    py_density: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    """用途: 绘制并保存 x 方向双面板 profile 图.

    参数:
    - `cu_x`: np.ndarray, Cu x 坐标.
    - `staggered_sz`: np.ndarray, staggered Cu Sz.
    - `px_x, px_density`: np.ndarray, O px 的 x 坐标和 hole density.
    - `py_x, py_density`: np.ndarray, O py 的 x 坐标和 hole density.
    - `output_path`: Path, 输出图片路径.
    - `dpi`: int, 输出图片 dpi.

    返回:
    - None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True)

    axes[0].plot(cu_x, staggered_sz, marker="o", linewidth=1.8, markersize=4.5, color="#1f77b4")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    axes[0].set_ylabel(r"$(-1)^{x+y}\langle S^z_d\rangle$")
    axes[0].set_title("Cu staggered spin profile")
    axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    axes[1].plot(px_x, px_density, marker="s", linewidth=1.7, markersize=4.2, label=r"O $p_x$", color="#d62728")
    axes[1].plot(py_x, py_density, marker="^", linewidth=1.7, markersize=4.2, label=r"O $p_y$", color="#2ca02c")
    axes[1].set_xlabel("x lattice coordinate")
    axes[1].set_ylabel("O hole density")
    axes[1].set_title("O orbital hole density profile")
    axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    axes[1].legend(frameon=False)

    x_min = min(float(np.min(cu_x)), float(np.min(px_x)), float(np.min(py_x)))
    x_max = max(float(np.max(cu_x)), float(np.max(px_x)), float(np.max(py_x)))
    axes[1].set_xlim(x_min - 0.25, x_max + 0.25)
    axes[1].set_xticks(np.arange(np.floor(x_min), np.ceil(x_max) + 1, 1.0))

    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """用途: 命令行入口, 读取 JSON 并绘制 x 方向 profile.

    参数:
    - 无.

    返回:
    - None.
    """
    args = parse_arguments()
    configure_matplotlib_font()
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name("emery_x_staggered_sz_hole_density.png")
    )
    data = load_measurement_json(input_path)
    values = parse_emery_observable_values(data)
    lx, ly = infer_lattice_shape(values, args.lx, args.ly)
    lookup = build_value_lookup(values)
    cu_x, staggered_sz = compute_staggered_cu_sz_profile(lookup, lx, ly)
    px_x, px_density, py_x, py_density = compute_o_hole_density_profiles(lookup, lx, ly)
    plot_x_profiles(cu_x, staggered_sz, px_x, px_density, py_x, py_density, output_path, args.dpi)
    print(f"Saved Emery x profiles to: {output_path}")


if __name__ == "__main__":
    main()
