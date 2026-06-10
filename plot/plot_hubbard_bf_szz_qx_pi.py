#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用途: 从 Hubbard_bf `measure` 输出绘制 Szz(qx, pi) 曲线.

输入文件:
- `block_binning_mean.json`, 需要包含 `Szzq_nx_ny` 格式的 observable key.

输出文件:
- 默认保存为输入 JSON 同目录下的 `hubbard_bf_szz_qx_pi.png`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SZZQ_KEY_PATTERN = re.compile(r"^Szzq_(\d+)_(\d+)$")


class SzzMomentumValue(NamedTuple):
    """用途: 保存单个 Szz(q) observable 的 momentum index 和数值."""

    nx: int
    ny: int
    value: float


def parse_arguments() -> argparse.Namespace:
    """用途: 解析命令行参数.

    参数:
    - 无.

    返回:
    - argparse.Namespace, 包含输入 JSON, 可选 `lx/ly`, 输出路径和 dpi.
    """
    parser = argparse.ArgumentParser(
        description="读取 Hubbard_bf measure 的 block_binning_mean.json 并绘制 Szz(qx, pi)."
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
        help="输出图片路径. 默认保存到输入文件同目录的 hubbard_bf_szz_qx_pi.png.",
    )
    parser.add_argument(
        "--lx",
        type=int,
        default=None,
        help="x 方向长度. 未给出时从 Szzq key 自动推断.",
    )
    parser.add_argument(
        "--ly",
        type=int,
        default=None,
        help="y 方向长度. 未给出时从 Szzq key 自动推断.",
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
    """用途: 读取 Hubbard_bf measure 输出的 JSON 文件.

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


def parse_szzq_observables(data: Dict[str, Any]) -> List[SzzMomentumValue]:
    """用途: 从 JSON 字典中提取所有 `Szzq_nx_ny` observable.

    参数:
    - `data`: Dict[str, Any], `block_binning_mean.json` 的内容.

    返回:
    - `List[SzzMomentumValue]`: 按 key 解析得到的 momentum index 和数值.
    """
    values: List[SzzMomentumValue] = []
    for key, raw_value in data.items():
        match = SZZQ_KEY_PATTERN.match(key)
        if match is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 的值不能转换为 float: {raw_value!r}") from exc
        values.append(SzzMomentumValue(int(match.group(1)), int(match.group(2)), value))
    if not values:
        raise ValueError("输入 JSON 中没有找到 `Szzq_nx_ny` observable.")
    return values


def infer_lattice_size(
    values: List[SzzMomentumValue],
    lx: Optional[int],
    ly: Optional[int],
) -> Tuple[int, int]:
    """用途: 从 `Szzq_nx_ny` key 推断或校验 momentum 网格尺寸.

    参数:
    - `values`: List[SzzMomentumValue], 已解析的 Szz(q) 数据.
    - `lx`: Optional[int], 用户指定的 x 方向长度.
    - `ly`: Optional[int], 用户指定的 y 方向长度.

    返回:
    - `Tuple[int, int]`: `(lx, ly)`.
    """
    inferred_lx = max(item.nx for item in values) + 1
    inferred_ly = max(item.ny for item in values) + 1
    final_lx = inferred_lx if lx is None else lx
    final_ly = inferred_ly if ly is None else ly
    if final_lx < inferred_lx:
        raise ValueError(f"--lx={final_lx} 小于 JSON 中最大 nx+1={inferred_lx}.")
    if final_ly < inferred_ly:
        raise ValueError(f"--ly={final_ly} 小于 JSON 中最大 ny+1={inferred_ly}.")
    return final_lx, final_ly


def extract_szz_qx_pi_half_line(
    values: List[SzzMomentumValue],
    lx: int,
    ly: int,
) -> Tuple[List[float], List[float]]:
    """用途: 提取 `qy=pi` 且 `0 <= qx <= pi` 的 `Szz(qx, pi)` 数据.

    数学公式:
    - `qx / pi = 2 * nx / lx`.
    - `qy = pi` 要求 `ny = ly / 2`, 因此当前实现要求 `ly` 为偶数.
    - `0 <= qx <= pi` 对应 `nx = 0, 1, ..., lx / 2`, 因此当前实现要求 `lx` 为偶数.

    参数:
    - `values`: List[SzzMomentumValue], 已解析的 Szz(q) 数据.
    - `lx, ly`: int, 晶格尺寸.

    返回:
    - `Tuple[List[float], List[float]]`: `(qx_over_pi, szz_values)`, 均按 `nx=0..lx/2` 排序.
    """
    if lx % 2 != 0:
        raise ValueError(f"0 <= qx <= pi 的端点 qx=pi 需要偶数 lx, 当前 lx={lx}.")
    if ly % 2 != 0:
        raise ValueError(f"qy=pi 需要偶数 ly, 当前 ly={ly}.")
    target_ny = ly // 2
    max_nx = lx // 2
    value_by_index = {(item.nx, item.ny): item.value for item in values}
    qx_over_pi: List[float] = []
    szz_values: List[float] = []
    missing_keys: List[str] = []
    for nx in range(max_nx + 1):
        key = (nx, target_ny)
        if key not in value_by_index:
            missing_keys.append(f"Szzq_{nx}_{target_ny}")
            continue
        qx_over_pi.append(2.0 * nx / lx)
        szz_values.append(value_by_index[key])
    if missing_keys:
        raise ValueError("缺少 qy=pi 线上的 observable: " + ", ".join(missing_keys))
    return qx_over_pi, szz_values


def format_tick_label(value: float) -> str:
    """用途: 将 `qx / pi` 数值格式化为简洁刻度标签.

    参数:
    - `value`: float, 横轴数值.

    返回:
    - `str`: 刻度标签.
    """
    if math.isclose(value, round(value), abs_tol=1.0e-12):
        return str(int(round(value)))
    return f"{value:.3g}"


def plot_szz_qx_pi(
    qx_over_pi: List[float],
    szz_values: List[float],
    output_path: Path,
    dpi: int,
) -> None:
    """用途: 绘制并保存 `Szz(qx, pi)` 曲线.

    参数:
    - `qx_over_pi`: List[float], 横轴 `qx / pi`, 只包含 `0..1`.
    - `szz_values`: List[float], 纵轴 `Szz(qx, pi)`.
    - `output_path`: Path, 输出图片路径.
    - `dpi`: int, 输出图片 dpi.

    返回:
    - None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    axis.plot(qx_over_pi, szz_values, marker="o", linewidth=1.8, markersize=4.5)
    axis.set_xlabel(r"$q_x / \pi$")
    axis.set_ylabel(r"$S^{zz}(q_x,\pi)$")
    axis.set_title(r"Hubbard $S^{zz}(q_x,\pi)$")
    axis.set_xlim(0.0, 1.0)
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    axis.set_xticks(qx_over_pi)
    axis.set_xticklabels([format_tick_label(value) for value in qx_over_pi])
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """用途: 命令行入口, 读取 JSON, 提取 `Szz(qx, pi)`, 并保存图像.

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
        else input_path.with_name("hubbard_bf_szz_qx_pi.png")
    )
    data = load_measurement_json(input_path)
    values = parse_szzq_observables(data)
    lx, ly = infer_lattice_size(values, args.lx, args.ly)
    qx_over_pi, szz_values = extract_szz_qx_pi_half_line(values, lx, ly)
    plot_szz_qx_pi(qx_over_pi, szz_values, output_path, args.dpi)
    print(f"Saved Hubbard_bf Szz(qx, pi) plot to: {output_path}")


if __name__ == "__main__":
    main()
