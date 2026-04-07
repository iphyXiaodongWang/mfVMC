#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用途:
- 绘制 |staggered_mz| 随 doping 的变化图.
- x 轴正方向表示 hole doping, 负方向表示 electron doping.
- 支持多条 hole 曲线和多条 electron 曲线, 并自动添加图例.

输入数据格式:
- CSV 文件至少包含两列:
  - doping: 浓度
  - abs_staggered_mz: |staggered_mz|

说明:
- 对于 `--hole-series`, doping 会被强制映射到正半轴: +abs(doping).
- 对于 `--electron-series`, doping 会被强制映射到负半轴: -abs(doping).
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_arguments() -> argparse.Namespace:
    """用途: 解析命令行参数.

    参数:
    - 无.

    返回:
    - argparse.Namespace, 包含输入曲线、列名、输出路径等参数.
    """
    parser = argparse.ArgumentParser(
        description="绘制双向 doping 的 |staggered_mz| 曲线图, 并输出 SVG."
    )
    parser.add_argument(
        "--hole-series",
        action="append",
        default=[],
        help="hole 曲线定义, 格式为 '标签:CSV路径'. 可重复传入.",
    )
    parser.add_argument(
        "--electron-series",
        action="append",
        default=[],
        help="electron 曲线定义, 格式为 '标签:CSV路径'. 可重复传入.",
    )
    parser.add_argument(
        "--doping-col",
        type=str,
        default="doping",
        help="doping 列名, 默认 doping.",
    )
    parser.add_argument(
        "--mz-col",
        type=str,
        default="abs_staggered_mz",
        help="|staggered_mz| 列名, 默认 abs_staggered_mz.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="|staggered mz| vs doping",
        help="图标题.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plot/abs_staggered_mz_bipolar.svg",
        help="输出 SVG 路径, 默认 plot/abs_staggered_mz_bipolar.svg.",
    )
    return parser.parse_args()


def parse_series_spec(series_spec: str) -> Tuple[str, Path]:
    """用途: 解析单条曲线定义字符串.

    参数:
    - series_spec: str, 格式为 '标签:CSV路径'.

    返回:
    - tuple[str, Path], (label, csv_path).
    """
    if ":" not in series_spec:
        raise ValueError(
            f"无效曲线定义 '{series_spec}', 需要 '标签:CSV路径' 格式."
        )
    label, raw_path = series_spec.split(":", 1)
    label = label.strip()
    csv_path = Path(raw_path.strip())
    if label == "":
        raise ValueError(f"无效曲线定义 '{series_spec}', 标签不能为空.")
    return label, csv_path


def read_curve_data(
    csv_path: Path,
    doping_col: str,
    mz_col: str,
    force_sign: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """用途: 读取单条曲线数据并规范 doping 符号.

    参数:
    - csv_path: Path, CSV 文件路径.
    - doping_col: str, doping 列名.
    - mz_col: str, |staggered_mz| 列名.
    - force_sign: int, +1 表示强制到正半轴, -1 表示强制到负半轴.

    返回:
    - tuple[np.ndarray, np.ndarray], (x_doping, y_abs_mz), 均按 x 升序排列.
    """
    if force_sign not in (+1, -1):
        raise ValueError("force_sign 只能是 +1 或 -1.")
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")

    x_vals: List[float] = []
    y_vals: List[float] = []

    with csv_path.open("r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 无表头: {csv_path}")
        if doping_col not in reader.fieldnames:
            raise KeyError(f"{csv_path} 中缺少列: {doping_col}")
        if mz_col not in reader.fieldnames:
            raise KeyError(f"{csv_path} 中缺少列: {mz_col}")

        for row in reader:
            raw_doping = row.get(doping_col, "").strip()
            raw_mz = row.get(mz_col, "").strip()
            if raw_doping == "" or raw_mz == "":
                continue
            doping = abs(float(raw_doping)) * float(force_sign)
            abs_mz = abs(float(raw_mz))
            x_vals.append(doping)
            y_vals.append(abs_mz)

    if len(x_vals) == 0:
        raise ValueError(f"CSV 中没有有效数据: {csv_path}")

    x_arr = np.array(x_vals, dtype=float)
    y_arr = np.array(y_vals, dtype=float)
    order = np.argsort(x_arr)
    return x_arr[order], y_arr[order]


def draw_plot(
    hole_series: List[Tuple[str, Path]],
    electron_series: List[Tuple[str, Path]],
    doping_col: str,
    mz_col: str,
    title: str,
    output_path: Path,
) -> None:
    """用途: 绘制并保存双向 doping 的 |staggered_mz| 图.

    参数:
    - hole_series: list[(str, Path)], hole 曲线集合.
    - electron_series: list[(str, Path)], electron 曲线集合.
    - doping_col: str, doping 列名.
    - mz_col: str, |staggered_mz| 列名.
    - title: str, 图标题.
    - output_path: Path, 输出 SVG 文件路径.

    返回:
    - None.
    """
    fig, axis = plt.subplots(figsize=(8, 5), dpi=160)

    for label, csv_path in hole_series:
        x_vals, y_vals = read_curve_data(csv_path, doping_col, mz_col, force_sign=+1)
        axis.plot(x_vals, y_vals, marker="o", linewidth=1.6, markersize=4.5, label=f"{label} (hole)")

    for label, csv_path in electron_series:
        x_vals, y_vals = read_curve_data(csv_path, doping_col, mz_col, force_sign=-1)
        axis.plot(
            x_vals,
            y_vals,
            marker="s",
            linewidth=1.6,
            markersize=4.5,
            linestyle="--",
            label=f"{label} (electron)",
        )

    axis.axvline(0.0, color="gray", linewidth=1.0, linestyle=":")
    axis.grid(alpha=0.3)
    axis.set_xlabel("doping (hole > 0, electron < 0)")
    axis.set_ylabel("|staggered mz|")
    axis.set_title(title)
    axis.legend(frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """用途: 主流程入口.

    参数:
    - 无.

    返回:
    - None.
    """
    args = parse_arguments()

    hole_series = [parse_series_spec(item) for item in args.hole_series]
    electron_series = [parse_series_spec(item) for item in args.electron_series]

    if len(hole_series) == 0 and len(electron_series) == 0:
        raise ValueError("至少需要提供一条曲线: --hole-series 或 --electron-series.")

    output_path = Path(args.output)
    if output_path.suffix.lower() != ".svg":
        output_path = output_path.with_suffix(".svg")

    draw_plot(
        hole_series=hole_series,
        electron_series=electron_series,
        doping_col=args.doping_col,
        mz_col=args.mz_col,
        title=args.title,
        output_path=output_path,
    )
    print(f"[OK] 图已保存: {output_path}")


if __name__ == "__main__":
    main()
