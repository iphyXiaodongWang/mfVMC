#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制 |staggered_mz| 随 doping 变化的双向图:
1) x > 0 表示 hole doping;
2) x < 0 表示 electron doping;
3) 支持多条 hole 曲线, 并带图例;
4) 输出为 SVG 矢量图.

数据兼容规则:
1) 输入可以是目录或 CSV 文件;
2) 若输入是目录, 默认读取 summary_min_sector_staggered_mz_S_pi_pi.csv;
3) 若 CSV 没有 doping 列, 则自动使用 Ndefect / L^2 计算 doping.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def normalize_key(raw_key: str) -> str:
    """
    归一化列名字符串, 便于兼容引号与大小写差异.

    参数:
    - raw_key: str, 原始列名.

    返回:
    - str, 归一化后的列名.
    """
    return raw_key.strip().strip('"').strip("'").strip().lower()


def parse_series_entry(series_entry: str) -> Tuple[str, Path, Optional[int]]:
    """
    解析单条曲线定义.

    支持两种格式:
    1) "label:path/to/file_or_dir"
    2) "label:path/to/file_or_dir:lattice_size"

    参数:
    - series_entry: str, 单条输入字符串.

    返回:
    - tuple[str, Path, Optional[int]], (曲线标签, 路径, 可选 L).
    """
    if ":" not in series_entry:
        raise ValueError(
            f"无效输入 '{series_entry}', 需要格式 label:path 或 label:path:L"
        )

    pieces = series_entry.split(":")
    series_label = pieces[0].strip()
    if series_label == "":
        raise ValueError(f"无效输入 '{series_entry}', label 不能为空")

    lattice_size: Optional[int] = None
    if len(pieces) >= 3 and re.fullmatch(r"\d+", pieces[-1].strip()):
        lattice_size = int(pieces[-1].strip())
        path_text = ":".join(pieces[1:-1]).strip()
    else:
        path_text = ":".join(pieces[1:]).strip()

    if path_text == "":
        raise ValueError(f"无效输入 '{series_entry}', path 不能为空")

    return series_label, Path(path_text).expanduser().resolve(), lattice_size


def infer_lattice_size_from_path(path_obj: Path) -> Optional[int]:
    """
    从路径中推断格点线长 L.

    支持匹配:
    - .../L_20/...
    - .../L20/...

    参数:
    - path_obj: Path, 目录或文件路径.

    返回:
    - Optional[int], 推断得到的 L, 失败返回 None.
    """
    match_obj = re.search(r"[\\/]+L_?(\d+)(?:[\\/]+|$)", str(path_obj))
    if match_obj is None:
        return None
    return int(match_obj.group(1))


def build_lattice_legend_label(
    series_label: str,
    raw_path: Path,
    lattice_size: Optional[int],
) -> str:
    """
    构造图例标签.

    规则:
    1) 优先使用显式传入的 lattice_size;
    2) 若缺失, 尝试从路径推断 L;
    3) 若仍失败, 回退到原始 series_label.

    参数:
    - series_label: str, 原始标签.
    - raw_path: Path, 数据路径.
    - lattice_size: Optional[int], 可选 L.

    返回:
    - str, 图例标签, 优先为 "L=xx".
    """
    final_lattice_size = lattice_size
    if final_lattice_size is None or final_lattice_size <= 0:
        final_lattice_size = infer_lattice_size_from_path(raw_path)

    if final_lattice_size is None or final_lattice_size <= 0:
        return series_label
    return f"L={final_lattice_size}"


def resolve_summary_csv_path(raw_path: Path, summary_name: str) -> Path:
    """
    解析输入路径为实际 CSV 文件路径.

    参数:
    - raw_path: Path, 用户输入路径.
    - summary_name: str, 当 raw_path 是目录时使用的默认文件名.

    返回:
    - Path, 实际 CSV 文件路径.
    """
    if raw_path.is_dir():
        return (raw_path / summary_name).resolve()
    return raw_path.resolve()


def parse_float_or_nan(raw_value: str) -> float:
    """
    将字符串转换为 float, 失败返回 NaN.

    参数:
    - raw_value: str, 原始字符串.

    返回:
    - float, 数值或 NaN.
    """
    text = raw_value.strip().strip('"').strip("'")
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def extract_column_value(
    row_dict: Dict[str, str],
    key_map: Dict[str, str],
    target_key: str,
) -> str:
    """
    从 CSV 当前行中按规范化列名提取字段值.

    参数:
    - row_dict: Dict[str, str], 当前行字典.
    - key_map: Dict[str, str], 规范化列名到原列名的映射.
    - target_key: str, 目标列名.

    返回:
    - str, 字段值, 若列不存在则返回空字符串.
    """
    norm_key = normalize_key(target_key)
    raw_key = key_map.get(norm_key)
    if raw_key is None:
        return ""
    return row_dict.get(raw_key, "")


def load_one_series(
    series_label: str,
    raw_path: Path,
    lattice_size: Optional[int],
    doping_col: str,
    ndefect_col: str,
    staggered_mz_col: str,
    doping_type: str,
    summary_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取并转换单条曲线.

    转换公式:
    - 若存在 doping 列: doping = |doping| 或 -|doping|;
    - 若无 doping 列: doping = Ndefect / L^2, 再按正负方向映射;
    - y 轴统一使用 |staggered_mz|.

    参数:
    - series_label: str, 曲线标签.
    - raw_path: Path, 目录或 CSV 路径.
    - lattice_size: Optional[int], 可选 L.
    - doping_col: str, doping 列名.
    - ndefect_col: str, Ndefect 列名.
    - staggered_mz_col: str, staggered_mz 列名.
    - doping_type: str, "hole" 或 "electron".
    - summary_name: str, 目录输入时读取的默认文件名.

    返回:
    - tuple[np.ndarray, np.ndarray], (x_doping, y_abs_staggered_mz).
    """
    csv_path = resolve_summary_csv_path(raw_path, summary_name)
    if not csv_path.is_file():
        raise FileNotFoundError(f"[{series_label}] 文件不存在: {csv_path}")

    if doping_type not in ("hole", "electron"):
        raise ValueError(f"[{series_label}] doping_type 仅支持 hole/electron")
    sign = 1.0 if doping_type == "hole" else -1.0

    if lattice_size is None:
        lattice_size = infer_lattice_size_from_path(csv_path)

    rows_xy: List[Tuple[float, float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"[{series_label}] CSV 无表头: {csv_path}")

        key_map: Dict[str, str] = {
            normalize_key(raw_key): raw_key for raw_key in reader.fieldnames
        }

        has_doping = normalize_key(doping_col) in key_map
        has_ndefect = normalize_key(ndefect_col) in key_map

        if not has_doping and not has_ndefect:
            raise KeyError(
                f"[{series_label}] {csv_path} 既没有 '{doping_col}' 也没有 '{ndefect_col}'"
            )

        if not has_doping and (lattice_size is None or lattice_size <= 0):
            raise ValueError(
                f"[{series_label}] {csv_path} 需使用 Ndefect 转 doping, 但缺少有效 L; "
                "请在 series 末尾添加 ':L'"
            )

        has_mz = normalize_key(staggered_mz_col) in key_map
        has_abs_mz = normalize_key("abs_staggered_mz") in key_map
        if not has_mz and not has_abs_mz:
            raise KeyError(
                f"[{series_label}] {csv_path} 缺少 '{staggered_mz_col}' 或 'abs_staggered_mz' 列"
            )

        for row_dict in reader:
            if has_doping:
                raw_doping = extract_column_value(row_dict, key_map, doping_col)
                doping_value = parse_float_or_nan(raw_doping)
            else:
                raw_ndefect = extract_column_value(row_dict, key_map, ndefect_col)
                ndefect_value = parse_float_or_nan(raw_ndefect)
                doping_value = ndefect_value / float(lattice_size * lattice_size)

            if has_mz:
                raw_mz = extract_column_value(row_dict, key_map, staggered_mz_col)
                mz_value = parse_float_or_nan(raw_mz)
            else:
                raw_mz = extract_column_value(row_dict, key_map, "abs_staggered_mz")
                mz_value = parse_float_or_nan(raw_mz)

            if math.isnan(doping_value) or math.isnan(mz_value):
                continue

            x_value = sign * abs(doping_value)
            y_value = abs(mz_value)
            rows_xy.append((x_value, y_value))

    if len(rows_xy) == 0:
        raise ValueError(f"[{series_label}] {csv_path} 没有可用数值行")

    doping_to_values: Dict[float, List[float]] = defaultdict(list)
    for x_value, y_value in rows_xy:
        doping_to_values[x_value].append(y_value)

    sorted_x = np.array(sorted(doping_to_values.keys()), dtype=float)
    sorted_y = np.array(
        [float(np.mean(doping_to_values[x_val])) for x_val in sorted_x],
        dtype=float,
    )
    return sorted_x, sorted_y


def collect_series_data(
    hole_series_inputs: List[str],
    electron_series_inputs: List[str],
    doping_col: str,
    ndefect_col: str,
    staggered_mz_col: str,
    summary_name: str,
) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
    """
    收集所有曲线数据.

    参数:
    - hole_series_inputs: List[str], hole 曲线定义列表.
    - electron_series_inputs: List[str], electron 曲线定义列表.
    - doping_col: str, doping 列名.
    - ndefect_col: str, Ndefect 列名.
    - staggered_mz_col: str, staggered_mz 列名.
    - summary_name: str, 目录输入时默认 summary 名称.

    返回:
    - List[Tuple[str, str, np.ndarray, np.ndarray]]
      - (曲线标签, doping_type, x数组, y数组).
    """
    all_series: List[Tuple[str, str, np.ndarray, np.ndarray]] = []

    for series_entry in hole_series_inputs:
        series_label, raw_path, lattice_size = parse_series_entry(series_entry)
        x_values, y_values = load_one_series(
            series_label=series_label,
            raw_path=raw_path,
            lattice_size=lattice_size,
            doping_col=doping_col,
            ndefect_col=ndefect_col,
            staggered_mz_col=staggered_mz_col,
            doping_type="hole",
            summary_name=summary_name,
        )
        legend_label = build_lattice_legend_label(
            series_label=series_label,
            raw_path=raw_path,
            lattice_size=lattice_size,
        )
        all_series.append((legend_label, "hole", x_values, y_values))

    for series_entry in electron_series_inputs:
        series_label, raw_path, lattice_size = parse_series_entry(series_entry)
        x_values, y_values = load_one_series(
            series_label=series_label,
            raw_path=raw_path,
            lattice_size=lattice_size,
            doping_col=doping_col,
            ndefect_col=ndefect_col,
            staggered_mz_col=staggered_mz_col,
            doping_type="electron",
            summary_name=summary_name,
        )
        legend_label = build_lattice_legend_label(
            series_label=series_label,
            raw_path=raw_path,
            lattice_size=lattice_size,
        )
        all_series.append((legend_label, "electron", x_values, y_values))

    if len(all_series) == 0:
        raise ValueError("至少需要提供一条 --hole-series 或 --electron-series")

    return all_series


def plot_series_data(
    all_series: List[Tuple[str, str, np.ndarray, np.ndarray]],
    output_path: Path,
    title: str,
) -> None:
    """
    绘制并保存图像.

    参数:
    - all_series: List[Tuple[str, str, np.ndarray, np.ndarray]], 全部曲线.
    - output_path: Path, 输出路径.
    - title: str, 图标题.

    返回:
    - None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)

    for series_label, doping_type, x_values, y_values in all_series:
        marker_style = "o" if doping_type == "hole" else "s"
        line_style = "-" if doping_type == "hole" else "--"
        axis.plot(
            x_values,
            y_values,
            marker=marker_style,
            linestyle=line_style,
            linewidth=1.8,
            markersize=5.2,
            label=series_label,
        )

    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axis.set_xlabel("doping (+: hole, -: electron)")
    axis.set_ylabel(r"$m_z$")
    if title.strip() != "":
        axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)

    figure.savefig(output_path, format="svg")
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器.

    参数:
    - 无.

    返回:
    - argparse.ArgumentParser, 参数解析器.
    """
    parser = argparse.ArgumentParser(
        description="绘制双向 doping 的 |staggered_mz| 曲线图, 输出 SVG."
    )
    parser.add_argument(
        "--hole-series",
        action="append",
        default=[],
        help=(
            'hole 曲线, 可重复传入, 格式 "label:path" 或 "label:path:L"; '
            "path 可以是目录或 CSV"
        ),
    )
    parser.add_argument(
        "--electron-series",
        action="append",
        default=[],
        help=(
            'electron 曲线, 可重复传入, 格式 "label:path" 或 "label:path:L"; '
            "path 可以是目录或 CSV"
        ),
    )
    parser.add_argument(
        "--doping-col",
        default="doping",
        help="CSV 中 doping 列名, 默认: doping",
    )
    parser.add_argument(
        "--ndefect-col",
        default="Ndefect",
        help="CSV 中 Ndefect 列名, 默认: Ndefect",
    )
    parser.add_argument(
        "--mz-col",
        default="staggered_mz",
        help="CSV 中 staggered_mz 列名, 默认: staggered_mz",
    )
    parser.add_argument(
        "--summary-name",
        default="summary_min_sector_staggered_mz_S_pi_pi.csv",
        help="series 为目录时默认读取的文件名",
    )
    parser.add_argument(
        "--title",
        default="",
        help="图标题, 默认留空表示不显示标题",
    )
    parser.add_argument(
        "--output",
        default="plot/abs_staggered_mz_bipolar.svg",
        help="输出 SVG 文件路径, 默认: plot/abs_staggered_mz_bipolar.svg",
    )
    return parser


def main() -> None:
    """
    主函数: 读取参数, 汇总数据并绘图.

    参数:
    - 无.

    返回:
    - None.
    """
    args = build_argument_parser().parse_args()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.suffix.lower() != ".svg":
        output_path = output_path.with_suffix(".svg")

    all_series = collect_series_data(
        hole_series_inputs=args.hole_series,
        electron_series_inputs=args.electron_series,
        doping_col=args.doping_col,
        ndefect_col=args.ndefect_col,
        staggered_mz_col=args.mz_col,
        summary_name=args.summary_name,
    )

    plot_series_data(
        all_series=all_series,
        output_path=output_path,
        title=args.title,
    )
    print(f"[OK] 输出完成: {output_path}")


if __name__ == "__main__":
    main()
