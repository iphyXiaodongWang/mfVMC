#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用途: 对比 Hubbard PH 与 non-PH 结果的 x 方向 charge/spin profile.

输入:
- PH 结果根目录, 默认 `results/workflow/PH/Hubbard`.
- non-PH 结果根目录, 默认 `results/workflow/organize`.

输出:
- 每个 case 的 `profile_comparison/charge_x_profile.png`.
- 每个 case 的 `profile_comparison/staggered_sz_x_profile.png`.
- 汇总 CSV `hubbard_ph_vs_nonph_x_profiles.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Observable_Key_Pattern = re.compile(r"^(n|Sz)_(\d+)_(\d+)$")


@dataclass(frozen=True)
class HubbardObservableValue:
    """用途: 保存一个 Hubbard 格点 observable.

    参数:
    - observable: str, observable 类型, 目前为 `"n"` 或 `"Sz"`.
    - x_coord: int, JSON key 中的 x 坐标, 通常为 1-based.
    - y_coord: int, JSON key 中的 y 坐标, 通常为 1-based.
    - value: float, observable 的测量均值.

    返回:
    - dataclass 实例, 不直接返回其它值.
    """

    observable: str
    x_coord: int
    y_coord: int
    value: float


@dataclass(frozen=True)
class HubbardProfile:
    """用途: 保存一个 Hubbard 结果的 x 方向 profile.

    参数:
    - x_positions: np.ndarray, x 坐标数组.
    - average_charge: np.ndarray, 每个 x 的平均 charge.
    - staggered_sz: np.ndarray, 每个 x 的带符号 staggered Sz.

    返回:
    - dataclass 实例, 不直接返回其它值.
    """

    x_positions: np.ndarray
    average_charge: np.ndarray
    staggered_sz: np.ndarray


def parse_arguments() -> argparse.Namespace:
    """用途: 解析命令行参数.

    参数:
    - 无, 从命令行读取参数.

    返回:
    - argparse.Namespace, 包含 ph_root, nonph_root, output_csv, dpi.
    """
    parser = argparse.ArgumentParser(
        description="对比 Hubbard PH 与 non-PH 的 x 方向 charge/staggered Sz profile."
    )
    parser.add_argument(
        "--ph_root",
        type=str,
        default="results/workflow/PH/Hubbard",
        help="PH workflow 的 Hubbard 结果根目录.",
    )
    parser.add_argument(
        "--nonph_root",
        type=str,
        default="results/workflow/organize",
        help="non-PH workflow organize 结果根目录.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="输出 CSV 路径. 默认写入 ph_root/hubbard_ph_vs_nonph_x_profiles.csv.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="输出图片 dpi.",
    )
    args = parser.parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    return args


def configure_matplotlib_font() -> None:
    """用途: 配置 matplotlib 字体, 保证中文和负号能正常显示.

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
    """用途: 读取 `block_binning_mean.json`.

    参数:
    - input_path: Path, JSON 文件路径.

    返回:
    - Dict[str, Any], JSON 顶层字典.
    """
    with input_path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(f"输入 JSON 顶层必须是 object: {input_path}")
    return data


def parse_hubbard_observable_values(data: Mapping[str, Any]) -> List[HubbardObservableValue]:
    """用途: 从 JSON 字典中解析 Hubbard 格点 density/spin observable.

    参数:
    - data: Mapping[str, Any], `block_binning_mean.json` 内容.

    返回:
    - List[HubbardObservableValue], 已解析的 `n_x_y` 与 `Sz_x_y` 数据.
    """
    values: List[HubbardObservableValue] = []
    for key, raw_value in data.items():
        match = Observable_Key_Pattern.match(key)
        if match is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 的值不能转换为 float: {raw_value!r}") from exc
        values.append(
            HubbardObservableValue(
                observable=match.group(1),
                x_coord=int(match.group(2)),
                y_coord=int(match.group(3)),
                value=value,
            )
        )
    if not values:
        raise ValueError("输入 JSON 中没有找到 Hubbard density/spin observable.")
    return values


def infer_lattice_shape(values: Iterable[HubbardObservableValue]) -> Tuple[int, int, int, int]:
    """用途: 从 observable 坐标推断 lattice shape 与坐标偏移.

    参数:
    - values: Iterable[HubbardObservableValue], 已解析的 observable.

    返回:
    - Tuple[int, int, int, int], `(lx, ly, x_min, y_min)`.
    """
    value_list = list(values)
    x_values = [item.x_coord for item in value_list]
    y_values = [item.y_coord for item in value_list]
    if not x_values or not y_values:
        raise ValueError("无法从 observable 推断 lattice shape.")
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    return x_max - x_min + 1, y_max - y_min + 1, x_min, y_min


def build_observable_lookup(
    values: Iterable[HubbardObservableValue],
) -> Dict[Tuple[str, int, int], float]:
    """用途: 将 observable 列表转换为按 `(observable, x, y)` 查询的字典.

    参数:
    - values: Iterable[HubbardObservableValue], 已解析 observable.

    返回:
    - Dict[Tuple[str, int, int], float], observable 查询表.
    """
    return {
        (item.observable, item.x_coord, item.y_coord): item.value
        for item in values
    }


def compute_hubbard_x_profile(data: Mapping[str, Any]) -> HubbardProfile:
    """用途: 计算 Hubbard x 方向平均 charge 与带符号 staggered Sz.

    数学公式:
    - `n_avg(x) = (1 / Ly) * sum_y <n(x,y)>`.
    - `m_stag(x) = (1 / Ly) * sum_y (-1)^(x+y) * <Sz(x,y)>`.

    参数:
    - data: Mapping[str, Any], `block_binning_mean.json` 内容.

    返回:
    - HubbardProfile, 包含 x 坐标, 平均 charge, 以及带符号 staggered Sz.
    """
    values = parse_hubbard_observable_values(data)
    lx, ly, x_min, y_min = infer_lattice_shape(values)
    lookup = build_observable_lookup(values)

    x_positions = np.arange(x_min, x_min + lx, dtype=float)
    average_charge = np.zeros(lx, dtype=float)
    staggered_sz = np.zeros(lx, dtype=float)
    for x_index, x_coord in enumerate(range(x_min, x_min + lx)):
        charge_values = []
        staggered_values = []
        for y_coord in range(y_min, y_min + ly):
            charge_key = ("n", x_coord, y_coord)
            spin_key = ("Sz", x_coord, y_coord)
            if charge_key not in lookup:
                raise ValueError(f"缺少 observable: n_{x_coord}_{y_coord}")
            if spin_key not in lookup:
                raise ValueError(f"缺少 observable: Sz_{x_coord}_{y_coord}")
            charge_values.append(lookup[charge_key])
            stagger_sign = -1.0 if (x_coord + y_coord) % 2 else 1.0
            staggered_values.append(stagger_sign * lookup[spin_key])
        average_charge[x_index] = float(np.mean(charge_values))
        staggered_sz[x_index] = float(np.mean(staggered_values))
    return HubbardProfile(
        x_positions=x_positions,
        average_charge=average_charge,
        staggered_sz=staggered_sz,
    )


def assert_matching_x_positions(case_name: str, ph_profile: HubbardProfile, nonph_profile: HubbardProfile) -> None:
    """用途: 检查 PH 与 non-PH 的 x 坐标是否完全一致.

    参数:
    - case_name: str, 当前 case 名称, 用于错误信息.
    - ph_profile: HubbardProfile, PH profile.
    - nonph_profile: HubbardProfile, non-PH profile.

    返回:
    - None, 若不一致则抛出 ValueError.
    """
    if ph_profile.x_positions.shape != nonph_profile.x_positions.shape:
        raise ValueError(f"{case_name}: PH 与 non-PH 的 Lx 不一致.")
    if not np.allclose(ph_profile.x_positions, nonph_profile.x_positions):
        raise ValueError(f"{case_name}: PH 与 non-PH 的 x 坐标不一致.")


def plot_two_series_profile(
    x_positions: np.ndarray,
    nonph_values: np.ndarray,
    ph_values: np.ndarray,
    y_label: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """用途: 绘制 PH 与 non-PH 的单个 x-profile 对比图.

    参数:
    - x_positions: np.ndarray, x 坐标.
    - nonph_values: np.ndarray, non-PH profile 数值.
    - ph_values: np.ndarray, PH profile 数值.
    - y_label: str, y 轴标签.
    - title: str, 图片标题.
    - output_path: Path, 输出图片路径.
    - dpi: int, 图片 dpi.

    返回:
    - None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.plot(
        x_positions,
        nonph_values,
        marker="o",
        linewidth=1.8,
        markersize=4.2,
        label="non-PH",
        color="#1f77b4",
    )
    axis.plot(
        x_positions,
        ph_values,
        marker="s",
        linewidth=1.8,
        markersize=4.0,
        label="PH",
        color="#d62728",
    )
    axis.set_xlabel("x")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.set_xticks(x_positions)
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def find_case_directories(ph_root: Path) -> List[Path]:
    """用途: 查找 PH Hubbard 结果根目录中的 case 目录.

    参数:
    - ph_root: Path, PH 结果根目录.

    返回:
    - List[Path], 按名称排序后的 case 目录列表.
    """
    case_dirs = [
        child
        for child in ph_root.iterdir()
        if child.is_dir() and (child / "logs" / "block_binning_mean.json").exists()
    ]
    return sorted(case_dirs, key=lambda item: item.name)


def build_profile_csv_rows(
    case_name: str,
    nonph_profile: HubbardProfile,
    ph_profile: HubbardProfile,
) -> List[Dict[str, float | str]]:
    """用途: 为一个 case 构造 CSV 输出行.

    参数:
    - case_name: str, case 名称.
    - nonph_profile: HubbardProfile, non-PH profile.
    - ph_profile: HubbardProfile, PH profile.

    返回:
    - List[Dict[str, float | str]], 每个 x 坐标一行.
    """
    rows: List[Dict[str, float | str]] = []
    for index, x_coord in enumerate(ph_profile.x_positions):
        rows.append(
            {
                "case": case_name,
                "x": float(x_coord),
                "charge_nonph": float(nonph_profile.average_charge[index]),
                "charge_ph": float(ph_profile.average_charge[index]),
                "staggered_sz_nonph": float(nonph_profile.staggered_sz[index]),
                "staggered_sz_ph": float(ph_profile.staggered_sz[index]),
            }
        )
    return rows


def write_profile_csv(output_csv: Path, rows: List[Dict[str, float | str]]) -> None:
    """用途: 写出所有 case 的 x-profile 汇总 CSV.

    参数:
    - output_csv: Path, 输出 CSV 路径.
    - rows: List[Dict[str, float | str]], 待写出的数据行.

    返回:
    - None.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "x",
        "charge_nonph",
        "charge_ph",
        "staggered_sz_nonph",
        "staggered_sz_ph",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_one_case(case_dir: Path, nonph_root: Path, dpi: int) -> List[Dict[str, float | str]]:
    """用途: 处理一个 case, 生成两张对比图并返回 CSV 行.

    参数:
    - case_dir: Path, PH case 目录.
    - nonph_root: Path, non-PH organize 根目录.
    - dpi: int, 输出图片 dpi.

    返回:
    - List[Dict[str, float | str]], 当前 case 的 profile CSV 行.
    """
    case_name = case_dir.name
    ph_json_path = case_dir / "logs" / "block_binning_mean.json"
    nonph_json_path = nonph_root / case_name / "block_binning_mean.json"
    if not nonph_json_path.exists():
        raise FileNotFoundError(f"缺少 non-PH JSON: {nonph_json_path}")

    ph_profile = compute_hubbard_x_profile(load_measurement_json(ph_json_path))
    nonph_profile = compute_hubbard_x_profile(load_measurement_json(nonph_json_path))
    assert_matching_x_positions(case_name, ph_profile, nonph_profile)

    output_dir = case_dir / "profile_comparison"
    plot_two_series_profile(
        x_positions=ph_profile.x_positions,
        nonph_values=nonph_profile.average_charge,
        ph_values=ph_profile.average_charge,
        y_label=r"$\langle n(x) \rangle_y$",
        title=f"{case_name}: average charge profile",
        output_path=output_dir / "charge_x_profile.png",
        dpi=dpi,
    )
    plot_two_series_profile(
        x_positions=ph_profile.x_positions,
        nonph_values=nonph_profile.staggered_sz,
        ph_values=ph_profile.staggered_sz,
        y_label=r"$\langle (-1)^{x+y} S^z(x,y) \rangle_y$",
        title=f"{case_name}: signed staggered Sz profile",
        output_path=output_dir / "staggered_sz_x_profile.png",
        dpi=dpi,
    )
    return build_profile_csv_rows(case_name, nonph_profile, ph_profile)


def main() -> None:
    """用途: 命令行入口, 批量生成 Hubbard PH/non-PH x-profile 对比图.

    参数:
    - 无, 从命令行读取参数.

    返回:
    - None.
    """
    args = parse_arguments()
    configure_matplotlib_font()

    ph_root = Path(args.ph_root)
    nonph_root = Path(args.nonph_root)
    output_csv = Path(args.output_csv) if args.output_csv else ph_root / "hubbard_ph_vs_nonph_x_profiles.csv"

    case_dirs = find_case_directories(ph_root)
    if not case_dirs:
        raise FileNotFoundError(f"没有在 PH 根目录中找到 case JSON: {ph_root}")

    all_rows: List[Dict[str, float | str]] = []
    for case_dir in case_dirs:
        case_rows = process_one_case(case_dir, nonph_root, args.dpi)
        all_rows.extend(case_rows)
        print(f"[OK] {case_dir.name}: generated profile comparison plots.")

    write_profile_csv(output_csv, all_rows)
    print(f"[OK] wrote CSV: {output_csv}")
    print(f"[OK] total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
