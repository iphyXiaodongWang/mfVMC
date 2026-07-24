#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用途: 绘制 PBC twist Emery measure 输出的 x 方向 staggered m_z 和 charge profiles."""

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


EMERY_OBSERVABLE_KEY_PATTERN = re.compile(
    r"^(n|Sz)_(d|px|py)_(-?\d+)_(\d+)$"
)


class EmeryObservableValue(NamedTuple):
    """用途: 保存一个已经解析的 Emery 局域 observable."""

    observable: str
    orbital: str
    x_cell: int
    y_cell: int
    value: float


class OrbitalChargeProfile(NamedTuple):
    """用途: 保存一个 orbital 的实际 x 坐标和 y 向平均 charge."""

    x_positions: np.ndarray
    values: np.ndarray


def parse_arguments() -> argparse.Namespace:
    """用途: 解析绘图脚本的命令行参数.

    参数:
    - 无.

    返回:
    - `argparse.Namespace`: 输入 JSON、输出 PNG、可选 lattice size 和 dpi.
    """
    parser = argparse.ArgumentParser(
        description=(
            "读取 twist Emery block_binning_mean.json, 绘制 x 方向 "
            "staggered Cu-d m_z 和 d/px/py charge profiles."
        )
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
        help="输出 PNG 路径. 默认保存到输入 JSON 同目录.",
    )
    parser.add_argument(
        "--lx",
        type=int,
        default=None,
        help="Cu unit cell 的 x 方向长度. 默认从 JSON keys 推断.",
    )
    parser.add_argument(
        "--ly",
        type=int,
        default=None,
        help="Cu unit cell 的 y 方向长度. 默认从 JSON keys 推断.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="输出图片 dpi, 默认 220.",
    )
    arguments = parser.parse_args()
    if arguments.lx is not None and arguments.lx <= 0:
        raise ValueError("--lx must be positive.")
    if arguments.ly is not None and arguments.ly <= 0:
        raise ValueError("--ly must be positive.")
    if arguments.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    return arguments


def configure_matplotlib_font() -> None:
    """用途: 配置 Matplotlib 字体和负号显示.

    参数:
    - 无.

    返回:
    - `None`.
    """
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_measurement_json(input_path: Path) -> Dict[str, Any]:
    """用途: 读取 twist Emery measure 生成的 JSON.

    参数:
    - `input_path::Path`: `block_binning_mean.json` 路径.

    返回:
    - `Dict[str, Any]`: observable 名称到均值的映射.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"输入 JSON 不存在: {input_path}")
    with input_path.open("r", encoding="utf-8") as file_object:
        measurement_data = json.load(file_object)
    if not isinstance(measurement_data, dict):
        raise ValueError(f"输入 JSON 顶层必须是 object: {input_path}")
    return measurement_data


def parse_emery_observable_values(
    measurement_data: Dict[str, Any],
) -> List[EmeryObservableValue]:
    """用途: 从 measure JSON 中解析 Emery 局域 density 和 spin observables.

    参数:
    - `measurement_data::Dict[str, Any]`: JSON 顶层字典.

    返回:
    - `List[EmeryObservableValue]`: 匹配 `n/Sz_orbital_x_y` 的数值.
    """
    observable_values: List[EmeryObservableValue] = []
    for observable_name, raw_value in measurement_data.items():
        key_match = EMERY_OBSERVABLE_KEY_PATTERN.match(observable_name)
        if key_match is None:
            continue
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{observable_name} 的值不能转换为 float: {raw_value!r}"
            ) from error
        observable_values.append(
            EmeryObservableValue(
                observable=key_match.group(1),
                orbital=key_match.group(2),
                x_cell=int(key_match.group(3)),
                y_cell=int(key_match.group(4)),
                value=numeric_value,
            )
        )
    if not observable_values:
        raise ValueError("输入 JSON 中没有 Emery density/spin observables.")
    return observable_values


def infer_lattice_shape(
    observable_values: Iterable[EmeryObservableValue],
    lx: Optional[int],
    ly: Optional[int],
) -> Tuple[int, int]:
    """用途: 从 Cu-d observable keys 推断或校验 PBC Cu unit-cell 尺寸.

    参数:
    - `observable_values::Iterable[EmeryObservableValue]`: 已解析 observables.
    - `lx, ly::Optional[int]`: 用户指定尺寸, `None` 表示自动推断.

    返回:
    - `Tuple[int, int]`: `(lx, ly)`.
    """
    value_list = list(observable_values)
    d_values = [value for value in value_list if value.orbital == "d"]
    if not d_values:
        raise ValueError("无法推断 lattice shape: 缺少 d orbital observables.")
    inferred_lx = max(value.x_cell for value in d_values)
    inferred_ly = max(value.y_cell for value in d_values)
    final_lx = inferred_lx if lx is None else lx
    final_ly = inferred_ly if ly is None else ly
    if final_lx < inferred_lx:
        raise ValueError(
            f"--lx={final_lx} 小于 JSON 中最大 d orbital x={inferred_lx}."
        )
    if final_ly < inferred_ly:
        raise ValueError(
            f"--ly={final_ly} 小于 JSON 中最大 d orbital y={inferred_ly}."
        )
    return final_lx, final_ly


def build_value_lookup(
    observable_values: Iterable[EmeryObservableValue],
) -> Dict[Tuple[str, str, int, int], float]:
    """用途: 构造 `(observable, orbital, x, y)` 到数值的查询表.

    参数:
    - `observable_values::Iterable[EmeryObservableValue]`: 已解析 observables.

    返回:
    - `Dict[Tuple[str, str, int, int], float]`: 局域 observable 查询表.
    """
    return {
        (
            observable_value.observable,
            observable_value.orbital,
            observable_value.x_cell,
            observable_value.y_cell,
        ): observable_value.value
        for observable_value in observable_values
    }


def compute_staggered_cu_mz_profile(
    value_lookup: Dict[Tuple[str, str, int, int], float],
    lx: int,
    ly: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """用途: 计算每个 Cu column 的 staggered d-orbital magnetization.

    科学计算公式:
    - `m_stag(x) = (1/Ly) * sum_y (-1)^(x+y) * <S^z_d(x,y)>`.

    参数:
    - `value_lookup::Dict`: 局域 observable 查询表.
    - `lx, ly::int`: Cu unit-cell 尺寸.

    返回:
    - `Tuple[np.ndarray, np.ndarray]`: Cu x 坐标和 `m_stag(x)`.
    """
    x_positions = np.arange(1, lx + 1, dtype=float)
    staggered_mz = np.zeros(lx, dtype=float)
    for x_cell in range(1, lx + 1):
        column_values = []
        for y_cell in range(1, ly + 1):
            observable_key = ("Sz", "d", x_cell, y_cell)
            if observable_key not in value_lookup:
                raise ValueError(f"缺少 observable: Sz_d_{x_cell}_{y_cell}")
            staggered_sign = -1.0 if (x_cell + y_cell) % 2 else 1.0
            column_values.append(
                staggered_sign * value_lookup[observable_key]
            )
        staggered_mz[x_cell - 1] = float(np.mean(column_values))
    return x_positions, staggered_mz


def compute_orbital_charge_profiles(
    value_lookup: Dict[Tuple[str, str, int, int], float],
    lx: int,
    ly: int,
) -> Dict[str, OrbitalChargeProfile]:
    """用途: 计算 d、p_x、p_y 三个 orbital 的 x 方向平均 charge.

    科学计算公式:
    - `n_orb(x) = (1/Ly) * sum_y <n_orb(x,y)>`.
    - 实际横坐标为 `x_d=x`, `x_px=x+1/2`, `x_py=x`.

    参数:
    - `value_lookup::Dict`: 局域 observable 查询表.
    - `lx, ly::int`: Cu unit-cell 尺寸.

    返回:
    - `Dict[str, OrbitalChargeProfile]`: `d/px/py` 三个 charge profiles.
    """
    charge_profiles: Dict[str, OrbitalChargeProfile] = {}
    for orbital in ("d", "px", "py"):
        x_offset = 0.5 if orbital == "px" else 0.0
        x_positions = np.asarray(
            [x_cell + x_offset for x_cell in range(1, lx + 1)],
            dtype=float,
        )
        average_values = np.zeros(lx, dtype=float)
        for x_cell in range(1, lx + 1):
            column_values = []
            for y_cell in range(1, ly + 1):
                observable_key = ("n", orbital, x_cell, y_cell)
                if observable_key not in value_lookup:
                    raise ValueError(
                        f"缺少 observable: n_{orbital}_{x_cell}_{y_cell}"
                    )
                column_values.append(value_lookup[observable_key])
            average_values[x_cell - 1] = float(np.mean(column_values))
        charge_profiles[orbital] = OrbitalChargeProfile(
            x_positions=x_positions,
            values=average_values,
        )
    return charge_profiles


def plot_x_profiles(
    cu_x_positions: np.ndarray,
    staggered_mz: np.ndarray,
    charge_profiles: Dict[str, OrbitalChargeProfile],
    output_path: Path,
    dpi: int,
) -> None:
    """用途: 绘制并保存 staggered m_z 和 orbital-resolved charge 双面板图.

    参数:
    - `cu_x_positions::np.ndarray`: Cu-d 的 x 坐标.
    - `staggered_mz::np.ndarray`: staggered Cu-d magnetization.
    - `charge_profiles::Dict[str, OrbitalChargeProfile]`: 三轨道 charge.
    - `output_path::Path`: 输出 PNG 路径.
    - `dpi::int`: 输出分辨率.

    返回:
    - `None`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)

    axes[0].plot(
        cu_x_positions,
        staggered_mz,
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        color="#1f77b4",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    axes[0].set_ylabel(r"$m_{\mathrm{stag}}(x)$")
    axes[0].set_title("Cu-d staggered magnetization")
    axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    plot_styles = {
        "d": ("o", "#1f77b4", r"Cu $d$"),
        "px": ("s", "#d62728", r"O $p_x$"),
        "py": ("^", "#2ca02c", r"O $p_y$"),
    }
    for orbital in ("d", "px", "py"):
        marker, color, label = plot_styles[orbital]
        profile = charge_profiles[orbital]
        axes[1].plot(
            profile.x_positions,
            profile.values,
            marker=marker,
            linewidth=1.7,
            markersize=4.2,
            color=color,
            label=label,
        )
    axes[1].set_xlabel("x lattice coordinate")
    axes[1].set_ylabel(r"$\langle n_{\mathrm{orb}}\rangle_y$")
    axes[1].set_title("Orbital-resolved charge profiles")
    axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    axes[1].legend(frameon=False)

    maximum_x = max(
        float(np.max(profile.x_positions))
        for profile in charge_profiles.values()
    )
    axes[1].set_xlim(
        float(np.min(cu_x_positions)) - 0.25,
        maximum_x + 0.25,
    )
    axes[1].set_xticks(
        np.arange(
            np.floor(float(np.min(cu_x_positions))),
            np.ceil(maximum_x) + 1.0,
            1.0,
        )
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """用途: CLI 入口, 读取 measure JSON 并生成 x-profile PNG.

    参数:
    - 无.

    返回:
    - `None`.
    """
    arguments = parse_arguments()
    configure_matplotlib_font()
    input_path = Path(arguments.input)
    output_path = (
        Path(arguments.output)
        if arguments.output
        else input_path.with_name("twist_emery_x_profiles.png")
    )
    measurement_data = load_measurement_json(input_path)
    observable_values = parse_emery_observable_values(measurement_data)
    lx, ly = infer_lattice_shape(
        observable_values,
        arguments.lx,
        arguments.ly,
    )
    value_lookup = build_value_lookup(observable_values)
    cu_x_positions, staggered_mz = compute_staggered_cu_mz_profile(
        value_lookup,
        lx,
        ly,
    )
    charge_profiles = compute_orbital_charge_profiles(value_lookup, lx, ly)
    plot_x_profiles(
        cu_x_positions,
        staggered_mz,
        charge_profiles,
        output_path,
        arguments.dpi,
    )
    print(f"Saved twist Emery x profiles to: {output_path}")


if __name__ == "__main__":
    main()
