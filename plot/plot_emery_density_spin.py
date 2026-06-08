#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用途: 从 Emery `measure` 输出绘制三带格点上的 density 和 spin 分布.

输入文件:
- `block_binning_mean.json`, 其中 observable key 来自 `Emery.jl`:
  `n_d_x_y`, `Sz_d_x_y`, `n_px_x_y`, `Sz_px_x_y`, `n_py_x_y`, `Sz_py_x_y`.

输出文件:
- `emery_density_distribution.png`
- `emery_spin_distribution.png`
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


EMERY_KEY_PATTERN = re.compile(r"^(n|Sz)_(d|px|py)_(-?\d+)_(\d+)$")
ORBITAL_MARKERS = {"d": "o", "px": "s", "py": "^"}
ORBITAL_LABELS = {"d": "Cu d", "px": "O px", "py": "O py"}


class EmeryObservableValue(NamedTuple):
    """用途: 保存单个 Emery orbital observable 的解析结果."""

    observable: str
    orbital: str
    x_cell: int
    y_cell: int
    x_plot: float
    y_plot: float
    value: float


def parse_arguments() -> argparse.Namespace:
    """用途: 解析命令行参数.

    参数:
    - 无.

    返回:
    - argparse.Namespace, 包含输入 JSON, 输出目录, 可选 `lx/ly`, dpi 和数值标注开关.
    """
    parser = argparse.ArgumentParser(
        description="读取 Emery measure 的 block_binning_mean.json 并绘制 density/spin 分布图"
    )
    parser.add_argument(
        "input",
        type=str,
        help="输入 JSON 路径, 例如 logs/block_binning_mean.json",
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
    parser.add_argument(
        "--annotate_values",
        action="store_true",
        help="在每个 orbital 附近标注数值. 大体系可能较拥挤.",
    )
    args = parser.parse_args()
    if args.lx is not None and args.lx <= 0:
        raise ValueError("--lx must be positive")
    if args.ly is not None and args.ly <= 0:
        raise ValueError("--ly must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    return args


def configure_matplotlib_font() -> None:
    """用途: 配置 matplotlib 字体, 保证中文标题或报错文本可显示.

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


def parse_emery_measure_key(key: str) -> Optional[Tuple[str, str, int, int]]:
    """用途: 解析 Emery observable key.

    参数:
    - `key`: str, 例如 `n_d_1_1`, `Sz_px_0_3`.

    返回:
    - `None`: key 不属于 Emery density/spin observable.
    - `(observable, orbital, x_cell, y_cell)`: 解析成功时的元组.
    """
    match = EMERY_KEY_PATTERN.match(key)
    if match is None:
        return None
    observable = match.group(1)
    orbital = match.group(2)
    x_cell = int(match.group(3))
    y_cell = int(match.group(4))
    return observable, orbital, x_cell, y_cell


def compute_emery_orbital_position(orbital: str, x_cell: int, y_cell: int) -> Tuple[float, float]:
    """用途: 将 Emery orbital cell 坐标转换成绘图坐标.

    参数:
    - `orbital`: str, 取值为 `d`, `px`, `py`.
    - `x_cell`: int, Emery.jl 中 observable key 的 x 坐标.
    - `y_cell`: int, Emery.jl 中 observable key 的 y 坐标.

    返回:
    - `(x_plot, y_plot)`: float 元组.

    公式:
    - Cu `d(x,y)` 位于 `(x, y)`;
    - O `px(x,y)` 位于 `(x + 0.5, y)`;
    - O `py(x,y)` 位于 `(x, y + 0.5)`.
    """
    if orbital == "d":
        return float(x_cell), float(y_cell)
    if orbital == "px":
        return float(x_cell) + 0.5, float(y_cell)
    if orbital == "py":
        return float(x_cell), float(y_cell) + 0.5
    raise ValueError(f"未知 Emery orbital: {orbital}")


def collect_emery_observable_values(
    data: Dict[str, Any],
    observable: str,
) -> list[EmeryObservableValue]:
    """用途: 从 JSON 字典中收集指定 Emery observable 的所有 orbital 值.

    参数:
    - `data`: Dict[str, Any], measure 均值字典.
    - `observable`: str, 取值为 `n` 或 `Sz`.

    返回:
    - `list[EmeryObservableValue]`: 按 JSON 原始顺序保存的解析结果.
    """
    if observable not in ("n", "Sz"):
        raise ValueError("observable must be 'n' or 'Sz'")

    values: list[EmeryObservableValue] = []
    for key, raw_value in data.items():
        parsed = parse_emery_measure_key(key)
        if parsed is None:
            continue
        found_observable, orbital, x_cell, y_cell = parsed
        if found_observable != observable:
            continue
        x_plot, y_plot = compute_emery_orbital_position(orbital, x_cell, y_cell)
        values.append(
            EmeryObservableValue(
                observable=found_observable,
                orbital=orbital,
                x_cell=x_cell,
                y_cell=y_cell,
                x_plot=x_plot,
                y_plot=y_plot,
                value=float(raw_value),
            )
        )
    return values


def infer_emery_shape(
    values: Iterable[EmeryObservableValue],
    lx_user: Optional[int],
    ly_user: Optional[int],
) -> Tuple[int, int]:
    """用途: 从 observable key 推断或校验 Emery Cu cell 尺寸.

    参数:
    - `values`: Iterable[EmeryObservableValue], 已解析的 observable 值.
    - `lx_user`: Optional[int], 用户指定的 Lx.
    - `ly_user`: Optional[int], 用户指定的 Ly.

    返回:
    - `(lx, ly)`: Emery Cu cell 尺寸.
    """
    value_list = list(values)
    if not value_list:
        raise ValueError("没有可用于推断尺寸的 Emery observable")

    inferred_lx = max(item.x_cell for item in value_list)
    inferred_ly = max(item.y_cell for item in value_list)
    lx = lx_user if lx_user is not None else inferred_lx
    ly = ly_user if ly_user is not None else inferred_ly

    for item in value_list:
        if item.orbital == "px":
            valid_x = 0 <= item.x_cell <= lx
        else:
            valid_x = 1 <= item.x_cell <= lx
        valid_y = 1 <= item.y_cell <= ly
        if not (valid_x and valid_y):
            raise ValueError(
                "observable 坐标超出 Emery 尺寸: "
                f"{item.observable}_{item.orbital}_{item.x_cell}_{item.y_cell}, "
                f"Lx={lx}, Ly={ly}"
            )
    return lx, ly


def setup_emery_axis(axis: plt.Axes, lx: int, ly: int, title_text: str) -> None:
    """用途: 配置 Emery 三带格点图的坐标轴.

    参数:
    - `axis`: plt.Axes, 目标坐标轴.
    - `lx, ly`: int, Cu cell 尺寸.
    - `title_text`: str, 图标题.

    返回:
    - None.
    """
    axis.set_xlim(0.0, lx + 0.8)
    axis.set_ylim(0.5, ly + 0.9)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title_text)
    axis.set_xticks(np.arange(0.5, lx + 1.0, 0.5))
    axis.set_yticks(np.arange(1.0, ly + 1.0, 1.0))
    axis.grid(color="#d0d0d0", linewidth=0.6, alpha=0.7)


def annotate_values(axis: plt.Axes, values: list[EmeryObservableValue], fmt: str) -> None:
    """用途: 在 orbital 旁边标注 observable 数值.

    参数:
    - `axis`: plt.Axes, 目标坐标轴.
    - `values`: list[EmeryObservableValue], 待标注数据.
    - `fmt`: str, Python 格式化字符串.

    返回:
    - None.
    """
    for item in values:
        axis.text(
            item.x_plot + 0.04,
            item.y_plot + 0.04,
            fmt.format(item.value),
            fontsize=7,
            ha="left",
            va="bottom",
            color="#222222",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 0.15},
        )


def plot_emery_observable_map(
    values: list[EmeryObservableValue],
    lx: int,
    ly: int,
    output_path: Path,
    title_text: str,
    colorbar_label: str,
    cmap_name: str,
    use_diverging_norm: bool,
    annotate: bool,
    dpi: int,
) -> None:
    """用途: 绘制 Emery orbital-resolved observable 分布图.

    参数:
    - `values`: list[EmeryObservableValue], 待绘制的 orbital observable.
    - `lx, ly`: int, Emery Cu cell 尺寸.
    - `output_path`: Path, 输出 PNG 路径.
    - `title_text`: str, 图标题.
    - `colorbar_label`: str, colorbar 标签.
    - `cmap_name`: str, matplotlib colormap 名称.
    - `use_diverging_norm`: bool, True 时以 0 为中心使用发散色标.
    - `annotate`: bool, 是否标注数值.
    - `dpi`: int, 输出图片 dpi.

    返回:
    - None.
    """
    if not values:
        raise ValueError(f"没有可绘制的数据: {output_path}")

    figure_width = max(7.0, 0.55 * lx + 2.0)
    figure_height = max(4.0, 0.75 * ly + 1.5)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    setup_emery_axis(axis, lx, ly, title_text)

    raw_values = np.array([item.value for item in values], dtype=float)
    if use_diverging_norm:
        vmax_abs = float(np.max(np.abs(raw_values)))
        vmax_abs = max(vmax_abs, 1e-12)
        norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)
    else:
        vmin = float(np.min(raw_values))
        vmax = float(np.max(raw_values))
        if abs(vmax - vmin) <= 1e-12:
            vmin -= 0.5
            vmax += 0.5
        norm = Normalize(vmin=vmin, vmax=vmax)

    scatter_for_colorbar = None
    for orbital in ("d", "px", "py"):
        orbital_values = [item for item in values if item.orbital == orbital]
        if not orbital_values:
            continue
        x_values = [item.x_plot for item in orbital_values]
        y_values = [item.y_plot for item in orbital_values]
        color_values = [item.value for item in orbital_values]
        scatter = axis.scatter(
            x_values,
            y_values,
            c=color_values,
            cmap=cmap_name,
            norm=norm,
            s=130 if orbital == "d" else 95,
            marker=ORBITAL_MARKERS[orbital],
            edgecolors="black",
            linewidths=0.65,
            label=ORBITAL_LABELS[orbital],
            zorder=3,
        )
        scatter_for_colorbar = scatter

    if annotate:
        annotate_values(axis, values, "{:+.3f}" if use_diverging_norm else "{:.3f}")

    if scatter_for_colorbar is not None:
        colorbar = figure.colorbar(scatter_for_colorbar, ax=axis, pad=0.02)
        colorbar.set_label(colorbar_label)

    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
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
    configure_matplotlib_font()
    data = load_measurement_json(input_path)
    density_values = collect_emery_observable_values(data, "n")
    spin_values = collect_emery_observable_values(data, "Sz")
    if not density_values:
        raise ValueError("未找到 Emery density key, 需要 n_d_x_y/n_px_x_y/n_py_x_y 格式")
    if not spin_values:
        raise ValueError("未找到 Emery spin key, 需要 Sz_d_x_y/Sz_px_x_y/Sz_py_x_y 格式")

    lx, ly = infer_emery_shape(
        [*density_values, *spin_values],
        lx_user=args.lx,
        ly_user=args.ly,
    )

    density_output = output_dir / "emery_density_distribution.png"
    spin_output = output_dir / "emery_spin_distribution.png"
    plot_emery_observable_map(
        values=density_values,
        lx=lx,
        ly=ly,
        output_path=density_output,
        title_text="Emery density <n>",
        colorbar_label="<n>",
        cmap_name="viridis",
        use_diverging_norm=False,
        annotate=args.annotate_values,
        dpi=args.dpi,
    )
    plot_emery_observable_map(
        values=spin_values,
        lx=lx,
        ly=ly,
        output_path=spin_output,
        title_text="Emery spin <S_z>",
        colorbar_label="<S_z>",
        cmap_name="coolwarm",
        use_diverging_norm=True,
        annotate=args.annotate_values,
        dpi=args.dpi,
    )

    print(f"[OK] input: {input_path}")
    print(f"[OK] Emery shape: Lx={lx}, Ly={ly}")
    print(f"[OK] output: {density_output}")
    print(f"[OK] output: {spin_output}")


if __name__ == "__main__":
    main()
