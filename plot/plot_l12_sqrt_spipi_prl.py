"""绘制 L=12 下 sqrt(S(pi,pi)) 随 doping 变化的 PRL 风格图."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


Default_Result_Root = Path("results") / "L_12"
Default_Output_Stem = "summary_doping_sqrt_spipi_prl"
Lattice_Size = 12


def parse_arguments() -> argparse.Namespace:
    """用途: 解析命令行参数.

    参数:
    - 无.

    返回:
    - argparse.Namespace, 包含 result_root, output_stem, lattice_size 等参数.
    """
    parser = argparse.ArgumentParser(
        description="绘制不同(J2,J3)参数下 sqrt(S(pi,pi)) 随 doping 的变化."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Default_Result_Root,
        help="结果目录, 默认 results/L_12.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default=Default_Output_Stem,
        help="输出文件名主干, 默认 summary_doping_sqrt_spipi_prl.",
    )
    parser.add_argument(
        "--lattice-size",
        type=int,
        default=Lattice_Size,
        help="方格子线性尺寸 L, 默认 12.",
    )
    return parser.parse_args()


def parse_j2_j3_from_summary_name(summary_path: Path) -> tuple[float, float] | None:
    """用途: 从 summary_J2_x_J3_y.csv 文件名解析 J2 和 J3.

    参数:
    - summary_path: Path, summary CSV 文件路径.

    返回:
    - tuple[float, float] | None, 解析成功返回 (J2, J3), 否则返回 None.
    """
    match_obj = re.fullmatch(
        r"summary_J2_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_J3_"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.csv",
        summary_path.name,
    )
    if match_obj is None:
        return None
    return float(match_obj.group(1)), float(match_obj.group(2))


def convert_s_pi_pi_to_sqrt_observable(s_pi_pi_value: float) -> float:
    """用途: 将 S(pi,pi) 转换为 sqrt(S(pi,pi)).

    参数:
    - s_pi_pi_value: float, 原始 S(pi,pi) 数值.

    返回:
    - float, sqrt(S(pi,pi)) 数值.

    公式:
    - y = sqrt(max(S(pi,pi), 0)).
    """
    return math.sqrt(max(float(s_pi_pi_value), 0.0))


def load_curve_from_summary(
    summary_path: Path,
    lattice_size: int,
) -> dict[str, object]:
    """用途: 从单个 summary CSV 读取一条 doping 曲线.

    参数:
    - summary_path: Path, 包含 Ndefect 和 S_pi_pi 列的 CSV 文件.
    - lattice_size: int, 方格子线性尺寸 L.

    返回:
    - dict[str, object], 包含 label, j2, j3, doping_values, sqrt_spipi_values.

    公式:
    - Nsite = L * L.
    - doping = Ndefect / Nsite.
    - y = sqrt(max(S(pi,pi), 0)).
    """
    parsed_values = parse_j2_j3_from_summary_name(summary_path)
    if parsed_values is None:
        raise ValueError(f"无法从文件名解析 J2/J3: {summary_path.name}")

    j2_value, j3_value = parsed_values
    n_site = lattice_size * lattice_size
    doping_values: list[float] = []
    sqrt_spipi_values: list[float] = []

    with summary_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required_columns = {"Ndefect", "S_pi_pi"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"文件缺少必要列 {sorted(missing_columns)}: {summary_path}"
            )

        for row_dict in reader:
            if row_dict.get("Ndefect", "") == "" or row_dict.get("S_pi_pi", "") == "":
                continue
            n_defect = int(float(row_dict["Ndefect"]))
            s_pi_pi_value = float(row_dict["S_pi_pi"])
            doping_values.append(n_defect / n_site)
            sqrt_spipi_values.append(convert_s_pi_pi_to_sqrt_observable(s_pi_pi_value))

    if not doping_values:
        raise ValueError(f"未读取到有效数据: {summary_path}")

    order_array = np.argsort(np.asarray(doping_values, dtype=float))
    return {
        "label": rf"$J_2={j2_value:g},\ J_3={j3_value:g}$",
        "j2": j2_value,
        "j3": j3_value,
        "doping_values": np.asarray(doping_values, dtype=float)[order_array],
        "sqrt_spipi_values": np.asarray(sqrt_spipi_values, dtype=float)[order_array],
    }


def collect_curves(result_root: Path, lattice_size: int) -> list[dict[str, object]]:
    """用途: 收集 result_root 下所有 summary_J2_*_J3_*.csv 曲线.

    参数:
    - result_root: Path, 结果根目录.
    - lattice_size: int, 方格子线性尺寸 L.

    返回:
    - list[dict[str, object]], 每个元素是一条可绘制曲线.
    """
    if not result_root.is_dir():
        raise FileNotFoundError(f"结果目录不存在: {result_root}")

    summary_paths = sorted(result_root.glob("summary_J2_*_J3_*.csv"))
    curves = [
        load_curve_from_summary(summary_path, lattice_size)
        for summary_path in summary_paths
        if parse_j2_j3_from_summary_name(summary_path) is not None
    ]
    curves.sort(key=lambda curve: (float(curve["j2"]), float(curve["j3"])))
    if not curves:
        raise RuntimeError(f"未找到可绘制的 summary CSV: {result_root}")
    return curves


def configure_prl_style() -> None:
    """用途: 设置接近 PRL 的 matplotlib 全局绘图风格.

    参数:
    - 无.

    返回:
    - None.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.75,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.minor.width": 0.55,
            "ytick.minor.width": 0.55,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_curves(curves: list[dict[str, object]], output_stem_path: Path) -> None:
    """用途: 绘制并保存 sqrt(S(pi,pi)) 随 doping 变化的图.

    参数:
    - curves: list[dict[str, object]], collect_curves 返回的曲线列表.
    - output_stem_path: Path, 不含扩展名的输出路径.

    返回:
    - None.
    """
    configure_prl_style()

    color_values = ["#000000", "#4C78A8", "#D55E00", "#009E73", "#7A3E9D"]
    marker_values = ["o", "s", "^", "D", "v"]
    linestyle_values = ["-", "--", "-.", ":", (0, (4.0, 1.4, 1.0, 1.4))]

    fig, axis = plt.subplots(figsize=(3.45, 2.45), constrained_layout=True)

    for curve_index, curve in enumerate(curves):
        axis.plot(
            curve["doping_values"],
            curve["sqrt_spipi_values"],
            marker=marker_values[curve_index % len(marker_values)],
            linestyle=linestyle_values[curve_index % len(linestyle_values)],
            color=color_values[curve_index % len(color_values)],
            linewidth=1.05,
            markersize=3.4,
            markerfacecolor=color_values[curve_index % len(color_values)],
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=str(curve["label"]),
        )

    axis.set_xlabel(r"doping $\delta$")
    axis.set_ylabel(r"$\sqrt{S(\pi,\pi)}$")
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.minorticks_on()
    axis.tick_params(which="both", top=True, right=True)
    axis.legend(
        frameon=False,
        loc="upper right",
        handlelength=1.8,
        handletextpad=0.45,
        borderaxespad=0.25,
        labelspacing=0.25,
    )

    output_stem_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """用途: 程序入口, 读取数据并生成 PRL 风格图片.

    参数:
    - 无.

    返回:
    - None.
    """
    args = parse_arguments()
    result_root = args.result_root
    output_stem_path = result_root / args.output_stem
    curves = collect_curves(result_root, args.lattice_size)
    plot_curves(curves, output_stem_path)
    print(f"[OK] PNG: {output_stem_path.with_suffix('.png')}")
    print(f"[OK] PDF: {output_stem_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
