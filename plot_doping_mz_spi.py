import argparse
import csv
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_arguments():
    """用途: 解析命令行参数. 参数: 无. 返回: argparse.Namespace."""
    parser = argparse.ArgumentParser(
        description="在同一张图中绘制不同(J2, J3)下 staggered_mz 与 S(pi,pi) 随 doping 的变化."
    )
    parser.add_argument(
        "result_root",
        type=str,
        help="结果根目录, 例如 results/L_16",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=None,
        help="线性尺寸L (方格子时 Lx=Ly=L). 若不提供, 尝试从目录名 L_xx 解析.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图片路径. 默认保存到 result_root/summary_doping_mz_spi.png",
    )
    parser.add_argument(
        "--signed-mz",
        action="store_true",
        help="若设置, 绘制带符号 mz. 默认绘制 |mz|.",
    )
    return parser.parse_args()


def infer_l_from_root(result_root, cli_l):
    """用途: 推断系统线性尺寸L. 参数: result_root字符串, cli_l整数或None. 返回: 正整数L."""
    if cli_l is not None:
        if cli_l <= 0:
            raise ValueError("--L 必须为正整数.")
        return cli_l

    root_name = os.path.basename(os.path.normpath(result_root))
    match = re.match(r"^L_(\d+)$", root_name)
    if match:
        return int(match.group(1))

    raise ValueError("无法从目录名解析L, 请显式传入 --L.")


def parse_j2_j3_from_folder(folder_name):
    """用途: 从文件夹名解析J2与J3. 参数: folder_name字符串. 返回: (j2, j3) 浮点数或None."""
    match = re.match(
        r"^J2_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_J3_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
        folder_name,
    )
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def load_summary_rows(csv_path):
    """用途: 读取summary表. 参数: csv_path字符串. 返回: 行字典列表."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_curve_from_rows(rows, l_value, use_abs_mz):
    """用途: 从CSV行构建一条曲线.

    参数:
    - rows: Dict列表, 每行需包含 Ndefect, staggered_mz, S_pi_pi 及可选标准误列.
    - l_value: 整数L.
    - use_abs_mz: 布尔值, 是否使用|staggered_mz|.

    返回:
    - dict, 含x, mz, mz_se, spi, spi_se数组.

    公式:
    - doping = Ndefect / (L * L)
    """
    x_vals = []
    mz_vals = []
    mz_se_vals = []
    spi_vals = []
    spi_se_vals = []

    n_site = l_value * l_value

    for row in rows:
        if row.get("Ndefect", "") == "":
            continue
        n_defect = int(float(row["Ndefect"]))
        doping = n_defect / n_site

        if row.get("staggered_mz", "") == "" or row.get("S_pi_pi", "") == "":
            continue

        mz = float(row["staggered_mz"])
        if use_abs_mz:
            mz = abs(mz)
        spi = float(row["S_pi_pi"])

        mz_se = np.nan
        spi_se = np.nan
        if row.get("staggered_mz_se", "") != "":
            mz_se = float(row["staggered_mz_se"])
        if row.get("S_pi_pi_se", "") != "":
            spi_se = float(row["S_pi_pi_se"])

        x_vals.append(doping)
        mz_vals.append(mz)
        mz_se_vals.append(mz_se)
        spi_vals.append(spi)
        spi_se_vals.append(spi_se)

    if not x_vals:
        return None

    order = np.argsort(np.array(x_vals))
    return {
        "x": np.array(x_vals)[order],
        "mz": np.array(mz_vals)[order],
        "mz_se": np.array(mz_se_vals)[order],
        "spi": np.array(spi_vals)[order],
        "spi_se": np.array(spi_se_vals)[order],
    }


def collect_all_curves(result_root, l_value, use_abs_mz):
    """用途: 收集所有(J2,J3)曲线. 参数: result_root字符串, l_value整数, use_abs_mz布尔. 返回: 曲线列表."""
    curves = []
    for name in sorted(os.listdir(result_root)):
        folder_path = os.path.join(result_root, name)
        if not os.path.isdir(folder_path):
            continue

        parsed = parse_j2_j3_from_folder(name)
        if parsed is None:
            continue

        j2, j3 = parsed
        csv_path = os.path.join(folder_path, "summary_min_sector_staggered_mz_S_pi_pi.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] 缺少文件, 已跳过: {csv_path}")
            continue

        rows = load_summary_rows(csv_path)
        curve = build_curve_from_rows(rows, l_value, use_abs_mz)
        if curve is None:
            print(f"[WARN] 文件无有效数据, 已跳过: {csv_path}")
            continue

        curve["label"] = f"J2={j2:g}, J3={j3:g}"
        curves.append(curve)

    return curves


def plot_curves(curves, output_path, use_abs_mz):
    """用途: 绘图并保存. 参数: curves列表, output_path字符串, use_abs_mz布尔. 返回: None."""
    if not curves:
        raise RuntimeError("没有可绘制的数据. 请检查输入目录及summary文件.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150, constrained_layout=False)
    ax_mz, ax_spi = axes

    for curve in curves:
        yerr_mz = None if np.isnan(curve["mz_se"]).all() else curve["mz_se"]
        yerr_spi = None if np.isnan(curve["spi_se"]).all() else curve["spi_se"]

        ax_mz.errorbar(
            curve["x"],
            curve["mz"],
            yerr=yerr_mz,
            marker="o",
            linestyle="-",
            capsize=3,
            linewidth=1.5,
            markersize=4,
            label=curve["label"],
        )
        ax_spi.errorbar(
            curve["x"],
            curve["spi"],
            yerr=yerr_spi,
            marker="o",
            linestyle="-",
            capsize=3,
            linewidth=1.5,
            markersize=4,
            label=curve["label"],
        )

    ax_mz.set_xlabel("doping")
    ax_spi.set_xlabel("doping")
    ax_mz.set_ylabel("|mz|" if use_abs_mz else "mz")
    ax_spi.set_ylabel("S(pi, pi)")
    ax_mz.set_title("doping vs mz")
    ax_spi.set_title("doping vs S(pi, pi)")
    ax_mz.grid(alpha=0.3)
    ax_spi.grid(alpha=0.3)

    handles, labels = ax_mz.get_legend_handles_labels()
    fig.subplots_adjust(right=0.80, wspace=0.28)
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    """用途: 程序入口. 参数: 无. 返回: None."""
    args = parse_arguments()
    result_root = args.result_root
    l_value = infer_l_from_root(result_root, args.L)

    if args.output is None:
        output_path = os.path.join(result_root, "summary_doping_mz_spi.png")
    else:
        output_path = args.output

    use_abs_mz = not args.signed_mz
    curves = collect_all_curves(result_root, l_value, use_abs_mz)
    plot_curves(curves, output_path, use_abs_mz)
    print(f"[OK] 图已保存: {output_path}")


if __name__ == "__main__":
    main()
