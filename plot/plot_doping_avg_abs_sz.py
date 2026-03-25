import argparse
import csv
import json
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_arguments():
    """用途: 解析命令行参数. 参数: 无. 返回: argparse.Namespace."""
    parser = argparse.ArgumentParser(
        description="绘制不同(J2, J3)下 average |Sz| 随 doping 的变化曲线. 这里 average |Sz| 定义为 sum_i |Sz_i| / L^2."
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
        help="输出图片路径. 默认保存到 result_root/summary_doping_avg_abs_sz.png",
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
    with open(csv_path, "r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows.append(row)
    return rows


def compute_avg_abs_sz_from_json(sz_json_path, l_value):
    """用途: 从 Sz.json 计算 average |Sz| = sum_i |Sz_i| / L^2.

    参数:
    - sz_json_path: Sz.json 文件路径字符串.
    - l_value: 系统线性尺寸L.

    返回:
    - float, average |Sz|.
    """
    with open(sz_json_path, "r", encoding="utf-8") as file_obj:
        sz_dict = json.load(file_obj)

    abs_sum = 0.0
    for value in sz_dict.values():
        abs_sum += abs(float(value))

    return abs_sum / float(l_value * l_value)


def build_curve_from_summary(folder_path, rows, l_value):
    """用途: 从单个参数目录的summary文件与对应Sz.json构造曲线.

    参数:
    - folder_path: 单个 J2/J3 参数目录路径字符串.
    - rows: summary CSV 行字典列表.
    - l_value: 系统线性尺寸L.

    返回:
    - dict或None, 含x和avg_abs_sz数组. 若无有效数据则返回None.
    """
    x_vals = []
    avg_abs_sz_vals = []
    n_site = l_value * l_value

    for row in rows:
        if row.get("Ndefect", "") == "" or row.get("best_sz", "") == "":
            continue

        n_defect = int(float(row["Ndefect"]))
        best_sz = int(float(row["best_sz"]))
        doping = n_defect / float(n_site)

        sz_json_path = os.path.join(
            folder_path,
            f"Ndefect{n_defect}",
            "logs",
            f"target_sz_{best_sz}",
            "Sz.json",
        )
        if not os.path.exists(sz_json_path):
            print(f"[WARN] 缺少 Sz.json, 已跳过该数据点: {sz_json_path}")
            continue

        avg_abs_sz = compute_avg_abs_sz_from_json(sz_json_path, l_value)
        x_vals.append(doping)
        avg_abs_sz_vals.append(avg_abs_sz)

    if not x_vals:
        return None

    order = np.argsort(np.array(x_vals))
    return {
        "x": np.array(x_vals)[order],
        "avg_abs_sz": np.array(avg_abs_sz_vals)[order],
    }


def collect_all_curves(result_root, l_value):
    """用途: 收集所有(J2, J3)曲线. 参数: result_root字符串, l_value整数. 返回: 曲线列表."""
    curves = []
    for name in sorted(os.listdir(result_root)):
        folder_path = os.path.join(result_root, name)
        if not os.path.isdir(folder_path):
            continue

        parsed = parse_j2_j3_from_folder(name)
        if parsed is None:
            continue

        csv_path = os.path.join(folder_path, "summary_min_sector_staggered_mz_S_pi_pi.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] 缺少 summary 文件, 已跳过: {csv_path}")
            continue

        rows = load_summary_rows(csv_path)
        curve = build_curve_from_summary(folder_path, rows, l_value)
        if curve is None:
            print(f"[WARN] 文件无有效数据, 已跳过: {csv_path}")
            continue

        j2, j3 = parsed
        curve["label"] = f"J2={j2:g}, J3={j3:g}"
        curves.append(curve)

    return curves


def plot_curves(curves, output_path):
    """用途: 绘图并保存. 参数: curves列表, output_path字符串. 返回: None."""
    if not curves:
        raise RuntimeError("没有可绘制的数据. 请检查输入目录、summary文件与Sz.json.")

    plt.figure(figsize=(7, 5), dpi=150)
    for curve in curves:
        plt.plot(
            curve["x"],
            curve["avg_abs_sz"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label=curve["label"],
        )

    plt.xlabel("doping")
    plt.ylabel("average |Sz|")
    plt.title("doping vs average |Sz|")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    """用途: 程序入口. 参数: 无. 返回: None."""
    args = parse_arguments()
    result_root = args.result_root
    l_value = infer_l_from_root(result_root, args.L)

    if args.output is None:
        output_path = os.path.join(result_root, "summary_doping_avg_abs_sz.png")
    else:
        output_path = args.output

    curves = collect_all_curves(result_root, l_value)
    plot_curves(curves, output_path)
    print(f"[OK] 图已保存: {output_path}")


if __name__ == "__main__":
    main()
