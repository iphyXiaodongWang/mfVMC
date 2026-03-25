#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_arguments():
    """用途: 解析命令行参数.

    参数:
    - 无.

    返回:
    - argparse.Namespace, 包含:
      - root: 结果根目录路径, 其下应包含 `logs/defect_seed_*`.
      - output_prefix: 输出文件名前缀字符串. 若为空, 则默认使用 `defect_seed_average`.
    """

    parser = argparse.ArgumentParser(
        description=(
            "对多个 defect_seed 的 best-sector 测量结果做平均, "
            "并绘制 staggered_mz 与 S(pi,pi) 的带 error bar 图."
        )
    )
    parser.add_argument(
        "root",
        type=str,
        help="结果根目录, 例如 results/test_defect_average",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="defect_seed_average",
        help="输出文件名前缀, 默认 defect_seed_average",
    )
    return parser.parse_args()


def parse_sector_min_energy_file(summary_path):
    """用途: 从 sector_min_energy.txt 中选出最优 target_sz.

    参数:
    - summary_path: Path, 指向 `sector_min_energy.txt`.

    返回:
    - tuple[int, float], `(best_sz, best_energy)`.
    """

    best_sz = None
    best_energy = None

    with summary_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            sz_value = int(parts[0])
            energy_value = float(parts[1])
            if best_energy is None or energy_value < best_energy:
                best_sz = sz_value
                best_energy = energy_value

    if best_sz is None or best_energy is None:
        raise ValueError(f"未能从 {summary_path} 解析 best sector.")

    return best_sz, best_energy


def parse_block_binning_table(txt_path):
    """用途: 解析 defect_block_binning.txt 中各 observable 的 Mean 与 SE.

    参数:
    - txt_path: Path, 指向 `defect_block_binning.txt`.

    返回:
    - dict[str, dict[str, float]], 形如:
      `{"staggered_mz": {"Mean": ..., "SE": ...}, "S_pi_pi": {...}}`.
    """

    observable_table = {}

    with txt_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            observable_name = parts[0]
            observable_table[observable_name] = {
                "Mean": float(parts[1]),
                "SE": float(parts[2]),
            }

    return observable_table


def collect_seed_rows(root_path):
    """用途: 收集每个 defect_seed 的 best-sector 测量结果.

    参数:
    - root_path: Path, 结果根目录.

    返回:
    - list[dict], 每个元素包含:
      - defect_seed: int
      - best_sz: int
      - best_energy: float
      - target_folder: str
      - staggered_mz: float
      - staggered_mz_mc_se: float
      - S_pi_pi: float
      - S_pi_pi_mc_se: float
    """

    logs_path = root_path / "logs"
    if not logs_path.is_dir():
        raise FileNotFoundError(f"未找到 logs 目录: {logs_path}")

    seed_rows = []
    seed_dirs = sorted(
        path for path in logs_path.iterdir() if path.is_dir() and path.name.startswith("defect_seed_")
    )
    if not seed_dirs:
        raise FileNotFoundError(f"未找到任何 defect_seed 目录: {logs_path}")

    for seed_dir in seed_dirs:
        defect_seed = int(seed_dir.name.replace("defect_seed_", ""))
        summary_path = seed_dir / "sector_min_energy.txt"
        if not summary_path.is_file():
            raise FileNotFoundError(f"缺少文件: {summary_path}")

        best_sz, best_energy = parse_sector_min_energy_file(summary_path)
        target_folder = seed_dir / f"target_sz_{best_sz}"
        json_path = target_folder / "defect_block_binning_mean.json"
        txt_path = target_folder / "defect_block_binning.txt"

        if not json_path.is_file():
            raise FileNotFoundError(f"缺少文件: {json_path}")
        if not txt_path.is_file():
            raise FileNotFoundError(f"缺少文件: {txt_path}")

        with json_path.open("r", encoding="utf-8") as file:
            mean_data = json.load(file)
        block_table = parse_block_binning_table(txt_path)

        seed_rows.append(
            {
                "defect_seed": defect_seed,
                "best_sz": best_sz,
                "best_energy": best_energy,
                "target_folder": str(target_folder.relative_to(root_path)),
                "staggered_mz": float(mean_data["staggered_mz"]),
                "staggered_mz_mc_se": float(block_table["staggered_mz"]["SE"]),
                "S_pi_pi": float(mean_data["S_pi_pi"]),
                "S_pi_pi_mc_se": float(block_table["S_pi_pi"]["SE"]),
            }
        )

    return seed_rows


def compute_jackknife_mean_and_se(values):
    """用途: 对多个 defect_seed 的 observable 做 leave-one-out jackknife.

    参数:
    - values: list[float], 不同 defect_seed 上同一 observable 的数值.

    返回:
    - tuple[float, float], `(mean_value, jackknife_se)`.

    公式:
    - 设总样本数为 `N`, 原始均值为 `mean = sum(values) / N`.
    - leave-one-out 均值为
      `mean_i = (sum(values) - values[i]) / (N - 1)`.
    - jackknife 标准误为
      `sqrt((N-1)/N * sum_i (mean_i - average(mean_i))^2)`.
    """

    n_value = len(values)
    if n_value == 0:
        raise ValueError("values 不能为空.")

    mean_value = sum(values) / n_value
    if n_value == 1:
        return mean_value, math.nan

    total_value = sum(values)
    leave_one_out_means = [
        (total_value - single_value) / (n_value - 1) for single_value in values
    ]
    mean_leave_one_out = sum(leave_one_out_means) / n_value
    variance_term = sum(
        (single_mean - mean_leave_one_out) ** 2 for single_mean in leave_one_out_means
    )
    jackknife_se = math.sqrt((n_value - 1) / n_value * variance_term)
    return mean_value, jackknife_se


def build_average_summary(seed_rows):
    """用途: 构造整体平均汇总结果.

    参数:
    - seed_rows: list[dict], 来自 `collect_seed_rows`.

    返回:
    - dict, 包含 defect_seed 平均后的结果.
    """

    staggered_values = [row["staggered_mz"] for row in seed_rows]
    spi_values = [row["S_pi_pi"] for row in seed_rows]
    energy_values = [row["best_energy"] for row in seed_rows]

    staggered_mean, staggered_jk_se = compute_jackknife_mean_and_se(staggered_values)
    spi_mean, spi_jk_se = compute_jackknife_mean_and_se(spi_values)
    energy_mean, energy_jk_se = compute_jackknife_mean_and_se(energy_values)

    average_mc_se_staggered = sum(row["staggered_mz_mc_se"] for row in seed_rows) / len(seed_rows)
    average_mc_se_spi = sum(row["S_pi_pi_mc_se"] for row in seed_rows) / len(seed_rows)

    return {
        "n_defect_seed": len(seed_rows),
        "staggered_mz_mean": staggered_mean,
        "staggered_mz_jackknife_se": staggered_jk_se,
        "staggered_mz_average_mc_se": average_mc_se_staggered,
        "S_pi_pi_mean": spi_mean,
        "S_pi_pi_jackknife_se": spi_jk_se,
        "S_pi_pi_average_mc_se": average_mc_se_spi,
        "best_energy_mean": energy_mean,
        "best_energy_jackknife_se": energy_jk_se,
    }


def write_seed_summary_csv(output_path, seed_rows):
    """用途: 将每个 defect_seed 的结果写入 CSV.

    参数:
    - output_path: Path, 输出 CSV 路径.
    - seed_rows: list[dict], 每个 seed 的汇总结果.

    返回:
    - None.
    """

    fieldnames = [
        "defect_seed",
        "best_sz",
        "best_energy",
        "target_folder",
        "staggered_mz",
        "staggered_mz_mc_se",
        "S_pi_pi",
        "S_pi_pi_mc_se",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seed_rows)


def write_average_summary_csv(output_path, average_summary):
    """用途: 将 defect_seed 平均结果写入单行 CSV.

    参数:
    - output_path: Path, 输出 CSV 路径.
    - average_summary: dict, 来自 `build_average_summary`.

    返回:
    - None.
    """

    fieldnames = [
        "n_defect_seed",
        "staggered_mz_mean",
        "staggered_mz_jackknife_se",
        "staggered_mz_average_mc_se",
        "S_pi_pi_mean",
        "S_pi_pi_jackknife_se",
        "S_pi_pi_average_mc_se",
        "best_energy_mean",
        "best_energy_jackknife_se",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(average_summary)


def plot_observable_with_average(ax, seed_rows, observable_key, error_key, ylabel):
    """用途: 绘制单个 observable 的各 defect_seed 散点及平均 error bar.

    参数:
    - ax: matplotlib.axes.Axes, 当前子图对象.
    - seed_rows: list[dict], 每个 seed 的汇总结果.
    - observable_key: str, observable 的键名.
    - error_key: str, 平均误差键名.
    - ylabel: str, y 轴标题.

    返回:
    - None.
    """

    seed_values = [row["defect_seed"] for row in seed_rows]
    observable_values = [row[observable_key] for row in seed_rows]

    mean_value, error_value = compute_jackknife_mean_and_se(observable_values)
    mean_x_position = max(seed_values) + 1

    ax.scatter(seed_values, observable_values, color="tab:blue", s=45, label="Per seed")
    ax.errorbar(
        [mean_x_position],
        [mean_value],
        yerr=[error_value],
        fmt="o",
        color="tab:red",
        capsize=4,
        markersize=6,
        label="Average",
    )
    ax.axhline(mean_value, color="tab:red", linestyle="--", linewidth=1)
    ax.set_xticks(seed_values + [mean_x_position])
    ax.set_xticklabels([str(seed) for seed in seed_values] + ["avg"])
    ax.set_xlabel("defect_seed")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()


def plot_average_figure(output_path, seed_rows):
    """用途: 绘制 staggered_mz 与 S(pi,pi) 的按 defect_seed 平均图.

    参数:
    - output_path: Path, 输出图片路径.
    - seed_rows: list[dict], 每个 seed 的汇总结果.

    返回:
    - None.
    """

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_observable_with_average(
        axes[0],
        seed_rows,
        observable_key="staggered_mz",
        error_key="staggered_mz_jackknife_se",
        ylabel="staggered_mz",
    )
    axes[0].set_title("Average staggered_mz over defect_seed")

    plot_observable_with_average(
        axes[1],
        seed_rows,
        observable_key="S_pi_pi",
        error_key="S_pi_pi_jackknife_se",
        ylabel="S(pi,pi)",
    )
    axes[1].set_title("Average S(pi,pi) over defect_seed")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    """用途: 执行多 defect_seed 平均与绘图主流程.

    参数:
    - 无.

    返回:
    - None.
    """

    args = parse_arguments()
    root_path = Path(args.root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"未找到目录: {root_path}")

    seed_rows = collect_seed_rows(root_path)
    average_summary = build_average_summary(seed_rows)

    seed_summary_csv = root_path / f"{args.output_prefix}_seed_summary.csv"
    average_summary_csv = root_path / f"{args.output_prefix}_average_summary.csv"
    figure_path = root_path / f"{args.output_prefix}_mz_spi.png"

    write_seed_summary_csv(seed_summary_csv, seed_rows)
    write_average_summary_csv(average_summary_csv, average_summary)
    plot_average_figure(figure_path, seed_rows)

    print(f"[INFO] Wrote per-seed summary to: {seed_summary_csv}")
    print(f"[INFO] Wrote average summary to: {average_summary_csv}")
    print(f"[INFO] Wrote figure to: {figure_path}")


if __name__ == "__main__":
    main()
