#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import re
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
      - root: 结果根目录, 可以是 `results/L_16/defect_average` 或 `.../auto_submit`.
      - lattice_size: 可选的线性尺寸 L. 若不提供, 则尝试从路径中解析.
      - output_prefix: 输出文件名前缀.
    """

    parser = argparse.ArgumentParser(
        description=(
            "对不同 Ndefect 下的多个 defect_seed 结果做平均, "
            "汇总 |staggered_mz|, S(pi,pi), 以及 "
            "sum_i |<Sz_i>| / L^2, 并保存均值、标准差与 doping 图."
        )
    )
    parser.add_argument(
        "root",
        type=str,
        help="结果根目录, 例如 results/L_16/defect_average",
    )
    parser.add_argument(
        "--L",
        dest="lattice_size",
        type=int,
        default=None,
        help="晶格线性尺寸 L. 若不提供, 脚本会尝试从路径中解析.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="defect_doping_average_abs_mean_sz",
        help="输出文件名前缀, 默认 defect_doping_average_abs_mean_sz",
    )
    return parser.parse_args()


def resolve_data_root(root_path):
    """用途: 解析真正存放 Ndefect 子目录的数据根目录.

    参数:
    - root_path: Path, 用户提供的根路径.

    返回:
    - Path, 真正的数据根目录. 若 root_path 下存在 `auto_submit`, 则优先使用它.
    """

    if not root_path.is_dir():
        raise FileNotFoundError(f"未找到目录: {root_path}")

    ndefect_dirs = [path for path in root_path.iterdir() if path.is_dir() and path.name.startswith("Ndefect")]
    if ndefect_dirs:
        return root_path

    auto_submit_path = root_path / "auto_submit"
    if auto_submit_path.is_dir():
        return auto_submit_path

    raise FileNotFoundError(
        f"在 {root_path} 下未找到 Ndefect 子目录, 也未找到 auto_submit 子目录."
    )


def infer_lattice_size(data_root, cli_lattice_size):
    """用途: 推断晶格线性尺寸 L.

    参数:
    - data_root: Path, 真正的数据根目录.
    - cli_lattice_size: Optional[int], 命令行显式传入的 L.

    返回:
    - int, 晶格线性尺寸 L.
    """

    if cli_lattice_size is not None:
        return cli_lattice_size

    for path_part in data_root.parts:
        match = re.fullmatch(r"L_(\d+)", path_part)
        if match:
            return int(match.group(1))

    raise ValueError("无法从路径解析 L, 请显式传入 --L.")


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
    """用途: 解析 defect_block_binning.txt 中 observable 的 Mean 与 SE.

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


def compute_average_abs_mean_sz_from_json(sz_json_path, lattice_size):
    """用途: 从 Sz.json 计算 sum_i |<Sz_i>| / L^2.

    参数:
    - sz_json_path: Path, 指向 `Sz.json`.
    - lattice_size: int, 晶格线性尺寸 L.

    返回:
    - float, `sum_i |<Sz_i>| / L^2`.

    公式:
    - 设 `Sz.json` 中存储的是非 defect 格点的 `<S_i^z>`.
    - 则脚本计算:
      `average_abs_mean_sz = (1 / L^2) * sum_{i in non-defect} |<S_i^z>|`.
    - 由于 defect 格点不在 JSON 中, 这里等价于把 defect 位置贡献视为 0.
    """

    with sz_json_path.open("r", encoding="utf-8") as file:
        sz_dict = json.load(file)

    normalization = float(lattice_size * lattice_size)
    return sum(abs(float(value)) for value in sz_dict.values()) / normalization


def collect_seed_rows_for_ndefect(data_root, lattice_size, ndefect):
    """用途: 收集单个 Ndefect 下各 defect_seed 的 best-sector 测量结果.

    参数:
    - data_root: Path, 数据根目录.
    - lattice_size: int, 晶格线性尺寸 L.
    - ndefect: int, 当前缺陷数量.

    返回:
    - list[dict], 每个元素对应一个 defect_seed 的汇总结果.
    """

    ndefect_root = data_root / f"Ndefect{ndefect}"
    logs_root = ndefect_root / "logs"
    if not logs_root.is_dir():
        print(f"[WARN] 缺少 logs 目录, 已跳过: {logs_root}")
        return []

    seed_rows = []
    seed_dirs = sorted(
        path for path in logs_root.iterdir() if path.is_dir() and path.name.startswith("defect_seed_")
    )

    for seed_dir in seed_dirs:
        defect_seed = int(seed_dir.name.replace("defect_seed_", ""))
        summary_path = seed_dir / "sector_min_energy.txt"
        if not summary_path.is_file():
            print(f"[WARN] 缺少文件, 已跳过: {summary_path}")
            continue

        try:
            best_sz, best_energy = parse_sector_min_energy_file(summary_path)
        except ValueError as error:
            print(f"[WARN] {error} 已跳过该 seed.")
            continue

        target_folder = seed_dir / f"target_sz_{best_sz}"
        block_txt_path = target_folder / "defect_block_binning.txt"
        sz_json_path = target_folder / "Sz.json"
        if not block_txt_path.is_file():
            print(f"[WARN] 缺少文件, 已跳过: {block_txt_path}")
            continue
        if not sz_json_path.is_file():
            print(f"[WARN] 缺少文件, 已跳过: {sz_json_path}")
            continue

        block_table = parse_block_binning_table(block_txt_path)
        required_observables = ["staggered_mz", "S_pi_pi"]
        missing_observables = [name for name in required_observables if name not in block_table]
        if missing_observables:
            print(
                f"[WARN] {block_txt_path} 缺少观测量 {missing_observables}, 已跳过该 seed."
            )
            continue

        doping = ndefect / float(lattice_size * lattice_size)
        staggered_mz = float(block_table["staggered_mz"]["Mean"])
        average_abs_mean_sz = compute_average_abs_mean_sz_from_json(sz_json_path, lattice_size)

        seed_rows.append(
            {
                "ndefect": ndefect,
                "doping": doping,
                "defect_seed": defect_seed,
                "best_sz": best_sz,
                "best_energy": best_energy,
                "target_folder": str(target_folder.relative_to(data_root)),
                "staggered_mz": staggered_mz,
                "abs_staggered_mz": abs(staggered_mz),
                "S_pi_pi": float(block_table["S_pi_pi"]["Mean"]),
                "average_abs_mean_sz": average_abs_mean_sz,
            }
        )

    return seed_rows


def collect_all_seed_rows(data_root, lattice_size):
    """用途: 收集所有 Ndefect 下的 per-seed 数据.

    参数:
    - data_root: Path, 数据根目录.
    - lattice_size: int, 晶格线性尺寸 L.

    返回:
    - list[dict], 所有 Ndefect 和 defect_seed 的明细数据.
    """

    all_seed_rows = []
    ndefect_dirs = sorted(
        (path for path in data_root.iterdir() if path.is_dir() and path.name.startswith("Ndefect")),
        key=lambda path: int(path.name.replace("Ndefect", "")),
    )

    for ndefect_dir in ndefect_dirs:
        ndefect = int(ndefect_dir.name.replace("Ndefect", ""))
        all_seed_rows.extend(collect_seed_rows_for_ndefect(data_root, lattice_size, ndefect))

    if not all_seed_rows:
        raise RuntimeError("没有收集到任何可用的 defect_seed 测量数据.")

    return all_seed_rows


def compute_mean_and_std(values):
    """用途: 计算均值与样本标准差.

    参数:
    - values: list[float], 待统计数值列表.

    返回:
    - tuple[float, float], `(mean_value, std_value)`.

    公式:
    - 均值:
      `mean = (1 / N) * sum_i x_i`
    - 样本标准差:
      `std = sqrt(sum_i (x_i - mean)^2 / (N - 1))`, 当 `N < 2` 时返回 `NaN`.
    """

    n_value = len(values)
    if n_value == 0:
        raise ValueError("values 不能为空.")

    mean_value = sum(values) / n_value
    if n_value < 2:
        return mean_value, math.nan

    variance = sum((value - mean_value) ** 2 for value in values) / (n_value - 1)
    return mean_value, math.sqrt(variance)


def build_average_rows(seed_rows):
    """用途: 按 Ndefect 汇总不同 defect_seed 的均值与标准差.

    参数:
    - seed_rows: list[dict], 所有 per-seed 明细数据.

    返回:
    - list[dict], 每个元素对应一个 Ndefect 的汇总结果.
    """

    grouped_rows = {}
    for row in seed_rows:
        grouped_rows.setdefault(row["ndefect"], []).append(row)

    average_rows = []
    for ndefect in sorted(grouped_rows):
        rows = grouped_rows[ndefect]
        doping = rows[0]["doping"]

        abs_staggered_mean, abs_staggered_std = compute_mean_and_std(
            [row["abs_staggered_mz"] for row in rows]
        )
        spi_mean, spi_std = compute_mean_and_std([row["S_pi_pi"] for row in rows])
        abs_mean_sz_mean, abs_mean_sz_std = compute_mean_and_std(
            [row["average_abs_mean_sz"] for row in rows]
        )

        average_rows.append(
            {
                "ndefect": ndefect,
                "doping": doping,
                "n_seed_used": len(rows),
                "abs_staggered_mz_mean": abs_staggered_mean,
                "abs_staggered_mz_std": abs_staggered_std,
                "S_pi_pi_mean": spi_mean,
                "S_pi_pi_std": spi_std,
                "average_abs_mean_sz_mean": abs_mean_sz_mean,
                "average_abs_mean_sz_std": abs_mean_sz_std,
            }
        )

    return average_rows


def write_seed_summary_csv(output_path, seed_rows):
    """用途: 将 per-seed 明细结果写入 CSV.

    参数:
    - output_path: Path, 输出 CSV 路径.
    - seed_rows: list[dict], 所有 per-seed 明细数据.

    返回:
    - None.
    """

    fieldnames = [
        "ndefect",
        "doping",
        "defect_seed",
        "best_sz",
        "best_energy",
        "target_folder",
        "staggered_mz",
        "abs_staggered_mz",
        "S_pi_pi",
        "average_abs_mean_sz",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seed_rows)


def write_average_summary_csv(output_path, average_rows):
    """用途: 将按 Ndefect 汇总后的均值和标准差写入 CSV.

    参数:
    - output_path: Path, 输出 CSV 路径.
    - average_rows: list[dict], 按 Ndefect 汇总后的结果.

    返回:
    - None.
    """

    fieldnames = [
        "ndefect",
        "doping",
        "n_seed_used",
        "abs_staggered_mz_mean",
        "abs_staggered_mz_std",
        "S_pi_pi_mean",
        "S_pi_pi_std",
        "average_abs_mean_sz_mean",
        "average_abs_mean_sz_std",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(average_rows)


def plot_average_rows(output_path, average_rows):
    """用途: 绘制不同 doping 下三个观测量的均值与标准差误差棒图.

    参数:
    - output_path: Path, 输出图片路径.
    - average_rows: list[dict], 按 Ndefect 汇总后的结果.

    返回:
    - None.
    """

    doping_values = [row["doping"] for row in average_rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].errorbar(
        doping_values,
        [row["abs_staggered_mz_mean"] for row in average_rows],
        yerr=[row["abs_staggered_mz_std"] for row in average_rows],
        fmt="o-",
        capsize=4,
        linewidth=1.2,
        markersize=5,
    )
    axes[0].set_title("Average |staggered mz|")
    axes[0].set_xlabel("doping")
    axes[0].set_ylabel("|staggered mz|")
    axes[0].grid(alpha=0.25, linestyle="--")

    axes[1].errorbar(
        doping_values,
        [row["S_pi_pi_mean"] for row in average_rows],
        yerr=[row["S_pi_pi_std"] for row in average_rows],
        fmt="o-",
        capsize=4,
        linewidth=1.2,
        markersize=5,
    )
    axes[1].set_title("Average S(pi,pi)")
    axes[1].set_xlabel("doping")
    axes[1].set_ylabel("S(pi,pi)")
    axes[1].grid(alpha=0.25, linestyle="--")

    axes[2].errorbar(
        doping_values,
        [row["average_abs_mean_sz_mean"] for row in average_rows],
        yerr=[row["average_abs_mean_sz_std"] for row in average_rows],
        fmt="o-",
        capsize=4,
        linewidth=1.2,
        markersize=5,
    )
    axes[2].set_title(r"Average $|\langle S_i^z \rangle|$")
    axes[2].set_xlabel("doping")
    axes[2].set_ylabel(r"$\sum_i |\langle S_i^z \rangle| / L^2$")
    axes[2].grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    """用途: 执行不同 doping 下的 defect_seed 平均与绘图主流程.

    参数:
    - 无.

    返回:
    - None.
    """

    args = parse_arguments()
    root_path = Path(args.root).resolve()
    data_root = resolve_data_root(root_path)
    lattice_size = infer_lattice_size(data_root, args.lattice_size)

    seed_rows = collect_all_seed_rows(data_root, lattice_size)
    average_rows = build_average_rows(seed_rows)

    seed_summary_csv = root_path / f"{args.output_prefix}_seed_summary.csv"
    average_summary_csv = root_path / f"{args.output_prefix}_average_summary.csv"
    figure_path = root_path / f"{args.output_prefix}_summary.png"

    write_seed_summary_csv(seed_summary_csv, seed_rows)
    write_average_summary_csv(average_summary_csv, average_rows)
    plot_average_rows(figure_path, average_rows)

    print(f"[INFO] 数据根目录: {data_root}")
    print(f"[INFO] 写入 per-seed 明细: {seed_summary_csv}")
    print(f"[INFO] 写入平均汇总: {average_summary_csv}")
    print(f"[INFO] 写入图片: {figure_path}")


if __name__ == "__main__":
    main()
