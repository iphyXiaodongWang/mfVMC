#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用途:
- 批量读取 defect_average 目录下每个 Ndefect 的 sq_map_average.npz。
- 使用 sq_mean 绘制 S(q) 热图。
- 全部图共享同一套绝对值色标, 便于跨 Ndefect 比较强度。

输入目录约定:
- results/L_xx/defect_average/{hole,electron}/Ndefect*/logs/average_picture2/sq_map_average.npz
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NDEFECT_PATTERN = re.compile(r"Ndefect(\d+)")


@dataclass
class SqMapRecord:
    """
    用途: 保存单个 Ndefect 的 S(q) 图所需信息。

    参数:
    - phase: str, "hole" 或 "electron"。
    - ndefect: int, 缺陷数。
    - signed_doping: float, 带符号 doping。
    - n_seed_used: int, 参与平均的 seed 数。
    - kx_grid: np.ndarray, kx 网格。
    - ky_grid: np.ndarray, ky 网格。
    - sq_mean: np.ndarray, 平均 S(q)。
    - source_npz: Path, 源 npz 路径。

    返回:
    - SqMapRecord 实例。
    """

    phase: str
    ndefect: int
    signed_doping: float
    n_seed_used: int
    kx_grid: np.ndarray
    ky_grid: np.ndarray
    sq_mean: np.ndarray
    source_npz: Path


def parse_arguments() -> argparse.Namespace:
    """
    用途: 解析命令行参数。

    参数:
    - 无。

    返回:
    - argparse.Namespace, 包含:
      - root: str, defect_average 根目录。
      - output: str, 输出图目录。
      - dpi: int, 图片 dpi。
      - cmap: str, colormap 名称。
    """

    parser = argparse.ArgumentParser(
        description="批量绘制每个 Ndefect 的 defect-average S(q) 图。"
    )
    parser.add_argument(
        "root",
        type=str,
        help="defect_average 根目录, 例如 results/L_20/defect_average",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plot/defect_average_sq_maps",
        help="输出目录, 默认 plot/defect_average_sq_maps",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="输出图片 dpi, 默认 180",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="颜色映射, 默认 viridis",
    )
    args = parser.parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi 必须为正整数。")
    return args


def load_sq_records(root: Path) -> list[SqMapRecord]:
    """
    用途: 扫描 root 并加载所有可用的 sq_map_average.npz。

    参数:
    - root: Path, defect_average 根目录。

    返回:
    - list[SqMapRecord], 按 phase 与 Ndefect 排序后的记录列表。
    """

    records: list[SqMapRecord] = []
    for phase in ("hole", "electron"):
        phase_root = root / phase
        if not phase_root.is_dir():
            continue
        ndefect_dirs = [
            one_path
            for one_path in phase_root.iterdir()
            if one_path.is_dir() and NDEFECT_PATTERN.fullmatch(one_path.name)
        ]
        ndefect_dirs.sort(
            key=lambda one_path: int(NDEFECT_PATTERN.fullmatch(one_path.name).group(1))
        )
        for ndefect_dir in ndefect_dirs:
            ndefect = int(NDEFECT_PATTERN.fullmatch(ndefect_dir.name).group(1))
            npz_path = (
                ndefect_dir
                / "logs"
                / "average_picture2"
                / "sq_map_average.npz"
            )
            if not npz_path.is_file():
                continue
            npz_data = np.load(npz_path, allow_pickle=True)
            sq_mean = np.asarray(npz_data["sq_mean"], dtype=float)
            record = SqMapRecord(
                phase=phase,
                ndefect=ndefect,
                signed_doping=float(npz_data["signed_doping"][0]),
                n_seed_used=int(npz_data["n_seed_used"][0]),
                kx_grid=np.asarray(npz_data["kx_grid"], dtype=float),
                ky_grid=np.asarray(npz_data["ky_grid"], dtype=float),
                sq_mean=sq_mean,
                source_npz=npz_path,
            )
            records.append(record)
    records.sort(key=lambda one_record: (one_record.phase, one_record.ndefect))
    return records


def build_pi_ticks(max_k: float) -> tuple[list[float], list[str]]:
    """
    用途: 生成 0, pi, 2pi 的坐标刻度标签。

    参数:
    - max_k: float, 坐标上限。

    返回:
    - tuple[list[float], list[str]], (刻度位置, 刻度标签)。
    """

    tick_positions = [0.0, np.pi, min(2.0 * np.pi, max_k)]
    tick_labels = ["0", r"$\pi$", r"$2\pi$" if max_k >= 2.0 * np.pi - 1e-10 else f"{max_k:.3f}"]
    return tick_positions, tick_labels


def draw_one_sq_map(
    record: SqMapRecord,
    output_path: Path,
    cmap_name: str,
    dpi: int,
    vmin: float,
    vmax: float,
) -> None:
    """
    用途: 绘制并保存单个 Ndefect 的 S(q) 图。

    参数:
    - record: SqMapRecord, 单个记录。
    - output_path: Path, 输出 png 路径。
    - cmap_name: str, colormap 名称。
    - dpi: int, 图片 dpi。
    - vmin: float, 色标下限。
    - vmax: float, 色标上限。

    返回:
    - None。
    """

    figure, axis = plt.subplots(figsize=(5.2, 4.2), dpi=dpi, constrained_layout=True)
    max_kx = float(np.max(record.kx_grid))
    max_ky = float(np.max(record.ky_grid))
    image = axis.imshow(
        record.sq_mean.T,
        origin="lower",
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
        extent=(0.0, max_kx, 0.0, max_ky),
        aspect="equal",
        interpolation="nearest",
    )
    axis.scatter([np.pi], [np.pi], c="white", marker="x", s=56, linewidths=1.6, zorder=4)
    axis.set_xlabel(r"$k_x$")
    axis.set_ylabel(r"$k_y$")
    axis.set_title(
        f"{record.phase}  Ndefect={record.ndefect}  doping={record.signed_doping:+.3f}  n_seed={record.n_seed_used}",
        fontsize=9.5,
    )
    xticks, xlabels = build_pi_ticks(max_kx)
    yticks, ylabels = build_pi_ticks(max_ky)
    axis.set_xticks(xticks)
    axis.set_xticklabels(xlabels)
    axis.set_yticks(yticks)
    axis.set_yticklabels(ylabels)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("S(q)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="png")
    plt.close(figure)


def write_index_csv(output_path: Path, records: list[SqMapRecord], image_paths: list[Path]) -> None:
    """
    用途: 输出索引 CSV, 记录每张图对应的 metadata 与路径。

    参数:
    - output_path: Path, 索引 CSV 输出路径。
    - records: list[SqMapRecord], 记录列表。
    - image_paths: list[Path], 生成图片路径列表, 顺序与 records 对齐。

    返回:
    - None。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "phase",
                "ndefect",
                "signed_doping",
                "n_seed_used",
                "image_path",
                "source_npz",
            ],
        )
        writer.writeheader()
        for record, image_path in zip(records, image_paths):
            writer.writerow(
                {
                    "phase": record.phase,
                    "ndefect": record.ndefect,
                    "signed_doping": f"{record.signed_doping:.6f}",
                    "n_seed_used": record.n_seed_used,
                    "image_path": str(image_path),
                    "source_npz": str(record.source_npz),
                }
            )


def main() -> None:
    """
    用途: 主流程。

    参数:
    - 无。

    返回:
    - None。
    """

    args = parse_arguments()
    root_path = Path(args.root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"目录不存在: {root_path}")
    output_root = Path(args.output).expanduser().resolve()

    records = load_sq_records(root=root_path)
    if len(records) == 0:
        raise RuntimeError(f"未找到可用 sq_map_average.npz: {root_path}")

    global_max = max(float(np.max(one_record.sq_mean)) for one_record in records)
    global_vmax = max(global_max, 1e-12)
    global_vmin = 0.0

    image_paths: list[Path] = []
    for record in records:
        image_path = output_root / record.phase / f"Ndefect{record.ndefect:02d}_sq_mean.png"
        draw_one_sq_map(
            record=record,
            output_path=image_path,
            cmap_name=args.cmap,
            dpi=args.dpi,
            vmin=global_vmin,
            vmax=global_vmax,
        )
        image_paths.append(image_path)
        print(
            f"[OK] phase={record.phase}, Ndefect={record.ndefect}, "
            f"n_seed={record.n_seed_used}, image={image_path}"
        )

    index_csv_path = output_root / "sq_map_image_index.csv"
    write_index_csv(output_path=index_csv_path, records=records, image_paths=image_paths)
    print(f"[OK] 统一色标范围: vmin={global_vmin:.8f}, vmax={global_vmax:.8f}")
    print(f"[OK] 索引文件: {index_csv_path}")


if __name__ == "__main__":
    main()
