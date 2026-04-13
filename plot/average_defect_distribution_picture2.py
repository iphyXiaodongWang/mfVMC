#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用途:
- 对 results/.../defect_average 下不同 defect_seed 的结果做平均。
- 输出 picture2 需要的两类核心数据:
  1) staggered_mz 等标量统计;
  2) S(q) 二维图的 seed 平均结果。

设计说明:
- 每个 defect_seed 先由 sector_min_energy.txt 自动选择 best target_sz。
- 标量从 best sector 的 defect_block_binning_mean.json 读取。
- S(q) 采用“先逐 seed 计算 S(q), 再对 S(q) 做平均”的方式, 避免不同 defect 分布时 SS 键集合不一致带来的偏差。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np


SS_KEY_PATTERN = re.compile(r"SS_(\d+)_(\d+)_(\d+)_(\d+)")
NDEFECT_PATTERN = re.compile(r"Ndefect(\d+)")
SEED_PATTERN = re.compile(r"defect_seed_(\d+)")


SEED_FIELDNAMES = [
    "phase",
    "phase_sign",
    "ndefect",
    "doping_abs",
    "signed_doping",
    "defect_seed",
    "best_sz",
    "best_energy_sector",
    "energy_measure",
    "staggered_mz",
    "abs_staggered_mz",
    "S_pi_pi",
    "sq_pipi",
    "n_site",
    "ss_term_count",
    "target_folder",
]


AVERAGE_FIELDNAMES = [
    "phase",
    "phase_sign",
    "ndefect",
    "doping_abs",
    "signed_doping",
    "n_seed_used",
    "staggered_mz_mean",
    "staggered_mz_std",
    "staggered_mz_se",
    "abs_staggered_mz_mean",
    "abs_staggered_mz_std",
    "abs_staggered_mz_se",
    "S_pi_pi_mean",
    "S_pi_pi_std",
    "S_pi_pi_se",
    "sq_pipi_mean",
    "sq_pipi_std",
    "sq_pipi_se",
    "best_energy_sector_mean",
    "best_energy_sector_std",
    "best_energy_sector_se",
    "energy_measure_mean",
    "energy_measure_std",
    "energy_measure_se",
    "n_site_mean",
    "n_site_std",
    "n_site_se",
]


def parse_arguments() -> argparse.Namespace:
    """
    用途: 解析命令行参数。

    参数:
    - 无。

    返回:
    - argparse.Namespace, 包含:
      - root: str, 输入根目录。可传:
        - results/L_20/defect_average
        - results/L_20/defect_average/hole
        - results/L_20/defect_average/electron
      - lattice_size: int | None, 晶格线性尺寸 L。为空时从路径自动推断。
      - ndefect_list: str, 仅处理的 Ndefect 列表, 例如 "6,18,42"。空字符串表示全处理。
      - output_dirname: str, 每个 Ndefect 下 logs 内输出目录名。
      - output_prefix: str, phase 层级汇总文件名前缀。
      - chunk_size: int, 计算 S(q) 时 q 点分块大小。
      - strict: bool, 严格模式。遇到缺失文件直接报错。
    """

    parser = argparse.ArgumentParser(
        description=(
            "平均 defect_seed 的 staggered_mz 与 S(q), 并导出 picture2 可用汇总文件。"
        )
    )
    parser.add_argument(
        "root",
        type=str,
        help="输入根目录, 例如 results/L_20/defect_average 或其 hole/electron 子目录。",
    )
    parser.add_argument(
        "--L",
        dest="lattice_size",
        type=int,
        default=None,
        help="晶格线性尺寸 L, 为空时从路径自动推断。",
    )
    parser.add_argument(
        "--ndefect-list",
        type=str,
        default="",
        help='可选, 指定要处理的 Ndefect 列表, 例如 "6,18,42"。',
    )
    parser.add_argument(
        "--output-dirname",
        type=str,
        default="average_picture2",
        help="每个 Ndefect/logs 下输出目录名, 默认 average_picture2。",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="picture2_defect_average",
        help="phase 级汇总 CSV 文件名前缀。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="S(q) 分块计算大小, 默认32。",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式: 任意 seed 缺文件或格式异常时立即报错。",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size 必须为正整数。")
    if args.lattice_size is not None and args.lattice_size <= 0:
        raise ValueError("--L 必须为正整数。")
    return args


def parse_ndefect_filter(raw_text: str) -> set[int] | None:
    """
    用途: 解析 --ndefect-list 参数。

    参数:
    - raw_text: str, 逗号分隔字符串, 例如 "6,18,42"。

    返回:
    - set[int] | None:
      - None: 表示不过滤, 处理全部 Ndefect。
      - set[int]: 仅处理集合中的 Ndefect。
    """

    stripped = raw_text.strip()
    if stripped == "":
        return None
    result: set[int] = set()
    for one_token in stripped.split(","):
        token = one_token.strip()
        if token == "":
            continue
        result.add(int(token))
    if len(result) == 0:
        return None
    return result


def resolve_phase_roots(root_path: Path) -> list[tuple[str, Path]]:
    """
    用途: 识别需要处理的 phase 根目录。

    参数:
    - root_path: Path, 用户输入路径。

    返回:
    - list[tuple[str, Path]], 每个元素是 (phase_label, phase_root_path)。
      - 若 root_path 下有 hole/electron 子目录, 则返回这些 phase。
      - 若 root_path 本身就是 Ndefect 根目录, 则返回单元素列表。
    """

    if not root_path.is_dir():
        raise FileNotFoundError(f"目录不存在: {root_path}")

    phase_candidates: list[tuple[str, Path]] = []
    for phase_name in ("hole", "electron"):
        phase_path = root_path / phase_name
        if phase_path.is_dir():
            phase_candidates.append((phase_name, phase_path))
    if len(phase_candidates) > 0:
        return phase_candidates

    return [(root_path.name, root_path)]


def resolve_data_root(phase_root: Path) -> Path:
    """
    用途: 解析真正包含 Ndefect 子目录的数据根目录。

    参数:
    - phase_root: Path, phase 根目录。

    返回:
    - Path, 包含 Ndefect* 子目录的数据根目录。
    """

    ndefect_dirs = get_ndefect_dirs(phase_root)
    if len(ndefect_dirs) > 0:
        return phase_root

    auto_submit_path = phase_root / "auto_submit"
    if auto_submit_path.is_dir():
        auto_dirs = get_ndefect_dirs(auto_submit_path)
        if len(auto_dirs) > 0:
            return auto_submit_path

    raise FileNotFoundError(f"在 {phase_root} 下未找到 Ndefect* 目录。")


def infer_lattice_size(data_root: Path, cli_lattice_size: int | None) -> int:
    """
    用途: 推断晶格线性尺寸 L。

    参数:
    - data_root: Path, 数据根目录。
    - cli_lattice_size: int | None, 命令行显式传入 L。

    返回:
    - int, 晶格线性尺寸 L。
    """

    if cli_lattice_size is not None:
        return int(cli_lattice_size)
    for one_part in data_root.parts:
        match_obj = re.fullmatch(r"L_(\d+)", one_part)
        if match_obj is not None:
            return int(match_obj.group(1))
    raise ValueError(f"无法从路径推断 L, 请显式传入 --L。路径: {data_root}")


def detect_phase_sign(phase_label: str) -> int:
    """
    用途: 根据 phase 名称推断 signed doping 的符号。

    参数:
    - phase_label: str, phase 名称。

    返回:
    - int:
      - electron: -1
      - 其他: +1
    """

    lower_text = phase_label.lower()
    if "electron" in lower_text:
        return -1
    return 1


def get_ndefect_dirs(data_root: Path) -> list[Path]:
    """
    用途: 获取并排序 Ndefect 子目录。

    参数:
    - data_root: Path, 数据根目录。

    返回:
    - list[Path], 按 Ndefect 数值升序排列。
    """

    matched_dirs: list[Path] = []
    for one_path in data_root.iterdir():
        if not one_path.is_dir():
            continue
        if NDEFECT_PATTERN.fullmatch(one_path.name) is None:
            continue
        matched_dirs.append(one_path)
    matched_dirs.sort(key=lambda one_path: int(NDEFECT_PATTERN.fullmatch(one_path.name).group(1)))
    return matched_dirs


def parse_sector_min_energy_file(summary_path: Path) -> tuple[int, float]:
    """
    用途: 从 sector_min_energy.txt 解析 best target_sz 与对应能量。

    参数:
    - summary_path: Path, sector_min_energy.txt 路径。

    返回:
    - tuple[int, float], (best_sz, best_energy_sector)。
    """

    best_sz: int | None = None
    best_energy: float | None = None
    with summary_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            one_line = raw_line.strip()
            if one_line == "" or one_line.startswith("#"):
                continue
            parts = one_line.split()
            if len(parts) < 2:
                continue
            sz_value = int(parts[0])
            energy_value = float(parts[1])
            if best_energy is None or energy_value < best_energy:
                best_sz = sz_value
                best_energy = energy_value
    if best_sz is None or best_energy is None:
        raise ValueError(f"无法解析 best sector: {summary_path}")
    return best_sz, best_energy


def parse_seed_id(seed_dir_name: str) -> int:
    """
    用途: 从 defect_seed_xxx 目录名提取 seed id。

    参数:
    - seed_dir_name: str, 目录名, 例如 defect_seed_2。

    返回:
    - int, seed id。
    """

    match_obj = SEED_PATTERN.fullmatch(seed_dir_name)
    if match_obj is None:
        raise ValueError(f"非法 seed 目录名: {seed_dir_name}")
    return int(match_obj.group(1))


def get_required_float(raw_dict: dict, key_name: str, file_path: Path) -> float:
    """
    用途: 从字典中读取必需浮点字段。

    参数:
    - raw_dict: dict, 输入字典。
    - key_name: str, 字段名。
    - file_path: Path, 发生错误时用于报错的文件路径。

    返回:
    - float, 转换后的数值。
    """

    if key_name not in raw_dict:
        raise KeyError(f"{file_path} 缺少字段 {key_name}")
    return float(raw_dict[key_name])


def get_optional_float(raw_dict: dict, key_name: str) -> float:
    """
    用途: 从字典中读取可选浮点字段, 缺失时返回 NaN。

    参数:
    - raw_dict: dict, 输入字典。
    - key_name: str, 字段名。

    返回:
    - float, 转换后的数值或 NaN。
    """

    if key_name not in raw_dict:
        return math.nan
    return float(raw_dict[key_name])


def read_block_scalar_json(json_path: Path) -> dict:
    """
    用途: 读取 defect_block_binning_mean.json 的标量信息。

    参数:
    - json_path: Path, JSON 文件路径。

    返回:
    - dict, 含关键字段:
      - staggered_mz: float
      - S_pi_pi: float
      - energy_measure: float
    """

    raw_data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError(f"JSON 不是字典结构: {json_path}")
    return {
        "staggered_mz": get_required_float(raw_data, "staggered_mz", json_path),
        "S_pi_pi": get_required_float(raw_data, "S_pi_pi", json_path),
        "energy_measure": get_optional_float(raw_data, "E"),
    }


def build_k_grid_2d(lattice_x: int, lattice_y: int, center: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    用途: 构造二维动量网格 KX, KY。

    参数:
    - lattice_x: int, x 方向线性尺寸 Lx。
    - lattice_y: int, y 方向线性尺寸 Ly。
    - center: bool, 是否平移到 [-pi, pi) 区间。

    返回:
    - tuple[np.ndarray, np.ndarray], shape 均为 (Lx, Ly)。

    公式:
    - kx(i) = 2*pi*i/Lx
    - ky(j) = 2*pi*j/Ly
    """

    kx_values = 2.0 * np.pi / lattice_x * np.arange(lattice_x, dtype=float)
    ky_values = 2.0 * np.pi / lattice_y * np.arange(lattice_y, dtype=float)
    if center:
        kx_values = (kx_values + np.pi) % (2.0 * np.pi) - np.pi
        ky_values = (ky_values + np.pi) % (2.0 * np.pi) - np.pi
    grid_kx, grid_ky = np.meshgrid(kx_values, ky_values, indexing="ij")
    return grid_kx, grid_ky


def parse_ss_key(raw_key: str) -> tuple[int, int, int, int] | None:
    """
    用途: 解析 SS_all.json 的键名。

    参数:
    - raw_key: str, 形如 SS_x0_y0_x1_y1。

    返回:
    - tuple[int, int, int, int] | None, 解析失败返回 None。
    """

    match_obj = SS_KEY_PATTERN.fullmatch(raw_key)
    if match_obj is None:
        return None
    return tuple(int(match_obj.group(index)) for index in range(1, 5))


def load_ss_terms(
    ss_all_json_path: Path,
    lattice_x: int,
    lattice_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    用途: 从 SS_all.json 读取可用的 (delta_x, delta_y, ss_value) 项, 并统计 Nsite。

    参数:
    - ss_all_json_path: Path, SS_all.json 路径。
    - lattice_x: int, Lx。
    - lattice_y: int, Ly。

    返回:
    - tuple[np.ndarray, np.ndarray, np.ndarray, int]:
      - delta_x_array: np.ndarray[float], 每项 x0-x1
      - delta_y_array: np.ndarray[float], 每项 y0-y1
      - ss_value_array: np.ndarray[float], 每项 SS 值
      - n_site: int, 在可用项中出现过的格点数
    """

    raw_dict = json.loads(ss_all_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw_dict, dict):
        raise ValueError(f"JSON 不是字典结构: {ss_all_json_path}")

    delta_x_list: list[float] = []
    delta_y_list: list[float] = []
    ss_value_list: list[float] = []
    site_flag = np.zeros((lattice_x, lattice_y), dtype=bool)

    for raw_key, raw_value in raw_dict.items():
        parsed = parse_ss_key(str(raw_key))
        if parsed is None:
            continue
        x0, y0, x1, y1 = parsed
        if not (0 <= x0 < lattice_x and 0 <= y0 < lattice_y and 0 <= x1 < lattice_x and 0 <= y1 < lattice_y):
            continue
        try:
            ss_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        delta_x_list.append(float(x0 - x1))
        delta_y_list.append(float(y0 - y1))
        ss_value_list.append(ss_value)
        site_flag[x0, y0] = True
        site_flag[x1, y1] = True

    if len(ss_value_list) == 0:
        raise ValueError(f"SS_all.json 无可用 SS 项: {ss_all_json_path}")

    delta_x_array = np.asarray(delta_x_list, dtype=float)
    delta_y_array = np.asarray(delta_y_list, dtype=float)
    ss_value_array = np.asarray(ss_value_list, dtype=float)
    n_site = int(np.count_nonzero(site_flag))
    return delta_x_array, delta_y_array, ss_value_array, n_site


def compute_sq_map_from_terms(
    grid_kx: np.ndarray,
    grid_ky: np.ndarray,
    delta_x_array: np.ndarray,
    delta_y_array: np.ndarray,
    ss_value_array: np.ndarray,
    n_site: int,
    chunk_size: int,
) -> np.ndarray:
    """
    用途: 根据 SS 项计算单个 seed 的 S(q) 二维矩阵。

    参数:
    - grid_kx: np.ndarray, KX 网格, shape=(Lx, Ly)。
    - grid_ky: np.ndarray, KY 网格, shape=(Lx, Ly)。
    - delta_x_array: np.ndarray, 每个 SS 项的 x0-x1。
    - delta_y_array: np.ndarray, 每个 SS 项的 y0-y1。
    - ss_value_array: np.ndarray, 每个 SS 项的权重。
    - n_site: int, 当前 seed 的有效格点数。
    - chunk_size: int, q 点分块大小。

    返回:
    - np.ndarray, S(q) 矩阵, shape=(Lx, Ly)。

    公式:
    - S(q) = [2 * sum_{(r0,r1)} cos(q·(r0-r1)) * SS(r0,r1) + Nsite*0.75] / (Lx*Ly)^2
    """

    flat_kx = grid_kx.ravel()
    flat_ky = grid_ky.ravel()
    flat_sq = np.zeros_like(flat_kx, dtype=float)
    lattice_x, lattice_y = grid_kx.shape
    normalization = float((lattice_x * lattice_y) ** 2)

    start_index = 0
    while start_index < flat_kx.shape[0]:
        end_index = min(start_index + chunk_size, flat_kx.shape[0])
        chunk_kx = flat_kx[start_index:end_index][:, None]
        chunk_ky = flat_ky[start_index:end_index][:, None]
        phase_matrix = chunk_kx * delta_x_array[None, :] + chunk_ky * delta_y_array[None, :]
        chunk_sum = (2.0 * np.cos(phase_matrix) * ss_value_array[None, :]).sum(axis=1)
        flat_sq[start_index:end_index] = chunk_sum
        start_index = end_index

    flat_sq = (flat_sq + float(n_site) * 0.75) / normalization
    return flat_sq.reshape(grid_kx.shape)


def compute_mean_std_and_se(values: Iterable[float]) -> tuple[float, float, float]:
    """
    用途: 计算标量均值、样本标准差、标准误。

    参数:
    - values: Iterable[float], 输入数值序列。

    返回:
    - tuple[float, float, float], (mean, std, se)。

    公式:
    - mean = (1/N) * sum_i x_i
    - std  = sqrt(sum_i (x_i - mean)^2 / (N-1)), N<2 时为 NaN
    - se   = std / sqrt(N), N<2 时为 NaN
    """

    value_list = [float(one_value) for one_value in values]
    number_value = len(value_list)
    if number_value == 0:
        raise ValueError("values 不能为空。")
    mean_value = sum(value_list) / number_value
    if number_value < 2:
        return mean_value, math.nan, math.nan
    variance = sum((one_value - mean_value) ** 2 for one_value in value_list) / (number_value - 1)
    std_value = math.sqrt(variance)
    se_value = std_value / math.sqrt(number_value)
    return mean_value, std_value, se_value


def compute_array_mean_std_and_se(
    array_sum: np.ndarray,
    array_sum_square: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    用途: 基于逐项和与平方和计算矩阵均值、样本标准差、标准误。

    参数:
    - array_sum: np.ndarray, 各元素求和结果。
    - array_sum_square: np.ndarray, 各元素平方求和结果。
    - count: int, 样本数 N。

    返回:
    - tuple[np.ndarray, np.ndarray, np.ndarray], (mean_array, std_array, se_array)。

    公式:
    - mean = sum / N
    - variance = (sum_sq - N*mean^2)/(N-1), N<2 时为 NaN
    - std = sqrt(max(variance, 0))
    - se = std / sqrt(N)
    """

    if count <= 0:
        raise ValueError("count 必须为正整数。")
    mean_array = array_sum / float(count)
    if count < 2:
        nan_array = np.full_like(mean_array, np.nan, dtype=float)
        return mean_array, nan_array, nan_array
    variance_array = (array_sum_square - float(count) * mean_array**2) / float(count - 1)
    variance_array = np.maximum(variance_array, 0.0)
    std_array = np.sqrt(variance_array)
    se_array = std_array / math.sqrt(float(count))
    return mean_array, std_array, se_array


def write_csv_rows(output_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """
    用途: 将字典行写入 CSV 文件。

    参数:
    - output_path: Path, 输出 CSV 路径。
    - fieldnames: list[str], 列名顺序。
    - rows: list[dict], 行数据。

    返回:
    - None。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_sq_npz(
    output_path: Path,
    grid_kx: np.ndarray,
    grid_ky: np.ndarray,
    sq_mean: np.ndarray,
    sq_std: np.ndarray,
    sq_se: np.ndarray,
    n_seed_used: int,
    phase_label: str,
    phase_sign: int,
    ndefect: int,
    doping_abs: float,
    signed_doping: float,
    pi_x_index: int,
    pi_y_index: int,
) -> None:
    """
    用途: 保存每个 Ndefect 的平均 S(q) 结果到 npz。

    参数:
    - output_path: Path, 输出 npz 路径。
    - grid_kx, grid_ky: np.ndarray, 动量网格。
    - sq_mean, sq_std, sq_se: np.ndarray, 平均与误差矩阵。
    - n_seed_used: int, 有效 seed 数。
    - phase_label: str, phase 名称。
    - phase_sign: int, phase 对应 doping 符号。
    - ndefect: int, 当前缺陷数。
    - doping_abs: float, 绝对 doping。
    - signed_doping: float, 带符号 doping。
    - pi_x_index, pi_y_index: int, 对应 (pi, pi) 的索引。

    返回:
    - None。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        kx_grid=grid_kx,
        ky_grid=grid_ky,
        sq_mean=sq_mean,
        sq_std=sq_std,
        sq_se=sq_se,
        n_seed_used=np.asarray([n_seed_used], dtype=int),
        phase_label=np.asarray([phase_label], dtype=object),
        phase_sign=np.asarray([phase_sign], dtype=int),
        ndefect=np.asarray([ndefect], dtype=int),
        doping_abs=np.asarray([doping_abs], dtype=float),
        signed_doping=np.asarray([signed_doping], dtype=float),
        pi_x_index=np.asarray([pi_x_index], dtype=int),
        pi_y_index=np.asarray([pi_y_index], dtype=int),
    )


def process_one_ndefect(
    data_root: Path,
    phase_label: str,
    phase_sign: int,
    ndefect: int,
    lattice_size: int,
    grid_kx: np.ndarray,
    grid_ky: np.ndarray,
    pi_x_index: int,
    pi_y_index: int,
    output_dirname: str,
    chunk_size: int,
    strict: bool,
) -> tuple[list[dict], dict] | None:
    """
    用途: 处理单个 Ndefect 下的所有 defect_seed, 输出标量与 S(q) 平均结果。

    参数:
    - data_root: Path, 数据根目录(含 Ndefect* 子目录)。
    - phase_label: str, phase 名称。
    - phase_sign: int, doping 符号。
    - ndefect: int, 当前缺陷数。
    - lattice_size: int, L。
    - grid_kx, grid_ky: np.ndarray, 动量网格。
    - pi_x_index, pi_y_index: int, (pi, pi) 索引。
    - output_dirname: str, logs 下输出目录名。
    - chunk_size: int, S(q) 分块大小。
    - strict: bool, 严格模式。

    返回:
    - tuple[list[dict], dict] | None:
      - list[dict]: 此 Ndefect 的每个 seed 标量记录。
      - dict: 此 Ndefect 的平均标量记录。
      - 若无可用 seed, 返回 None。
    """

    ndefect_root = data_root / f"Ndefect{ndefect}"
    logs_root = ndefect_root / "logs"
    if not logs_root.is_dir():
        message = f"[WARN] 缺少 logs 目录, 已跳过: {logs_root}"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return None

    seed_dirs = [one_path for one_path in logs_root.iterdir() if one_path.is_dir() and SEED_PATTERN.fullmatch(one_path.name)]
    seed_dirs.sort(key=lambda one_path: parse_seed_id(one_path.name))
    if len(seed_dirs) == 0:
        message = f"[WARN] 未找到 defect_seed 目录, 已跳过: {logs_root}"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return None

    doping_abs = float(ndefect) / float(lattice_size * lattice_size)
    signed_doping = float(phase_sign) * doping_abs

    seed_rows: list[dict] = []
    sq_sum: np.ndarray | None = None
    sq_sum_square: np.ndarray | None = None

    for seed_dir in seed_dirs:
        try:
            defect_seed = parse_seed_id(seed_dir.name)
            summary_path = seed_dir / "sector_min_energy.txt"
            if not summary_path.is_file():
                raise FileNotFoundError(f"缺少文件: {summary_path}")
            best_sz, best_energy_sector = parse_sector_min_energy_file(summary_path)

            target_dir = seed_dir / f"target_sz_{best_sz}"
            block_json_path = target_dir / "defect_block_binning_mean.json"
            ss_all_json_path = target_dir / "SS_all.json"
            if not block_json_path.is_file():
                raise FileNotFoundError(f"缺少文件: {block_json_path}")
            if not ss_all_json_path.is_file():
                raise FileNotFoundError(f"缺少文件: {ss_all_json_path}")

            scalar_data = read_block_scalar_json(block_json_path)
            delta_x_array, delta_y_array, ss_value_array, n_site = load_ss_terms(
                ss_all_json_path=ss_all_json_path,
                lattice_x=lattice_size,
                lattice_y=lattice_size,
            )
            sq_map = compute_sq_map_from_terms(
                grid_kx=grid_kx,
                grid_ky=grid_ky,
                delta_x_array=delta_x_array,
                delta_y_array=delta_y_array,
                ss_value_array=ss_value_array,
                n_site=n_site,
                chunk_size=chunk_size,
            )
            sq_pipi = float(sq_map[pi_x_index, pi_y_index])

            if sq_sum is None:
                sq_sum = np.zeros_like(sq_map, dtype=float)
                sq_sum_square = np.zeros_like(sq_map, dtype=float)
            sq_sum += sq_map
            sq_sum_square += sq_map**2

            seed_rows.append(
                {
                    "phase": phase_label,
                    "phase_sign": phase_sign,
                    "ndefect": ndefect,
                    "doping_abs": doping_abs,
                    "signed_doping": signed_doping,
                    "defect_seed": defect_seed,
                    "best_sz": best_sz,
                    "best_energy_sector": float(best_energy_sector),
                    "energy_measure": float(scalar_data["energy_measure"]),
                    "staggered_mz": float(scalar_data["staggered_mz"]),
                    "abs_staggered_mz": abs(float(scalar_data["staggered_mz"])),
                    "S_pi_pi": float(scalar_data["S_pi_pi"]),
                    "sq_pipi": sq_pipi,
                    "n_site": n_site,
                    "ss_term_count": int(ss_value_array.shape[0]),
                    "target_folder": str(target_dir.relative_to(data_root)),
                }
            )
        except Exception as error:
            message = f"[WARN] Ndefect={ndefect}, seed={seed_dir.name} 处理失败: {error}"
            if strict:
                raise RuntimeError(message) from error
            print(message)
            continue

    if len(seed_rows) == 0:
        message = f"[WARN] Ndefect={ndefect} 没有可用 seed, 已跳过。"
        if strict:
            raise RuntimeError(message)
        print(message)
        return None

    sq_mean, sq_std, sq_se = compute_array_mean_std_and_se(
        array_sum=sq_sum,
        array_sum_square=sq_sum_square,
        count=len(seed_rows),
    )

    output_dir = logs_root / output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "seed_scalar_summary.csv", SEED_FIELDNAMES, seed_rows)

    save_sq_npz(
        output_path=output_dir / "sq_map_average.npz",
        grid_kx=grid_kx,
        grid_ky=grid_ky,
        sq_mean=sq_mean,
        sq_std=sq_std,
        sq_se=sq_se,
        n_seed_used=len(seed_rows),
        phase_label=phase_label,
        phase_sign=phase_sign,
        ndefect=ndefect,
        doping_abs=doping_abs,
        signed_doping=signed_doping,
        pi_x_index=pi_x_index,
        pi_y_index=pi_y_index,
    )

    n_site_values = [float(one_row["n_site"]) for one_row in seed_rows]
    staggered_values = [float(one_row["staggered_mz"]) for one_row in seed_rows]
    abs_staggered_values = [float(one_row["abs_staggered_mz"]) for one_row in seed_rows]
    spi_values = [float(one_row["S_pi_pi"]) for one_row in seed_rows]
    sq_pipi_values = [float(one_row["sq_pipi"]) for one_row in seed_rows]
    energy_sector_values = [float(one_row["best_energy_sector"]) for one_row in seed_rows]
    energy_measure_values = [float(one_row["energy_measure"]) for one_row in seed_rows if not math.isnan(float(one_row["energy_measure"]))]

    staggered_mean, staggered_std, staggered_se = compute_mean_std_and_se(staggered_values)
    abs_staggered_mean, abs_staggered_std, abs_staggered_se = compute_mean_std_and_se(abs_staggered_values)
    spi_mean, spi_std, spi_se = compute_mean_std_and_se(spi_values)
    sq_pipi_mean, sq_pipi_std, sq_pipi_se = compute_mean_std_and_se(sq_pipi_values)
    energy_sector_mean, energy_sector_std, energy_sector_se = compute_mean_std_and_se(energy_sector_values)
    n_site_mean, n_site_std, n_site_se = compute_mean_std_and_se(n_site_values)

    if len(energy_measure_values) == 0:
        energy_measure_mean = math.nan
        energy_measure_std = math.nan
        energy_measure_se = math.nan
    else:
        energy_measure_mean, energy_measure_std, energy_measure_se = compute_mean_std_and_se(energy_measure_values)

    average_row = {
        "phase": phase_label,
        "phase_sign": phase_sign,
        "ndefect": ndefect,
        "doping_abs": doping_abs,
        "signed_doping": signed_doping,
        "n_seed_used": len(seed_rows),
        "staggered_mz_mean": staggered_mean,
        "staggered_mz_std": staggered_std,
        "staggered_mz_se": staggered_se,
        "abs_staggered_mz_mean": abs_staggered_mean,
        "abs_staggered_mz_std": abs_staggered_std,
        "abs_staggered_mz_se": abs_staggered_se,
        "S_pi_pi_mean": spi_mean,
        "S_pi_pi_std": spi_std,
        "S_pi_pi_se": spi_se,
        "sq_pipi_mean": sq_pipi_mean,
        "sq_pipi_std": sq_pipi_std,
        "sq_pipi_se": sq_pipi_se,
        "best_energy_sector_mean": energy_sector_mean,
        "best_energy_sector_std": energy_sector_std,
        "best_energy_sector_se": energy_sector_se,
        "energy_measure_mean": energy_measure_mean,
        "energy_measure_std": energy_measure_std,
        "energy_measure_se": energy_measure_se,
        "n_site_mean": n_site_mean,
        "n_site_std": n_site_std,
        "n_site_se": n_site_se,
    }

    write_csv_rows(output_dir / "scalar_average_summary.csv", AVERAGE_FIELDNAMES, [average_row])
    (output_dir / "scalar_average_summary.json").write_text(
        json.dumps(average_row, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[INFO] phase={phase_label}, Ndefect={ndefect}, "
        f"n_seed={len(seed_rows)}, output={output_dir}"
    )
    return seed_rows, average_row


def main() -> None:
    """
    用途: 执行 defect 分布平均主流程, 输出 phase 级与 Ndefect 级汇总文件。

    参数:
    - 无。

    返回:
    - None。
    """

    args = parse_arguments()
    root_path = Path(args.root).expanduser().resolve()
    phase_roots = resolve_phase_roots(root_path)
    ndefect_filter = parse_ndefect_filter(args.ndefect_list)

    all_seed_rows_combined: list[dict] = []
    all_average_rows_combined: list[dict] = []

    for phase_label, phase_root in phase_roots:
        data_root = resolve_data_root(phase_root)
        lattice_size = infer_lattice_size(data_root, args.lattice_size)
        phase_sign = detect_phase_sign(phase_label)

        grid_kx, grid_ky = build_k_grid_2d(lattice_x=lattice_size, lattice_y=lattice_size, center=False)
        kx_values = grid_kx[:, 0]
        ky_values = grid_ky[0, :]
        pi_x_index = int(np.argmin(np.abs(kx_values - np.pi)))
        pi_y_index = int(np.argmin(np.abs(ky_values - np.pi)))

        phase_seed_rows: list[dict] = []
        phase_average_rows: list[dict] = []

        for ndefect_dir in get_ndefect_dirs(data_root):
            ndefect = int(NDEFECT_PATTERN.fullmatch(ndefect_dir.name).group(1))
            if ndefect_filter is not None and ndefect not in ndefect_filter:
                continue
            result = process_one_ndefect(
                data_root=data_root,
                phase_label=phase_label,
                phase_sign=phase_sign,
                ndefect=ndefect,
                lattice_size=lattice_size,
                grid_kx=grid_kx,
                grid_ky=grid_ky,
                pi_x_index=pi_x_index,
                pi_y_index=pi_y_index,
                output_dirname=args.output_dirname,
                chunk_size=args.chunk_size,
                strict=args.strict,
            )
            if result is None:
                continue
            one_seed_rows, one_average_row = result
            phase_seed_rows.extend(one_seed_rows)
            phase_average_rows.append(one_average_row)

        if len(phase_seed_rows) == 0 or len(phase_average_rows) == 0:
            raise RuntimeError(f"phase={phase_label} 未收集到有效数据。")

        phase_seed_rows.sort(key=lambda one_row: (int(one_row["ndefect"]), int(one_row["defect_seed"])))
        phase_average_rows.sort(key=lambda one_row: int(one_row["ndefect"]))

        seed_csv_path = phase_root / f"{args.output_prefix}_seed_scalar_summary.csv"
        avg_csv_path = phase_root / f"{args.output_prefix}_scalar_average_summary.csv"
        write_csv_rows(seed_csv_path, SEED_FIELDNAMES, phase_seed_rows)
        write_csv_rows(avg_csv_path, AVERAGE_FIELDNAMES, phase_average_rows)
        print(f"[INFO] 写入 phase seed 汇总: {seed_csv_path}")
        print(f"[INFO] 写入 phase 平均汇总: {avg_csv_path}")

        all_seed_rows_combined.extend(phase_seed_rows)
        all_average_rows_combined.extend(phase_average_rows)

    if len(phase_roots) > 1:
        all_seed_rows_combined.sort(key=lambda one_row: (float(one_row["signed_doping"]), int(one_row["defect_seed"])))
        all_average_rows_combined.sort(key=lambda one_row: float(one_row["signed_doping"]))
        seed_combined_path = root_path / f"{args.output_prefix}_seed_scalar_summary_combined.csv"
        avg_combined_path = root_path / f"{args.output_prefix}_scalar_average_summary_combined.csv"
        write_csv_rows(seed_combined_path, SEED_FIELDNAMES, all_seed_rows_combined)
        write_csv_rows(avg_combined_path, AVERAGE_FIELDNAMES, all_average_rows_combined)
        print(f"[INFO] 写入 combined seed 汇总: {seed_combined_path}")
        print(f"[INFO] 写入 combined 平均汇总: {avg_combined_path}")


if __name__ == "__main__":
    main()
