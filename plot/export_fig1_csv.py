# -*- coding: utf-8 -*-

"""
用途: 按照 picture2.py 当前实际使用口径, 整理 fig1 所需数据并导出为 CSV。

导出内容:
- 不同 doping 下的 VMC20, VMC12, DMRG12 的 sqrt(S(pi,pi))
- 不同 doping 下的 VMC20, VMC12, DMRG12 的 |stagger mz|
- VMC20 的上述两个量对应 standard error, 便于后续直接画 error bar

说明:
- 为了完整保留 picture2 中 hole / electron 两条分支在 doping=0 处的独立数据点,
  输出表额外保留 carrier_type 列。
- hole 分支使用正 doping, electron 分支使用负 doping。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


def find_project_root(explicit_project_root: str = "") -> Path:
    """
    用途: 定位项目根目录, 需同时包含 plot 和 results 子目录。

    参数:
    - explicit_project_root: str, 手动指定项目根目录; 为空时自动向上查找。

    返回:
    - Path, 项目根目录绝对路径。
    """
    if explicit_project_root.strip() != "":
        root = Path(explicit_project_root).expanduser().resolve()
        if (root / "plot").is_dir() and (root / "results").is_dir():
            return root
        raise FileNotFoundError(f"手动指定路径不是项目根目录: {root}")

    current_path = Path.cwd().resolve()
    candidate_roots = [current_path, *current_path.parents]
    candidate_roots.append(Path(r"D:/study/研究生/科研/VMC/HKJ_s/mfVMC"))
    for root in candidate_roots:
        if (root / "plot").is_dir() and (root / "results").is_dir():
            return root

    raise FileNotFoundError("无法自动定位项目根目录, 请在脚本中设置 explicit_project_root。")


def parse_float(raw_text: str) -> float:
    """
    用途: 将字符串转换为 float。

    参数:
    - raw_text: str, 原始文本。

    返回:
    - float, 转换后的数值。
    """
    return float(str(raw_text).strip())


def convert_s_pi_pi_to_sqrt_observable(s_pi_pi_value: float) -> float:
    """
    用途: 将 S(pi,pi) 转换为 sqrt(S(pi,pi))。

    参数:
    - s_pi_pi_value: float, S(pi,pi) 数值。

    返回:
    - float, sqrt(S(pi,pi)) 数值。

    公式:
    - y = sqrt(max(S(pi,pi), 0))
    """
    clipped_value = max(float(s_pi_pi_value), 0.0)
    return math.sqrt(clipped_value)


def convert_s_pi_pi_se_to_sqrt_observable_se(
    s_pi_pi_value: float,
    s_pi_pi_se_value: float,
) -> float:
    """
    用途: 将 S(pi,pi) 的 standard error 转换为 sqrt(S(pi,pi)) 的 standard error。

    参数:
    - s_pi_pi_value: float, S(pi,pi) 均值。
    - s_pi_pi_se_value: float, S(pi,pi) 的 standard error。

    返回:
    - float, sqrt(S(pi,pi)) 的 standard error。

    公式:
    - 令 y = sqrt(S), 则一阶误差传播为
      sigma_y ~= sigma_S / (2 * sqrt(S))

    说明:
    - 若原始误差为 NaN, 则直接返回 NaN。
    - 若 S <= 0 且误差为 0, 则返回 0。
    - 若 S <= 0 但误差非 0, 则返回 NaN, 避免给出不可靠结果。
    """
    if math.isnan(s_pi_pi_se_value):
        return float("nan")

    clipped_value = max(float(s_pi_pi_value), 0.0)
    if clipped_value > 0.0:
        return float(s_pi_pi_se_value) / (2.0 * math.sqrt(clipped_value))
    if math.isclose(clipped_value, 0.0) and math.isclose(float(s_pi_pi_se_value), 0.0):
        return 0.0
    return float("nan")


def build_signed_doping(doping_abs: float, carrier_type: str) -> float:
    """
    用途: 根据载流子类型生成带符号的 doping。

    参数:
    - doping_abs: float, 非负的 doping 绝对值。
    - carrier_type: str, "hole" 或 "electron"。

    返回:
    - float, hole 为正, electron 为负的 signed doping。
    """
    if carrier_type == "hole":
        return abs(doping_abs)
    if carrier_type == "electron":
        return -abs(doping_abs)
    raise ValueError(f"不支持的 carrier_type: {carrier_type}")


def format_float_text(value: float) -> str:
    """
    用途: 将 float 以稳定文本格式写入 CSV。

    参数:
    - value: float, 待格式化数值。

    返回:
    - str, 适合写入 CSV 的字符串。
    """
    numeric_value = float(value)
    if math.isnan(numeric_value):
        return ""
    return str(numeric_value)


def format_signed_doping_text(signed_doping: float, carrier_type: str) -> str:
    """
    用途: 生成 signed doping 的输出文本, 并在零点保留 hole / electron 区分。

    参数:
    - signed_doping: float, 带符号 doping。
    - carrier_type: str, "hole" 或 "electron"。

    返回:
    - str, 用于 CSV 的 signed doping 文本。
    """
    if math.isclose(signed_doping, 0.0, abs_tol=1e-15):
        return "-0.0" if carrier_type == "electron" else "0.0"
    return format_float_text(signed_doping)


def load_csv_table(csv_path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    """
    用途: 读取文本表格为字典列表。

    参数:
    - csv_path: Path, 输入文件路径。
    - delimiter: str, 字段分隔符。

    返回:
    - list[dict[str, str]], 每行对应的字段字典。
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"文件缺少表头: {csv_path}")
        return list(reader)


def load_vmc20_series(
    csv_path: Path,
) -> dict[tuple[str, float], dict[str, float]]:
    """
    用途: 读取 VMC20 defect_average 数据。

    参数:
    - csv_path: Path, VMC20 汇总 CSV 路径。

    返回:
    - dict[tuple[str, float], dict[str, float]]
      - key: (carrier_type, signed_doping)
      - value: {
          "abs_staggered_mz": float,
          "abs_staggered_mz_se": float,
          "sqrt_s_pi_pi": float,
          "sqrt_s_pi_pi_se": float,
        }
    """
    series_map: dict[tuple[str, float], dict[str, float]] = {}
    for row_dict in load_csv_table(csv_path, delimiter=","):
        carrier_type = str(row_dict["phase"]).strip()
        if carrier_type not in ("hole", "electron"):
            continue
        signed_doping = parse_float(row_dict["signed_doping"])
        abs_staggered_mz = parse_float(row_dict["abs_staggered_mz_mean"])
        abs_staggered_mz_se = parse_float(row_dict["abs_staggered_mz_se"])
        s_pi_pi_mean = parse_float(row_dict["S_pi_pi_mean"])
        s_pi_pi_se = parse_float(row_dict["S_pi_pi_se"])
        sqrt_s_pi_pi = convert_s_pi_pi_to_sqrt_observable(s_pi_pi_mean)
        sqrt_s_pi_pi_se = convert_s_pi_pi_se_to_sqrt_observable_se(
            s_pi_pi_value=s_pi_pi_mean,
            s_pi_pi_se_value=s_pi_pi_se,
        )
        series_map[(carrier_type, signed_doping)] = {
            "abs_staggered_mz": abs_staggered_mz,
            "abs_staggered_mz_se": abs_staggered_mz_se,
            "sqrt_s_pi_pi": sqrt_s_pi_pi,
            "sqrt_s_pi_pi_se": sqrt_s_pi_pi_se,
        }
    return series_map


def load_vmc12_series(
    csv_path: Path,
    carrier_type: str,
) -> dict[tuple[str, float], dict[str, float]]:
    """
    用途: 读取 VMC12 benchmark 数据。

    参数:
    - csv_path: Path, VMC12 benchmark CSV 路径。
    - carrier_type: str, "hole" 或 "electron"。

    返回:
    - dict[tuple[str, float], dict[str, float]]
      - key: (carrier_type, signed_doping)
      - value: {"abs_staggered_mz": float, "sqrt_s_pi_pi": float}
    """
    series_map: dict[tuple[str, float], dict[str, float]] = {}
    for row_dict in load_csv_table(csv_path, delimiter=","):
        doping_abs = parse_float(row_dict["doping"])
        signed_doping = build_signed_doping(doping_abs=doping_abs, carrier_type=carrier_type)
        abs_staggered_mz = abs(parse_float(row_dict["staggered_mz"]))
        sqrt_s_pi_pi = convert_s_pi_pi_to_sqrt_observable(
            parse_float(row_dict["S_pi_pi"])
        )
        series_map[(carrier_type, signed_doping)] = {
            "abs_staggered_mz": abs_staggered_mz,
            "sqrt_s_pi_pi": sqrt_s_pi_pi,
        }
    return series_map


def load_dmrg12_series(
    spipi_path: Path,
    mz_path: Path,
    carrier_type: str,
    lattice_size: int,
) -> dict[tuple[str, float], dict[str, float]]:
    """
    用途: 读取 DMRG12 benchmark 的 S(pi,pi) 与 staggered mz, 并按 Ndefect 合并。

    参数:
    - spipi_path: Path, S(pi,pi) 数据文件路径。
    - mz_path: Path, mz 数据文件路径。
    - carrier_type: str, "hole" 或 "electron"。
    - lattice_size: int, 格子线长 L, 用于计算 doping = Ndefect / L^2。

    返回:
    - dict[tuple[str, float], dict[str, float]]
      - key: (carrier_type, signed_doping)
      - value: {"abs_staggered_mz": float, "sqrt_s_pi_pi": float}
    """
    spipi_rows = load_csv_table(spipi_path, delimiter="\t")
    mz_rows = load_csv_table(mz_path, delimiter="\t")

    ndefect_to_sqrt_s_pi_pi: dict[int, float] = {}
    for row_dict in spipi_rows:
        ndefect = int(parse_float(row_dict["Ndefect"]))
        ndefect_to_sqrt_s_pi_pi[ndefect] = convert_s_pi_pi_to_sqrt_observable(
            parse_float(row_dict["S(pi,pi)"])
        )

    ndefect_to_abs_staggered_mz: dict[int, float] = {}
    for row_dict in mz_rows:
        ndefect = int(parse_float(row_dict["Ndefect"]))
        ndefect_to_abs_staggered_mz[ndefect] = abs(parse_float(row_dict["mz"]))

    common_ndefect_list = sorted(
        set(ndefect_to_sqrt_s_pi_pi.keys()) & set(ndefect_to_abs_staggered_mz.keys())
    )
    if len(common_ndefect_list) == 0:
        raise ValueError(f"DMRG 数据无法按 Ndefect 对齐: {spipi_path}, {mz_path}")

    series_map: dict[tuple[str, float], dict[str, float]] = {}
    for ndefect in common_ndefect_list:
        doping_abs = float(ndefect) / float(lattice_size * lattice_size)
        signed_doping = build_signed_doping(doping_abs=doping_abs, carrier_type=carrier_type)
        series_map[(carrier_type, signed_doping)] = {
            "abs_staggered_mz": ndefect_to_abs_staggered_mz[ndefect],
            "sqrt_s_pi_pi": ndefect_to_sqrt_s_pi_pi[ndefect],
        }
    return series_map


def merge_fig1_rows(
    vmc20_map: dict[tuple[str, float], dict[str, float]],
    vmc12_map: dict[tuple[str, float], dict[str, float]],
    dmrg12_map: dict[tuple[str, float], dict[str, float]],
) -> list[dict[str, str]]:
    """
    用途: 将 3 组数据源按 (carrier_type, signed_doping) 合并为 CSV 行。

    参数:
    - vmc20_map: dict, VMC20 数据映射。
    - vmc12_map: dict, VMC12 数据映射。
    - dmrg12_map: dict, DMRG12 数据映射。

    返回:
    - list[dict[str, str]], 适合 DictWriter 写出的行列表。
    """
    all_keys = set(vmc20_map.keys()) | set(vmc12_map.keys()) | set(dmrg12_map.keys())
    carrier_sort_order = {"electron": 0, "hole": 1}
    sorted_keys = sorted(
        all_keys,
        key=lambda item: (item[1], carrier_sort_order.get(item[0], 99)),
    )

    merged_rows: list[dict[str, str]] = []
    for carrier_type, signed_doping in sorted_keys:
        vmc20_value = vmc20_map.get((carrier_type, signed_doping), {})
        vmc12_value = vmc12_map.get((carrier_type, signed_doping), {})
        dmrg12_value = dmrg12_map.get((carrier_type, signed_doping), {})
        merged_rows.append(
            {
                "carrier_type": carrier_type,
                "signed_doping": format_signed_doping_text(
                    signed_doping=signed_doping,
                    carrier_type=carrier_type,
                ),
                "vmc20_sqrt_s_pi_pi": format_float_text(vmc20_value["sqrt_s_pi_pi"])
                if "sqrt_s_pi_pi" in vmc20_value
                else "",
                "vmc20_sqrt_s_pi_pi_se": format_float_text(
                    vmc20_value["sqrt_s_pi_pi_se"]
                )
                if "sqrt_s_pi_pi_se" in vmc20_value
                else "",
                "vmc20_abs_staggered_mz": format_float_text(
                    vmc20_value["abs_staggered_mz"]
                )
                if "abs_staggered_mz" in vmc20_value
                else "",
                "vmc20_abs_staggered_mz_se": format_float_text(
                    vmc20_value["abs_staggered_mz_se"]
                )
                if "abs_staggered_mz_se" in vmc20_value
                else "",
                "vmc12_sqrt_s_pi_pi": format_float_text(vmc12_value["sqrt_s_pi_pi"])
                if "sqrt_s_pi_pi" in vmc12_value
                else "",
                "vmc12_abs_staggered_mz": format_float_text(
                    vmc12_value["abs_staggered_mz"]
                )
                if "abs_staggered_mz" in vmc12_value
                else "",
                "dmrg12_sqrt_s_pi_pi": format_float_text(dmrg12_value["sqrt_s_pi_pi"])
                if "sqrt_s_pi_pi" in dmrg12_value
                else "",
                "dmrg12_abs_staggered_mz": format_float_text(
                    dmrg12_value["abs_staggered_mz"]
                )
                if "abs_staggered_mz" in dmrg12_value
                else "",
            }
        )
    return merged_rows


def write_fig1_csv(output_path: Path, row_dict_list: list[dict[str, str]]) -> None:
    """
    用途: 将 fig1 数据写出为 CSV 文件。

    参数:
    - output_path: Path, 输出 CSV 路径。
    - row_dict_list: list[dict[str, str]], 待写入的行列表。

    返回:
    - None。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "carrier_type",
        "signed_doping",
        "vmc20_sqrt_s_pi_pi",
        "vmc20_sqrt_s_pi_pi_se",
        "vmc20_abs_staggered_mz",
        "vmc20_abs_staggered_mz_se",
        "vmc12_sqrt_s_pi_pi",
        "vmc12_abs_staggered_mz",
        "dmrg12_sqrt_s_pi_pi",
        "dmrg12_abs_staggered_mz",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_dict_list)


def main() -> None:
    """
    用途: 汇总 3 组 fig1 数据并输出 results/fig1.csv。

    参数:
    - 无。

    返回:
    - None。
    """
    explicit_project_root = ""
    project_root = find_project_root(explicit_project_root=explicit_project_root)

    vmc20_map = load_vmc20_series(
        project_root
        / "results/L_20/defect_average/picture2_defect_average_scalar_average_summary_combined.csv"
    )
    vmc12_map: dict[tuple[str, float], dict[str, float]] = {}
    vmc12_map.update(
        load_vmc12_series(
            project_root
            / "results/benchmark_domain/auto_submit/best_sector_observables_vs_doping.csv",
            carrier_type="hole",
        )
    )
    vmc12_map.update(
        load_vmc12_series(
            project_root
            / "results/benchmark_domain/auto_submit_electron/best_sector_observables_vs_doping.csv",
            carrier_type="electron",
        )
    )

    dmrg12_map: dict[tuple[str, float], dict[str, float]] = {}
    dmrg12_map.update(
        load_dmrg12_series(
            spipi_path=project_root / "results/benchmark_domain/DMRG/data/Spipi.txt",
            mz_path=project_root / "results/benchmark_domain/DMRG.txt",
            carrier_type="hole",
            lattice_size=12,
        )
    )
    dmrg12_map.update(
        load_dmrg12_series(
            spipi_path=project_root
            / "results/benchmark_domain/DMRG/data_electron/Spipi.txt",
            mz_path=project_root / "results/benchmark_domain/DMRG_electron.txt",
            carrier_type="electron",
            lattice_size=12,
        )
    )

    row_dict_list = merge_fig1_rows(
        vmc20_map=vmc20_map,
        vmc12_map=vmc12_map,
        dmrg12_map=dmrg12_map,
    )
    output_path = project_root / "results/fig1.csv"
    write_fig1_csv(output_path=output_path, row_dict_list=row_dict_list)
    print(f"[OK] 已写出: {output_path}")
    print(f"[OK] 行数: {len(row_dict_list)}")


if __name__ == "__main__":
    main()
