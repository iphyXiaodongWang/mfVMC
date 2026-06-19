#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用途: 组装 picture4 的两个子图 (a)(b).

子图说明:
- panel (a): 直接根据 PTMC 原始 csv 数据重绘 Spin Glass order vs temperature.
- panel (b): 读取指定 target_sz_4/Sz.json, 使用统一箭头图逻辑绘制.

最终排版:
- 单行左右排布, 左侧为 (a), 右侧为 (b).

输出:
- results/picture4.png
- results/picture4.pdf
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from plot_domain_sz_arrow import (
    build_diverging_colormap,
    build_shared_domain_norm,
    draw_domain_sz_panel,
    load_dmrg_txt_matrix,
    load_sz_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL_A_DATA_DIR = Path(
    r"D:\study\研究生\科研\VMC\spin_model\classical_MC\PTMC\results\large_size\J2_0.5_J3_1.25\doping0.090\data"
)
PANEL_B_SOURCE_DIR = (
    PROJECT_ROOT
    / "results/L_20/defect_average/hole/Ndefect36/logs/defect_seed_3/target_sz_-2"
)
PANEL_B_LATTICE_SIZE = 20
PANEL_B_DMRG_TXT_PATH = PROJECT_ROOT / (
    "results/benchmark_domain/"
    "M D=10000_MPSdefect_Heisenberg_OBC_12_12_15series_S1"
    "[12.0, 12.0, 1.0, 1.0, 0.5, 1.25, 0.3].txt"
)
PANEL_B_DMRG_OUTPUT_PATH = (
    PROJECT_ROOT / "results/benchmark_domain/DMRG_hole_Ndefect15_domain_sz_arrow.png"
)
PANEL_B_DMRG_LATTICE_SIZE_X = 12
PANEL_B_DMRG_LATTICE_SIZE_Y = 12
PANEL_B_DMRG_DEFECT_LOCATIONS = [
    (4, 2),
    (10, 7),
    (7, 10),
    (5, 8),
    (2, 11),
    (12, 6),
    (11, 1),
    (4, 12),
    (9, 2),
    (2, 8),
    (10, 10),
    (9, 5),
    (2, 4),
    (5, 4),
    (5, 10),
]
PANEL_C_SOURCE_DIR = PROJECT_ROOT / "results/benchmark_domain/auto_submit/Ndefect15"
PANEL_C_LATTICE_SIZE = 12
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_PNG_PATH = OUTPUT_DIR / "picture4.png"
OUTPUT_PDF_PATH = OUTPUT_DIR / "picture4.pdf"
PANEL_A_TITLE = ""
PANEL_A_XLABEL = "$T$"
PANEL_A_YLABEL = "$q_{SG}^{(2)}$"
PANEL_A_SAMPLING_FILENAME = "sampling_results.csv"
PANEL_A_REPLICA_SAMPLING_GLOB = "sampling_results_seed*_ladder*_replica*.csv"
PANEL_A_OBSERVABLE_COLUMN = "sg_q2_0_mean"
PANEL_A_OBSERVABLE_ERROR_COLUMN = "sg_q2_0_err"
PANEL_A_REPLICA_TEMPERATURE_ATOL = 1.0e-12
PANEL_A_SIZE_DIR_PATTERN = re.compile(r"^Lx(\d+)_Ly(\d+)_Lz(\d+)_Ndefect\d+$")
PANEL_A_LINE_MARKER = "o"
PANEL_A_LINE_MARKER_SIZE = 2.7
PANEL_A_LINE_WIDTH = 1.15
PANEL_A_LEGEND_FONTSIZE = 22
PANEL_A_LEGEND_LOC = "upper right"
PANEL_A_LEGEND_BBOX_TO_ANCHOR = (0.99, 0.50)
PANEL_A_LABEL_FONTSIZE = 24
PANEL_A_TITLE_FONTSIZE = 22
PANEL_A_TICK_FONTSIZE = 22
PANEL_A_GRID_ALPHA = 0.20
PANEL_A_GRID_LINEWIDTH = 0.60
PANEL_A_XLIM = [0.1, 0.9]
PANEL_A_YLIM = [0.0, 0.18]
PANEL_A_XTICKS = [0.2, 0.4, 0.6, 0.8]
PANEL_A_YTICKS = [0.0, 0.06, 0.12, 0.18]
PANEL_A_XMINOR_TICKS = [0.1, 0.3, 0.5, 0.7, 0.9]
PANEL_A_YMINOR_TICKS = [0.03, 0.09, 0.15]
PANEL_A_XLABEL_X = 0.50
PANEL_A_XLABEL_Y = -0.03
PANEL_A_YLABEL_X = -0.03
PANEL_A_YLABEL_Y = 0.50
PANEL_A_INSET_ENABLED = True
PANEL_A_INSET_TARGET_TEMPERATURE = 0.1
PANEL_A_INSET_BOUNDS = [0.35, 0.58, 0.60, 0.39]
PANEL_A_INSET_XLABEL = r"$1/L$"
PANEL_A_INSET_YLABEL = "$q_{SG}^{(2)}$"
PANEL_A_INSET_XLABEL_X = 0.45
PANEL_A_INSET_XLABEL_Y = -0.06
PANEL_A_INSET_YLABEL_X = -0.08
PANEL_A_INSET_YLABEL_Y = 0.50
PANEL_A_INSET_XLIM = [0.00, 0.13]
PANEL_A_INSET_YLIM = [0.07, 0.17]
PANEL_A_INSET_XTICKS = [0.00, 0.10]
PANEL_A_INSET_YTICKS = [0.08, 0.16]
PANEL_A_INSET_XMINOR_TICKS = [0.05]
PANEL_A_INSET_YMINOR_TICKS = [0.12]
PANEL_A_INSET_LABEL_FONTSIZE = 24
PANEL_A_INSET_TICK_FONTSIZE = 22
PANEL_A_INSET_MINOR_TICK_LENGTH = 2.0
PANEL_A_INSET_MINOR_TICK_WIDTH = 0.6
PANEL_A_INSET_MARKER_SIZE = 18.0
PANEL_A_INSET_MARKER_COLOR = "black"
PANEL_A_INSET_ERRORBAR_ENABLED = True
PANEL_A_INSET_ERRORBAR_COLOR = "black"
PANEL_A_INSET_ERRORBAR_ALPHA = 1.0
PANEL_A_INSET_ERRORBAR_LINEWIDTH = 0.8
PANEL_A_INSET_ERRORBAR_CAPSIZE = 2.0
PANEL_A_INSET_ERRORBAR_CAPTHICK = 0.8
PANEL_A_INSET_ERRORBAR_ZORDER = 2.5
PANEL_A_INSET_FIT_LINEWIDTH = 1.0
PANEL_A_INSET_FIT_COLOR = "black"
PANEL_A_INSET_BORDER_LINEWIDTH = 0.8
PANEL_A_MINOR_TICK_LENGTH = 2.0
PANEL_A_MINOR_TICK_WIDTH = 0.6
PANEL_LABEL_COLOR = "black"
PANEL_LABEL_FONTSIZE = 24.0
PANEL_LABEL_HORIZONTAL_ALIGNMENT = "left"
PANEL_LABEL_VERTICAL_ALIGNMENT = "bottom"
PANEL_LABEL_DEFAULT_X = 0.015
PANEL_LABEL_DEFAULT_Y = 1.005
PANEL_A_LABEL_X = 0.015
PANEL_A_LABEL_Y = 1.005
PANEL_B_LABEL_X = 0.015
PANEL_B_LABEL_Y = 1.005
APS_SERIF_FONT_FAMILY = [
    "Times New Roman",
    "Times",
    "Nimbus Roman No9 L",
    "DejaVu Serif",
]
PANEL_DOMAIN_ARROW_SCALE = 5.0
PANEL_B_ARROW_LENGTH_SHRINK_FACTOR = 1.20
PANEL_DOMAIN_ARROW_WIDTH = 0.008
PANEL_DOMAIN_ARROW_ALPHA = 0.9
PANEL_DOMAIN_ARROW_ANGLES = "uv"
PANEL_DOMAIN_ARROW_SCALE_UNITS = None
PANEL_DOMAIN_ARROW_HEADWIDTH = 7.5
PANEL_DOMAIN_ARROW_HEADLENGTH = 5.0
PANEL_DOMAIN_ARROW_HEADAXISLENGTH = 4.5
PANEL_DOMAIN_ARROW_MINLENGTH = 1.0
PANEL_DOMAIN_ARROW_ZORDER = 5.0
PANEL_DOMAIN_COLORMAP_NAME = "PuOr"  # "BrBG"
PANEL_DOMAIN_NEGATIVE_COLOR = "#F2B134"
PANEL_DOMAIN_ZERO_COLOR = "#F7F7F7"
PANEL_DOMAIN_POSITIVE_COLOR = "#2B7BFF"
PANEL_DOMAIN_SITE_MARKER_SIZE = 10.0
PANEL_DOMAIN_SITE_MARKER_FACE_COLOR = "none"
PANEL_DOMAIN_SITE_MARKER_EDGE_COLOR = "black"
PANEL_DOMAIN_SITE_MARKER_ALPHA = 0.8
PANEL_DOMAIN_SITE_MARKER_LINEWIDTH = 1.3
PANEL_DOMAIN_DEFECT_MARKER_SIZE = PANEL_DOMAIN_SITE_MARKER_SIZE + 15.0
PANEL_DOMAIN_DEFECT_MARKER_FACE_COLOR = "none"
PANEL_DOMAIN_DEFECT_MARKER_EDGE_COLOR = "#ff0000"
PANEL_DOMAIN_DEFECT_MARKER_ALPHA = 0.7
PANEL_DOMAIN_DEFECT_MARKER_LINEWIDTH = 1.3
PANEL_DOMAIN_GRID_COLOR = "black"
PANEL_DOMAIN_GRID_ALPHA = 0.5
PANEL_DOMAIN_GRID_LINEWIDTH = 1.2
PANEL_DOMAIN_SHOW_PERIODIC_BOUNDARY_BONDS = True
PANEL_DOMAIN_PERIODIC_BOUNDARY_STUB_LENGTH = 0.75
PANEL_DOMAIN_SHOW_COLORBAR = False
PANEL_DOMAIN_COLORBAR_LABEL = "domain = (-1)^(x+y) * <Sz>"
PANEL_DOMAIN_COLORBAR_WIDTH_RATIO = 0.040
PANEL_DOMAIN_COLORBAR_PAD_RATIO = 0.025
PANEL_DOMAIN_COLORBAR_LABEL_FONTSIZE = 8.8
PANEL_DOMAIN_COLORBAR_TICK_FONTSIZE = 8.0
GRID_WSPACE = 0.00
GRID_HSPACE = 0.00
TOP_PANEL_GAP_WIDTH_RATIO = 0.025
BOTTOM_PANEL_GAP_WIDTH_RATIO = 0.05
BOTTOM_ROW_HEIGHT_RATIO = 1.00
PANEL_ROW_GAP_RATIO = 0.1
PANEL_A_WIDTH_RATIO = 1.3
PANEL_A_HEIGHT_RATIO = 1.3
PANEL_B_SIDE_RATIO = 1.3
FIGURE_BASE_WIDTH = 10.6
FIGURE_MIN_HEIGHT = 7.2
FIGURE_MAX_HEIGHT = 9.4
FIGURE_LEFT_MARGIN = 0.025
FIGURE_RIGHT_MARGIN = 0.995
FIGURE_BOTTOM_MARGIN = 0.070
FIGURE_TOP_MARGIN = 0.985
BASE_SQUARE_PANEL_SIDE_INCH = 3.90
IMAGE_WHITE_BACKGROUND_THRESHOLD = 0.02
IMAGE_CROP_PADDING_PIXELS = 2


def configure_aps_style() -> None:
    """用途: 配置接近 APS/PRL 的 matplotlib 字体风格.

    参数:
    - 无.

    返回:
    - None.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": APS_SERIF_FONT_FAMILY,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_panel_b_dmrg_sz_matrix(
    txt_path: Path,
    lattice_size_x: int,
    lattice_size_y: int,
    defect_locations: list[tuple[int, int]],
) -> np.ndarray:
    """用途: 读取子图 (b) 的 DMRG <Sz> 矩阵.

    参数:
    - txt_path: Path, DMRG 一维 txt 文件路径.
    - lattice_size_x: int, x 方向系统尺寸 Lx.
    - lattice_size_y: int, y 方向系统尺寸 Ly.
    - defect_locations: list[tuple[int, int]], defect 坐标列表, 使用 1-based index.

    返回:
    - np.ndarray, DMRG <Sz> 二维矩阵.
    """
    return load_dmrg_txt_matrix(
        txt_path=txt_path,
        lattice_size_x=lattice_size_x,
        lattice_size_y=lattice_size_y,
        defect_locations=defect_locations,
    )


def load_panel_b_sz_matrix(target_dir: Path, lattice_size: int) -> np.ndarray:
    """用途: 读取子图 (b) 的 VMC <Sz> 矩阵.

    参数:
    - target_dir: Path, 含有 Sz.json 的目录.
    - lattice_size: int, 系统线性尺寸 L.

    返回:
    - np.ndarray, VMC <Sz> 二维矩阵.
    """
    if not target_dir.is_dir():
        raise NotADirectoryError(f"子图 (b) 数据目录不存在: {target_dir}")

    return load_sz_matrix(target_dir, lattice_size)


def load_panel_c_sz_matrix(txt_path: Path) -> np.ndarray:
    """用途: 读取子图 (c) 的 DMRG <Sz> 矩阵.

    参数:
    - txt_path: Path, DMRG 一维 txt 文件路径.

    返回:
    - np.ndarray, DMRG <Sz> 二维矩阵.
    """
    return load_panel_b_dmrg_sz_matrix(
        txt_path=txt_path,
        lattice_size_x=PANEL_B_DMRG_LATTICE_SIZE_X,
        lattice_size_y=PANEL_B_DMRG_LATTICE_SIZE_Y,
        defect_locations=PANEL_B_DMRG_DEFECT_LOCATIONS,
    )


def load_panel_d_sz_matrix(target_dir: Path, lattice_size: int) -> np.ndarray:
    """用途: 读取子图 (d) 的 VMC <Sz> 矩阵.

    参数:
    - target_dir: Path, 含有 Sz.json 的目录.
    - lattice_size: int, 系统线性尺寸 L.

    返回:
    - np.ndarray, VMC <Sz> 二维矩阵.
    """
    if not target_dir.is_dir():
        raise NotADirectoryError(f"子图 (d) 数据目录不存在: {target_dir}")

    return load_sz_matrix(target_dir, lattice_size)


def resolve_panel_c_target_dir(source_dir: Path) -> Path:
    """用途: 解析子图 (c) 的实际数据目录.

    参数:
    - source_dir: Path, 数据源目录.
      支持两种形式:
      1. 直接传入含有 Sz.json 的 target_sz_* 目录.
      2. 传入 auto_submit/Ndefect* 根目录, 程序自动读取最低能量 sector.

    返回:
    - Path, 实际含有 Sz.json 的 target_sz_* 目录路径.
    """
    if not source_dir.is_dir():
        raise NotADirectoryError(f"子图 (c) 数据源目录不存在: {source_dir}")

    if (source_dir / "Sz.json").is_file():
        return source_dir

    sector_summary_path = source_dir / "logs/sector_min_energy.txt"
    if not sector_summary_path.is_file():
        raise FileNotFoundError(
            "子图 (c) 数据源目录下既没有 Sz.json, 也没有 logs/sector_min_energy.txt: "
            f"{source_dir}"
        )

    sector_table = np.genfromtxt(
        sector_summary_path,
        names=True,
        delimiter="\t",
        dtype=None,
        encoding="utf-8-sig",
        comments="#",
    )
    sector_table = np.atleast_1d(sector_table)
    if sector_table.size == 0:
        raise ValueError(f"sector_min_energy.txt 为空: {sector_summary_path}")

    energy_values = np.asarray(sector_table["min_energy"], dtype=float)
    sz_values = np.asarray(sector_table["sz"], dtype=int)
    best_index = int(np.nanargmin(energy_values))
    best_sz = int(sz_values[best_index])
    resolved_target_dir = source_dir / "logs" / f"target_sz_{best_sz}"
    if not (resolved_target_dir / "Sz.json").is_file():
        raise FileNotFoundError(
            "最低能量 sector 对应的 Sz.json 不存在: "
            f"{resolved_target_dir / 'Sz.json'}"
        )
    return resolved_target_dir


def load_image_array(image_path: Path):
    """用途: 读取已有位图文件为图像数组.

    参数:
    - image_path: Path, PNG 等位图文件路径.

    返回:
    - np.ndarray, matplotlib 可直接 imshow 的图像数组.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"缺少图片文件: {image_path}")
    return mpimg.imread(image_path)


def load_panel_a_series(data_dir: Path) -> list[dict[str, object]]:
    """用途: 从 PTMC 多个尺寸目录读取 a 图所需的 sg_q2_0_mean vs T 曲线.

    参数:
    - data_dir: Path, 包含多个 Lx*_Ly*_Lz*_Ndefect* 子目录的数据根目录.

    返回:
    - list[dict[str, object]], 每个元素包含:
      - label: str, 图例标签, 例如 "L=8".
      - size: int, 线性尺寸 L.
      - temperature_values: np.ndarray, 温度数组.
      - observable_values: np.ndarray, sg_q2_0_mean 数组.
      - observable_error_values: np.ndarray, sg_q2_0_err 数组.
    """
    if not data_dir.is_dir():
        raise NotADirectoryError(f"子图 (a) 数据目录不存在: {data_dir}")

    panel_a_series: list[dict[str, object]] = []
    for one_size_dir in sorted(data_dir.iterdir()):
        if not one_size_dir.is_dir():
            continue
        matched = PANEL_A_SIZE_DIR_PATTERN.match(one_size_dir.name)
        if matched is None:
            continue

        sampling_results_path = one_size_dir / PANEL_A_SAMPLING_FILENAME
        if not sampling_results_path.is_file():
            continue

        size_value = int(matched.group(1))
        replica_statistics = load_panel_a_replica_statistics(one_size_dir)
        if replica_statistics is None:
            sampling_table = np.genfromtxt(
                sampling_results_path,
                names=True,
                delimiter=",",
                dtype=None,
                encoding="utf-8-sig",
            )
            sampling_table = np.atleast_1d(sampling_table)
            if sampling_table.size == 0:
                raise ValueError(f"sampling_results.csv 为空: {sampling_results_path}")

            temperature_values = np.asarray(sampling_table["T"], dtype=float)
            observable_values = np.asarray(
                sampling_table[PANEL_A_OBSERVABLE_COLUMN],
                dtype=float,
            )
            if PANEL_A_OBSERVABLE_ERROR_COLUMN in sampling_table.dtype.names:
                observable_error_values = np.asarray(
                    sampling_table[PANEL_A_OBSERVABLE_ERROR_COLUMN],
                    dtype=float,
                )
            else:
                observable_error_values = np.zeros_like(observable_values, dtype=float)
            sort_indices = np.argsort(temperature_values)
            temperature_values = temperature_values[sort_indices]
            observable_values = observable_values[sort_indices]
            observable_error_values = observable_error_values[sort_indices]
        else:
            (
                temperature_values,
                observable_values,
                observable_error_values,
            ) = replica_statistics
        panel_a_series.append(
            {
                "label": f"$L={size_value}$",
                "size": size_value,
                "temperature_values": temperature_values,
                "observable_values": observable_values,
                "observable_error_values": observable_error_values,
            }
        )

    if len(panel_a_series) == 0:
        raise FileNotFoundError(
            f"未在子图 (a) 数据目录下找到有效的 {PANEL_A_SAMPLING_FILENAME}: {data_dir}"
        )

    panel_a_series.sort(key=lambda one_series: int(one_series["size"]))
    return panel_a_series


def load_panel_a_replica_statistics(
    one_size_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """用途: 从单个尺寸目录的 replica csv 重构 disorder average 与标准误.

    参数:
    - one_size_dir: Path, 单个系统尺寸目录, 其中可包含多个
      `sampling_results_seed*_ladder*_replica*.csv`.

    返回:
    - tuple[np.ndarray, np.ndarray, np.ndarray] | None:
      - 若存在 replica 文件, 返回
        (temperature_values, observable_mean_values, observable_standard_error_values).
      - 若不存在 replica 文件, 返回 None.
    """
    replica_csv_paths = sorted(one_size_dir.glob(PANEL_A_REPLICA_SAMPLING_GLOB))
    if len(replica_csv_paths) == 0:
        return None

    reference_temperature_values = None
    replica_observable_rows = []

    for replica_csv_path in replica_csv_paths:
        replica_table = np.genfromtxt(
            replica_csv_path,
            names=True,
            delimiter=",",
            dtype=None,
            encoding="utf-8-sig",
        )
        replica_table = np.atleast_1d(replica_table)
        if replica_table.size == 0:
            raise ValueError(f"replica sampling csv 为空: {replica_csv_path}")
        if PANEL_A_OBSERVABLE_COLUMN not in replica_table.dtype.names:
            raise KeyError(
                f"replica sampling csv 缺少列 {PANEL_A_OBSERVABLE_COLUMN}: {replica_csv_path}"
            )

        temperature_values = np.asarray(replica_table["T"], dtype=float)
        observable_values = np.asarray(
            replica_table[PANEL_A_OBSERVABLE_COLUMN],
            dtype=float,
        )
        sort_indices = np.argsort(temperature_values)
        temperature_values = temperature_values[sort_indices]
        observable_values = observable_values[sort_indices]

        if reference_temperature_values is None:
            reference_temperature_values = temperature_values
        else:
            if (
                reference_temperature_values.shape != temperature_values.shape
                or not np.allclose(
                    reference_temperature_values,
                    temperature_values,
                    rtol=0.0,
                    atol=PANEL_A_REPLICA_TEMPERATURE_ATOL,
                )
            ):
                raise ValueError(
                    "replica sampling csv 的温度网格不一致, 无法直接做 disorder average: "
                    f"{replica_csv_path}"
                )
        replica_observable_rows.append(observable_values)

    observable_matrix = np.asarray(replica_observable_rows, dtype=float)
    observable_mean_values = np.mean(observable_matrix, axis=0)
    if observable_matrix.shape[0] >= 2:
        observable_standard_error_values = np.std(
            observable_matrix,
            axis=0,
            ddof=1,
        ) / np.sqrt(float(observable_matrix.shape[0]))
    else:
        observable_standard_error_values = np.zeros_like(
            observable_mean_values,
            dtype=float,
        )

    return (
        np.asarray(reference_temperature_values, dtype=float),
        np.asarray(observable_mean_values, dtype=float),
        np.asarray(observable_standard_error_values, dtype=float),
    )


def build_panel_a_inset_data(
    panel_a_series: list[dict[str, object]],
    target_temperature: float,
) -> dict[str, np.ndarray | float]:
    """用途: 提取 a 图 inset 所需的 1/L 外推数据并执行线性拟合.

    参数:
    - panel_a_series: list[dict[str, object]], a 图多尺寸曲线数据列表.
    - target_temperature: float, 目标温度 T. 程序对每个尺寸选择最接近该温度的数据点.

    返回:
    - dict[str, np.ndarray | float], 包含:
      - inverse_size_values: np.ndarray, 横坐标 1/L 数组.
      - order_values: np.ndarray, 纵坐标 spin glass order 数组.
      - order_error_values: np.ndarray, 纵向误差棒所需的误差数组.
      - nearest_temperature_values: np.ndarray, 每个尺寸实际选中的最近温度.
      - fit_slope: float, 线性拟合斜率.
      - fit_intercept: float, 线性拟合在 x=0 处的截距.
      - fit_x_values: np.ndarray, 拟合直线绘图用横坐标数组.
      - fit_y_values: np.ndarray, 拟合直线绘图用纵坐标数组.
    """
    if len(panel_a_series) < 2:
        raise ValueError("a 图 inset 至少需要两个系统尺寸才能做线性拟合.")

    inverse_size_values = []
    order_values = []
    order_error_values = []
    nearest_temperature_values = []

    for one_series in panel_a_series:
        temperature_values = np.asarray(one_series["temperature_values"], dtype=float)
        observable_values = np.asarray(one_series["observable_values"], dtype=float)
        observable_error_values = np.asarray(
            one_series.get(
                "observable_error_values",
                np.zeros_like(observable_values, dtype=float),
            ),
            dtype=float,
        )
        if temperature_values.size == 0 or observable_values.size == 0:
            raise ValueError("a 图曲线数据为空, 无法构造 inset.")

        nearest_index = int(np.argmin(np.abs(temperature_values - target_temperature)))
        if "size" in one_series:
            size_value = int(one_series["size"])
        else:
            label_text = str(one_series.get("label", ""))
            matched = re.search(r"L\s*=\s*(\d+)", label_text)
            if matched is None:
                raise KeyError("a 图曲线数据缺少 size 字段, 且 label 中无法解析 L.")
            size_value = int(matched.group(1))
        inverse_size_values.append(1.0 / float(size_value))
        order_values.append(float(observable_values[nearest_index]))
        order_error_values.append(float(observable_error_values[nearest_index]))
        nearest_temperature_values.append(float(temperature_values[nearest_index]))

    inverse_size_array = np.asarray(inverse_size_values, dtype=float)
    order_array = np.asarray(order_values, dtype=float)
    order_error_array = np.asarray(order_error_values, dtype=float)
    nearest_temperature_array = np.asarray(nearest_temperature_values, dtype=float)
    sort_indices = np.argsort(inverse_size_array)
    inverse_size_array = inverse_size_array[sort_indices]
    order_array = order_array[sort_indices]
    order_error_array = order_error_array[sort_indices]
    nearest_temperature_array = nearest_temperature_array[sort_indices]

    fit_slope, fit_intercept = np.polyfit(inverse_size_array, order_array, deg=1)
    fit_x_max = float(np.max(inverse_size_array)) * 1.05
    fit_x_values = np.array([0.0, fit_x_max], dtype=float)
    fit_y_values = fit_slope * fit_x_values + fit_intercept
    return {
        "inverse_size_values": inverse_size_array,
        "order_values": order_array,
        "order_error_values": order_error_array,
        "nearest_temperature_values": nearest_temperature_array,
        "fit_slope": float(fit_slope),
        "fit_intercept": float(fit_intercept),
        "fit_x_values": fit_x_values,
        "fit_y_values": fit_y_values,
    }


def draw_panel_a_inset(
    parent_axis,
    panel_a_series: list[dict[str, object]],
) -> dict[str, object] | None:
    """用途: 在 a 图中绘制 1/L 外推 inset.

    参数:
    - parent_axis: matplotlib.axes.Axes, a 图主坐标轴对象.
    - panel_a_series: list[dict[str, object]], a 图多尺寸曲线数据列表.

    返回:
    - dict[str, object] | None:
      - 若启用 inset, 返回包含 inset 轴对象, error bar 和拟合结果的字典.
      - 若未启用 inset, 返回 None.
    """
    if not PANEL_A_INSET_ENABLED:
        return None

    inset_data = build_panel_a_inset_data(
        panel_a_series,
        target_temperature=PANEL_A_INSET_TARGET_TEMPERATURE,
    )
    inset_axis = parent_axis.inset_axes(PANEL_A_INSET_BOUNDS)
    errorbar_container = None
    if PANEL_A_INSET_ERRORBAR_ENABLED:
        errorbar_container = inset_axis.errorbar(
            inset_data["inverse_size_values"],
            inset_data["order_values"],
            yerr=inset_data["order_error_values"],
            fmt="none",
            ecolor=PANEL_A_INSET_ERRORBAR_COLOR,
            elinewidth=PANEL_A_INSET_ERRORBAR_LINEWIDTH,
            capsize=PANEL_A_INSET_ERRORBAR_CAPSIZE,
            capthick=PANEL_A_INSET_ERRORBAR_CAPTHICK,
            alpha=PANEL_A_INSET_ERRORBAR_ALPHA,
            zorder=PANEL_A_INSET_ERRORBAR_ZORDER,
        )
    inset_axis.scatter(
        inset_data["inverse_size_values"],
        inset_data["order_values"],
        s=PANEL_A_INSET_MARKER_SIZE,
        c=PANEL_A_INSET_MARKER_COLOR,
        zorder=3,
    )
    inset_axis.plot(
        inset_data["fit_x_values"],
        inset_data["fit_y_values"],
        color=PANEL_A_INSET_FIT_COLOR,
        linewidth=PANEL_A_INSET_FIT_LINEWIDTH,
        zorder=2,
    )
    inset_axis.set_xlabel(PANEL_A_INSET_XLABEL, fontsize=PANEL_A_INSET_LABEL_FONTSIZE)
    inset_axis.set_ylabel(PANEL_A_INSET_YLABEL, fontsize=PANEL_A_INSET_LABEL_FONTSIZE)
    inset_axis.xaxis.set_label_coords(PANEL_A_INSET_XLABEL_X, PANEL_A_INSET_XLABEL_Y)
    inset_axis.yaxis.set_label_coords(PANEL_A_INSET_YLABEL_X, PANEL_A_INSET_YLABEL_Y)
    inset_axis.tick_params(labelsize=PANEL_A_INSET_TICK_FONTSIZE)

    x_max = float(np.max(inset_data["fit_x_values"]))
    y_min = min(
        float(np.min(inset_data["order_values"])),
        float(np.min(inset_data["fit_y_values"])),
    )
    y_max = max(
        float(np.max(inset_data["order_values"])),
        float(np.max(inset_data["fit_y_values"])),
    )
    y_padding = 0.08 * max(1e-6, y_max - y_min)
    if PANEL_A_INSET_XLIM is not None:
        inset_axis.set_xlim(PANEL_A_INSET_XLIM)
    else:
        inset_axis.set_xlim(0.0, x_max)
    if PANEL_A_INSET_YLIM is not None:
        inset_axis.set_ylim(PANEL_A_INSET_YLIM)
    else:
        inset_axis.set_ylim(y_min - y_padding, y_max + y_padding)
    if PANEL_A_INSET_XTICKS is not None:
        inset_axis.set_xticks(PANEL_A_INSET_XTICKS)
    if PANEL_A_INSET_YTICKS is not None:
        inset_axis.set_yticks(PANEL_A_INSET_YTICKS)
    if PANEL_A_INSET_XMINOR_TICKS is not None:
        inset_axis.set_xticks(PANEL_A_INSET_XMINOR_TICKS, minor=True)
    if PANEL_A_INSET_YMINOR_TICKS is not None:
        inset_axis.set_yticks(PANEL_A_INSET_YMINOR_TICKS, minor=True)
    inset_axis.tick_params(
        axis="both",
        which="minor",
        length=PANEL_A_INSET_MINOR_TICK_LENGTH,
        width=PANEL_A_INSET_MINOR_TICK_WIDTH,
    )
    for one_spine in inset_axis.spines.values():
        one_spine.set_linewidth(PANEL_A_INSET_BORDER_LINEWIDTH)

    return {
        "axis": inset_axis,
        "errorbar_container": errorbar_container,
        **inset_data,
    }


def trim_image_whitespace(panel_image: np.ndarray) -> np.ndarray:
    """用途: 裁掉位图四周近似纯白的边缘, 减少无效留白.

    参数:
    - panel_image: np.ndarray, 输入图像数组, 支持灰度图或 RGB(A) 图.

    返回:
    - np.ndarray, 裁边后的图像数组. 若未检测到有效内容, 则返回原图.
    """
    image_array = np.asarray(panel_image)
    if image_array.ndim == 2:
        content_mask = np.abs(image_array - 1.0) > IMAGE_WHITE_BACKGROUND_THRESHOLD
    else:
        rgb_array = image_array[..., :3]
        content_mask = (
            np.max(np.abs(rgb_array - 1.0), axis=2) > IMAGE_WHITE_BACKGROUND_THRESHOLD
        )

    content_coordinates = np.argwhere(content_mask)
    if content_coordinates.size == 0:
        return image_array

    row_start = max(0, int(content_coordinates[:, 0].min()) - IMAGE_CROP_PADDING_PIXELS)
    row_end = min(
        image_array.shape[0],
        int(content_coordinates[:, 0].max()) + IMAGE_CROP_PADDING_PIXELS + 1,
    )
    col_start = max(0, int(content_coordinates[:, 1].min()) - IMAGE_CROP_PADDING_PIXELS)
    col_end = min(
        image_array.shape[1],
        int(content_coordinates[:, 1].max()) + IMAGE_CROP_PADDING_PIXELS + 1,
    )
    return image_array[row_start:row_end, col_start:col_end]


def compute_panel_width_ratios(panel_images) -> list[float]:
    """用途: 根据图像宽高比估计子图的相对宽度.

    参数:
    - panel_images: Iterable[np.ndarray], 若干子图对应的图像数组.

    返回:
    - List[float], 每个子图的宽度比例 w/h.
    """
    width_ratios: list[float] = []
    for panel_image in panel_images:
        panel_height = max(1, int(panel_image.shape[0]))
        panel_width = max(1, int(panel_image.shape[1]))
        width_ratios.append(float(panel_width) / float(panel_height))
    return width_ratios


def compute_row_height_ratios(top_panel_image, bottom_panel_images) -> list[float]:
    """用途: 根据图像宽高比估计两行布局的相对高度.

    参数:
    - top_panel_image: np.ndarray, 第一行单图对应的图像数组.
    - bottom_panel_images: Iterable[np.ndarray], 第二行两张子图的图像数组.

    返回:
    - List[float], [上排高度比例, 下排高度比例].
    """
    _ = top_panel_image
    _ = bottom_panel_images
    return [BOTTOM_ROW_HEIGHT_RATIO, BOTTOM_ROW_HEIGHT_RATIO]


def compute_figure_size(
    row_total_width_ratios: list[float],
    row_height_ratios: list[float],
) -> tuple[float, float]:
    """用途: 根据网格比例估计 figure 尺寸.

    参数:
    - row_total_width_ratios: List[float], 各行总宽度比例.
    - row_height_ratios: List[float], 上下两行高度比例.

    返回:
    - tuple[float, float], figure 的 (width, height), 单位为英寸.
    """
    figure_width = FIGURE_BASE_WIDTH
    total_width_ratio = max(1e-12, float(max(row_total_width_ratios)))
    total_height_ratio = max(1e-12, float(sum(row_height_ratios)))
    raw_figure_height = figure_width * total_height_ratio / total_width_ratio
    figure_height = min(FIGURE_MAX_HEIGHT, max(FIGURE_MIN_HEIGHT, raw_figure_height))
    return figure_width, figure_height


def compute_square_panel_positions() -> dict[str, list[float]]:
    """用途: 计算当前 a/b 两个子图在 figure 中的最终位置.

    参数:
    - 无.

    返回:
    - dict[str, list[float]], 两个子图的 [x0, y0, width, height] 位置字典.
    """
    panel_side_inch = float(BASE_SQUARE_PANEL_SIDE_INCH)
    panel_a_width_inch = max(1e-6, float(PANEL_A_WIDTH_RATIO) * panel_side_inch)
    panel_a_height_inch = max(1e-6, float(PANEL_A_HEIGHT_RATIO) * panel_side_inch)
    panel_b_side_inch = max(1e-6, float(PANEL_B_SIDE_RATIO) * panel_side_inch)
    column_gap_inch = float(TOP_PANEL_GAP_WIDTH_RATIO) * panel_side_inch
    if PANEL_DOMAIN_SHOW_COLORBAR:
        colorbar_pad_inch = float(PANEL_DOMAIN_COLORBAR_PAD_RATIO) * panel_side_inch
        colorbar_width_inch = float(PANEL_DOMAIN_COLORBAR_WIDTH_RATIO) * panel_side_inch
    else:
        colorbar_pad_inch = 0.0
        colorbar_width_inch = 0.0
    left_margin_inch = float(FIGURE_BASE_WIDTH) * float(FIGURE_LEFT_MARGIN)
    right_margin_inch = float(FIGURE_BASE_WIDTH) * float(1.0 - FIGURE_RIGHT_MARGIN)
    bottom_margin_inch = float(FIGURE_MAX_HEIGHT) * float(FIGURE_BOTTOM_MARGIN)
    top_margin_inch = float(FIGURE_MAX_HEIGHT) * float(1.0 - FIGURE_TOP_MARGIN)

    content_height_inch = max(panel_a_height_inch, panel_b_side_inch)
    content_width_inch = (
        panel_a_width_inch
        + column_gap_inch
        + panel_b_side_inch
        + colorbar_pad_inch
        + colorbar_width_inch
    )
    figure_width_inch = left_margin_inch + content_width_inch + right_margin_inch
    figure_height_inch = bottom_margin_inch + content_height_inch + top_margin_inch

    panel_a_x_inch = left_margin_inch
    panel_b_x_inch = panel_a_x_inch + panel_a_width_inch + column_gap_inch
    panel_a_y_inch = bottom_margin_inch + 0.5 * (
        content_height_inch - panel_a_height_inch
    )
    panel_b_y_inch = bottom_margin_inch + 0.5 * (
        content_height_inch - panel_b_side_inch
    )
    colorbar_x_inch = panel_b_x_inch + panel_b_side_inch + colorbar_pad_inch
    layout_dict = {
        "figure_size": [figure_width_inch, figure_height_inch],
        "panel_a": [
            panel_a_x_inch / figure_width_inch,
            panel_a_y_inch / figure_height_inch,
            panel_a_width_inch / figure_width_inch,
            panel_a_height_inch / figure_height_inch,
        ],
        "panel_b": [
            panel_b_x_inch / figure_width_inch,
            panel_b_y_inch / figure_height_inch,
            panel_b_side_inch / figure_width_inch,
            panel_b_side_inch / figure_height_inch,
        ],
    }
    if PANEL_DOMAIN_SHOW_COLORBAR:
        layout_dict["colorbar"] = [
            colorbar_x_inch / figure_width_inch,
            panel_b_y_inch / figure_height_inch,
            colorbar_width_inch / figure_width_inch,
            panel_b_side_inch / figure_height_inch,
        ]
    return layout_dict


def compute_panel_b_arrow_spacing_scale(
    square_panel_positions: dict[str, list[float]],
    panel_shape: tuple[int, int],
) -> float:
    """用途: 按相邻 site 的显示间距计算 b 图箭头尺寸缩放比例.

    参数:
    - square_panel_positions: dict[str, list[float]], compute_square_panel_positions() 返回的布局字典.
    - panel_shape: tuple[int, int], b 图矩阵形状, 格式为 (Lx, Ly).

    返回:
    - float, b 图相邻 site 显示间距相对于参考 12x12 方图的比值.
    """
    figure_width_inch = float(square_panel_positions["figure_size"][0])
    panel_width_inch = float(square_panel_positions["panel_b"][2]) * figure_width_inch
    panel_site_spacing_inch = panel_width_inch / (float(panel_shape[0]) + 0.2)
    reference_panel_site_spacing_inch = float(BASE_SQUARE_PANEL_SIDE_INCH) / (
        float(PANEL_C_LATTICE_SIZE) + 0.2
    )
    return max(1e-6, panel_site_spacing_inch / reference_panel_site_spacing_inch)


def add_panel_label(axis, panel_label: str) -> None:
    """用途: 在指定坐标轴左上角添加子图编号.

    参数:
    - axis: matplotlib.axes.Axes, 目标坐标轴对象.
    - panel_label: str, 子图编号字符串, 例如 '(a)'.

    返回:
    - None.
    """
    if panel_label == "(a)":
        label_x = PANEL_A_LABEL_X
        label_y = PANEL_A_LABEL_Y
    elif panel_label == "(b)":
        label_x = PANEL_B_LABEL_X
        label_y = PANEL_B_LABEL_Y
    else:
        label_x = PANEL_LABEL_DEFAULT_X
        label_y = PANEL_LABEL_DEFAULT_Y

    axis.text(
        label_x,
        label_y,
        panel_label,
        transform=axis.transAxes,
        ha=PANEL_LABEL_HORIZONTAL_ALIGNMENT,
        va=PANEL_LABEL_VERTICAL_ALIGNMENT,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="normal",
        color=PANEL_LABEL_COLOR,
        clip_on=False,
    )


def draw_panel_a_series(
    axis,
    panel_a_series: list[dict[str, object]],
    panel_label: str,
) -> dict[str, object] | None:
    """用途: 在指定坐标轴中绘制 a 图的 sg_q2_0_mean vs T 多尺寸折线图.

    参数:
    - axis: matplotlib.axes.Axes, 目标坐标轴对象.
    - panel_a_series: list[dict[str, object]], a 图曲线数据列表.
    - panel_label: str, 子图编号字符串, 例如 '(a)'.

    返回:
    - dict[str, object] | None:
      - 若启用 inset, 返回 inset 对应的 artist 与拟合结果字典.
      - 若未启用 inset, 返回 None.
    """
    axis.set_aspect("auto")
    axis.set_title(PANEL_A_TITLE, fontsize=PANEL_A_TITLE_FONTSIZE, pad=6.0)
    axis.set_xlabel(PANEL_A_XLABEL, fontsize=PANEL_A_LABEL_FONTSIZE)
    axis.xaxis.set_label_coords(PANEL_A_XLABEL_X, PANEL_A_XLABEL_Y)
    axis.set_ylabel(PANEL_A_YLABEL, fontsize=PANEL_A_LABEL_FONTSIZE)
    axis.yaxis.set_label_coords(PANEL_A_YLABEL_X, PANEL_A_YLABEL_Y)
    axis.tick_params(labelsize=PANEL_A_TICK_FONTSIZE)

    for one_series in panel_a_series:
        axis.plot(
            np.asarray(one_series["temperature_values"], dtype=float),
            np.asarray(one_series["observable_values"], dtype=float),
            marker=PANEL_A_LINE_MARKER,
            markersize=PANEL_A_LINE_MARKER_SIZE,
            linewidth=PANEL_A_LINE_WIDTH,
            label=str(one_series["label"]),
        )

    axis.margins(x=0.03, y=0.05)
    if PANEL_A_XLIM is not None:
        axis.set_xlim(PANEL_A_XLIM)
    if PANEL_A_YLIM is not None:
        axis.set_ylim(PANEL_A_YLIM)
    if PANEL_A_XTICKS is not None:
        axis.set_xticks(PANEL_A_XTICKS)
    if PANEL_A_YTICKS is not None:
        axis.set_yticks(PANEL_A_YTICKS)
    if PANEL_A_XMINOR_TICKS is not None:
        axis.set_xticks(PANEL_A_XMINOR_TICKS, minor=True)
    if PANEL_A_YMINOR_TICKS is not None:
        axis.set_yticks(PANEL_A_YMINOR_TICKS, minor=True)
    axis.tick_params(
        axis="both",
        which="minor",
        length=PANEL_A_MINOR_TICK_LENGTH,
        width=PANEL_A_MINOR_TICK_WIDTH,
    )
    axis.legend(
        loc=PANEL_A_LEGEND_LOC,
        bbox_to_anchor=PANEL_A_LEGEND_BBOX_TO_ANCHOR,
        frameon=True,
        fontsize=PANEL_A_LEGEND_FONTSIZE,
    )
    add_panel_label(axis, panel_label)
    return draw_panel_a_inset(axis, panel_a_series)


def hide_lattice_ticks(axis) -> None:
    """用途: 隐藏晶格图的坐标刻度标签和刻度线, 保留已有网格.

    参数:
    - axis: matplotlib.axes.Axes, 目标晶格图坐标轴对象.

    返回:
    - None.
    """
    axis.tick_params(
        axis="both",
        which="both",
        length=0.0,
        labelbottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
    )


def build_picture4_figure(
    top_panel_series,
    panel_b_sz_matrix,
):
    """用途: 构建仅含 a/b 两图的 picture4 figure.

    参数:
    - top_panel_series: list[dict[str, object]], 编号 (a) 的多尺寸曲线数据.
    - panel_b_sz_matrix: np.ndarray, 编号 (b) 的 <Sz> 矩阵, 显示在右侧.

    返回:
    - tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes], dict[str, object]]
      - figure: figure 对象.
      - panel_axes: [axis_panel_a, axis_panel_b].
      - artist_dict: 关键 artist 字典, 包含 b 图与共享 norm.
    """
    configure_aps_style()

    square_panel_positions = compute_square_panel_positions()
    panel_b_arrow_size_scale = compute_panel_b_arrow_spacing_scale(
        square_panel_positions=square_panel_positions,
        panel_shape=panel_b_sz_matrix.shape,
    )
    figure_size = tuple(square_panel_positions["figure_size"])
    shared_norm = build_shared_domain_norm([panel_b_sz_matrix])
    panel_b_colormap = build_diverging_colormap(
        negative_color=PANEL_DOMAIN_NEGATIVE_COLOR,
        positive_color=PANEL_DOMAIN_POSITIVE_COLOR,
        neutral_color=PANEL_DOMAIN_ZERO_COLOR,
        colormap_name=PANEL_DOMAIN_COLORMAP_NAME,
    )

    figure = plt.figure(figsize=figure_size, constrained_layout=False)
    figure.subplots_adjust(
        left=FIGURE_LEFT_MARGIN,
        right=FIGURE_RIGHT_MARGIN,
        bottom=FIGURE_BOTTOM_MARGIN,
        top=FIGURE_TOP_MARGIN,
    )
    axis_panel_a = figure.add_subplot(1, 2, 1)
    axis_panel_b = figure.add_subplot(1, 2, 2)

    panel_a_inset_artist = draw_panel_a_series(axis_panel_a, top_panel_series, "(a)")
    panel_b_artist = draw_domain_sz_panel(
        axis_panel_b,
        panel_b_sz_matrix,
        norm=shared_norm,
        colormap=panel_b_colormap,
        site_marker_size=PANEL_DOMAIN_SITE_MARKER_SIZE,
        site_marker_face_color=PANEL_DOMAIN_SITE_MARKER_FACE_COLOR,
        site_marker_edge_color=PANEL_DOMAIN_SITE_MARKER_EDGE_COLOR,
        site_marker_alpha=PANEL_DOMAIN_SITE_MARKER_ALPHA,
        site_marker_linewidth=PANEL_DOMAIN_SITE_MARKER_LINEWIDTH,
        defect_marker_size=PANEL_DOMAIN_DEFECT_MARKER_SIZE,
        defect_marker_face_color=PANEL_DOMAIN_DEFECT_MARKER_FACE_COLOR,
        defect_marker_edge_color=PANEL_DOMAIN_DEFECT_MARKER_EDGE_COLOR,
        defect_marker_alpha=PANEL_DOMAIN_DEFECT_MARKER_ALPHA,
        defect_marker_linewidth=PANEL_DOMAIN_DEFECT_MARKER_LINEWIDTH,
        grid_color=PANEL_DOMAIN_GRID_COLOR,
        grid_alpha=PANEL_DOMAIN_GRID_ALPHA,
        grid_linewidth=PANEL_DOMAIN_GRID_LINEWIDTH,
        show_periodic_boundary_bonds=PANEL_DOMAIN_SHOW_PERIODIC_BOUNDARY_BONDS,
        periodic_boundary_stub_length=PANEL_DOMAIN_PERIODIC_BOUNDARY_STUB_LENGTH,
        quiver_scale=PANEL_DOMAIN_ARROW_SCALE
        / panel_b_arrow_size_scale
        * PANEL_B_ARROW_LENGTH_SHRINK_FACTOR,
        quiver_width=PANEL_DOMAIN_ARROW_WIDTH * panel_b_arrow_size_scale,
        quiver_alpha=PANEL_DOMAIN_ARROW_ALPHA,
        quiver_angles=PANEL_DOMAIN_ARROW_ANGLES,
        quiver_scale_units=PANEL_DOMAIN_ARROW_SCALE_UNITS,
        quiver_headwidth=PANEL_DOMAIN_ARROW_HEADWIDTH * panel_b_arrow_size_scale,
        quiver_headlength=PANEL_DOMAIN_ARROW_HEADLENGTH * panel_b_arrow_size_scale,
        quiver_headaxislength=PANEL_DOMAIN_ARROW_HEADAXISLENGTH
        * panel_b_arrow_size_scale,
        quiver_minlength=PANEL_DOMAIN_ARROW_MINLENGTH * panel_b_arrow_size_scale,
        colormap_name=PANEL_DOMAIN_COLORMAP_NAME,
    )
    if hasattr(panel_b_artist["quiver"], "set_zorder"):
        panel_b_artist["quiver"].set_zorder(PANEL_DOMAIN_ARROW_ZORDER)
    hide_lattice_ticks(axis_panel_b)
    add_panel_label(axis_panel_b, "(b)")
    axis_panel_a.set_position(square_panel_positions["panel_a"])
    axis_panel_b.set_position(square_panel_positions["panel_b"])
    shared_colorbar = None
    if PANEL_DOMAIN_SHOW_COLORBAR:
        colorbar_axis = figure.add_axes(square_panel_positions["colorbar"])
        shared_colorbar = figure.colorbar(
            panel_b_artist["quiver"],
            cax=colorbar_axis,
        )
        shared_colorbar.set_label(
            PANEL_DOMAIN_COLORBAR_LABEL,
            fontsize=PANEL_DOMAIN_COLORBAR_LABEL_FONTSIZE,
        )
        shared_colorbar.ax.tick_params(labelsize=PANEL_DOMAIN_COLORBAR_TICK_FONTSIZE)
    artist_dict = {
        "panel_a_inset": panel_a_inset_artist,
        "panel_b": panel_b_artist,
        "colorbar": shared_colorbar,
        "shared_norm": shared_norm,
    }
    return (
        figure,
        [axis_panel_a, axis_panel_b],
        artist_dict,
    )


def draw_picture4(
    top_panel_series,
    panel_b_sz_matrix,
) -> None:
    """用途: 将 a/b 两个子图排版为 picture4 并保存.

    参数:
    - top_panel_series: list[dict[str, object]], 编号 (a) 的多尺寸曲线数据.
    - panel_b_sz_matrix: np.ndarray, 编号 (b) 的 <Sz> 矩阵.

    返回:
    - None.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure, _, _ = build_picture4_figure(
        top_panel_series=top_panel_series,
        panel_b_sz_matrix=panel_b_sz_matrix,
    )
    figure.savefig(OUTPUT_DIR / "picture4.svg", dpi=260, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG_PATH, dpi=260, bbox_inches="tight")
    figure.savefig(OUTPUT_PDF_PATH, dpi=260, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """用途: 主入口, 生成 picture4 的 PNG/PDF.

    参数:
    - 无.

    返回:
    - None.
    """
    top_panel_series = load_panel_a_series(PANEL_A_DATA_DIR)
    panel_b_sz_matrix = load_panel_b_sz_matrix(
        PANEL_B_SOURCE_DIR,
        PANEL_B_LATTICE_SIZE,
    )
    draw_picture4(
        top_panel_series=top_panel_series,
        panel_b_sz_matrix=panel_b_sz_matrix,
    )

    print(f"[OK] panel_a data: {PANEL_A_DATA_DIR}")
    print(f"[OK] panel_b source: {PANEL_B_SOURCE_DIR}")
    print(f"[OK] output png: {OUTPUT_PNG_PATH}")
    print(f"[OK] output pdf: {OUTPUT_PDF_PATH}")


if __name__ == "__main__":
    main()
