#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用途: 组装 picture4 的四个子图 (a)(b)(c)(d).

子图说明:
- top-left panel: 直接引用 PTMC 输出目录中的 Spin Glass order vs temperature 预览图.
- top-right panel: 读取指定 target_sz_4/Sz.json, 使用统一箭头图逻辑绘制.
- bottom-left panel: 读取 DMRG txt 数据, 重建二维 <Sz> 后使用统一箭头图逻辑绘制.
- bottom-right panel: 读取指定 Sz.json, 使用统一箭头图逻辑绘制.

最终排版:
- 第一行放 top-left 与 top-right panel, 编号为 (a)(b).
- 第二行左侧放 DMRG panel, 编号为 (c).
- 第二行右侧放 VMC panel, 编号为 (d).

输出:
- results/picture4.png
- results/picture4.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from plot_domain_sz_arrow import (
    build_shared_domain_norm,
    draw_domain_sz_panel,
    load_dmrg_txt_matrix,
    load_sz_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL_B_SOURCE_DIR = (
    PROJECT_ROOT
    / "results/L_20/defect_average/hole/Ndefect54/logs/defect_seed_3/target_sz_4"
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
PANEL_A_PREVIEW_PATH = Path(
    r"D:\study\研究生\科研\VMC\spin_model\classical_MC\PTMC\results\large_size\J2_0.25_J3_1.25\doping0.080\data\preview_sg_q2_0_mean_vs_T_doping0.080.png"
)
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_PNG_PATH = OUTPUT_DIR / "picture4.png"
OUTPUT_PDF_PATH = OUTPUT_DIR / "picture4.pdf"
PANEL_LABEL_COLOR = "#C62828"
PANEL_LABEL_FONTSIZE = 15.0
PANEL_LABEL_X = 0.015
PANEL_LABEL_Y = 0.985
APS_SERIF_FONT_FAMILY = [
    "Times New Roman",
    "Times",
    "Nimbus Roman No9 L",
    "DejaVu Serif",
]
GRID_WSPACE = 0.00
GRID_HSPACE = 0.00
TOP_PANEL_GAP_WIDTH_RATIO = 0.05
BOTTOM_PANEL_GAP_WIDTH_RATIO = 0.05
BOTTOM_ROW_HEIGHT_RATIO = 1.00
BOTTOM_PANEL_TARGET_GAP = 0.010
FIGURE_BASE_WIDTH = 10.6
FIGURE_MIN_HEIGHT = 7.2
FIGURE_MAX_HEIGHT = 9.4
FIGURE_LEFT_MARGIN = 0.055
FIGURE_RIGHT_MARGIN = 0.985
FIGURE_BOTTOM_MARGIN = 0.070
FIGURE_TOP_MARGIN = 0.985
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


def add_panel_label(axis, panel_label: str) -> None:
    """用途: 在指定坐标轴左上角添加子图编号.

    参数:
    - axis: matplotlib.axes.Axes, 目标坐标轴对象.
    - panel_label: str, 子图编号字符串, 例如 '(a)'.

    返回:
    - None.
    """
    axis.text(
        PANEL_LABEL_X,
        PANEL_LABEL_Y,
        panel_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="normal",
        color=PANEL_LABEL_COLOR,
    )


def draw_image_panel(
    axis,
    panel_image,
    panel_label: str,
) -> None:
    """用途: 在指定坐标轴中绘制位图子图并添加编号.

    参数:
    - axis: matplotlib.axes.Axes, 目标坐标轴对象.
    - panel_image: np.ndarray, 要显示的图像数组.
    - panel_label: str, 子图编号字符串, 例如 '(a)'.
    返回:
    - None.
    """
    axis.imshow(panel_image)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    add_panel_label(axis, panel_label)


def build_picture4_figure(
    top_panel_image,
    top_right_panel_sz_matrix,
    bottom_left_panel_sz_matrix,
    bottom_right_panel_sz_matrix,
):
    """用途: 构建 picture4 的 figure 与四个子图坐标轴.

    参数:
    - top_panel_image: np.ndarray, 第一行子图, 即编号 (a) 的图像数组.
    - top_right_panel_sz_matrix: np.ndarray, 第一行右侧子图, 即编号 (b) 的 <Sz> 矩阵.
    - bottom_left_panel_sz_matrix: np.ndarray, 第二行左侧子图, 即编号 (c) 的 <Sz> 矩阵.
    - bottom_right_panel_sz_matrix: np.ndarray, 第二行右侧子图, 即编号 (d) 的 <Sz> 矩阵.

    返回:
    - tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes], dict[str, object]]
      - figure: figure 对象.
      - panel_axes: [axis_top_left, axis_top_right, axis_bottom_left, axis_bottom_right].
      - artist_dict: 关键 artist 字典, 包含三张箭头图和共享 norm.
    """
    configure_aps_style()

    top_width_ratios = compute_panel_width_ratios(
        [top_panel_image, top_right_panel_sz_matrix]
    )
    bottom_width_ratios = compute_panel_width_ratios(
        [bottom_left_panel_sz_matrix, bottom_right_panel_sz_matrix]
    )
    row_height_ratios = compute_row_height_ratios(
        top_panel_image,
        [top_right_panel_sz_matrix, bottom_left_panel_sz_matrix, bottom_right_panel_sz_matrix],
    )
    figure_size = compute_figure_size(
        [
            top_width_ratios[0] + TOP_PANEL_GAP_WIDTH_RATIO + top_width_ratios[1],
            bottom_width_ratios[0]
            + BOTTOM_PANEL_GAP_WIDTH_RATIO
            + bottom_width_ratios[1],
        ],
        row_height_ratios,
    )
    shared_norm = build_shared_domain_norm(
        [top_right_panel_sz_matrix, bottom_left_panel_sz_matrix, bottom_right_panel_sz_matrix]
    )

    figure = plt.figure(figsize=figure_size, constrained_layout=False)
    outer_grid_spec = figure.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=row_height_ratios,
        hspace=GRID_HSPACE,
    )
    figure.subplots_adjust(
        left=FIGURE_LEFT_MARGIN,
        right=FIGURE_RIGHT_MARGIN,
        bottom=FIGURE_BOTTOM_MARGIN,
        top=FIGURE_TOP_MARGIN,
    )
    top_grid_spec = outer_grid_spec[0, 0].subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=[
            top_width_ratios[0],
            TOP_PANEL_GAP_WIDTH_RATIO,
            top_width_ratios[1],
        ],
        wspace=0.0,
    )
    bottom_grid_spec = outer_grid_spec[1, 0].subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=[
            bottom_width_ratios[0],
            BOTTOM_PANEL_GAP_WIDTH_RATIO,
            bottom_width_ratios[1],
        ],
        wspace=0.0,
    )

    axis_top_left = figure.add_subplot(top_grid_spec[0, 0])
    axis_top_right = figure.add_subplot(top_grid_spec[0, 2])
    axis_bottom_left = figure.add_subplot(bottom_grid_spec[0, 0])
    axis_bottom_right = figure.add_subplot(bottom_grid_spec[0, 2])

    draw_image_panel(axis_top_left, top_panel_image, "(a)")
    top_right_artist = draw_domain_sz_panel(
        axis_top_right,
        top_right_panel_sz_matrix,
        norm=shared_norm,
    )
    bottom_left_artist = draw_domain_sz_panel(
        axis_bottom_left,
        bottom_left_panel_sz_matrix,
        norm=shared_norm,
    )
    bottom_right_artist = draw_domain_sz_panel(
        axis_bottom_right,
        bottom_right_panel_sz_matrix,
        norm=shared_norm,
    )
    add_panel_label(axis_top_right, "(b)")
    add_panel_label(axis_bottom_left, "(c)")
    add_panel_label(axis_bottom_right, "(d)")
    figure.canvas.draw()
    bottom_left_axis_position = axis_bottom_left.get_position()
    bottom_right_axis_position = axis_bottom_right.get_position()
    combined_center_x = 0.5 * (
        float(bottom_left_axis_position.x0) + float(bottom_right_axis_position.x1)
    )
    combined_total_width = (
        float(bottom_left_axis_position.width)
        + float(bottom_right_axis_position.width)
        + BOTTOM_PANEL_TARGET_GAP
    )
    bottom_left_new_x0 = combined_center_x - 0.5 * combined_total_width
    bottom_right_new_x0 = (
        bottom_left_new_x0
        + float(bottom_left_axis_position.width)
        + BOTTOM_PANEL_TARGET_GAP
    )
    axis_bottom_left.set_position(
        [
            bottom_left_new_x0,
            float(bottom_left_axis_position.y0),
            float(bottom_left_axis_position.width),
            float(bottom_left_axis_position.height),
        ]
    )
    axis_bottom_right.set_position(
        [
            bottom_right_new_x0,
            float(bottom_right_axis_position.y0),
            float(bottom_right_axis_position.width),
            float(bottom_right_axis_position.height),
        ]
    )
    artist_dict = {
        "top_right": top_right_artist,
        "bottom_left": bottom_left_artist,
        "bottom_right": bottom_right_artist,
        "colorbar": None,
        "shared_norm": shared_norm,
    }
    return (
        figure,
        [axis_top_left, axis_top_right, axis_bottom_left, axis_bottom_right],
        artist_dict,
    )


def draw_picture4(
    top_panel_image,
    top_right_panel_sz_matrix,
    bottom_left_panel_sz_matrix,
    bottom_right_panel_sz_matrix,
) -> None:
    """用途: 将三个子图排版为 picture4 并保存.

    参数:
    - top_panel_image: np.ndarray, 第一行子图, 即编号 (a) 的图像数组.
    - top_right_panel_sz_matrix: np.ndarray, 第一行右侧子图, 即编号 (b) 的 <Sz> 矩阵.
    - bottom_left_panel_sz_matrix: np.ndarray, 第二行左侧子图, 即编号 (c) 的 <Sz> 矩阵.
    - bottom_right_panel_sz_matrix: np.ndarray, 第二行右侧子图, 即编号 (d) 的 <Sz> 矩阵.

    返回:
    - None.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure, _, _ = build_picture4_figure(
        top_panel_image=top_panel_image,
        top_right_panel_sz_matrix=top_right_panel_sz_matrix,
        bottom_left_panel_sz_matrix=bottom_left_panel_sz_matrix,
        bottom_right_panel_sz_matrix=bottom_right_panel_sz_matrix,
    )
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
    top_panel_image = trim_image_whitespace(load_image_array(PANEL_A_PREVIEW_PATH))
    top_right_panel_sz_matrix = load_panel_b_sz_matrix(
        PANEL_B_SOURCE_DIR,
        PANEL_B_LATTICE_SIZE,
    )
    bottom_left_panel_sz_matrix = load_panel_c_sz_matrix(PANEL_B_DMRG_TXT_PATH)
    bottom_right_panel_target_dir = resolve_panel_c_target_dir(PANEL_C_SOURCE_DIR)
    bottom_right_panel_sz_matrix = load_panel_d_sz_matrix(
        bottom_right_panel_target_dir, PANEL_C_LATTICE_SIZE
    )
    draw_picture4(
        top_panel_image=top_panel_image,
        top_right_panel_sz_matrix=top_right_panel_sz_matrix,
        bottom_left_panel_sz_matrix=bottom_left_panel_sz_matrix,
        bottom_right_panel_sz_matrix=bottom_right_panel_sz_matrix,
    )

    print(f"[OK] panel_a preview: {PANEL_A_PREVIEW_PATH}")
    print(f"[OK] panel_b source: {PANEL_B_SOURCE_DIR}")
    print(f"[OK] panel_c dmrg txt: {PANEL_B_DMRG_TXT_PATH}")
    print(f"[OK] panel_d source: {PANEL_C_SOURCE_DIR}")
    print(f"[OK] panel_d target: {bottom_right_panel_target_dir}")
    print(f"[OK] output png: {OUTPUT_PNG_PATH}")
    print(f"[OK] output pdf: {OUTPUT_PDF_PATH}")


if __name__ == "__main__":
    main()
