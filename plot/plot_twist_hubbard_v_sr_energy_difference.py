"""整理并绘制重新 SR 优化后 Stripe-AFM 能量差随 V 的变化。"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt


PLOT_DIRECTORY = Path(__file__).resolve().parent
if str(PLOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PLOT_DIRECTORY))

from plot_twist_hubbard_v_fixed_energy_difference import (  # noqa: E402
    BASELINE_RESULT_ROOT,
    PROJECT_ROOT,
    collect_energy_difference_rows,
    create_energy_difference_figure,
    write_energy_difference_csv,
)


SR_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "twist_Hubbard"
    / "V_dependence"
    / "tx1ty1"
    / "SR"
)
OUTPUT_DIRECTORY = SR_RESULT_ROOT / "energy_difference_plots"
SR_PLOT_TITLE = "SR-optimized Stripe-AFM energy difference"


def collect_sr_energy_difference_rows():
    """用途: 汇总重新 SR 优化后两种 Stripe 相对 AFM 的五点能量差。

    参数:
    - 无。

    返回:
    - `list[dict[str, float | str]]`, 两种 Stripe 各五个 V 点的 records。
    """
    return collect_energy_difference_rows(
        SR_RESULT_ROOT,
        BASELINE_RESULT_ROOT,
    )


def create_sr_energy_difference_figure(rows):
    """用途: 创建与 fixed-parameter 阶段同样式的 SR 能量差图。

    参数:
    - `rows`: `list[dict[str, object]]`, 两种 Stripe 的能量差 records。

    返回:
    - `tuple[Figure, Axes]`, matplotlib figure 和主坐标轴。
    """
    return create_energy_difference_figure(rows, title=SR_PLOT_TITLE)


if __name__ == "__main__":
    energy_difference_rows = collect_sr_energy_difference_rows()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    csv_output_path = write_energy_difference_csv(
        energy_difference_rows,
        OUTPUT_DIRECTORY / "stripe_afm_energy_difference.csv",
    )
    energy_difference_figure, _ = create_sr_energy_difference_figure(
        energy_difference_rows
    )
    pdf_output_path = OUTPUT_DIRECTORY / "stripe_afm_energy_difference.pdf"
    png_output_path = OUTPUT_DIRECTORY / "stripe_afm_energy_difference.png"
    energy_difference_figure.savefig(pdf_output_path)
    energy_difference_figure.savefig(png_output_path, dpi=300)
    plt.close(energy_difference_figure)
    print(f"Wrote {csv_output_path}")
    print(f"Wrote {pdf_output_path}")
    print(f"Wrote {png_output_path}")
