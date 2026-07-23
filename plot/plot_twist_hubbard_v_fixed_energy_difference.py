"""整理并绘制固定波函数参数下 Stripe-AFM 能量差随最近邻相互作用 V 的变化。"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "twist_Hubbard"
    / "V_dependence"
    / "tx1ty1"
    / "fixed_params"
)
BASELINE_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "twist_Hubbard"
    / "Energy_dependence"
    / "tx1ty1"
)
OUTPUT_DIRECTORY = FIXED_RESULT_ROOT / "energy_difference_plots"

LATTICE_SIZE_X = 16
LATTICE_SIZE_Y = 16
NUMBER_OF_SITES = LATTICE_SIZE_X * LATTICE_SIZE_Y
V_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0)
STRIPE_STATES = ("Stripe4", "Stripe8")

BASELINE_DIRECTORIES = {
    "AFM": "AFM_SR",
    "Stripe4": "Stripe4_SR",
    "Stripe8": "Stripe8_SR",
}

CSV_HEADERS = (
    "state",
    "V",
    "energy_difference",
    "energy_difference_se",
    "energy_difference_per_site",
    "energy_difference_per_site_se",
)

STATE_PLOT_STYLES = {
    "Stripe4": {"color": "#d95f02", "marker": "s", "linestyle": "--"},
    "Stripe8": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
}


def read_observable_statistics(block_binning_path, observable_name):
    """用途: 从 `block_binning.txt` 读取指定 observable 的 mean 和 SE。

    参数:
    - `block_binning_path`: `Path | str`, block-binning 文本文件路径。
    - `observable_name`: `str`, 需要读取的 observable 名称。

    返回:
    - `tuple[float, float]`, `(mean, standard_error)`。
    """
    block_binning_path = Path(block_binning_path)
    for line in block_binning_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if fields[0] != observable_name:
            continue
        mean = float(fields[1])
        standard_error = float(fields[2])
        if not math.isfinite(mean) or not math.isfinite(standard_error):
            raise ValueError(
                f"Observable {observable_name} contains a non-finite value: "
                f"mean={mean}, SE={standard_error}"
            )
        if standard_error < 0.0:
            raise ValueError(
                f"Observable {observable_name} has a negative SE: {standard_error}"
            )
        return mean, standard_error
    raise KeyError(f"Observable {observable_name} not found in {block_binning_path}")


def calculate_energy_difference(
    stripe_mean,
    stripe_standard_error,
    afm_mean,
    afm_standard_error,
    number_of_sites,
):
    """用途: 计算 Stripe-AFM 总能量差、每格点差值及其独立误差传播。

    科学公式:
    - `Delta_E = E_Stripe - E_AFM`。
    - `sigma_Delta_E = sqrt(sigma_Stripe^2 + sigma_AFM^2)`。
    - `Delta_e = Delta_E / N`, 其中 `N` 为格点数。

    参数:
    - `stripe_mean`: `float`, Stripe 总能量 mean。
    - `stripe_standard_error`: `float`, Stripe 总能量 SE。
    - `afm_mean`: `float`, AFM 总能量 mean。
    - `afm_standard_error`: `float`, AFM 总能量 SE。
    - `number_of_sites`: `int`, 晶格总格点数。

    返回:
    - `tuple[float, float, float, float]`, 依次为总能量差、总能量差 SE、
      每格点能量差、每格点能量差 SE。
    """
    if number_of_sites <= 0:
        raise ValueError("number_of_sites must be positive.")
    energy_difference = stripe_mean - afm_mean
    energy_difference_standard_error = math.hypot(
        stripe_standard_error,
        afm_standard_error,
    )
    return (
        energy_difference,
        energy_difference_standard_error,
        energy_difference / number_of_sites,
        energy_difference_standard_error / number_of_sites,
    )


def collect_energy_difference_rows(scan_result_root, baseline_result_root):
    """用途: 汇总 Stripe4/Stripe8 相对 AFM 的五点能量差数据。

    参数:
    - `scan_result_root`: `Path | str`, `V>0` 扫描结果根目录。
    - `baseline_result_root`: `Path | str`, 提供复用 `V=0` 数据的结果根目录。

    返回:
    - `list[dict[str, float | str]]`, 两种 Stripe 各五个 V 点的 tidy records。
    """
    scan_result_root = Path(scan_result_root)
    baseline_result_root = Path(baseline_result_root)
    rows = []
    for stripe_state in STRIPE_STATES:
        for nearest_neighbor_v in V_VALUES:
            if nearest_neighbor_v == 0.0:
                afm_case_directory = (
                    baseline_result_root / BASELINE_DIRECTORIES["AFM"] / "tp0.00"
                )
                stripe_case_directory = (
                    baseline_result_root
                    / BASELINE_DIRECTORIES[stripe_state]
                    / "tp0.00"
                )
            else:
                case_name = f"V{nearest_neighbor_v:.2f}"
                afm_case_directory = scan_result_root / "AFM" / case_name
                stripe_case_directory = scan_result_root / stripe_state / case_name

            for case_directory in (afm_case_directory, stripe_case_directory):
                nonempty_error_paths = [
                    error_path
                    for error_path in (case_directory / "slurm_out").glob("*.err")
                    if error_path.stat().st_size > 0
                ]
                if nonempty_error_paths:
                    raise RuntimeError(
                        f"Nonempty Slurm error log under {case_directory}: "
                        f"{nonempty_error_paths[0]}"
                    )

            afm_mean, afm_standard_error = read_observable_statistics(
                afm_case_directory / "logs" / "block_binning.txt",
                "E",
            )
            stripe_mean, stripe_standard_error = read_observable_statistics(
                stripe_case_directory / "logs" / "block_binning.txt",
                "E",
            )
            (
                energy_difference,
                energy_difference_standard_error,
                energy_difference_per_site,
                energy_difference_per_site_standard_error,
            ) = calculate_energy_difference(
                stripe_mean,
                stripe_standard_error,
                afm_mean,
                afm_standard_error,
                NUMBER_OF_SITES,
            )
            rows.append(
                {
                    "state": stripe_state,
                    "V": nearest_neighbor_v,
                    "energy_difference": energy_difference,
                    "energy_difference_se": energy_difference_standard_error,
                    "energy_difference_per_site": energy_difference_per_site,
                    "energy_difference_per_site_se": (
                        energy_difference_per_site_standard_error
                    ),
                }
            )
    return rows


def write_energy_difference_csv(rows, output_path):
    """用途: 将 Stripe-AFM 能量差 records 写为 UTF-8 CSV。

    参数:
    - `rows`: `list[dict[str, object]]`, 能量差 tidy records。
    - `output_path`: `Path | str`, 目标 CSV 路径。

    返回:
    - `Path`, 实际写出的 CSV 路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row[header] for header in CSV_HEADERS})
    return output_path


def create_energy_difference_figure(
    rows,
    title="Fixed-parameter Stripe-AFM energy difference",
):
    """用途: 创建两种 Stripe 相对 AFM 的总能量差误差棒图。

    参数:
    - `rows`: `list[dict[str, object]]`, 两种 Stripe 的能量差 records。
    - `title`: `str`, 图标题。

    返回:
    - `tuple[Figure, Axes]`, matplotlib figure 和主坐标轴。
    """
    figure, axes = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for stripe_state in STRIPE_STATES:
        state_rows = sorted(
            (row for row in rows if row["state"] == stripe_state),
            key=lambda row: row["V"],
        )
        axes.errorbar(
            [row["V"] for row in state_rows],
            [row["energy_difference"] for row in state_rows],
            yerr=[row["energy_difference_se"] for row in state_rows],
            label=rf"{stripe_state} $-$ AFM",
            linewidth=1.7,
            markersize=6.0,
            capsize=3.0,
            **STATE_PLOT_STYLES[stripe_state],
        )

    axes.axhline(0.0, color="0.35", linewidth=1.0, linestyle=":")
    axes.set_xlabel(r"$V/t$")
    axes.set_ylabel(r"$E_{\mathrm{Stripe}}-E_{\mathrm{AFM}}$")
    axes.set_xticks(V_VALUES)
    axes.set_title(title)
    axes.grid(axis="y", alpha=0.25)
    axes.legend(frameon=False)
    return figure, axes


if __name__ == "__main__":
    energy_difference_rows = collect_energy_difference_rows(
        FIXED_RESULT_ROOT,
        BASELINE_RESULT_ROOT,
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    csv_output_path = write_energy_difference_csv(
        energy_difference_rows,
        OUTPUT_DIRECTORY / "stripe_afm_energy_difference.csv",
    )
    energy_difference_figure, _ = create_energy_difference_figure(
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
