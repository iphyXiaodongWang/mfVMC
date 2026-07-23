"""整理并绘制 twist Hubbard 三种 SR 态的能量与序参量随 `|t'|` 的变化。"""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


STATE_DIRECTORIES = (
    ("AFM", "AFM_SR"),
    ("Stripe4", "Stripe4_SR"),
    ("Stripe8", "Stripe8_SR"),
)

TPRIME_VALUES = (0.00, 0.05, 0.10, 0.15, 0.20)

OBSERVABLE_NAMES = (
    "E",
    "E_hop",
    "E_hop_tx",
    "E_hop_ty",
    "E_hop_t2",
    "E_int",
    "E_int_charge",
    "E_int_spin",
    "E_pert_t_local_proj",
    "E_pert_t_local_proj_x",
    "E_pert_t_local_proj_y",
    "E_pert_t_local_proj_t2",
    "E_pert_J_local_proj",
    "E_pert_J_local_proj_x",
    "E_pert_J_local_proj_y",
    "E_pert_J_local_proj_t2",
)

CSV_HEADERS = ("state", "tprime_abs", "t2", "observable", "mean", "se")

ORDER_PARAMETER_SPECS = (
    ("Stripe4", "Stripe4_SR", ("Delta_c", "Delta_s")),
    ("Stripe8", "Stripe8_SR", ("Delta_c", "Delta_s")),
    ("AFM", "AFM_SR", ("Delta_AF",)),
)

ORDER_PARAMETER_CSV_HEADERS = (
    "state",
    "tprime_abs",
    "t2",
    "parameter",
    "value",
)

LATTICE_SIZE_X = 16
LATTICE_SIZE_Y = 16

MEASURED_ORDER_SPECS = (
    ("Stripe4", "Stripe4_SR", 4),
    ("Stripe8", "Stripe8_SR", 8),
    ("AFM", "AFM_SR", None),
)

MEASURED_ORDER_CSV_HEADERS = (
    "state",
    "tprime_abs",
    "t2",
    "observable",
    "wavevector",
    "value",
)

OBSERVABLE_PLOT_SPECS = {
    "E": ("energy_total", "Total Hubbard energy", r"$E$"),
    "E_hop": ("energy_hopping_total", "Total hopping energy", r"$E_{\mathrm{hop}}$"),
    "E_hop_tx": ("energy_hopping_x", "X-direction hopping energy", r"$E_{t_x}$"),
    "E_hop_ty": ("energy_hopping_y", "Y-direction hopping energy", r"$E_{t_y}$"),
    "E_hop_t2": ("energy_hopping_tprime", "Diagonal hopping energy", r"$E_{t'}$"),
    "E_int": ("energy_interaction_total", "Total interaction energy", r"$E_U$"),
    "E_int_charge": (
        "energy_interaction_charge",
        "Charge contribution to interaction energy",
        r"$E_{U,\mathrm{charge}}$",
    ),
    "E_int_spin": (
        "energy_interaction_spin",
        "Spin contribution to interaction energy",
        r"$E_{U,\mathrm{spin}}$",
    ),
    "E_pert_t_local_proj": (
        "energy_projected_hopping_total",
        "Total projected hopping energy",
        r"$E^{\mathrm{proj}}_t$",
    ),
    "E_pert_t_local_proj_x": (
        "energy_projected_hopping_x",
        "X-direction projected hopping energy",
        r"$E^{\mathrm{proj}}_{t_x}$",
    ),
    "E_pert_t_local_proj_y": (
        "energy_projected_hopping_y",
        "Y-direction projected hopping energy",
        r"$E^{\mathrm{proj}}_{t_y}$",
    ),
    "E_pert_t_local_proj_t2": (
        "energy_projected_hopping_tprime",
        "Diagonal projected hopping energy",
        r"$E^{\mathrm{proj}}_{t'}$",
    ),
    "E_pert_J_local_proj": (
        "energy_projected_exchange_total",
        "Total projected exchange energy",
        r"$E^{\mathrm{proj}}_J$",
    ),
    "E_pert_J_local_proj_x": (
        "energy_projected_exchange_x",
        "X-direction projected exchange energy",
        r"$E^{\mathrm{proj}}_{J_x}$",
    ),
    "E_pert_J_local_proj_y": (
        "energy_projected_exchange_y",
        "Y-direction projected exchange energy",
        r"$E^{\mathrm{proj}}_{J_y}$",
    ),
    "E_pert_J_local_proj_t2": (
        "energy_projected_exchange_tprime",
        "Diagonal projected exchange energy",
        r"$E^{\mathrm{proj}}_{J'}$",
    ),
}

STATE_PLOT_STYLES = {
    "AFM": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "Stripe4": {"color": "#d95f02", "marker": "s", "linestyle": "--"},
    "Stripe8": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
}


def read_block_binning_energy_values(block_binning_path):
    """用途: 读取一个 case 的 16 个能量分量 mean 和 standard error。

    参数:
    - `block_binning_path`: `Path`, `logs/block_binning.txt` 文件路径。

    返回:
    - `dict[str, tuple[float, float]]`, observable 到 `(mean, se)` 的映射。
    """
    block_binning_path = Path(block_binning_path)
    values = {}
    for line in block_binning_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if fields[0] in OBSERVABLE_NAMES:
            values[fields[0]] = (float(fields[1]), float(fields[2]))
    return values


def collect_energy_component_rows(root_directory):
    """用途: 整理 AFM、Stripe4 和 Stripe8 的五点能量 mean/SE 数据。

    参数:
    - `root_directory`: `Path`, `Energy_dependence/tx1ty1` 结果根目录。

    返回:
    - `list[dict[str, object]]`, 共 240 行的 tidy energy records。
    """
    root_directory = Path(root_directory)
    rows = []
    for state_name, state_directory_name in STATE_DIRECTORIES:
        for tprime_abs in TPRIME_VALUES:
            case_directory = root_directory / state_directory_name / f"tp{tprime_abs:.2f}"
            nonempty_error_paths = [
                error_path
                for error_path in (case_directory / "slurm_out").glob("*.err")
                if error_path.stat().st_size > 0
            ]
            if nonempty_error_paths:
                raise RuntimeError(
                    f"Nonempty Slurm error log under {state_directory_name}/{case_directory.name}: "
                    f"{nonempty_error_paths[0]}"
                )
            values = read_block_binning_energy_values(
                case_directory / "logs" / "block_binning.txt"
            )
            for observable_name in OBSERVABLE_NAMES:
                mean, standard_error = values[observable_name]
                if (
                    not math.isfinite(mean)
                    or not math.isfinite(standard_error)
                    or standard_error < 0.0
                ):
                    raise ValueError(
                        f"Invalid energy statistics for {state_directory_name}/"
                        f"{case_directory.name}/{observable_name}: "
                        f"mean={mean}, se={standard_error}"
                    )
                rows.append(
                    {
                        "state": state_name,
                        "tprime_abs": tprime_abs,
                        "t2": -tprime_abs,
                        "observable": observable_name,
                        "mean": mean,
                        "se": standard_error,
                    }
                )
    return rows


def write_energy_component_csv(rows, output_path):
    """用途: 将三种态的 energy mean/SE records 写为 UTF-8 tidy CSV。

    参数:
    - `rows`: `list[dict[str, object]]`, `collect_energy_component_rows` 的返回值。
    - `output_path`: `Path`, 目标 CSV 文件路径。

    返回:
    - `Path`, 实际写出的 CSV 路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def create_energy_component_figure(rows, observable_name):
    """用途: 为单个能量分量构造三种 SR 态的五点 errorbar figure。

    参数:
    - `rows`: `list[dict[str, object]]`, tidy energy records。
    - `observable_name`: `str`, `OBSERVABLE_PLOT_SPECS` 中的能量字段名。

    返回:
    - `tuple[Figure, Axes]`, matplotlib figure 和主坐标轴。
    """
    if observable_name not in OBSERVABLE_PLOT_SPECS:
        raise ValueError(f"Unknown energy observable: {observable_name}")

    _, title, energy_symbol = OBSERVABLE_PLOT_SPECS[observable_name]
    figure, axes = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for state_name, _ in STATE_DIRECTORIES:
        state_rows = sorted(
            (
                row
                for row in rows
                if row["state"] == state_name and row["observable"] == observable_name
            ),
            key=lambda row: row["tprime_abs"],
        )
        x_values = [row["tprime_abs"] for row in state_rows]
        mean_values = [row["mean"] for row in state_rows]
        standard_errors = [row["se"] for row in state_rows]
        data_line, _, _ = axes.errorbar(
            x_values,
            mean_values,
            yerr=standard_errors,
            linewidth=1.6,
            markersize=5.5,
            capsize=3.0,
            elinewidth=1.0,
            **STATE_PLOT_STYLES[state_name],
        )
        data_line.set_label(state_name)

    axes.set_title(title)
    axes.set_xlabel(r"$|t'|$")
    axes.set_ylabel(f"{energy_symbol} for 16×16 system")
    axes.set_xticks(TPRIME_VALUES)
    axes.grid(True, alpha=0.25)
    axes.legend(frameon=False)
    return figure, axes


def save_all_energy_component_figures(rows, output_directory):
    """用途: 将 16 个能量分量分别保存为 PNG 和 PDF。

    参数:
    - `rows`: `list[dict[str, object]]`, tidy energy records。
    - `output_directory`: `Path`, 图片输出目录。

    返回:
    - `list[Path]`, 依 observable 顺序排列的 32 个 PNG/PDF 路径。
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for observable_name in OBSERVABLE_NAMES:
        filename_stem, _, _ = OBSERVABLE_PLOT_SPECS[observable_name]
        figure, _ = create_energy_component_figure(rows, observable_name)
        try:
            png_path = output_directory / f"{filename_stem}.png"
            pdf_path = output_directory / f"{filename_stem}.pdf"
            figure.savefig(png_path, dpi=220)
            figure.savefig(pdf_path)
            output_paths.extend((png_path, pdf_path))
        finally:
            plt.close(figure)
    return output_paths


def collect_order_parameter_rows(root_directory):
    """用途: 读取三种 SR 态在五个扫描点的最优变分序参量。

    参数:
    - `root_directory`: `Path`, `Energy_dependence/tx1ty1` 结果根目录。

    返回:
    - `list[dict[str, object]]`, 共 25 行的 tidy parameter records。其中
      `Delta_c` 保留 `min_params.json` 中的原始符号。
    """
    root_directory = Path(root_directory)
    rows = []
    for state_name, state_directory_name, parameter_names in ORDER_PARAMETER_SPECS:
        for tprime_abs in TPRIME_VALUES:
            case_name = f"tp{tprime_abs:.2f}"
            parameter_path = (
                root_directory
                / state_directory_name
                / case_name
                / "logs"
                / "min_params.json"
            )
            parameter_values = json.loads(parameter_path.read_text(encoding="utf-8"))
            for parameter_name in parameter_names:
                if parameter_name not in parameter_values:
                    raise ValueError(
                        f"Missing order parameter for {state_directory_name}/"
                        f"{case_name}/{parameter_name}"
                    )
                value = float(parameter_values[parameter_name])
                if not math.isfinite(value):
                    raise ValueError(
                        f"Invalid order parameter for {state_directory_name}/"
                        f"{case_name}/{parameter_name}: value={value}"
                    )
                rows.append(
                    {
                        "state": state_name,
                        "tprime_abs": tprime_abs,
                        "t2": -tprime_abs,
                        "parameter": parameter_name,
                        "value": value,
                    }
                )
    return rows


def write_order_parameter_csv(rows, output_path):
    """用途: 将三种态的最优变分序参量写为 UTF-8 tidy CSV。

    参数:
    - `rows`: `list[dict[str, object]]`, `collect_order_parameter_rows` 的返回值。
    - `output_path`: `Path`, 目标 CSV 文件路径。

    返回:
    - `Path`, 实际写出的 CSV 路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ORDER_PARAMETER_CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _plot_order_parameter_series(axes, rows, state_name, parameter_name, label):
    """用途: 在指定坐标轴上绘制一个态的一组最优变分参数曲线。

    参数:
    - `axes`: `Axes`, 目标 matplotlib 坐标轴。
    - `rows`: `list[dict[str, object]]`, tidy parameter records。
    - `state_name`: `str`, `AFM`、`Stripe4` 或 `Stripe8`。
    - `parameter_name`: `str`, `Delta_c`、`Delta_s` 或 `Delta_AF`。
    - `label`: `str`, 图例标签。

    返回:
    - `Line2D`, 新绘制的数据曲线。
    """
    series_rows = sorted(
        (
            row
            for row in rows
            if row["state"] == state_name and row["parameter"] == parameter_name
        ),
        key=lambda row: row["tprime_abs"],
    )
    line, = axes.plot(
        [row["tprime_abs"] for row in series_rows],
        [row["value"] for row in series_rows],
        linewidth=1.6,
        markersize=5.5,
        label=label,
        **STATE_PLOT_STYLES[state_name],
    )
    return line


def create_charge_order_figure(rows):
    """用途: 构造 Stripe4 和 Stripe8 带符号 `Delta_c` 的对比图。

    参数:
    - `rows`: `list[dict[str, object]]`, tidy parameter records。

    返回:
    - `tuple[Figure, Axes]`, matplotlib figure 和主坐标轴。
    """
    figure, axes = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    _plot_order_parameter_series(axes, rows, "Stripe4", "Delta_c", r"Stripe4 $\Delta_c$")
    _plot_order_parameter_series(axes, rows, "Stripe8", "Delta_c", r"Stripe8 $\Delta_c$")
    axes.axhline(0.0, color="0.45", linewidth=1.0, alpha=0.7, zorder=0)
    axes.set_title("Charge order parameter")
    axes.set_xlabel(r"$|t'|$")
    axes.set_ylabel(r"Signed $\Delta_c$")
    axes.set_xticks(TPRIME_VALUES)
    axes.grid(True, alpha=0.25)
    axes.legend(frameon=False)
    return figure, axes


def create_spin_and_afm_order_figure(rows):
    """用途: 构造两个 Stripe `Delta_s` 与 AFM `Delta_AF` 的对比图。

    参数:
    - `rows`: `list[dict[str, object]]`, tidy parameter records。

    返回:
    - `tuple[Figure, Axes]`, matplotlib figure 和主坐标轴。
    """
    figure, axes = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    _plot_order_parameter_series(axes, rows, "Stripe4", "Delta_s", r"Stripe4 $\Delta_s$")
    _plot_order_parameter_series(axes, rows, "Stripe8", "Delta_s", r"Stripe8 $\Delta_s$")
    _plot_order_parameter_series(axes, rows, "AFM", "Delta_AF", r"AFM $\Delta_{AF}$")
    axes.set_title("Spin and AFM order parameters")
    axes.set_xlabel(r"$|t'|$")
    axes.set_ylabel("Order parameter")
    axes.set_xticks(TPRIME_VALUES)
    axes.grid(True, alpha=0.25)
    axes.legend(frameon=False)
    return figure, axes


def save_order_parameter_figures(rows, output_directory):
    """用途: 将两张序参量图分别保存为 PNG 和 PDF。

    参数:
    - `rows`: `list[dict[str, object]]`, tidy parameter records。
    - `output_directory`: `Path`, 图片输出目录。

    返回:
    - `list[Path]`, 两张图对应的 4 个 PNG/PDF 路径。
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_specs = (
        ("order_parameter_charge", create_charge_order_figure),
        ("order_parameter_spin_and_afm", create_spin_and_afm_order_figure),
    )
    output_paths = []
    for filename_stem, figure_factory in figure_specs:
        figure, _ = figure_factory(rows)
        try:
            png_path = output_directory / f"{filename_stem}.png"
            pdf_path = output_directory / f"{filename_stem}.pdf"
            figure.savefig(png_path, dpi=220)
            figure.savefig(pdf_path)
            output_paths.extend((png_path, pdf_path))
        finally:
            plt.close(figure)
    return output_paths


def _compute_measurement_x_profiles(data):
    """用途: 从 16×16 measure 数据计算 x 向平均 charge 和 staggered spin profile。

    科学计算公式:
    - `n_bar(x) = sum_y <n(x,y)> / Ly`。
    - `m_stag(x) = sum_y (-1)^(x+y) <Sz(x,y)> / Ly`。

    参数:
    - `data`: `dict[str, object]`, `block_binning_mean.json` 顶层字典。

    返回:
    - `tuple[list[float], list[float]]`, `(average_charge, staggered_spin)`。
    """
    average_charge = []
    staggered_spin = []
    for x_coord in range(1, LATTICE_SIZE_X + 1):
        charge_sum = 0.0
        staggered_spin_sum = 0.0
        for y_coord in range(1, LATTICE_SIZE_Y + 1):
            charge_key = f"n_{x_coord}_{y_coord}"
            spin_key = f"Sz_{x_coord}_{y_coord}"
            for observable_key in (charge_key, spin_key):
                if observable_key not in data:
                    raise ValueError(f"Missing site measurement: {observable_key}")
                value = float(data[observable_key])
                if not math.isfinite(value):
                    raise ValueError(
                        f"Invalid site measurement: {observable_key}={value}"
                    )
            charge_sum += float(data[charge_key])
            staggered_sign = -1.0 if (x_coord + y_coord) % 2 else 1.0
            staggered_spin_sum += staggered_sign * float(data[spin_key])
        average_charge.append(charge_sum / LATTICE_SIZE_Y)
        staggered_spin.append(staggered_spin_sum / LATTICE_SIZE_Y)
    return average_charge, staggered_spin


def _compute_fourier_amplitude(profile, wavevector):
    """用途: 计算实数 x-profile 在目标波矢处的 Fourier 振幅大小。

    科学计算公式:
    - `M(q) = (2 / Lx) * |sum_x f(x) exp(-i q x)|`。

    参数:
    - `profile`: `list[float]`, 长度为 `Lx` 的实数 profile。
    - `wavevector`: `float`, 目标 x 方向波矢 `q`。

    返回:
    - `float`, 非负 Fourier 振幅。
    """
    fourier_sum = 0.0j
    for x_coord, value in enumerate(profile, start=1):
        phase = -wavevector * x_coord
        fourier_sum += value * complex(math.cos(phase), math.sin(phase))
    return 2.0 * abs(fourier_sum) / LATTICE_SIZE_X


def compute_stripe_measured_order(data, stripe_period):
    """用途: 由 measure 格点均值计算 Stripe charge/spin modulation 大小。

    科学计算公式:
    - charge 波矢 `q_c = 2π / lambda`。
    - spin 波矢 `q_s = π / lambda`，作用于 staggered spin profile。
    - `M_c = 2 |sum_x n_bar(x) exp(-i q_c x)| / Lx`。
    - `M_s = 2 |sum_x m_stag(x) exp(-i q_s x)| / Lx`。

    参数:
    - `data`: `dict[str, object]`, `block_binning_mean.json` 顶层字典。
    - `stripe_period`: `int`, Stripe charge 周期 `lambda`。

    返回:
    - `dict[str, float]`, `charge_modulation` 和 `spin_modulation`。
    """
    if stripe_period <= 0:
        raise ValueError(f"stripe_period must be positive: {stripe_period}")
    average_charge, staggered_spin = _compute_measurement_x_profiles(data)
    charge_wavevector = 2.0 * math.pi / stripe_period
    spin_wavevector = math.pi / stripe_period
    return {
        "charge_modulation": _compute_fourier_amplitude(
            average_charge,
            charge_wavevector,
        ),
        "spin_modulation": _compute_fourier_amplitude(
            staggered_spin,
            spin_wavevector,
        ),
    }


def compute_afm_measured_order(data):
    """用途: 由 measure 格点均值计算 AFM staggered order 大小。

    科学计算公式:
    - `M_AF = |sum_(x,y) (-1)^(x+y) <Sz(x,y)>| / (Lx*Ly)`。

    参数:
    - `data`: `dict[str, object]`, `block_binning_mean.json` 顶层字典。

    返回:
    - `float`, 非负 AFM order 大小。
    """
    _, staggered_spin = _compute_measurement_x_profiles(data)
    return abs(sum(staggered_spin) / LATTICE_SIZE_X)


def collect_measured_order_rows(root_directory):
    """用途: 整理三个态、五个扫描点的实际 measure 序参量大小。

    参数:
    - `root_directory`: `Path`, `Energy_dependence/tx1ty1` 结果根目录。

    返回:
    - `list[dict[str, object]]`, 共 25 行的 tidy measured-order records。
    """
    root_directory = Path(root_directory)
    rows = []
    for state_name, state_directory_name, stripe_period in MEASURED_ORDER_SPECS:
        for tprime_abs in TPRIME_VALUES:
            case_name = f"tp{tprime_abs:.2f}"
            measurement_path = (
                root_directory
                / state_directory_name
                / case_name
                / "logs"
                / "block_binning_mean.json"
            )
            data = json.loads(measurement_path.read_text(encoding="utf-8"))
            if stripe_period is None:
                rows.append(
                    {
                        "state": state_name,
                        "tprime_abs": tprime_abs,
                        "t2": -tprime_abs,
                        "observable": "afm_order",
                        "wavevector": "(pi,pi)",
                        "value": compute_afm_measured_order(data),
                    }
                )
                continue

            measured_values = compute_stripe_measured_order(data, stripe_period)
            rows.extend(
                (
                    {
                        "state": state_name,
                        "tprime_abs": tprime_abs,
                        "t2": -tprime_abs,
                        "observable": "charge_modulation",
                        "wavevector": f"2*pi/{stripe_period}",
                        "value": measured_values["charge_modulation"],
                    },
                    {
                        "state": state_name,
                        "tprime_abs": tprime_abs,
                        "t2": -tprime_abs,
                        "observable": "spin_modulation",
                        "wavevector": f"pi/{stripe_period}",
                        "value": measured_values["spin_modulation"],
                    },
                )
            )
    return rows


def write_measured_order_csv(rows, output_path):
    """用途: 将实际 measure 序参量写为 UTF-8 tidy CSV。

    参数:
    - `rows`: `list[dict[str, object]]`, `collect_measured_order_rows` 的返回值。
    - `output_path`: `Path`, 目标 CSV 文件路径。

    返回:
    - `Path`, 实际写出的 CSV 路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MEASURED_ORDER_CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _plot_measured_order_series(axes, rows, state_name, observable_name, label):
    """用途: 在指定坐标轴上绘制一个实际 measure 序参量的五点曲线。

    参数:
    - `axes`: `Axes`, 目标 matplotlib 坐标轴。
    - `rows`: `list[dict[str, object]]`, tidy measured-order records。
    - `state_name`: `str`, `AFM`、`Stripe4` 或 `Stripe8`。
    - `observable_name`: `str`, measure 序参量字段名。
    - `label`: `str`, 图例标签。

    返回:
    - `Line2D`, 新绘制的数据曲线。
    """
    series_rows = sorted(
        (
            row
            for row in rows
            if row["state"] == state_name and row["observable"] == observable_name
        ),
        key=lambda row: row["tprime_abs"],
    )
    line, = axes.plot(
        [row["tprime_abs"] for row in series_rows],
        [row["value"] for row in series_rows],
        linewidth=1.6,
        markersize=5.5,
        label=label,
        **STATE_PLOT_STYLES[state_name],
    )
    return line


def create_measured_charge_order_figure(rows):
    """用途: 构造两个 Stripe 实际 charge modulation 大小的对比图。

    参数:
    - `rows`: `list[dict[str, object]]`, tidy measured-order records。

    返回:
    - `tuple[Figure, Axes]`, matplotlib figure 和主坐标轴。
    """
    figure, axes = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    _plot_measured_order_series(
        axes, rows, "Stripe4", "charge_modulation", "Stripe4 charge modulation"
    )
    _plot_measured_order_series(
        axes, rows, "Stripe8", "charge_modulation", "Stripe8 charge modulation"
    )
    axes.set_title("Measured charge modulation")
    axes.set_xlabel(r"$|t'|$")
    axes.set_ylabel(r"$M_c$")
    axes.set_xticks(TPRIME_VALUES)
    axes.grid(True, alpha=0.25)
    axes.legend(frameon=False)
    return figure, axes


def create_measured_spin_and_afm_order_figure(rows):
    """用途: 构造两个 Stripe spin modulation 与 AFM order 大小的对比图。

    参数:
    - `rows`: `list[dict[str, object]]`, tidy measured-order records。

    返回:
    - `tuple[Figure, Axes]`, matplotlib figure 和主坐标轴。
    """
    figure, axes = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    _plot_measured_order_series(
        axes, rows, "Stripe4", "spin_modulation", "Stripe4 spin modulation"
    )
    _plot_measured_order_series(
        axes, rows, "Stripe8", "spin_modulation", "Stripe8 spin modulation"
    )
    _plot_measured_order_series(axes, rows, "AFM", "afm_order", "AFM order")
    axes.set_title("Measured spin modulation and AFM order")
    axes.set_xlabel(r"$|t'|$")
    axes.set_ylabel("Measured order magnitude")
    axes.set_xticks(TPRIME_VALUES)
    axes.grid(True, alpha=0.25)
    axes.legend(frameon=False)
    return figure, axes


def save_measured_order_figures(rows, output_directory):
    """用途: 将两张实际 measure 序参量图分别保存为 PNG 和 PDF。

    参数:
    - `rows`: `list[dict[str, object]]`, tidy measured-order records。
    - `output_directory`: `Path`, 图片输出目录。

    返回:
    - `list[Path]`, 两张图对应的 4 个 PNG/PDF 路径。
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_specs = (
        ("measured_order_charge", create_measured_charge_order_figure),
        (
            "measured_order_spin_and_afm",
            create_measured_spin_and_afm_order_figure,
        ),
    )
    output_paths = []
    for filename_stem, figure_factory in figure_specs:
        figure, _ = figure_factory(rows)
        try:
            png_path = output_directory / f"{filename_stem}.png"
            pdf_path = output_directory / f"{filename_stem}.pdf"
            figure.savefig(png_path, dpi=220)
            figure.savefig(pdf_path)
            output_paths.extend((png_path, pdf_path))
        finally:
            plt.close(figure)
    return output_paths


def main(root_directory=None, output_directory=None):
    """用途: 整理真实能量与序参量结果并导出 CSV、PNG 和 PDF。

    参数:
    - `root_directory`: `Path | None`, `tx1ty1` 结果根目录; `None` 时使用仓库默认路径。
    - `output_directory`: `Path | None`, 输出目录; `None` 时写入结果根目录下的
      `energy_component_plots`。

    返回:
    - `dict[str, object]`, 包含 `csv` 路径和 `figures` 路径列表。
    """
    repository_root = Path(__file__).resolve().parents[1]
    if root_directory is None:
        root_directory = (
            repository_root
            / "results"
            / "twist_Hubbard"
            / "Energy_dependence"
            / "tx1ty1"
        )
    root_directory = Path(root_directory)
    if output_directory is None:
        output_directory = root_directory / "energy_component_plots"
    output_directory = Path(output_directory)

    rows = collect_energy_component_rows(root_directory)
    csv_path = write_energy_component_csv(
        rows,
        output_directory / "energy_components_with_se.csv",
    )
    figure_paths = save_all_energy_component_figures(rows, output_directory)
    order_parameter_rows = collect_order_parameter_rows(root_directory)
    order_parameter_csv_path = write_order_parameter_csv(
        order_parameter_rows,
        output_directory / "order_parameters.csv",
    )
    figure_paths.extend(
        save_order_parameter_figures(order_parameter_rows, output_directory)
    )
    measured_order_rows = collect_measured_order_rows(root_directory)
    measured_order_csv_path = write_measured_order_csv(
        measured_order_rows,
        output_directory / "measured_order_parameters.csv",
    )
    figure_paths.extend(save_measured_order_figures(measured_order_rows, output_directory))
    print(f"energy rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"order parameter rows: {len(order_parameter_rows)}")
    print(f"order parameter CSV: {order_parameter_csv_path}")
    print(f"measured order rows: {len(measured_order_rows)}")
    print(f"measured order CSV: {measured_order_csv_path}")
    print(f"figures: {len(figure_paths)}")
    return {
        "csv": csv_path,
        "order_parameter_csv": order_parameter_csv_path,
        "measured_order_csv": measured_order_csv_path,
        "figures": figure_paths,
    }


if __name__ == "__main__":
    main()
