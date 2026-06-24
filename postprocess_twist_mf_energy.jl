using ArgParse
using JSON
using Printf

const DEFAULT_MF_ENERGY_OUTPUT_FILENAME = "mf_energy.json"

"""
用途: 获取 twist Hubbard MF 能量后处理脚本的默认输出路径。

参数:
- `input_json_path::AbstractString`: measure 输出的 JSON 文件路径, 通常为 `logs/block_binning_mean.json`。

返回:
- `String`: 与输入 JSON 同目录的 `mf_energy.json` 路径。
"""
function default_twist_mf_energy_output_path(input_json_path::AbstractString)::String
    input_directory = dirname(abspath(input_json_path))
    return joinpath(input_directory, DEFAULT_MF_ENERGY_OUTPUT_FILENAME)
end

"""
用途: 从 JSON 字典中读取必须存在的数值字段。

参数:
- `data::Dict`: `JSON.parsefile` 读取到的字典。
- `key::AbstractString`: 需要读取的字段名。

返回:
- `Float64`: 字段对应的数值。
"""
function read_required_float_field(data::AbstractDict, key::AbstractString)::Float64
    haskey(data, key) || error("Missing required field in measure JSON: $(key).")
    value = data[key]
    value isa Number || error("Field $(key) must be numeric, got $(typeof(value)).")
    return Float64(value)
end

"""
用途: 从 twist Hubbard measure JSON 中读取单个格点的平均电荷密度和 Sz。

参数:
- `data::Dict`: `JSON.parsefile` 读取到的字典。
- `x, y::Int`: 从 1 开始的晶格坐标。

返回:
- `NamedTuple`: 包含 `charge_density` 和 `spin_z`, 分别对应 `<n_j>` 和 `<S^z_j>`。
"""
function read_twist_site_density_and_spin(data::AbstractDict, x::Int, y::Int)
    charge_density = read_required_float_field(data, "n_$(x)_$(y)")
    spin_z = read_required_float_field(data, "Sz_$(x)_$(y)")
    return (; charge_density=charge_density, spin_z=spin_z)
end

"""
用途: 根据单格点 `<n_j>` 和 `<S^z_j>` 计算 MF 分解所需的一点密度。

参数:
- `charge_density::Float64`: 单格点平均总密度 `<n_j> = <n_{j up} + n_{j down}>`。
- `spin_z::Float64`: 单格点平均自旋 `<S^z_j> = (<n_{j up}> - <n_{j down}>) / 2`。

返回:
- `NamedTuple`: 包含 `spin_density`, `n_up`, `n_down`。

公式:
- TeX 中定义 `m_j = <n_{j up}> - <n_{j down}>`。
- 当前 measure 输出 `Sz_j = m_j / 2`, 因此 `m_j = 2 * Sz_j`。
- `<n_{j up}> = (n_j + m_j) / 2`, `<n_{j down}> = (n_j - m_j) / 2`。
"""
function compute_site_mean_field_densities(charge_density::Float64, spin_z::Float64)
    spin_density = 2.0 * spin_z
    n_up = 0.5 * (charge_density + spin_density)
    n_down = 0.5 * (charge_density - spin_density)
    return (; spin_density=spin_density, n_up=n_up, n_down=n_down)
end

"""
用途: 从 twist Hubbard measure JSON 计算单个态的 onsite interaction MF 诊断能量。

参数:
- `input_json_path::AbstractString`: `twist_Hubbard.jl --job measure` 生成的 `block_binning_mean.json`。
- `lx, ly::Int`: 晶格尺寸。
- `onsite_u::Float64`: Hubbard onsite 相互作用强度 `U`。

返回:
- `Dict{String, Any}`: 包含 `E_int_true`, `E_U_MF`, `E_U_corr`,
  `E_U_MF_charge`, `E_U_MF_spin` 和一致性检查字段。

公式:
- `E_U_MF = U * sum_j <n_{j up}> * <n_{j down}>`。
- `E_U_MF_charge = U / 4 * sum_j <n_j>^2`。
- `E_U_MF_spin = -U / 4 * sum_j <m_j>^2`。
- `E_U_corr = E_int_true - E_U_MF`。
"""
function compute_twist_mf_energy_from_json(
    input_json_path::AbstractString,
    lx::Int,
    ly::Int,
    onsite_u::Float64,
)::Dict{String,Any}
    lx > 0 || error("Lx must be positive, got $(lx).")
    ly > 0 || error("Ly must be positive, got $(ly).")
    isfile(input_json_path) || error("Input JSON file not found: $(input_json_path).")

    data = JSON.parsefile(input_json_path)
    e_int_true = read_required_float_field(data, "E_int")

    mf_density_product_sum = 0.0
    charge_density_square_sum = 0.0
    spin_density_square_sum = 0.0
    for x in 1:lx, y in 1:ly
        site_observables = read_twist_site_density_and_spin(data, x, y)
        site_densities = compute_site_mean_field_densities(
            site_observables.charge_density,
            site_observables.spin_z,
        )
        mf_density_product_sum += site_densities.n_up * site_densities.n_down
        charge_density_square_sum += site_observables.charge_density^2
        spin_density_square_sum += site_densities.spin_density^2
    end

    e_u_mf = onsite_u * mf_density_product_sum
    e_u_mf_charge = 0.25 * onsite_u * charge_density_square_sum
    e_u_mf_spin = -0.25 * onsite_u * spin_density_square_sum
    e_u_corr = e_int_true - e_u_mf
    consistency_error = e_u_mf_charge + e_u_mf_spin - e_u_mf

    output = Dict{String,Any}(
        "input_json" => abspath(input_json_path),
        "Lx" => lx,
        "Ly" => ly,
        "N_sites" => lx * ly,
        "U" => onsite_u,
        "E_int_true" => e_int_true,
        "E_U_MF" => e_u_mf,
        "E_U_corr" => e_u_corr,
        "E_U_MF_charge" => e_u_mf_charge,
        "E_U_MF_spin" => e_u_mf_spin,
        "E_U_MF_consistency_error" => consistency_error,
    )
    for optional_key in ("E", "E_hop", "count")
        if haskey(data, optional_key)
            output[optional_key] = data[optional_key]
        end
    end
    return output
end

"""
用途: 将 MF 能量后处理结果写入 JSON 文件。

参数:
- `result::Dict{String, Any}`: `compute_twist_mf_energy_from_json` 返回的结果。
- `input_json_path::AbstractString`: 输入 JSON 路径, 用于在未指定输出时推断默认目录。
- `output_path::AbstractString`: 输出路径; 空字符串表示使用输入同目录下的 `mf_energy.json`。

返回:
- `String`: 实际写出的输出文件路径。
"""
function write_twist_mf_energy_output(
    result::Dict{String,Any},
    input_json_path::AbstractString,
    output_path::AbstractString,
)::String
    actual_output_path = isempty(strip(output_path)) ?
                         default_twist_mf_energy_output_path(input_json_path) :
                         abspath(output_path)
    mkpath(dirname(actual_output_path))
    open(actual_output_path, "w") do io
        JSON.print(io, result)
        println(io)
    end
    return actual_output_path
end

"""
用途: 打印 MF 能量后处理结果摘要。

参数:
- `result::Dict{String, Any}`: `compute_twist_mf_energy_from_json` 返回的结果。
- `output_path::AbstractString`: 实际输出 JSON 路径。

返回:
- `nothing`。
"""
function print_twist_mf_energy_summary(result::Dict{String,Any}, output_path::AbstractString)::Nothing
    println("Twist Hubbard MF energy diagnostics")
    println("Input JSON: ", result["input_json"])
    println("Output JSON: ", output_path)
    @printf("E_int_true              %.12f\n", result["E_int_true"])
    @printf("E_U_MF                  %.12f\n", result["E_U_MF"])
    @printf("E_U_corr                %.12f\n", result["E_U_corr"])
    @printf("E_U_MF_charge           %.12f\n", result["E_U_MF_charge"])
    @printf("E_U_MF_spin             %.12f\n", result["E_U_MF_spin"])
    @printf("MF consistency error    %.12e\n", result["E_U_MF_consistency_error"])
    return nothing
end

"""
用途: 解析 twist Hubbard MF 能量后处理脚本的命令行参数。

参数:
- 无。参数来自 Julia 进程的 `ARGS`。

返回:
- `Dict{String, Any}`: `ArgParse.parse_args` 返回的参数字典。
"""
function parse_twist_mf_energy_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--json"
        help = "Input twist_Hubbard.jl measure JSON, usually logs/block_binning_mean.json"
        arg_type = String
        required = true
        "--Lx"
        help = "Lattice size in x direction"
        arg_type = Int
        required = true
        "--Ly"
        help = "Lattice size in y direction"
        arg_type = Int
        required = true
        "--U"
        help = "Hubbard onsite interaction U"
        arg_type = Float64
        required = true
        "--output"
        help = "Output JSON path. Default: mf_energy.json in the input JSON directory"
        arg_type = String
        default = ""
    end
    return parse_args(settings)
end

"""
用途: 运行 twist Hubbard MF 能量后处理命令行流程。

参数:
- 无。所有配置来自命令行参数。

返回:
- `nothing`。
"""
function main_twist_mf_energy_postprocess()::Nothing
    args = parse_twist_mf_energy_commandline()
    result = compute_twist_mf_energy_from_json(args["json"], args["Lx"], args["Ly"], args["U"])
    output_path = write_twist_mf_energy_output(result, args["json"], args["output"])
    print_twist_mf_energy_summary(result, output_path)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_twist_mf_energy_postprocess()
end
