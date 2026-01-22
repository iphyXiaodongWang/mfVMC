#!/usr/bin/env julia

using Printf
using JSON

const DEFAULT_OUTPUT_NAME = "min_params.json"

"""
    parse_header_line(header_line::AbstractString) -> Vector{String}

用途: 解析 sr_history 文件的表头, 返回参数名列表.
参数: header_line, String, 以 "#" 开头的表头行.
返回: Vector{String}, 参数名列表(不包含 Step/E_mean/E_err/GradNorm).
"""
function parse_header_line(header_line::AbstractString)::Vector{String}
    cleaned = replace(strip(header_line), r"^#\s*" => "")
    tokens = split(cleaned, r"\s+")
    if length(tokens) <= 4
        return String[]
    end
    return tokens[5:end]
end

"""
    parse_data_line(line::AbstractString) -> Vector{Float64}

用途: 解析一行数值数据, 返回浮点数组.
参数: line, String, 数据行(以空白分隔).
返回: Vector{Float64}, 当前行的数值序列.
"""
function parse_data_line(line::AbstractString)::Vector{Float64}
    tokens = split(strip(line), r"\s+")
    return parse.(Float64, tokens)
end

"""
    read_sr_history(file_path::AbstractString) -> (param_names::Vector{String}, rows::Vector)

用途: 读取 sr_history 文件, 提取参数名与每步数据.
参数: file_path, String, sr_history 文件路径.
返回: (param_names, rows), param_names 为参数名列表, rows 为包含 step/e_mean/params 的记录数组.
"""
function read_sr_history(file_path::AbstractString)
    param_names = String[]
    rows = Vector{NamedTuple{(:step, :e_mean, :params),Tuple{Float64,Float64,Vector{Float64}}}}()

    for line in eachline(file_path)
        stripped = strip(line)
        if isempty(stripped)
            continue
        end
        if startswith(stripped, "#")
            if isempty(param_names)
                param_names = parse_header_line(stripped)
            end
            continue
        end
        values = parse_data_line(stripped)
        if length(values) < 5
            continue
        end
        step = values[1]
        e_mean = values[2]
        params = values[5:end]
        push!(rows, (step=step, e_mean=e_mean, params=params))
    end

    return param_names, rows
end

"""
    find_min_energy(rows) -> (min_energy::Float64, min_step::Float64, min_params::Vector{Float64})

用途: 从数据记录中找出最小能量及其对应的参数.
参数: rows, 由 read_sr_history 返回的记录数组.
返回: (min_energy, min_step, min_params).
"""
function find_min_energy(rows)
    if isempty(rows)
        error("sr_history 文件中没有可用的数据行.")
    end

    min_row = rows[1]
    for row in rows[2:end]
        if row.e_mean < min_row.e_mean
            min_row = row
        end
    end

    return min_row.e_mean, min_row.step, min_row.params
end

"""
    build_param_dict(param_names::Vector{String}, params::Vector{Float64}) -> Dict{String, Float64}

用途: 将参数名与参数值组合为字典, 便于后续 JSON 输出.
参数: param_names, 参数名列表; params, 参数值列表.
返回: Dict{String, Float64}, 参数名到数值的映射.
"""
function build_param_dict(param_names::Vector{String}, params::Vector{Float64})::Dict{String,Float64}
    if length(param_names) != length(params)
        param_names = ["P_$i" for i in 1:length(params)]
    end

    param_dict = Dict{String,Float64}()
    for (name, value) in zip(param_names, params)
        param_dict[name] = value
    end
    return param_dict
end

"""
    write_json_file(output_path::AbstractString, param_names::Vector{String}, param_dict::Dict{String, Float64}) -> Nothing

用途: 将参数字典写入 JSON 文件(使用 JSON 库).
参数: output_path, 输出路径; param_names, 参数名顺序; param_dict, 参数字典.
返回: Nothing.
"""
function write_json_file(
    output_path::AbstractString,
    param_names::Vector{String},
    param_dict::Dict{String,Float64}
)::Nothing
    open(output_path, "w") do io
        JSON.print(io, param_dict)
        println(io)
    end
    return nothing
end

"""
    print_usage() -> Nothing

用途: 输出脚本用法说明.
参数: 无.
返回: Nothing.
"""
function print_usage()::Nothing
    println("Usage:")
    println("  julia extract_min_energy.jl <sr_history_path> [output_json_path]")
    println("Example:")
    println("  julia extract_min_energy.jl test/logs/sr_defect_history.txt")
    return nothing
end

"""
    main() -> Nothing

用途: 脚本入口, 读取 SR 输出文件并生成最小能量对应的参数 JSON.
参数: 无(使用 ARGS).
返回: Nothing.
"""
function main()::Nothing
    if length(ARGS) < 1
        print_usage()
        return nothing
    end

    input_path = ARGS[1]
    output_path = if length(ARGS) >= 2
        ARGS[2]
    else
        joinpath(dirname(input_path), DEFAULT_OUTPUT_NAME)
    end

    param_names, rows = read_sr_history(input_path)
    min_energy, min_step, min_params = find_min_energy(rows)
    param_dict = build_param_dict(param_names, min_params)
    write_json_file(output_path, param_names, param_dict)

    println("Min energy: $(min_energy)")
    println("Min step: $(min_step)")
    println("Saved params to: $(output_path)")
    return nothing
end

main()
