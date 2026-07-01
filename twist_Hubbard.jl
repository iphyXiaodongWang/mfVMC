using MPI
using Printf
using LinearAlgebra
using ArgParse
using JSON

push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using mfVMC

const ACTIVE_TWIST_PROJECTOR_DERIVATIVE_PARAM_NAMES = Ref{Union{Nothing,Vector{Symbol}}}(nothing)
const ACTIVE_TWIST_BACKFLOW_DERIVATIVE_PARAM_NAMES = Ref{Union{Nothing,Vector{Symbol}}}(nothing)

"""
用途: 将二维晶格坐标转换为本项目 Hubbard sampler 使用的一维 site index。

参数:
- `x, y::Int`: 从 1 开始的晶格坐标。
- `ly::Int`: y 方向长度。

返回:
- `Int`: 从 1 开始的 site index, 排列约定为 `site = (x - 1) * ly + y`。
"""
function twist_site_index(x::Int, y::Int, ly::Int)::Int
    return (x - 1) * ly + y
end

"""
用途: 判断当前进程是否为 MPI root rank。

参数:
- 无。内部读取 MPI 初始化状态和 `MPI.COMM_WORLD` rank。

返回:
- `Bool`: 未初始化 MPI 或当前 rank 为 0 时返回 `true`。
"""
function is_twist_root_rank()::Bool
    if !MPI.Initialized()
        return true
    end
    return MPI.Comm_rank(MPI.COMM_WORLD) == 0
end

"""
用途: 构造 twist Hubbard PBC 有限距离 Jastrow 的位移标签列表。

参数:
- `lx, ly::Int`: 二维晶格尺寸。
- `dx_max, dy_max::Int`: x/y 方向最大位移截断。

返回:
- `Vector{Tuple{Int, Int}}`: 按 `(dx, dy)` 排序的位移标签, 不包含 `(0, 0)`。
"""
function build_twist_jastrow_displacement_labels(
    lx::Int,
    ly::Int;
    dx_max::Int=div(lx, 2),
    dy_max::Int=div(ly, 2),
)::Vector{Tuple{Int,Int}}
    if lx <= 0 || ly <= 0
        error("lx and ly must be positive, got lx=$(lx), ly=$(ly).")
    end
    if dx_max < 0 || dy_max < 0
        error("dx_max and dy_max must be non-negative, got dx_max=$(dx_max), dy_max=$(dy_max).")
    end

    labels = Tuple{Int,Int}[]
    for dx in 0:min(dx_max, lx-1)
        for dy in 0:min(dy_max, div(ly, 2))
            if dx == 0 && dy == 0
                continue
            end
            push!(labels, (dx, dy))
        end
    end
    return labels
end

"""
用途: 为 twist Hubbard 有限距离 Jastrow 构造参数名。

参数:
- `dx, dy::Int`: x/y 方向位移。

返回:
- `Symbol`: 形如 `:vj_dx_dy` 的参数名。
"""
function build_twist_jastrow_param_name(dx::Int, dy::Int)::Symbol
    return Symbol("vj_$(dx)_$(dy)")
end

"""
用途: 将 twist Hubbard 二维 PBC 坐标转换成一维 site index。

参数:
- `x, y::Int`: 从 1 开始的格点坐标, 函数内部按 PBC wrap。
- `lx, ly::Int`: 晶格尺寸。

返回:
- `Int`: 从 1 开始的一维 site index。
"""
function twist_pbc_site_index(x::Int, y::Int, lx::Int, ly::Int)::Int
    return twist_site_index(mod(x - 1, lx) + 1, mod(y - 1, ly) + 1, ly)
end

"""
用途: 为给定 displacement 生成 twist Hubbard PBC Jastrow 的唯一无序 pair 集合。

数学公式:
- 对每个格点 `i = (x, y)`, 连接 `j = (x + dx, y ± dy)`。
- x/y 方向均使用 PBC。
- 每个 pair 规范化为 `(min(i,j), max(i,j))`, 因此正反方向共享同一个 `vj_dx_dy` 参数。

参数:
- `lx, ly::Int`: 二维晶格尺寸。
- `dx, dy::Int`: x/y 方向位移, 必须非负。

返回:
- `Vector{Tuple{Int, Int}}`: 去重并排序后的 site pair 列表。
"""
function build_twist_jastrow_pair_set_for_displacement(
    lx::Int,
    ly::Int,
    dx::Int,
    dy::Int,
)::Vector{Tuple{Int,Int}}
    if dx < 0 || dy < 0
        error("dx and dy must be non-negative, got dx=$(dx), dy=$(dy).")
    end

    y_offsets = dy == 0 ? (0,) : (dy, -dy)
    unique_pairs = Set{Tuple{Int,Int}}()
    for x in 1:lx, y in 1:ly
        site_index = twist_pbc_site_index(x, y, lx, ly)
        for offset_y in y_offsets
            neighbor_index = twist_pbc_site_index(x + dx, y + offset_y, lx, ly)
            if neighbor_index == site_index
                continue
            end
            push!(unique_pairs, (min(site_index, neighbor_index), max(site_index, neighbor_index)))
        end
    end
    return sort!(collect(unique_pairs))
end

"""
用途: 为给定 displacement 构造 twist Hubbard PBC Jastrow 对称邻接表。

参数:
- `lx, ly::Int`: 二维晶格尺寸。
- `dx, dy::Int`: x/y 方向位移。

返回:
- `Vector{Vector{Int}}`: 每个 site 的 Jastrow 邻居列表, 无自环且对称。
"""
function build_twist_jastrow_neighbor_table_for_displacement(
    lx::Int,
    ly::Int,
    dx::Int,
    dy::Int,
)::Vector{Vector{Int}}
    unique_pairs = build_twist_jastrow_pair_set_for_displacement(lx, ly, dx, dy)
    neighbor_table = [Int[] for _ in 1:(lx*ly)]
    for (site_i, site_j) in unique_pairs
        push!(neighbor_table[site_i], site_j)
        push!(neighbor_table[site_j], site_i)
    end
    for neighbors in neighbor_table
        sort!(neighbors)
    end
    return neighbor_table
end

"""
用途: 构造 twist Hubbard PBC 有限距离 Jastrow terms。

参数:
- `lx, ly::Int`: 二维晶格尺寸。
- `dx_max, dy_max::Int`: x/y 方向 Jastrow 截断。

返回:
- `Vector{JastrowProjectorTerm{Float64}}`: 初值均为 `0.0` 的有限距离 Jastrow terms。
"""
function build_twist_finite_distance_jastrow_terms(
    lx::Int,
    ly::Int;
    dx_max::Int=div(lx, 2),
    dy_max::Int=div(ly, 2),
)::Vector{JastrowProjectorTerm{Float64}}
    jastrow_terms = JastrowProjectorTerm{Float64}[]
    for (dx, dy) in build_twist_jastrow_displacement_labels(lx, ly; dx_max=dx_max, dy_max=dy_max)
        push!(
            jastrow_terms,
            JastrowProjectorTerm(
                param_name=build_twist_jastrow_param_name(dx, dy),
                v=0.0,
                site_to_neighbor_sites=build_twist_jastrow_neighbor_table_for_displacement(lx, ly, dx, dy),
            ),
        )
    end
    return jastrow_terms
end

"""
用途: 构造 twist Hubbard 使用的有限距离 Jastrow projector。

参数:
- `lx, ly::Int`: 二维晶格尺寸。
- `g::Float64`: Gutzwiller projector 参数。
- `jastrow_dx_max, jastrow_dy_max::Int`: Jastrow 位移截断范围。

返回:
- `CompositeProjector`: 包含 Gutzwiller 与 PBC 有限距离 density Jastrow terms 的 projector。
"""
function build_twist_projector(
    lx::Int,
    ly::Int,
    g::Float64;
    jastrow_dx_max::Int=div(lx, 2),
    jastrow_dy_max::Int=div(ly, 2),
)::CompositeProjector
    projector_terms = AbstractProjectorTerm[
        GutzwillerProjectorTerm(param_name=:g, g=g),
    ]
    append!(
        projector_terms,
        build_twist_finite_distance_jastrow_terms(
            lx,
            ly;
            dx_max=jastrow_dx_max,
            dy_max=jastrow_dy_max,
        ),
    )
    return CompositeProjector(projector_terms)
end

"""
用途: 解析固定参数字符串。

参数:
- `fixed_params_string::AbstractString`: 形如 `"Delta_AF=1.0,g=0.5"` 的逗号分隔字符串。

返回:
- `Dict{Symbol, Float64}`: 参数名到固定值的映射。
"""
function parse_twist_fixed_param_string(fixed_params_string::AbstractString)::Dict{Symbol,Float64}
    fixed_param_values = Dict{Symbol,Float64}()
    stripped_string = strip(fixed_params_string)
    if isempty(stripped_string)
        return fixed_param_values
    end

    for raw_assignment in split(stripped_string, ",")
        assignment = strip(raw_assignment)
        parts = split(assignment, "=")
        if length(parts) != 2
            error("Invalid fixed parameter assignment: $(assignment). Expected name=value.")
        end
        param_name = Symbol(strip(parts[1]))
        param_value = parse(Float64, strip(parts[2]))
        if haskey(fixed_param_values, param_name)
            error("Duplicate fixed parameter: $(param_name).")
        end
        fixed_param_values[param_name] = param_value
    end
    return fixed_param_values
end

"""
用途: 解析 active 参数名字符串。

参数:
- `param_names_string::AbstractString`: 形如 `"Delta_AF,g"` 的逗号分隔字符串。

返回:
- `Vector{Symbol}`: active 参数名列表, 空字符串返回空列表。
"""
function parse_twist_param_name_list(param_names_string::AbstractString)::Vector{Symbol}
    stripped_string = strip(param_names_string)
    if isempty(stripped_string)
        return Symbol[]
    end
    param_names = [Symbol(strip(raw_name)) for raw_name in split(stripped_string, ",")]
    if any(name -> isempty(String(name)), param_names)
        error("Active parameter list contains an empty parameter name.")
    end
    if length(unique(param_names)) != length(param_names)
        error("Active parameter list contains duplicate names: $(param_names).")
    end
    return param_names
end

"""
用途: 解析 twist Hubbard 命令行中的布尔字符串。

参数:
- `raw_value::AbstractString`: 支持 `true/false`, `1/0`, `yes/no`, `on/off`。
- `option_name::AbstractString`: 参数名, 用于错误信息。

返回:
- `Bool`: 解析后的布尔值。
"""
function parse_twist_bool_flag(raw_value::AbstractString, option_name::AbstractString)::Bool
    normalized_value = lowercase(strip(raw_value))
    if normalized_value in ("true", "t", "1", "yes", "y", "on")
        return true
    elseif normalized_value in ("false", "f", "0", "no", "n", "off")
        return false
    end
    error("Invalid value for $(option_name): $(raw_value).")
end

"""
用途: 从 JSON 文件读取 twist Hubbard 初始参数, 缺失参数使用当前默认值。

参数:
- `json_path::AbstractString`: JSON 文件路径。
- `param_names::Vector{Symbol}`: 当前参数名顺序。
- `default_params::Vector{Float64}`: 默认参数值。

返回:
- `Vector{Float64}`: 按 `param_names` 顺序排列的初始参数。
"""
function build_twist_init_params_from_json_with_defaults(
    json_path::AbstractString,
    param_names::Vector{Symbol},
    default_params::Vector{Float64},
)::Vector{Float64}
    isfile(json_path) || error("JSON file not found: $(json_path)")
    length(param_names) == length(default_params) ||
        error("param_names and default_params length mismatch.")

    raw_dict = JSON.parsefile(json_path)
    init_params = Float64[]
    for (param_index, param_name) in enumerate(param_names)
        key = String(param_name)
        if haskey(raw_dict, key)
            push!(init_params, Float64(raw_dict[key]))
        else
            push!(init_params, default_params[param_index])
        end
    end
    return init_params
end

"""
用途: 将 fixed 参数写入参数向量。

参数:
- `param_names::Vector{Symbol}`: 完整参数名列表。
- `init_params::Vector{Float64}`: 当前参数值。
- `fixed_param_values::Dict{Symbol, Float64}`: fixed 参数映射。

返回:
- `Vector{Float64}`: 应用 fixed 值后的参数向量。
"""
function apply_twist_fixed_params_to_values(
    param_names::Vector{Symbol},
    init_params::Vector{Float64},
    fixed_param_values::Dict{Symbol,Float64},
)::Vector{Float64}
    length(param_names) == length(init_params) ||
        error("param_names and init_params length mismatch.")
    param_index_map = Dict(name => index for (index, name) in enumerate(param_names))
    updated_params = copy(init_params)
    for (param_name, param_value) in fixed_param_values
        haskey(param_index_map, param_name) ||
            error("Unknown fixed parameter $(param_name). Available parameters: $(join(String.(param_names), ", ")).")
        updated_params[param_index_map[param_name]] = param_value
    end
    return updated_params
end

"""
用途: 根据 fixed/active 设置生成 SR 实际优化参数下标。

参数:
- `param_names::Vector{Symbol}`: 完整参数名列表。
- `fixed_param_values::Dict{Symbol, Float64}`: fixed 参数映射。
- `requested_active_param_names::Vector{Symbol}`: 用户显式指定的 active 参数名。

返回:
- `Vector{Int}`: 参与 SR 优化的参数下标。
"""
function build_twist_active_param_indices(
    param_names::Vector{Symbol},
    fixed_param_values::Dict{Symbol,Float64},
    requested_active_param_names::Vector{Symbol}=Symbol[],
)::Vector{Int}
    param_index_map = Dict(name => index for (index, name) in enumerate(param_names))
    for param_name in keys(fixed_param_values)
        haskey(param_index_map, param_name) ||
            error("Unknown fixed parameter $(param_name).")
    end

    if isempty(requested_active_param_names)
        return [
            index for (index, param_name) in enumerate(param_names)
                      if !haskey(fixed_param_values, param_name)
        ]
    end

    active_indices = Int[]
    for param_name in requested_active_param_names
        haskey(param_index_map, param_name) ||
            error("Unknown active parameter $(param_name).")
        if haskey(fixed_param_values, param_name)
            error("Parameter $(param_name) cannot be both fixed and active.")
        end
        push!(active_indices, param_index_map[param_name])
    end
    return active_indices
end

"""
用途: 将 SR active 参数合并回完整参数向量。

参数:
- `full_param_template::Vector{Float64}`: 完整参数模板。
- `active_param_indices::Vector{Int}`: active 参数在完整向量中的下标。
- `active_param_values::Vector{Float64}`: SR 当前 active 参数值。

返回:
- `Vector{Float64}`: 合并后的完整参数向量。
"""
function merge_twist_active_params_into_full(
    full_param_template::Vector{Float64},
    active_param_indices::Vector{Int},
    active_param_values::Vector{Float64},
)::Vector{Float64}
    length(active_param_indices) == length(active_param_values) ||
        error("active_param_indices and active_param_values length mismatch.")
    full_param_values = copy(full_param_template)
    for (active_offset, param_index) in enumerate(active_param_indices)
        full_param_values[param_index] = active_param_values[active_offset]
    end
    return full_param_values
end

"""
用途: 将未参与 SR 优化的参数补写进最优参数 JSON。

参数:
- `json_path::AbstractString`: `extract_min_energy` 生成的 JSON 文件路径。
- `full_param_names::Vector{Symbol}`: 完整参数名列表。
- `full_param_values::Vector{Float64}`: 完整参数值模板。
- `active_param_indices::Vector{Int}`: 参与 SR 优化的参数下标。

返回:
- `nothing`。
"""
function append_twist_inactive_params_to_json!(
    json_path::AbstractString,
    full_param_names::Vector{Symbol},
    full_param_values::Vector{Float64},
    active_param_indices::Vector{Int},
)::Nothing
    if !isfile(json_path)
        return nothing
    end
    param_dict = JSON.parsefile(json_path)
    active_param_index_set = Set(active_param_indices)
    for (param_index, param_name) in enumerate(full_param_names)
        if !(param_index in active_param_index_set)
            param_dict[String(param_name)] = full_param_values[param_index]
        end
    end
    open(json_path, "w") do io
        JSON.print(io, param_dict)
        println(io)
    end
    return nothing
end

"""
用途: 校验 active projector 参数名是否合法。

参数:
- `available_param_names::Vector{Symbol}`: projector 的完整参数名列表。
- `active_param_names::Union{Nothing, Vector{Symbol}}`: active 参数名; `nothing` 表示全部。

返回:
- `nothing`。若参数名重复或不存在会抛出异常。
"""
function validate_twist_active_projector_param_names!(
    available_param_names::Vector{Symbol},
    active_param_names::Union{Nothing,Vector{Symbol}},
)::Nothing
    if active_param_names === nothing
        return nothing
    end
    if length(unique(active_param_names)) != length(active_param_names)
        error("Duplicate active projector derivative parameters: $(active_param_names).")
    end
    available_param_name_set = Set(available_param_names)
    for active_param_name in active_param_names
        if !(active_param_name in available_param_name_set)
            error("Unknown active projector derivative parameter $(active_param_name).")
        end
    end
    return nothing
end

"""
用途: 设置当前 SR 中 projector 哪些参数参与导数计算。

参数:
- `projector_param_names::Vector{Symbol}`: 完整 projector 参数名。
- `active_projector_param_names::Union{Nothing, Vector{Symbol}}`: active projector 参数名; `nothing` 表示全部。

返回:
- `nothing`。该设置只影响本脚本中重定义的 `compute_grad_log_psi!`。
"""
function set_active_twist_projector_derivative_param_names!(
    projector_param_names::Vector{Symbol};
    active_projector_param_names::Union{Nothing,Vector{Symbol}}=nothing,
)::Nothing
    validate_twist_active_projector_param_names!(
        projector_param_names,
        active_projector_param_names,
    )
    ACTIVE_TWIST_PROJECTOR_DERIVATIVE_PARAM_NAMES[] =
        active_projector_param_names === nothing ? nothing : copy(active_projector_param_names)
    return nothing
end

"""
用途: 校验 active backflow 参数名是否合法。

参数:
- `available_param_names::Vector{Symbol}`: backflow 的完整参数名列表。
- `active_param_names::Union{Nothing, Vector{Symbol}}`: active 参数名; `nothing` 表示全部。

返回:
- `nothing`。若参数名重复或不存在会抛出异常。
"""
function validate_twist_active_backflow_param_names!(
    available_param_names::Vector{Symbol},
    active_param_names::Union{Nothing,Vector{Symbol}},
)::Nothing
    if active_param_names === nothing
        return nothing
    end
    if length(unique(active_param_names)) != length(active_param_names)
        error("Duplicate active backflow derivative parameters: $(active_param_names).")
    end
    available_param_name_set = Set(available_param_names)
    for active_param_name in active_param_names
        if !(active_param_name in available_param_name_set)
            error("Unknown active backflow derivative parameter $(active_param_name).")
        end
    end
    return nothing
end

"""
用途: 设置当前 SR 中 backflow 哪些参数参与导数计算。

参数:
- `backflow_param_names::Vector{Symbol}`: 完整 backflow 参数名。
- `active_backflow_param_names::Union{Nothing, Vector{Symbol}}`: active backflow 参数名; `nothing` 表示全部。

返回:
- `nothing`。该设置只影响本脚本中重定义的 `compute_grad_log_psi!`。
"""
function set_active_twist_backflow_derivative_param_names!(
    backflow_param_names::Vector{Symbol};
    active_backflow_param_names::Union{Nothing,Vector{Symbol}}=nothing,
)::Nothing
    validate_twist_active_backflow_param_names!(
        backflow_param_names,
        active_backflow_param_names,
    )
    ACTIVE_TWIST_BACKFLOW_DERIVATIVE_PARAM_NAMES[] =
        active_backflow_param_names === nothing ? nothing : copy(active_backflow_param_names)
    return nothing
end

"""
用途: 在 twist Hubbard 中覆盖 SR 的 log-derivative 计算, 支持只优化部分 projector 参数。

数学公式:
- 对 determinant 参数使用 `O_p = Tr(A^{-1} dA/dp)`。
- 对 projector 参数使用 `O_p = d log(P) / dp`, 但只保留 active projector 参数。
- 对 backflow 参数使用 `O_p = Tr(A^{-1} dA_b/dp)`, 并只保留 active backflow 参数。

参数:
- `vwf::mfVMC.VMC.vwf_det{T}`: determinant 波函数对象。

返回:
- `Vector{T}`: 与当前 SR active 参数顺序一致的 log-derivative 向量。
"""
function mfVMC.VMC.compute_grad_log_psi!(vwf::mfVMC.VMC.vwf_det{T}) where T
    ws = mfVMC.VMC.ensure_ws!(vwf)
    sampler = vwf.sampler
    a_inv = vwf.awf_inv

    wf_param_count = length(vwf.param_keys)
    projector_param_names_all = mfVMC.Projector.projector_param_names(vwf.projector)
    backflow_param_names_all = mfVMC.Backflow.backflow_param_names(vwf.backflow)
    active_projector_param_names =
        ACTIVE_TWIST_PROJECTOR_DERIVATIVE_PARAM_NAMES[] === nothing ?
        projector_param_names_all :
        ACTIVE_TWIST_PROJECTOR_DERIVATIVE_PARAM_NAMES[]
    active_backflow_param_names =
        ACTIVE_TWIST_BACKFLOW_DERIVATIVE_PARAM_NAMES[] === nothing ?
        backflow_param_names_all :
        ACTIVE_TWIST_BACKFLOW_DERIVATIVE_PARAM_NAMES[]

    resize!(
        ws.grad_buffer,
        wf_param_count + length(active_projector_param_names) + length(active_backflow_param_names),
    )
    o_vec = ws.grad_buffer
    fill!(o_vec, zero(T))

    mean_field_view = @view o_vec[1:wf_param_count]
    if mfVMC.Backflow.uses_backflow(vwf.backflow)
        mfVMC.VMC._compute_backflow_meanfield_gradient_from_selected_rows!(
            mean_field_view,
            ws.backflow_chain_rule_source_rows,
            ws.backflow_chain_rule_source_weights,
            a_inv,
            sampler.electron_locs,
            vwf.dUt_matrix,
            sampler.state,
            vwf.backflow,
        )
    else
        mfVMC.VMC._compute_dense_tensor_gradient!(mean_field_view, vwf)
    end

    if !isempty(active_projector_param_names)
        full_projector_derivatives = mfVMC.Projector.projector_log_derivative(vwf.projector, sampler)
        projector_derivative_map = Dict{Symbol,Float64}()
        for (param_name, derivative_value) in zip(projector_param_names_all, full_projector_derivatives)
            projector_derivative_map[param_name] = Float64(derivative_value)
        end
        for (active_offset, param_name) in enumerate(active_projector_param_names)
            o_vec[wf_param_count+active_offset] = T(projector_derivative_map[param_name])
        end
    end
    if !isempty(active_backflow_param_names)
        backflow_pairs = mfVMC.Backflow.build_backflow_derivative_orbitals(
            vwf.base_gs_U,
            sampler.state,
            vwf.backflow,
        )
        backflow_derivative_map = Dict{Symbol,T}()
        for (param_name, derivative_orbitals) in backflow_pairs
            backflow_derivative_map[param_name] =
                mfVMC.VMC._compute_orbital_log_derivative_from_selected_rows!(
                    ws.orbital_log_derivative_row_buffer,
                    a_inv,
                    sampler.electron_locs,
                    derivative_orbitals,
                )
        end
        backflow_offset = wf_param_count + length(active_projector_param_names)
        for (active_offset, param_name) in enumerate(active_backflow_param_names)
            haskey(backflow_derivative_map, param_name) ||
                error("Missing backflow derivative for active parameter $(param_name).")
            o_vec[backflow_offset+active_offset] = backflow_derivative_map[param_name]
        end
    end
    return o_vec
end

"""
用途: 构造指数衰减学习率函数。

参数:
- `lr_start::Float64`: 初始学习率。
- `lr_end::Float64`: 最后一步目标学习率。
- `n_steps::Int`: SR 总步数。

返回:
- `Function`: 形式为 `(lr0, step) -> lr` 的学习率函数。
"""
function build_twist_exponential_lr_func(
    lr_start::Float64,
    lr_end::Float64,
    n_steps::Int,
)::Function
    if n_steps <= 1
        return (lr0, step) -> lr_end
    end
    if lr_start == 0.0
        return (lr0, step) -> 0.0
    end
    if lr_start < 0.0 || lr_end < 0.0
        error("lr and lr_end must be non-negative.")
    end
    lr_decay_gamma = (lr_end / lr_start)^(1.0 / (n_steps - 1))
    return (lr0, step) -> lr0 * (lr_decay_gamma^(step - 1))
end

"""
用途: 构造 twist Hubbard 使用的 SR 参数对象。

参数:
- `meas_params::VMCParams`: SR 每一步内部使用的 VMC 采样参数。
- `n_steps::Int`: SR 总步数。
- `lr::Float64`: SR 学习率。
- `eigen_cutoff::Float64`: SR 矩阵特征值截断阈值, `0.0` 表示不做额外截断。

返回:
- `SRParams`: 已显式设置 `eigen_cutoff` 的 SR 参数对象。
"""
function build_twist_sr_params(
    meas_params::VMCParams,
    n_steps::Int,
    lr::Float64,
    eigen_cutoff::Float64,
)::SRParams
    return SRParams(
        vmc_params=meas_params,
        n_steps=n_steps,
        lr=lr,
        eigen_cutoff=eigen_cutoff,
    )
end

"""
用途: 构造 twist Hubbard physical Hamiltonian 的 hopping 和 interaction 项。

参数:
- `lx, ly::Int`: 晶格尺寸。
- `tx, ty::Float64`: x/y 方向最近邻 hopping。
- `t2::Float64`: 平移不破缺的对角次近邻 hopping。
- `onsite_u::Float64`: onsite Hubbard 相互作用强度。

返回:
- `NamedTuple`: 包含 `hopping_terms`, `tx_hopping_terms`, `ty_hopping_terms`,
  `t2_hopping_terms`, `interaction_terms`, `all_terms`。

数学公式:
- `H_hop = -sum_<ij>,sigma t_ij (c^dag_{i,sigma} c_{j,sigma} + h.c.)`。
- `H_int = U * sum_i n_{i,up} n_{i,dn}`。
"""
function build_twist_hamiltonian_terms(
    lx::Int,
    ly::Int,
    tx::Float64,
    ty::Float64,
    t2::Float64,
    onsite_u::Float64,
)
    n_sites = lx * ly
    hopping_bonds = build_twist_nearest_neighbor_bonds(lx, ly)
    tx_hopping_terms = OperatorTerm[]
    ty_hopping_terms = OperatorTerm[]
    t2_hopping_terms = OperatorTerm[]
    interaction_terms = OperatorTerm[]

    for (site_i, site_j) in hopping_bonds.x_bonds
        push!(tx_hopping_terms, OperatorTerm([:cdag_up, :c_up], [site_i, site_j], -tx))
        push!(tx_hopping_terms, OperatorTerm([:cdag_up, :c_up], [site_j, site_i], -tx))
        push!(tx_hopping_terms, OperatorTerm([:cdag_dn, :c_dn], [site_i, site_j], -tx))
        push!(tx_hopping_terms, OperatorTerm([:cdag_dn, :c_dn], [site_j, site_i], -tx))
    end
    for (site_i, site_j) in hopping_bonds.y_bonds
        push!(ty_hopping_terms, OperatorTerm([:cdag_up, :c_up], [site_i, site_j], -ty))
        push!(ty_hopping_terms, OperatorTerm([:cdag_up, :c_up], [site_j, site_i], -ty))
        push!(ty_hopping_terms, OperatorTerm([:cdag_dn, :c_dn], [site_i, site_j], -ty))
        push!(ty_hopping_terms, OperatorTerm([:cdag_dn, :c_dn], [site_j, site_i], -ty))
    end
    for (site_i, site_j) in hopping_bonds.diagonal_bonds
        push!(t2_hopping_terms, OperatorTerm([:cdag_up, :c_up], [site_i, site_j], -t2))
        push!(t2_hopping_terms, OperatorTerm([:cdag_up, :c_up], [site_j, site_i], -t2))
        push!(t2_hopping_terms, OperatorTerm([:cdag_dn, :c_dn], [site_i, site_j], -t2))
        push!(t2_hopping_terms, OperatorTerm([:cdag_dn, :c_dn], [site_j, site_i], -t2))
    end
    for site_i in 1:n_sites
        push!(interaction_terms, OperatorTerm([:n_up, :n_dn], [site_i, site_i], onsite_u))
    end

    hopping_terms = vcat(tx_hopping_terms, ty_hopping_terms, t2_hopping_terms)
    all_terms = vcat(hopping_terms, interaction_terms)
    return (;
        hopping_terms=hopping_terms,
        tx_hopping_terms=tx_hopping_terms,
        ty_hopping_terms=ty_hopping_terms,
        t2_hopping_terms=t2_hopping_terms,
        interaction_terms=interaction_terms,
        all_terms=all_terms,
    )
end

"""
用途: 计算一组 OperatorTerm 的局域能量之和。

参数:
- `terms::Vector{OperatorTerm}`: 待求和的 Hamiltonian 项。
- `model`: 当前 VMC model, 为了匹配 observable 函数签名保留, 本函数内部不使用。
- `vwf`: determinant 波函数对象。

返回:
- `Float64`: `sum_t E_t(C)` 的当前构型局域能量估计。
"""
function measure_twist_term_energy_sum(
    terms::Vector{OperatorTerm},
    model,
    vwf,
)::Float64
    energy = 0.0
    for term in terms
        energy += mfVMC.Model.compute_term_energy(term, vwf)
    end
    return energy
end

"""
用途: 计算 twist Hubbard 当前 VMC 构型上的 onsite interaction charge/spin 势能分解。

参数:
- `onsite_u::Float64`: Hubbard onsite 相互作用强度 `U`。
- `vwf`: determinant 波函数对象, 需要提供 `vwf.sampler.state`。

返回:
- `NamedTuple`: 包含
  - `charge::Float64`: `E_int_charge = U / 4 * sum_j n_j^2`。
  - `spin::Float64`: `E_int_spin = -U / 4 * sum_j m_j^2`。

数学公式:
- `n_j = n_{j up} + n_{j down}`。
- `m_j = n_{j up} - n_{j down}`。
- `n_{j up} n_{j down} = (n_j^2 - m_j^2) / 4`。
"""
function measure_twist_interaction_charge_spin_energy(
    onsite_u::Float64,
    vwf,
)
    charge_density_square_sum = 0.0
    spin_density_square_sum = 0.0

    for site_state in vwf.sampler.state
        n_up = (site_state & UP) != 0 ? 1.0 : 0.0
        n_down = (site_state & DN) != 0 ? 1.0 : 0.0
        charge_density = n_up + n_down
        spin_density = n_up - n_down
        charge_density_square_sum += charge_density^2
        spin_density_square_sum += spin_density^2
    end

    return (
        charge=0.25 * onsite_u * charge_density_square_sum,
        spin=-0.25 * onsite_u * spin_density_square_sum,
    )
end

"""
用途: 构造 twist Hubbard measure 使用的 observables。

参数:
- `lx, ly::Int`: 晶格尺寸。
- `hopping_terms::Vector{OperatorTerm}`: hopping Hamiltonian 项, 非空时加入 `:E_hop`。
- `tx_hopping_terms::Vector{OperatorTerm}`: x 方向最近邻 hopping 项, 非空时加入 `:E_hop_tx`。
- `ty_hopping_terms::Vector{OperatorTerm}`: y 方向最近邻 hopping 项, 非空时加入 `:E_hop_ty`。
- `t2_hopping_terms::Vector{OperatorTerm}`: 对角次近邻 hopping 项, 非空时加入 `:E_hop_t2`。
- `interaction_terms::Vector{OperatorTerm}`: interaction Hamiltonian 项, 非空时加入 `:E_int`。
- `onsite_u::Float64`: Hubbard onsite 相互作用强度, 用于加入 `:E_int_charge` 和 `:E_int_spin`。

返回:
- `Dict{Symbol, Function}`: 包含总能量, 可选分项能量, 每个 site 的 `Sz` 和密度 `n`。
"""
function definition_twist_observables(
    lx::Int,
    ly::Int;
    hopping_terms::Vector{OperatorTerm}=OperatorTerm[],
    tx_hopping_terms::Vector{OperatorTerm}=OperatorTerm[],
    ty_hopping_terms::Vector{OperatorTerm}=OperatorTerm[],
    t2_hopping_terms::Vector{OperatorTerm}=OperatorTerm[],
    interaction_terms::Vector{OperatorTerm}=OperatorTerm[],
    onsite_u::Float64=0.0,
)::Dict{Symbol,Function}
    observables = Dict{Symbol,Function}()
    observables[:E] = local_energy
    if !isempty(hopping_terms)
        hopping_terms_local = copy(hopping_terms)
        observables[:E_hop] = (model, vwf) -> measure_twist_term_energy_sum(
            hopping_terms_local,
            model,
            vwf,
        )
    end
    if !isempty(tx_hopping_terms)
        tx_hopping_terms_local = copy(tx_hopping_terms)
        observables[:E_hop_tx] = (model, vwf) -> measure_twist_term_energy_sum(
            tx_hopping_terms_local,
            model,
            vwf,
        )
    end
    if !isempty(ty_hopping_terms)
        ty_hopping_terms_local = copy(ty_hopping_terms)
        observables[:E_hop_ty] = (model, vwf) -> measure_twist_term_energy_sum(
            ty_hopping_terms_local,
            model,
            vwf,
        )
    end
    if !isempty(t2_hopping_terms)
        t2_hopping_terms_local = copy(t2_hopping_terms)
        observables[:E_hop_t2] = (model, vwf) -> measure_twist_term_energy_sum(
            t2_hopping_terms_local,
            model,
            vwf,
        )
    end
    if !isempty(interaction_terms)
        interaction_terms_local = copy(interaction_terms)
        observables[:E_int] = (model, vwf) -> measure_twist_term_energy_sum(
            interaction_terms_local,
            model,
            vwf,
        )
        observables[:E_int_charge] = (model, vwf) -> measure_twist_interaction_charge_spin_energy(
            onsite_u,
            vwf,
        ).charge
        observables[:E_int_spin] = (model, vwf) -> measure_twist_interaction_charge_spin_energy(
            onsite_u,
            vwf,
        ).spin
    end
    for x in 1:lx, y in 1:ly
        site = twist_site_index(x, y, ly)
        spin_key = Symbol("Sz_$(x)_$(y)")
        density_key = Symbol("n_$(x)_$(y)")
        observables[spin_key] = (model, vwf) -> get_Sz(vwf.sampler.state[site])
        observables[density_key] = (model, vwf) -> begin
            site_state = vwf.sampler.state[site]
            n_up = (site_state & UP) != 0 ? 1.0 : 0.0
            n_down = (site_state & DN) != 0 ? 1.0 : 0.0
            return n_up + n_down
        end
    end
    return observables
end

"""
用途: 解析 twist Hubbard 主程序的命令行参数。

参数:
- 无。参数来自 Julia 进程的 `ARGS`。

返回:
- `Dict{String, Any}`: `ArgParse.parse_args` 返回的参数字典。

说明:
- physical hopping 使用最近邻 `tx/ty` 和平移不破缺的对角次近邻 `t2`。
- mean-field ansatz 固定 `chi1x = 1`, 并优化 `chi1y` 与 `chi2`。
- 不包含 pairing 和 backflow 参数。
"""
function parse_twist_commandline()
    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--Lx"
        help = "Lattice size in X direction"
        arg_type = Int
        default = 8
        "--Ly"
        help = "Lattice size in Y direction"
        arg_type = Int
        default = 3
        "--tx"
        help = "Nearest-neighbor hopping amplitude in X direction"
        arg_type = Float64
        default = 1.0
        "--ty"
        help = "Nearest-neighbor hopping amplitude in Y direction"
        arg_type = Float64
        default = 1.0
        "--t2"
        help = "Translationally invariant next-nearest-neighbor diagonal hopping amplitude"
        arg_type = Float64
        default = 0.0
        "--U"
        help = "On-site interaction strength"
        arg_type = Float64
        default = 8.0
        "--bcx"
        help = "Mean-field boundary condition phase in X direction"
        arg_type = Float64
        default = 1.001
        "--bcy"
        help = "Mean-field boundary condition phase in Y direction"
        arg_type = Float64
        default = 0.999
        "--Delta_AF"
        help = "AFM order parameter"
        arg_type = Float64
        default = 3.0
        "--Delta_c"
        help = "Charge stripe order parameter"
        arg_type = Float64
        default = 3.0
        "--Delta_s"
        help = "Spin stripe order parameter"
        arg_type = Float64
        default = 3.0
        "--target_sz"
        help = "Target total Sz, represented as N_up - N_down"
        arg_type = Int
        default = 0
        "--nMC"
        help = "Number of Monte Carlo total samples"
        arg_type = Int
        default = 10000
        "--wMC"
        help = "Number of Monte Carlo warming up steps"
        arg_type = Int
        default = 100
        "--rMC"
        help = "Number of rebuild interval"
        arg_type = Int
        default = 100
        "--dMC"
        help = "Number of Monte Carlo decorrelation sweeps"
        arg_type = Int
        default = 1
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 5423
        "--nSR"
        help = "Total steps for SR"
        arg_type = Int
        default = 50
        "--lr"
        help = "SR learn rate"
        arg_type = Float64
        default = 0.04
        "--lr_end"
        help = "Target learning rate at the last SR step. Default follows --lr"
        arg_type = Float64
        default = NaN
        "--eigen_cutoff"
        help = "SR eigenvalue truncation cutoff. Keep lambda_i / lambda_max >= eigen_cutoff. 0 means no truncation."
        arg_type = Float64
        default = 0.0
        "--init_params_json"
        help = "Path to json file that provides initial parameters"
        arg_type = String
        default = ""
        "--fixed_params"
        help = "Comma-separated fixed parameter assignments, e.g. 'Delta_AF=1.0'"
        arg_type = String
        default = ""
        "--active_params"
        help = "Comma-separated parameter names optimized by SR. Empty means all non-fixed parameters."
        arg_type = String
        default = ""
        "--job"
        help = "Job to be done. Can be SR and measure"
        arg_type = String
        default = "SR"
        "--doping"
        help = "Doping level. This follows Hubbard_restricted.jl: N_e = N_sites * (1 + doping)."
        arg_type = Float64
        default = 0.125
        "--ansatz"
        help = "Ansatz type, can be 'AFM' or 'Stripe'"
        arg_type = String
        default = "Stripe"
        "--lambda"
        help = "Assumed stripe length"
        arg_type = Int
        default = 4
        "--stripe_center"
        help = "Stripe center type, can be 'site' or 'bond'"
        arg_type = String
        default = "site"
        "--g"
        help = "Gutzwiller projector parameter"
        arg_type = Float64
        default = 1.0
        "--enable_backflow"
        help = "Enable twist Hubbard nonPH backflow"
        arg_type = String
        default = "true"
        "--bf_epsilon"
        help = "Backflow epsilon initial value"
        arg_type = Float64
        default = 1.0
        "--bf_eta1"
        help = "Backflow eta1 initial value"
        arg_type = Float64
        default = 0.0
        "--bf_eta2"
        help = "Backflow eta2 initial value"
        arg_type = Float64
        default = 0.0
        "--bf_eta3"
        help = "Backflow eta3 initial value"
        arg_type = Float64
        default = 0.0
        "--bf_eta4"
        help = "Backflow eta4 initial value"
        arg_type = Float64
        default = 0.0
        "--jastrow_dx_max"
        help = "Maximum x displacement for finite-distance Jastrow projector"
        arg_type = Int
        default = 2
        "--jastrow_dy_max"
        help = "Maximum y displacement for finite-distance Jastrow projector"
        arg_type = Int
        default = 2
    end

    return parse_args(settings)
end

"""
用途: 保存 twist Hubbard nonPH mean-field ansatz 的参数。

字段:
- `lx, ly::Int`: 晶格尺寸。
- `bcx, bcy::Float64`: mean-field 边界条件因子。
- `chi_x, chi_y::Float64`: x/y 方向最近邻 hopping。
- `chi2::Float64`: 平移不破缺的对角次近邻 hopping。
- `delta_af::Float64`: AF staggered field 振幅。
- `delta_c, delta_s::Float64`: charge/spin stripe field 振幅。
- `stripe_wavevector::Float64`: stripe 波矢 `Q`。
- `stripe_center_offset::Float64`: stripe 中心偏移 `x0`。
"""
struct TwistHubbardNonPHParams
    lx::Int
    ly::Int
    bcx::Float64
    bcy::Float64
    chi_x::Float64
    chi_y::Float64
    chi2::Float64
    delta_af::Float64
    delta_c::Float64
    delta_s::Float64
    stripe_wavevector::Float64
    stripe_center_offset::Float64
end

"""
用途: 构造 `TwistHubbardNonPHParams` 参数对象。

参数:
- `lx, ly::Int`: 晶格尺寸。
- `bcx, bcy::Float64`: mean-field 边界条件因子。
- `chi_x, chi_y::Float64`: x/y 方向最近邻 hopping。
- `chi2::Float64`: 对角次近邻 hopping。
- `delta_af, delta_c, delta_s::Float64`: AF/charge stripe/spin stripe 场。
- `stripe_wavevector::Float64`: stripe 波矢 `Q`。
- `stripe_center_offset::Float64`: stripe 中心偏移 `x0`。

返回:
- `TwistHubbardNonPHParams`: twist Hubbard nonPH mean-field 参数对象。
"""
function TwistHubbardNonPHParams(;
    lx::Int,
    ly::Int,
    bcx::Float64=1.0,
    bcy::Float64=1.0,
    chi_x::Float64=1.0,
    chi_y::Float64=1.0,
    chi2::Float64=0.0,
    delta_af::Float64=0.0,
    delta_c::Float64=0.0,
    delta_s::Float64=0.0,
    stripe_wavevector::Float64=0.0,
    stripe_center_offset::Float64=0.0,
)
    return TwistHubbardNonPHParams(
        lx,
        ly,
        bcx,
        bcy,
        chi_x,
        chi_y,
        chi2,
        delta_af,
        delta_c,
        delta_s,
        stripe_wavevector,
        stripe_center_offset,
    )
end

"""
用途: 构造 PBC 下 x/y 最近邻和对角次近邻 bond 列表。

参数:
- `lx, ly::Int`: 晶格尺寸。

返回:
- `NamedTuple`: 包含 `x_bonds`, `y_bonds`, `diagonal_bonds`, 每个元素为代表方向 `(site_i, site_j)`。
"""
function build_twist_nearest_neighbor_bonds(lx::Int, ly::Int)
    if lx <= 0 || ly <= 0
        error("Lattice sizes must be positive, got lx=$(lx), ly=$(ly).")
    end

    x_bonds = Tuple{Int,Int}[]
    y_bonds = Tuple{Int,Int}[]
    diagonal_bonds = Tuple{Int,Int}[]
    local_idx(x, y) = mod(x - 1, lx) * ly + mod(y - 1, ly) + 1
    for y in 1:ly, x in 1:lx
        site_i = local_idx(x, y)
        push!(x_bonds, (site_i, local_idx(x + 1, y)))
        push!(y_bonds, (site_i, local_idx(x, y + 1)))
        push!(diagonal_bonds, (site_i, local_idx(x + 1, y + 1)))
        push!(diagonal_bonds, (site_i, local_idx(x + 1, y - 1)))
    end
    return (; x_bonds=x_bonds, y_bonds=y_bonds, diagonal_bonds=diagonal_bonds)
end

"""
用途: 将代表方向 hopping bond 展开成 backflow 使用的双向 source 数据。

参数:
- `source_bonds::Vector{Tuple{Int, Int}}`: 待写入的有向 source bond 列表。
- `source_amplitudes::Vector{Float64}`: 待写入的有向 source hopping 振幅列表。
- `representative_bonds::Vector{Tuple{Int, Int}}`: 无向 bond 的一个方向代表。
- `hopping_amplitude::Float64`: 该类 bond 对应的物理 hopping 振幅。

返回:
- `nothing`。函数会原地追加 `(i, j)` 与 `(j, i)` 两个方向。
"""
function append_twist_directed_backflow_sources!(
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{Float64},
    representative_bonds::Vector{Tuple{Int,Int}},
    hopping_amplitude::Float64,
)::Nothing
    for (site_i, site_j) in representative_bonds
        push!(source_bonds, (site_i, site_j))
        push!(source_amplitudes, hopping_amplitude)
        push!(source_bonds, (site_j, site_i))
        push!(source_amplitudes, hopping_amplitude)
    end
    return nothing
end

"""
用途: 根据 twist Hubbard 的 x/y/对角 hopping bond 生成 backflow source 数据。

参数:
- `hopping_bonds`: `build_twist_nearest_neighbor_bonds` 返回的 NamedTuple, 需要包含
  `x_bonds`, `y_bonds`, `diagonal_bonds`。
- `tx, ty, t2::Float64`: physical Hamiltonian 中 x/y 最近邻和对角次近邻 hopping。

返回:
- `Tuple{Vector{Tuple{Int, Int}}, Vector{Float64}}`: 有向 source bonds 和对应物理 hopping 振幅。
"""
function build_twist_backflow_source_data(
    hopping_bonds,
    tx::Float64,
    ty::Float64,
    t2::Float64,
)
    source_bonds = Tuple{Int,Int}[]
    source_amplitudes = Float64[]
    append_twist_directed_backflow_sources!(
        source_bonds,
        source_amplitudes,
        hopping_bonds.x_bonds,
        tx,
    )
    append_twist_directed_backflow_sources!(
        source_bonds,
        source_amplitudes,
        hopping_bonds.y_bonds,
        ty,
    )
    append_twist_directed_backflow_sources!(
        source_bonds,
        source_amplitudes,
        hopping_bonds.diagonal_bonds,
        t2,
    )
    return source_bonds, source_amplitudes
end

"""
用途: 构造 twist Hubbard nonPH 使用的 Eq.(5) composite backflow。

参数:
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向 source bond 列表。
- `source_amplitudes::Vector{Float64}`: 与 source bond 对齐的物理 hopping 振幅。
- `bf_epsilon, bf_eta1, bf_eta2, bf_eta3, bf_eta4::Float64`: backflow 参数初值。

返回:
- `CompositeBackflowTerm`: 参数顺序为 `bf_epsilon`, `bf_eta1`, `bf_eta2`, `bf_eta3`, `bf_eta4`。
"""
function build_twist_composite_backflow(
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{Float64},
    bf_epsilon::Float64,
    bf_eta1::Float64,
    bf_eta2::Float64,
    bf_eta3::Float64,
    bf_eta4::Float64,
)
    epsilon_terms = [
        BackflowEpsilonTerm(
            param_name=:bf_epsilon,
            epsilon_bf=bf_epsilon,
            group_names=Symbol[:hubbard],
        ),
    ]
    hubbard_group = mfVMC.Backflow.build_directed_backflow_source_group(
        :hubbard,
        source_bonds,
        source_amplitudes,
        BackflowEta1DoublonHoleTerm(
            param_name=:bf_eta1,
            eta1_bf=bf_eta1,
        ),
        BackflowEta2SpinExchangeTerm(
            param_name=:bf_eta2,
            eta2_bf=bf_eta2,
        ),
        BackflowEta3DoublonSingleTerm(
            param_name=:bf_eta3,
            eta3_bf=bf_eta3,
        ),
        BackflowEta4SingleHoleTerm(
            param_name=:bf_eta4,
            eta4_bf=bf_eta4,
        ),
    )
    return CompositeBackflowTerm(epsilon_terms, [hubbard_group])
end

"""
用途: 根据开关构造 twist Hubbard optional backflow。

参数:
- `enable_backflow::Bool`: 是否启用 backflow。
- `source_bonds, source_amplitudes`: backflow source 数据。
- `bf_epsilon, bf_eta1, bf_eta2, bf_eta3, bf_eta4::Float64`: backflow 参数初值。

返回:
- `AbstractBackflowTerm`: 开启时为 `CompositeBackflowTerm`, 关闭时为 `NoBackflowTerm()`。
"""
function build_twist_optional_backflow(
    enable_backflow::Bool,
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{Float64},
    bf_epsilon::Float64,
    bf_eta1::Float64,
    bf_eta2::Float64,
    bf_eta3::Float64,
    bf_eta4::Float64,
)
    if !enable_backflow
        return NoBackflowTerm()
    end
    return build_twist_composite_backflow(
        source_bonds,
        source_amplitudes,
        bf_epsilon,
        bf_eta1,
        bf_eta2,
        bf_eta3,
        bf_eta4,
    )
end

"""
用途: 构造 twist Hubbard 的 nonPH number-conserving mean-field Hamiltonian。

数学公式:
- 使用 spinful electron basis `row(i, up)=2i-1`, `row(i, down)=2i`。
- 最近邻 hopping 为
  `H_{i,i+x} = -chi_x`, `H_{i,i+y} = -chi_y`, 同时作用在 up/down block。
- 对角次近邻 hopping 为 `H_{i,i+x+y} = H_{i,i+x-y} = -chi2`。
- 对角场为
  `H_{i up,i up} += (+1) * (-1)^(x+y) * m_i / 2 + rho_i / 2`,
  `H_{i down,i down} += (-1) * (-1)^(x+y) * m_i / 2 + rho_i / 2`。
- 其中 `rho_i = Delta_c cos(Q (x - x0))`,
  `m_i = Delta_AF + Delta_s sin(Q/2 * (x - x0))`。

参数:
- `params::TwistHubbardNonPHParams`: twist Hubbard mean-field 参数。

返回:
- `Hermitian{Float64, Matrix{Float64}}`: 维度为 `(2N_sites, 2N_sites)` 的 Hamiltonian。
"""
function build_twist_hubbard_nonph_hamiltonian(
    params::TwistHubbardNonPHParams,
)
    lx = params.lx
    ly = params.ly
    n_sites = lx * ly
    hamiltonian = zeros(Float64, 2 * n_sites, 2 * n_sites)

    for x in 1:lx
        charge_field_x = params.delta_c * cos(params.stripe_wavevector * (x - params.stripe_center_offset))
        spin_field_x = params.delta_af + params.delta_s * sin(params.stripe_wavevector / 2 * (x - params.stripe_center_offset))

        for y in 1:ly
            site_i = twist_site_index(x, y, ly)
            staggered_sign = (-1)^(x + y)

            site_x = twist_site_index(x == lx ? 1 : x + 1, y, ly)
            site_y = twist_site_index(x, y == ly ? 1 : y + 1, ly)
            site_pp = twist_site_index(x == lx ? 1 : x + 1, y == ly ? 1 : y + 1, ly)
            site_pm = twist_site_index(x == lx ? 1 : x + 1, y == 1 ? ly : y - 1, ly)
            bc_x = x == lx ? params.bcx : 1.0
            bc_y = y == ly ? params.bcy : 1.0
            bc_pp = (x == lx ? params.bcx : 1.0) * (y == ly ? params.bcy : 1.0)
            bc_pm = (x == lx ? params.bcx : 1.0) * (y == 1 ? params.bcy : 1.0)

            add_term_ij_nonPH(hamiltonian, site_i, site_x, -params.chi_x * bc_x)
            add_term_ij_nonPH(hamiltonian, site_i, site_y, -params.chi_y * bc_y)
            add_term_ij_nonPH(hamiltonian, site_i, site_pp, -params.chi2 * bc_pp)
            add_term_ij_nonPH(hamiltonian, site_i, site_pm, -params.chi2 * bc_pm)

            up_row = 2 * (site_i - 1) + 1
            down_row = up_row + 1
            hamiltonian[up_row, up_row] += staggered_sign * spin_field_x / 2 + charge_field_x / 2
            hamiltonian[down_row, down_row] += -staggered_sign * spin_field_x / 2 + charge_field_x / 2
        end
    end

    return Hermitian(hamiltonian + hamiltonian')
end

"""
用途: 构造 twist Hubbard nonPH Hamiltonian 对单个优化参数的导数矩阵。

数学公式:
- 当前优化 `chi1y`, `chi2`, `Delta_AF`, `Delta_c`, `Delta_s`。
- 这些参数线性进入 Hamiltonian, 因此 `dH/dp` 可通过只把目标参数置为 `1.0`,
  其它优化场置为 `0.0`, 并保持 `Q, x0, bcx, bcy` 不变得到。

参数:
- `params::TwistHubbardNonPHParams`: 当前 ansatz 参数, 提供尺寸、边界和 stripe 几何。
- `param_name::Symbol`: 目标参数名, 支持 `:chi1y`, `:chi2`, `:Delta_AF`, `:Delta_c`, `:Delta_s`。

返回:
- `Matrix{Float64}`: 与 Hamiltonian 同维度的 `dH/dp` 矩阵。
"""
function build_twist_hubbard_nonph_dh_dparam(
    params::TwistHubbardNonPHParams,
    param_name::Symbol,
)::Matrix{Float64}
    if !(param_name in (:chi1y, :chi2, :Delta_AF, :Delta_c, :Delta_s))
        error("Unknown twist Hubbard mean-field parameter: $(param_name).")
    end

    derivative_params = TwistHubbardNonPHParams(
        lx=params.lx,
        ly=params.ly,
        bcx=params.bcx,
        bcy=params.bcy,
        chi_x=0.0,
        chi_y=param_name == :chi1y ? 1.0 : 0.0,
        chi2=param_name == :chi2 ? 1.0 : 0.0,
        delta_af=param_name == :Delta_AF ? 1.0 : 0.0,
        delta_c=param_name == :Delta_c ? 1.0 : 0.0,
        delta_s=param_name == :Delta_s ? 1.0 : 0.0,
        stripe_wavevector=params.stripe_wavevector,
        stripe_center_offset=params.stripe_center_offset,
    )
    return Matrix(build_twist_hubbard_nonph_hamiltonian(derivative_params))
end

"""
用途: 生成 twist Hubbard nonPH determinant 的占据轨道和 mean-field 参数导数。

数学公式:
- 对 number-conserving spinful Hamiltonian `H U = U epsilon` 对角化。
- nonPH determinant 固定真实电子数 `N_e`, 因此取最低 `N_e` 个单粒子轨道。
- 对 mean-field 参数的 determinant 导数使用 `dU/dp`, 后续 SR 中进入
  `O_p = Tr(A^{-1} dA/dp)`。

参数:
- `params::TwistHubbardNonPHParams`: twist Hubbard mean-field 参数。
- `param_names::Vector{Symbol}`: 需要求导的 mean-field 参数名。
- `n_occupied_orbitals::Int`: determinant 占据轨道数, nonPH 下等于真实电子数。

返回:
- `(epsilon, occupied_orbitals, d_ut_params)`: 本征值, 占据轨道矩阵, 以及按参数名索引的转置导数矩阵。
"""
function make_twist_hubbard_nonph_ansatz_and_derivs(
    params::TwistHubbardNonPHParams;
    param_names::Vector{Symbol}=Symbol[],
    n_occupied_orbitals::Int,
)
    hamiltonian = Matrix(build_twist_hubbard_nonph_hamiltonian(params))
    hamiltonian_derivatives = Dict{Symbol,Matrix{Float64}}()
    for param_name in param_names
        hamiltonian_derivatives[param_name] = build_twist_hubbard_nonph_dh_dparam(
            params,
            param_name,
        )
    end

    epsilon, full_orbitals, _, orbital_derivatives = compute_eig_and_dU_reg1(
        hamiltonian,
        hamiltonian_derivatives,
    )
    if n_occupied_orbitals < 0 || n_occupied_orbitals > size(full_orbitals, 2)
        error("n_occupied_orbitals=$(n_occupied_orbitals) is outside 0:$(size(full_orbitals, 2)).")
    end

    if is_twist_root_rank()
        eig_eq_error = norm(hamiltonian * full_orbitals - full_orbitals * Diagonal(epsilon))
        left_index = max(1, n_occupied_orbitals - 4)
        right_index = min(length(epsilon), n_occupied_orbitals + 4)
        println("Eigen equation error (HU - Uepsilon): ", eig_eq_error)
        println("epsilon is ", epsilon[left_index:right_index])
    end

    occupied_orbitals = real.(full_orbitals[:, 1:n_occupied_orbitals])
    d_ut_params = Dict{Symbol,Matrix{Float64}}()
    for param_name in param_names
        d_ut_params[param_name] = permutedims(real.(orbital_derivatives[param_name][:, 1:n_occupied_orbitals]))
    end
    return epsilon, occupied_orbitals, d_ut_params
end

"""
用途: 根据参数更新 twist Hubbard nonPH determinant 波函数和参数导数。

参数:
- `vwf`: determinant 波函数对象。
- `param_names::Vector{Symbol}`: 完整参数名列表, 顺序为 mean-field, projector。
- `params::Vector{Float64}`: 与 `param_names` 对齐的完整参数值。
- `lx, ly::Int`: 晶格尺寸。
- `bcx, bcy::Float64`: mean-field 边界条件因子。
- `tx, ty::Float64`: physical hopping, 保留用于接口兼容; mean-field gauge 固定为 `chi_x=1`, `chi_y=chi1y`。
- `n_occupied_orbitals::Int`: nonPH determinant 占据轨道数, 等于真实电子数。
- `nparams_proj::Int`: projector 参数数量。
- `nparams_backflow::Int`: backflow 参数数量。
- `stripe_wavevector::Float64`: stripe 波矢 `Q`。
- `stripe_center_offset::Float64`: stripe 中心偏移 `x0`。
- `active_wf_param_names::Union{Nothing, Vector{Symbol}}`: 参与求导和 SR 的 mean-field 参数名; `nothing` 表示全部。

返回:
- `nothing`。
"""
function update_twist_ansatz!(
    vwf,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx::Int,
    ly::Int,
    bcx::Float64,
    bcy::Float64,
    tx::Float64,
    ty::Float64,
    n_occupied_orbitals::Int;
    nparams_proj::Int=0,
    nparams_backflow::Int=0,
    stripe_wavevector::Float64=0.0,
    stripe_center_offset::Float64=0.0,
    active_wf_param_names::Union{Nothing,Vector{Symbol}}=nothing,
)::Nothing
    total_param_count = length(param_names)
    nparams_wf = total_param_count - nparams_proj - nparams_backflow
    wf_param_names = param_names[1:nparams_wf]
    wf_param_values = params[1:nparams_wf]
    projector_param_names = nparams_proj > 0 ? param_names[(nparams_wf+1):(nparams_wf+nparams_proj)] : Symbol[]
    projector_param_values = nparams_proj > 0 ? params[(nparams_wf+1):(nparams_wf+nparams_proj)] : Float64[]
    backflow_param_names = nparams_backflow > 0 ? param_names[(nparams_wf+nparams_proj+1):total_param_count] : Symbol[]
    backflow_param_values = nparams_backflow > 0 ? params[(nparams_wf+nparams_proj+1):total_param_count] : Float64[]

    derivative_wf_param_names = active_wf_param_names === nothing ? wf_param_names : active_wf_param_names
    wf_param_name_set = Set(wf_param_names)
    for derivative_param_name in derivative_wf_param_names
        if !(derivative_param_name in wf_param_name_set)
            error("Active mean-field parameter $(derivative_param_name) is not in full mean-field parameter list.")
        end
    end

    param_map = Dict{Symbol,Float64}(zip(wf_param_names, wf_param_values))
    twist_params = TwistHubbardNonPHParams(
        lx=lx,
        ly=ly,
        bcx=bcx,
        bcy=bcy,
        chi_x=1.0,
        chi_y=get(param_map, :chi1y, compute_twist_chi1y_initial_value(tx, ty)),
        chi2=get(param_map, :chi2, 0.0),
        delta_af=get(param_map, :Delta_AF, 0.0),
        delta_c=get(param_map, :Delta_c, 0.0),
        delta_s=get(param_map, :Delta_s, 0.0),
        stripe_wavevector=stripe_wavevector,
        stripe_center_offset=stripe_center_offset,
    )

    _, gs_u, d_ut_params = make_twist_hubbard_nonph_ansatz_and_derivs(
        twist_params;
        param_names=derivative_wf_param_names,
        n_occupied_orbitals=n_occupied_orbitals,
    )

    copyto!(vwf.base_gs_U, gs_u)
    copyto!(vwf.gs_U, gs_u)
    copyto!(vwf.gs_U_t, permutedims(gs_u))

    d_ut_matrix = zeros(Float64, size(gs_u, 2), size(gs_u, 1), length(derivative_wf_param_names))
    for (param_index, param_name) in enumerate(derivative_wf_param_names)
        d_ut_matrix[:, :, param_index] = d_ut_params[param_name]
    end
    update_vwf_params!(vwf, derivative_wf_param_names, d_ut_matrix)

    if !isempty(projector_param_names)
        update_vwf_projector_params!(vwf, projector_param_names, projector_param_values)
    end
    if !isempty(backflow_param_names)
        update_vwf_backflow_params!(vwf, backflow_param_names, backflow_param_values)
    end
    init_gswf!(vwf)
    return nothing
end

"""
用途: 根据 physical hopping 生成 twist mean-field 中 `chi1y` 的初值。

参数:
- `tx, ty::Float64`: physical Hamiltonian 中 x/y 方向最近邻 hopping。

返回:
- `Float64`: `chi1y = ty / tx`。这里固定 gauge 为 `chi1x = 1`。
"""
function compute_twist_chi1y_initial_value(tx::Float64, ty::Float64)::Float64
    if isapprox(tx, 0.0; atol=1.0e-12, rtol=0.0)
        error("twist Hubbard fixes chi1x=1 as gauge, so --tx must be nonzero to initialize chi1y=ty/tx.")
    end
    return ty / tx
end

"""
用途: 根据 physical hopping 生成 twist mean-field 中 `chi2` 的初值。

参数:
- `tx, t2::Float64`: physical Hamiltonian 中 x 方向最近邻 hopping 与对角次近邻 hopping。

返回:
- `Float64`: `chi2 = t2 / tx`。这里固定 gauge 为 `chi1x = 1`。
"""
function compute_twist_chi2_initial_value(tx::Float64, t2::Float64)::Float64
    if isapprox(tx, 0.0; atol=1.0e-12, rtol=0.0)
        error("twist Hubbard fixes chi1x=1 as gauge, so --tx must be nonzero to initialize chi2=t2/tx.")
    end
    return t2 / tx
end

"""
用途: 根据 `ansatz` 参数生成 twist Hubbard mean-field 参数名、初值和 stripe 几何。

参数:
- `args::Dict{String, Any}`: 命令行参数字典。

返回:
- `NamedTuple`: 包含 `wf_param_names`, `wf_init_params`, `stripe_wavevector`, `stripe_center_offset`。
"""
function build_twist_mean_field_parameter_setup(args)
    ansatz = args["ansatz"]
    chi1y_initial_value = compute_twist_chi1y_initial_value(args["tx"], args["ty"])
    chi2_initial_value = compute_twist_chi2_initial_value(args["tx"], args["t2"])
    if ansatz == "AFM"
        return (
            wf_param_names=[:chi1y, :chi2, :Delta_AF],
            wf_init_params=[chi1y_initial_value, chi2_initial_value, args["Delta_AF"]],
            stripe_wavevector=0.0,
            stripe_center_offset=0.0,
        )
    elseif ansatz == "Stripe"
        lambda = args["lambda"]
        stripe_center = args["stripe_center"]
        stripe_center_offset = if stripe_center == "site"
            0.0
        elseif stripe_center == "bond"
            0.5
        else
            error("Unknown stripe_center type: $(stripe_center)")
        end
        return (
            wf_param_names=[:chi1y, :chi2, :Delta_c, :Delta_s],
            wf_init_params=[chi1y_initial_value, chi2_initial_value, args["Delta_c"], args["Delta_s"]],
            stripe_wavevector=2π / lambda,
            stripe_center_offset=stripe_center_offset,
        )
    end
    error("Unknown ansatz type: $(ansatz)")
end

"""
用途: 运行 twist Hubbard nonPH VMC/SR 主流程。

参数:
- 无。所有配置来自命令行参数。

返回:
- `nothing`。

说明:
- sampler 使用 `config_Hubbard(...; ifPH=false)`, 因此 determinant row 直接对应真实电子轨道。
- determinant 列数取真实电子数 `N_e = N_sites * (1 + doping)`。
- 第一版没有 backflow, physical Hamiltonian 也没有次近邻 hopping。
"""
function main_twist()::Nothing
    args = parse_twist_commandline()

    session = init_mpi_session()
    rank = session.rank
    is_root = rank == session.root

    lx = args["Lx"]
    ly = args["Ly"]
    bcx = args["bcx"]
    bcy = args["bcy"]
    tx = args["tx"]
    ty = args["ty"]
    t2 = args["t2"]
    onsite_u = args["U"]
    target_sz = args["target_sz"]
    doping = args["doping"]
    nmc = args["nMC"]
    wmc = args["wMC"]
    rmc = args["rMC"]
    dmc = args["dMC"]
    n_steps = args["nSR"]
    lr = args["lr"]
    lr_end = isnan(args["lr_end"]) ? lr : args["lr_end"]
    eigen_cutoff = args["eigen_cutoff"]
    job = args["job"]
    init_params_json = args["init_params_json"]
    fixed_params_string = args["fixed_params"]
    active_params_string = args["active_params"]
    n_sites = lx * ly
    jastrow_dx_max = args["jastrow_dx_max"] < 0 ? div(lx, 2) : args["jastrow_dx_max"]
    jastrow_dy_max = args["jastrow_dy_max"] < 0 ? div(ly, 2) : args["jastrow_dy_max"]

    mean_field_setup = build_twist_mean_field_parameter_setup(args)
    wf_param_names = mean_field_setup.wf_param_names
    wf_init_params = mean_field_setup.wf_init_params
    stripe_wavevector = mean_field_setup.stripe_wavevector
    stripe_center_offset = mean_field_setup.stripe_center_offset

    meas_params = VMCParams(
        total_samples=nmc,
        warmup_steps=wmc,
        rebuild_every=rmc,
        decorr_steps=dmc,
        seed=args["seed"] + rank,
    )

    projector = build_twist_projector(
        lx,
        ly,
        args["g"];
        jastrow_dx_max=jastrow_dx_max,
        jastrow_dy_max=jastrow_dy_max,
    )
    hopping_bonds = build_twist_nearest_neighbor_bonds(lx, ly)
    source_bonds, source_amplitudes = build_twist_backflow_source_data(hopping_bonds, tx, ty, t2)
    backflow = build_twist_optional_backflow(
        parse_twist_bool_flag(args["enable_backflow"], "--enable_backflow"),
        source_bonds,
        source_amplitudes,
        args["bf_epsilon"],
        args["bf_eta1"],
        args["bf_eta2"],
        args["bf_eta3"],
        args["bf_eta4"],
    )
    proj_param_names = projector_param_names(projector)
    proj_init_params = projector_param_values(projector)
    nparams_proj = length(proj_param_names)
    backflow_param_name_list = backflow_param_names(backflow)
    backflow_init_params = backflow_param_values(backflow)
    nparams_backflow = length(backflow_param_name_list)

    init_params = vcat(wf_init_params, proj_init_params, backflow_init_params)
    param_names = vcat(wf_param_names, proj_param_names, backflow_param_name_list)

    if !isempty(init_params_json)
        init_params = build_twist_init_params_from_json_with_defaults(init_params_json, param_names, init_params)
        if is_root
            println("Loaded initial parameters from json: $(init_params_json)")
        end
    end

    fixed_param_values = parse_twist_fixed_param_string(fixed_params_string)
    requested_active_param_names = parse_twist_param_name_list(active_params_string)
    if !isempty(fixed_param_values)
        init_params = apply_twist_fixed_params_to_values(param_names, init_params, fixed_param_values)
    end
    active_param_indices = build_twist_active_param_indices(param_names, fixed_param_values, requested_active_param_names)
    if job == "SR" && isempty(active_param_indices)
        error("At least one parameter must remain active for SR optimization.")
    end

    uses_param_subset = length(active_param_indices) != length(param_names) || !isempty(requested_active_param_names)
    sr_param_names = uses_param_subset ? param_names[active_param_indices] : param_names
    sr_init_params = uses_param_subset ? init_params[active_param_indices] : init_params
    wf_param_name_set = Set(wf_param_names)
    projector_param_name_set = Set(proj_param_names)
    backflow_param_name_set = Set(backflow_param_name_list)
    active_wf_param_names = [name for name in sr_param_names if name in wf_param_name_set]
    active_projector_param_names = [name for name in sr_param_names if name in projector_param_name_set]
    active_backflow_param_names = [name for name in sr_param_names if name in backflow_param_name_set]
    set_active_twist_projector_derivative_param_names!(
        proj_param_names;
        active_projector_param_names=uses_param_subset ? active_projector_param_names : nothing,
    )
    set_active_twist_backflow_derivative_param_names!(
        backflow_param_name_list;
        active_backflow_param_names=uses_param_subset ? active_backflow_param_names : nothing,
    )

    term_setup = build_twist_hamiltonian_terms(lx, ly, tx, ty, t2, onsite_u)
    ham = GeneralModel(n_sites, term_setup.all_terms)

    electron_count_float = n_sites * (1 + doping)
    nelec = round(Int, electron_count_float)
    if !isapprox(electron_count_float, nelec; atol=1e-8, rtol=0.0)
        error("N_sites * (1 + doping) must be an integer, got $(electron_count_float).")
    end
    @assert (target_sz + nelec) % 2 == 0 "Wrong parity between target_sz and electron count."
    nup = (nelec + target_sz) ÷ 2
    ndn = nelec - nup
    if nup < 0 || ndn < 0 || nup > n_sites || ndn > n_sites
        error("Invalid nonPH particle numbers: nup=$(nup), ndn=$(ndn), N_sites=$(n_sites).")
    end

    sampler = config_Hubbard(n_sites, nup, ndn; ifPH=false)
    init_config_Hubbard!(sampler)

    vwf = vwf_det(zeros(Float64, 2 * n_sites, nelec), sampler; backflow=backflow)
    set_projector!(vwf, projector)
    kernel = HubbardKernel(conserve_sz=true)

    if is_root
        println("Initial parameters: $(init_params)")
        println("twist Hubbard nonPH particle numbers: N_up=$(nup), N_down=$(ndn), N_e=$(nelec)")
        println("Nearest-neighbor hopping: tx=$(tx), ty=$(ty)")
        println("Backflow enabled: $(mfVMC.Backflow.uses_backflow(backflow))")
        if !isempty(fixed_param_values)
            fixed_param_messages = [
                "$(String(param_name))=$(fixed_param_values[param_name])"
                for param_name in sort(collect(keys(fixed_param_values)); by=String)
            ]
            println("Fixed parameters: $(join(fixed_param_messages, ", "))")
        end
        if uses_param_subset
            active_param_names = param_names[active_param_indices]
            println("Active parameters: $(join(String.(active_param_names), ", "))")
        end
    end

    update_twist_ansatz!(
        vwf,
        param_names,
        init_params,
        lx,
        ly,
        bcx,
        bcy,
        tx,
        ty,
        nelec;
        nparams_proj=nparams_proj,
        nparams_backflow=nparams_backflow,
        stripe_wavevector=stripe_wavevector,
        stripe_center_offset=stripe_center_offset,
    )

    folder = "logs"
    mkpath(folder)

    if job == "SR"
        sr_params = build_twist_sr_params(meas_params, n_steps, lr, eigen_cutoff)
        exp_lr_func = build_twist_exponential_lr_func(lr, lr_end, n_steps)

        update_vwf_func! = (vwf, params) -> begin
            full_params = uses_param_subset ?
                          merge_twist_active_params_into_full(init_params, active_param_indices, params) :
                          params
            derivative_wf_param_names = uses_param_subset ? active_wf_param_names : nothing
            update_twist_ansatz!(
                vwf,
                param_names,
                full_params,
                lx,
                ly,
                bcx,
                bcy,
                tx,
                ty,
                nelec;
                nparams_proj=nparams_proj,
                nparams_backflow=nparams_backflow,
                stripe_wavevector=stripe_wavevector,
                stripe_center_offset=stripe_center_offset,
                active_wf_param_names=derivative_wf_param_names,
            )
        end

        run_sr_optimization(
            ham,
            vwf,
            kernel,
            sr_init_params,
            update_vwf_func!,
            sr_params;
            log_file=joinpath(folder, "sr_history.txt"),
            param_names=sr_param_names,
            lr_func=exp_lr_func,
        )
        if is_root
            extract_min_energy(joinpath(folder, "sr_history.txt"))
            append_twist_inactive_params_to_json!(
                joinpath(folder, "min_params.json"),
                param_names,
                init_params,
                active_param_indices,
            )
        end
    elseif job == "measure"
        results = run_simulation(
            ham,
            vwf,
            kernel,
            definition_twist_observables(
                lx,
                ly;
                hopping_terms=term_setup.hopping_terms,
                tx_hopping_terms=term_setup.tx_hopping_terms,
                ty_hopping_terms=term_setup.ty_hopping_terms,
                t2_hopping_terms=term_setup.t2_hopping_terms,
                interaction_terms=term_setup.interaction_terms,
                onsite_u=onsite_u,
            ),
            meas_params;
            history_observables=[
                :E,
                :E_hop,
                :E_hop_tx,
                :E_hop_ty,
                :E_hop_t2,
                :E_int,
                :E_int_charge,
                :E_int_spin,
            ],
        )
        if is_root && results !== nothing
            means = results[:means]
            mean_dict = Dict{Symbol,Any}()
            for (key, value) in means
                mean_dict[key] = value isa Number ? real(value) : value
            end

            histories = results[:histories]
            if !isempty(histories)
                mean_hist, se_dict, n_eff_dict, tau_int_dict, _ = blocking_binning(histories)
                txt_file = joinpath(folder, "block_binning.txt")
                open(txt_file, "w") do io
                    println(io, "# Observable\tMean\tSE\tN_eff\tTau_int")
                    for name in sort(collect(keys(mean_hist)))
                        mean_val = mean_hist[name]
                        se_val = se_dict[name]
                        n_eff_val = n_eff_dict[name]
                        tau_val = tau_int_dict[name]
                        if mean_val isa Number && se_val isa Number && n_eff_val isa Number && tau_val isa Number
                            @printf(
                                io,
                                "%s\t%.10f\t%.10f\t%.6f\t%.6f\n",
                                String(name),
                                mean_val,
                                se_val,
                                n_eff_val,
                                tau_val,
                            )
                        else
                            println(io, "$(String(name))\t$(mean_val)\t$(se_val)\t$(n_eff_val)\t$(tau_val)")
                        end
                    end
                end
            end

            mean_dict_str = Dict{String,Any}()
            for (key, value) in mean_dict
                mean_dict_str[String(key)] = value
            end
            open(joinpath(folder, "block_binning_mean.json"), "w") do io
                JSON.print(io, mean_dict_str)
            end
        end
    else
        error("Unknown job: $(job)")
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_twist()
end
