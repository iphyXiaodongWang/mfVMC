using MPI
using Random
using Printf
using DelimitedFiles
using LinearAlgebra
using Statistics
using ArgParse
using JSON
# using FFWT

# === 1. 环境设置 ===
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
push!(LOAD_PATH, @__DIR__)


using mfVMC
include("PartonSquare.jl")
using .PartonSquare


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--Lx"
        help = "Lattice size in X direction"
        arg_type = Int
        default = 8
        "--Ly"
        help = "Lattice size in Y direction"
        arg_type = Int
        default = 3
        "--t1"
        help = "Hopping amplitude"
        arg_type = Float64
        default = 1.0
        "--t2"
        help = "Next-nearest neighbor hopping amplitude"
        arg_type = Float64
        default = -0.2
        "--U"
        help = "On-site interaction strength"
        arg_type = Float64
        default = 8.0
        "--bcx"
        help = "Boundary condition phase in X (1.0 or -1.0)"
        arg_type = Float64
        default = 1.001
        "--bcy"
        help = "Boundary condition phase in Y (1.0 or -1.0)"
        arg_type = Float64
        default = 0.999
        "--etax"
        help = "MF parameters"
        arg_type = Float64
        default = 0.01
        "--etay"
        help = "MF parameters"
        arg_type = Float64
        default = 0.01
        "--chi2"
        help = "Next-nearest neighbor hopping in MF ansatz. Default follows --t2"
        arg_type = Float64
        default = -0.2
        "--Delta_AF"
        help = "AFM order parameters"
        arg_type = Float64
        default = 3.0
        "--Delta_c"
        help = "charge stripe order parameters"
        arg_type = Float64
        default = 3.0
        "--Delta_s"
        help = "spin stripe order parameters"
        arg_type = Float64
        default = 3.0
        "--mu"
        help = "chemical potential"
        arg_type = Float64
        default = -3.0
        "--target_sz"
        help = "target total sz"
        arg_type = Int
        default = 0
        "--nMC"
        help = "Number of Monte Carlo total_samples"
        arg_type = Int
        default = 10000
        "--wMC"
        help = "Number of Monte Carlo warnming up"
        arg_type = Int
        default = 100
        "--rMC"
        help = "Number of rebuild inserve"
        arg_type = Int
        default = 100
        "--dMC"
        help = "Number of Monte Carlo decorrelation sweeps"
        arg_type = Int
        default = 1
        "--seed"
        help = "random seed"
        arg_type = Int
        default = 5423
        "--nSR"
        help = "total steps for SR"
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
        "--init_params_json"
        help = "Path to json file that provides initial parameters"
        arg_type = String
        default = ""
        "--fixed_params"
        help = "Comma-separated fixed parameter assignments, e.g. 'mu=0.1,bf_epsilon=1.0'"
        arg_type = String
        default = ""
        "--active_params"
        help = "Comma-separated parameter names optimized by SR. Empty means all non-fixed parameters."
        arg_type = String
        default = ""
        "--enable_backflow"
        help = "Whether to enable backflow terms. Accepts true/false, 1/0, yes/no. Default is true."
        arg_type = String
        default = "true"
        "--job"
        help = "Job to be done. Can be SR and measure"
        arg_type = String
        default = "SR"
        "--doping"
        help = "Doping level"
        arg_type = Float64
        default = 0.125
        "--ansatz"
        help = "Ansatz type, can be 'AFM' or 'Stripe'"
        arg_type = String
        default = "Stripe"
        "--lambda"
        help = "assuming length of stripe"
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
        "--bf_epsilon"
        help = "Eq.(5) backflow epsilon parameter"
        arg_type = Float64
        default = 1.0
        "--bf_eta1"
        help = "Eq.(5) backflow eta1 doublon-hole hopping parameter"
        arg_type = Float64
        default = 0.0
        "--bf_eta2"
        help = "Eq.(5) backflow eta2 spin-exchange hopping parameter"
        arg_type = Float64
        default = 0.0
        "--bf_eta3"
        help = "Eq.(5) backflow eta3 mixed virtual hopping parameter"
        arg_type = Float64
        default = 0.0
    end

    return parse_args(s)
end

const ACTIVE_PROJECTOR_DERIVATIVE_PARAM_NAMES = Ref{Union{Nothing,Vector{Symbol}}}(nothing)
const ACTIVE_BACKFLOW_DERIVATIVE_PARAM_NAMES = Ref{Union{Nothing,Vector{Symbol}}}(nothing)

"""
用途: 检查一组 active derivative 参数名是否属于给定参数组。

参数:
- `available_param_names::Vector{Symbol}`: 当前参数组的完整参数名。
- `active_param_names::Union{Nothing, Vector{Symbol}}`: 需要参与 SR 求导的参数名; `nothing` 表示全部参与。
- `param_group_name::AbstractString`: 参数组名称, 用于错误信息。

返回:
- `nothing`。若 active 参数名重复或不存在会抛出异常。
"""
function validate_active_derivative_param_names!(
    available_param_names::Vector{Symbol},
    active_param_names::Union{Nothing,Vector{Symbol}},
    param_group_name::AbstractString,
)::Nothing
    if active_param_names === nothing
        return nothing
    end
    if length(unique(active_param_names)) != length(active_param_names)
        error("Duplicate active $(param_group_name) derivative parameters: $(active_param_names).")
    end

    available_param_name_set = Set(available_param_names)
    for active_param_name in active_param_names
        if !(active_param_name in available_param_name_set)
            error(
                "Unknown active $(param_group_name) derivative parameter $(active_param_name). " *
                "Available parameters: $(join(String.(available_param_names), ", ")).",
            )
        end
    end
    return nothing
end

"""
用途: 设置当前 SR 中 projector 与 backflow 哪些参数参与导数计算。

参数:
- `projector_param_names::Vector{Symbol}`: 完整 projector 参数名。
- `backflow_param_names::Vector{Symbol}`: 完整 backflow 参数名。
- `active_projector_param_names::Union{Nothing, Vector{Symbol}}`: active projector 参数名; `nothing` 表示全部。
- `active_backflow_param_names::Union{Nothing, Vector{Symbol}}`: active backflow 参数名; `nothing` 表示全部。

返回:
- `nothing`。该设置只影响本脚本中重定义的 `compute_grad_log_psi!`。
"""
function set_active_sr_derivative_param_names!(
    projector_param_names::Vector{Symbol},
    backflow_param_names::Vector{Symbol};
    active_projector_param_names::Union{Nothing,Vector{Symbol}}=nothing,
    active_backflow_param_names::Union{Nothing,Vector{Symbol}}=nothing,
)::Nothing
    validate_active_derivative_param_names!(
        projector_param_names,
        active_projector_param_names,
        "projector",
    )
    validate_active_derivative_param_names!(
        backflow_param_names,
        active_backflow_param_names,
        "backflow",
    )

    ACTIVE_PROJECTOR_DERIVATIVE_PARAM_NAMES[] =
        active_projector_param_names === nothing ? nothing : copy(active_projector_param_names)
    ACTIVE_BACKFLOW_DERIVATIVE_PARAM_NAMES[] =
        active_backflow_param_names === nothing ? nothing : copy(active_backflow_param_names)
    return nothing
end

"""
用途: 根据完整参数名和当前 active 设置返回真正参与 SR 求导的参数名。

参数:
- `all_param_names::Vector{Symbol}`: 某一参数组的完整参数名。
- `active_param_names::Union{Nothing, Vector{Symbol}}`: active 参数名; `nothing` 表示全部参与。

返回:
- `Vector{Symbol}`: 按 SR 参数顺序排列的 active 参数名。
"""
function get_active_derivative_param_names(
    all_param_names::Vector{Symbol},
    active_param_names::Union{Nothing,Vector{Symbol}},
)::Vector{Symbol}
    if active_param_names === nothing
        return all_param_names
    end
    return active_param_names
end

"""
用途: 在 `Hubbard_restricted.jl` 中覆盖 SR 的 log-derivative 计算, 支持只优化部分 projector/backflow 参数。

数学公式:
- 对 determinant 参数仍使用 `O_p = Tr(A^{-1} dA/dp)`。
- 对 projector 参数使用 `O_p = d log(P) / dp`, 但只保留 active projector 参数。
- 对 backflow 参数使用 `O_p = Tr(A^{-1} dA_b/dp)`, 但只保留 active backflow 参数。

参数:
- `vwf::mfVMC.VMC.vwf_det{T}`: determinant 波函数对象。

返回:
- `Vector{T}`: 与当前 SR active 参数顺序一致的 log-derivative 向量。
"""
function mfVMC.VMC.compute_grad_log_psi!(vwf::mfVMC.VMC.vwf_det{T}) where T
    ws = mfVMC.VMC.ensure_ws!(vwf)
    ss = vwf.sampler

    a_inv = vwf.awf_inv
    n_orbitals, n_electrons = size(a_inv)

    wf_param_count = length(vwf.param_keys)
    projector_param_names_all = mfVMC.Projector.projector_param_names(vwf.projector)
    backflow_param_names_all = mfVMC.Backflow.backflow_param_names(vwf.backflow)
    active_projector_param_names = get_active_derivative_param_names(
        projector_param_names_all,
        ACTIVE_PROJECTOR_DERIVATIVE_PARAM_NAMES[],
    )
    active_backflow_param_names = get_active_derivative_param_names(
        backflow_param_names_all,
        ACTIVE_BACKFLOW_DERIVATIVE_PARAM_NAMES[],
    )

    total_active_param_count =
        wf_param_count + length(active_projector_param_names) + length(active_backflow_param_names)
    resize!(ws.grad_buffer, total_active_param_count)
    o_vec = ws.grad_buffer
    fill!(o_vec, zero(T))

    has_active_backflow = mfVMC.Backflow.uses_backflow(vwf.backflow)
    for param_index in 1:wf_param_count
        dU_t = @view vwf.dUt_matrix[:, :, param_index]
        derivative_orbitals = if has_active_backflow
            mfVMC.Backflow.fill_backflow_chain_rule_orbitals!(
                ws.backflow_chain_rule_buffer,
                transpose(dU_t),
                ss.state,
                vwf.backflow,
            )
            ws.backflow_chain_rule_buffer
        else
            nothing
        end

        total_sum = zero(T)
        @inbounds for electron_index in 1:n_electrons
            row_index = ss.electron_locs[electron_index]
            column_sum = zero(T)
            if has_active_backflow
                @simd for orbital_index in 1:n_orbitals
                    column_sum += a_inv[orbital_index, electron_index] *
                                  derivative_orbitals[row_index, orbital_index]
                end
            else
                @simd for orbital_index in 1:n_orbitals
                    column_sum += a_inv[orbital_index, electron_index] *
                                  dU_t[orbital_index, row_index]
                end
            end
            total_sum += column_sum
        end
        o_vec[param_index] = total_sum
    end

    output_offset = wf_param_count
    if !isempty(active_projector_param_names)
        full_projector_derivatives = mfVMC.Projector.projector_log_derivative(vwf.projector, ss)
        projector_derivative_map = Dict{Symbol,Float64}()
        for (param_name, derivative_value) in zip(projector_param_names_all, full_projector_derivatives)
            projector_derivative_map[param_name] = Float64(derivative_value)
        end
        for (active_offset, param_name) in enumerate(active_projector_param_names)
            if !haskey(projector_derivative_map, param_name)
                error("Missing projector derivative for active parameter $(param_name).")
            end
            o_vec[output_offset + active_offset] = T(projector_derivative_map[param_name])
        end
        output_offset += length(active_projector_param_names)
    end

    if !isempty(active_backflow_param_names)
        backflow_pairs = mfVMC.Backflow.build_backflow_derivative_orbitals(
            vwf.base_gs_U,
            ss.state,
            vwf.backflow,
        )
        backflow_derivative_map = Dict{Symbol,Matrix{T}}()
        for derivative_pair in backflow_pairs
            backflow_derivative_map[first(derivative_pair)] = last(derivative_pair)
        end

        for (active_offset, param_name) in enumerate(active_backflow_param_names)
            if !haskey(backflow_derivative_map, param_name)
                error("Missing backflow derivative for active parameter $(param_name).")
            end
            derivative_orbitals = backflow_derivative_map[param_name]
            total_sum = zero(T)
            @inbounds for electron_index in 1:n_electrons
                row_index = ss.electron_locs[electron_index]
                column_sum = zero(T)
                @simd for orbital_index in 1:n_orbitals
                    column_sum += a_inv[orbital_index, electron_index] *
                                  derivative_orbitals[row_index, orbital_index]
                end
                total_sum += column_sum
            end
            o_vec[output_offset + active_offset] = total_sum
        end
    end

    return o_vec
end

# ==============================================================================
# 3. 辅助函数
# ==============================================================================

"""
用途: 将 Jastrow 位移标签规范化为只依赖距离的参数标签。

参数:
- `dx::Int`: 某个轴向最短镜像位移的 `x` 分量绝对值。
- `dy::Int`: 某个轴向最短镜像位移的 `y` 分量绝对值。

返回:
- `Tuple{Int, Int}`: 排序后的 `(min(dx, dy), max(dx, dy))` 标签, 使 `J_x_y = J_y_x`。
"""
function build_jastrow_canonical_displacement_label(
    dx::Int,
    dy::Int,
)::Tuple{Int,Int}
    if dx < 0 || dy < 0
        error("dx and dy must be non-negative, got dx=$(dx), dy=$(dy).")
    end
    return (min(dx, dy), max(dx, dy))
end

"""
用途: 枚举全位移 Jastrow 参数的合法距离标签。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。

返回:
- `Vector{Tuple{Int, Int}}`: 按稳定顺序排列的规范化距离标签列表, 其中
  `(dx, dy)` 表示 `min(|Delta x|, |Delta y|)` 与 `max(|Delta x|, |Delta y|)`,
  且排除 `(0, 0)`。

说明:
- 这里将 `(dx, dy)` 与 `(dy, dx)` 合并到同一个标签, 因此满足 `J_x_y = J_y_x`。
- 为了去除 `g` 与全位移 Jastrow 参数共同构成的一维冗余自由度, 这里统一移除
  稳定排序后的最后一个位移标签。
"""
function build_jastrow_displacement_labels(
    lx::Int,
    ly::Int,
)::Vector{Tuple{Int,Int}}
    if lx <= 0 || ly <= 0
        error("lx and ly must be positive, got lx=$(lx), ly=$(ly).")
    end

    displacement_label_set = Set{Tuple{Int,Int}}()
    max_dx = fld(lx, 2)
    max_dy = fld(ly, 2)
    for dx in 0:max_dx
        for dy in 0:max_dy
            if dx == 0 && dy == 0
                continue
            end
            push!(displacement_label_set, build_jastrow_canonical_displacement_label(dx, dy))
        end
    end
    displacement_labels = sort!(collect(displacement_label_set))
    if !isempty(displacement_labels)
        pop!(displacement_labels)
    end
    return displacement_labels
end

"""
用途: 根据 Jastrow 位移标签构造规范化参数名。

参数:
- `dx::Int`: 位移标签的第一个分量, 可以是轴向 `x` 分量或未排序距离分量。
- `dy::Int`: 位移标签的第二个分量, 可以是轴向 `y` 分量或未排序距离分量。

返回:
- `Symbol`: 形如 `:vj_dx_dy` 的参数名, 例如 `:vj_1_2`。
"""
function build_jastrow_param_name(
    dx::Int,
    dy::Int,
)::Symbol
    if dx < 0 || dy < 0
        error("dx and dy must be non-negative, got dx=$(dx), dy=$(dy).")
    end
    canonical_dx, canonical_dy = build_jastrow_canonical_displacement_label(dx, dy)
    return Symbol("vj_$(canonical_dx)_$(canonical_dy)")
end

"""
用途: 枚举某个规范化 Jastrow 距离标签对应的合法轴向位移。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。
- `dx::Int`: 位移标签的第一个分量, 可以是轴向 `x` 分量或未排序距离分量。
- `dy::Int`: 位移标签的第二个分量, 可以是轴向 `y` 分量或未排序距离分量。

返回:
- `Vector{Tuple{Int, Int}}`: 合法轴向位移列表, 每个元素为 `(axis_dx, axis_dy)`。

说明:
- 例如方形晶格上的标签 `(1, 2)` 会返回 `(1, 2)` 与 `(2, 1)`。
- 对长方形晶格, 若交换后的轴向位移超出某个方向的最短镜像范围, 会自动跳过。
"""
function build_jastrow_axis_displacement_variants(
    lx::Int,
    ly::Int,
    dx::Int,
    dy::Int,
)::Vector{Tuple{Int,Int}}
    if lx <= 0 || ly <= 0
        error("lx and ly must be positive, got lx=$(lx), ly=$(ly).")
    end
    if dx < 0 || dy < 0
        error("dx and dy must be non-negative, got dx=$(dx), dy=$(dy).")
    end
    canonical_dx, canonical_dy = build_jastrow_canonical_displacement_label(dx, dy)
    if canonical_dx == 0 && canonical_dy == 0
        error("Displacement (0, 0) is not allowed for Jastrow terms.")
    end

    candidate_set = Set{Tuple{Int,Int}}([
        (canonical_dx, canonical_dy),
        (canonical_dy, canonical_dx),
    ])
    axis_displacements = Tuple{Int,Int}[]
    max_dx = fld(lx, 2)
    max_dy = fld(ly, 2)
    for (axis_dx, axis_dy) in sort!(collect(candidate_set))
        if axis_dx <= max_dx && axis_dy <= max_dy
            push!(axis_displacements, (axis_dx, axis_dy))
        end
    end
    if isempty(axis_displacements)
        error("No valid axis displacement for canonical Jastrow label ($(canonical_dx), $(canonical_dy)) on lx=$(lx), ly=$(ly).")
    end
    return axis_displacements
end

"""
用途: 为给定的轴向 Jastrow 位移生成去重后的有向 PBC offset 列表。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。
- `axis_dx::Int`: `x` 方向绝对位移, 必须位于最短镜像范围内。
- `axis_dy::Int`: `y` 方向绝对位移, 必须位于最短镜像范围内。

返回:
- `Vector{Tuple{Int, Int}}`: 在模 `lx`、`ly` 意义下去重后的 offset 列表。

说明:
- 当 `axis_dx = 0` 或 `axis_dy = 0` 时, 对应方向不会重复枚举正负号。
- 当 `axis_dx = lx / 2` 或 `axis_dy = ly / 2` 时, 通过模运算后会自动去重。
"""
function build_jastrow_wrapped_offsets_for_axis_displacement(
    lx::Int,
    ly::Int,
    axis_dx::Int,
    axis_dy::Int,
)::Vector{Tuple{Int,Int}}
    if lx <= 0 || ly <= 0
        error("lx and ly must be positive, got lx=$(lx), ly=$(ly).")
    end
    if axis_dx < 0 || axis_dy < 0
        error("axis_dx and axis_dy must be non-negative, got axis_dx=$(axis_dx), axis_dy=$(axis_dy).")
    end
    if axis_dx == 0 && axis_dy == 0
        error("Displacement (0, 0) is not allowed for Jastrow terms.")
    end
    if axis_dx > fld(lx, 2) || axis_dy > fld(ly, 2)
        error("Axis displacement exceeds shortest-image range: axis_dx=$(axis_dx), axis_dy=$(axis_dy), lx=$(lx), ly=$(ly).")
    end

    sign_choices_x = axis_dx == 0 ? [1] : [1, -1]
    sign_choices_y = axis_dy == 0 ? [1] : [1, -1]
    wrapped_offset_set = Set{Tuple{Int,Int}}()
    for sign_x in sign_choices_x
        for sign_y in sign_choices_y
            push!(wrapped_offset_set, (mod(sign_x * axis_dx, lx), mod(sign_y * axis_dy, ly)))
        end
    end
    return sort!(collect(wrapped_offset_set))
end

"""
用途: 为给定的 Jastrow 距离标签生成去重后的有向 PBC offset 列表。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。
- `dx::Int`: 位移标签的第一个分量, 可以是轴向 `x` 分量或未排序距离分量。
- `dy::Int`: 位移标签的第二个分量, 可以是轴向 `y` 分量或未排序距离分量。

返回:
- `Vector{Tuple{Int, Int}}`: 合并 `(dx, dy)` 与 `(dy, dx)` 后, 在模 `lx`、`ly`
  意义下去重后的 offset 列表。
"""
function build_jastrow_wrapped_offsets_for_displacement(
    lx::Int,
    ly::Int,
    dx::Int,
    dy::Int,
)::Vector{Tuple{Int,Int}}
    wrapped_offset_set = Set{Tuple{Int,Int}}()
    axis_displacements = build_jastrow_axis_displacement_variants(lx, ly, dx, dy)
    for (axis_dx, axis_dy) in axis_displacements
        axis_offsets = build_jastrow_wrapped_offsets_for_axis_displacement(
            lx,
            ly,
            axis_dx,
            axis_dy,
        )
        union!(wrapped_offset_set, axis_offsets)
    end
    return sort!(collect(wrapped_offset_set))
end

"""
用途: 为给定的 Jastrow 距离标签构造唯一无序 pair 集合。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。
- `dx::Int`: 位移标签的第一个分量, 可以是轴向 `x` 分量或未排序距离分量。
- `dy::Int`: 位移标签的第二个分量, 可以是轴向 `y` 分量或未排序距离分量。

返回:
- `Vector{Tuple{Int, Int}}`: 经 `i < j` 规范化并按字典序排序后的唯一 pair 列表。
"""
function build_jastrow_pair_set_for_displacement(
    lx::Int,
    ly::Int,
    dx::Int,
    dy::Int,
)::Vector{Tuple{Int,Int}}
    wrapped_offsets = build_jastrow_wrapped_offsets_for_displacement(lx, ly, dx, dy)
    unique_pairs = Set{Tuple{Int,Int}}()
    for x in 1:lx
        for y in 1:ly
            site_index = (x - 1) * ly + y
            for (offset_x, offset_y) in wrapped_offsets
                neighbor_x = mod(x - 1 + offset_x, lx) + 1
                neighbor_y = mod(y - 1 + offset_y, ly) + 1
                neighbor_index = (neighbor_x - 1) * ly + neighbor_y
                if neighbor_index == site_index
                    continue
                end
                push!(unique_pairs, (min(site_index, neighbor_index), max(site_index, neighbor_index)))
            end
        end
    end
    return sort!(collect(unique_pairs))
end

"""
用途: 为给定的 Jastrow 距离标签构造对称邻接表。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。
- `dx::Int`: 位移标签的第一个分量, 可以是轴向 `x` 分量或未排序距离分量。
- `dy::Int`: 位移标签的第二个分量, 可以是轴向 `y` 分量或未排序距离分量。

返回:
- `Vector{Vector{Int}}`: 满足无自环、无重复、对称的邻接表。
"""
function build_jastrow_neighbor_table_for_displacement(
    lx::Int,
    ly::Int,
    dx::Int,
    dy::Int,
)::Vector{Vector{Int}}
    unique_pairs = build_jastrow_pair_set_for_displacement(lx, ly, dx, dy)
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
用途: 根据全位移分类批量构造 Jastrow terms、参数名与默认初值。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。
- `default_value::Float64`: 每个 `vj_dx_dy` 的默认初值。

返回:
- `Tuple{Vector{JastrowProjectorTerm{Float64}}, Vector{Symbol}, Vector{Float64}}`:
  `(jastrow_terms, param_names, init_params)`。
"""
function build_full_displacement_jastrow_terms(
    lx::Int,
    ly::Int;
    default_value::Float64=0.0,
)::Tuple{Vector{JastrowProjectorTerm{Float64}},Vector{Symbol},Vector{Float64}}
    displacement_labels = build_jastrow_displacement_labels(lx, ly)
    jastrow_terms = JastrowProjectorTerm{Float64}[]
    param_names = Symbol[]
    init_params = Float64[]

    for (dx, dy) in displacement_labels
        param_name = build_jastrow_param_name(dx, dy)
        neighbor_table = build_jastrow_neighbor_table_for_displacement(lx, ly, dx, dy)
        push!(
            jastrow_terms,
            JastrowProjectorTerm(
                param_name=param_name,
                v=default_value,
                site_to_neighbor_sites=neighbor_table,
            ),
        )
        push!(param_names, param_name)
        push!(init_params, default_value)
    end

    return jastrow_terms, param_names, init_params
end

"""
用途: 构造受约束 Hubbard 主程序使用的 projector。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。
- `g::Float64`: Gutzwiller projector 参数。
- `jastrow_default_value::Float64`: 全位移 Jastrow 参数的默认初值。

返回:
- `CompositeProjector`: 由一个 Gutzwiller term 和全部 `vj_dx_dy` Jastrow terms 组成的 projector。
"""
function build_restricted_projector(
    lx::Int,
    ly::Int,
    g::Float64;
    jastrow_default_value::Float64=0.0,
)::CompositeProjector
    jastrow_terms, _, _ = build_full_displacement_jastrow_terms(
        lx,
        ly;
        default_value=jastrow_default_value,
    )
    projector_terms = AbstractProjectorTerm[
        GutzwillerProjectorTerm(param_name=:g, g=g),
    ]
    append!(projector_terms, jastrow_terms)
    return CompositeProjector(projector_terms)
end

"""
用途: 构造受约束 Hubbard 主程序使用的完整 Eq.(5) composite backflow。

数学公式:
- `U_b = U_0 + delta U_epsilon + delta U_eta1 + delta U_eta2 + delta U_eta3`。
- `delta U_epsilon(i, sigma) = (bf_epsilon - 1) * xi_{i,sigma} * U_0(i, sigma)`。
- `delta U_eta1(i, sigma) = bf_eta1 * sum_j t_ij * D_i * H_j * U_0(j, sigma)`。
- `delta U_eta2(i, sigma) = bf_eta2 * sum_j t_ij *
   n_i_sigma h_i_-sigma n_j_-sigma h_j_sigma * U_0(j, sigma)`。
- `delta U_eta3(i, sigma) = bf_eta3 * sum_j t_ij *
   (D_i n_j_-sigma h_j_sigma + n_i_sigma h_i_-sigma H_j) * U_0(j, sigma)`。
- `xi_{i,sigma}` 在 `eta1`, `eta2`, `eta3` 任一局域 virtual hopping 条件非零时取 `1`。

参数:
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 与有向键对齐的 hopping 振幅 `t_ij`。
- `bf_epsilon::Float64`: `epsilon` backflow 参数, 退化值为 `1.0`。
- `bf_eta1::Float64`: `eta1` backflow 参数, 退化值为 `0.0`。
- `bf_eta2::Float64`: `eta2` backflow 参数, 退化值为 `0.0`。
- `bf_eta3::Float64`: `eta3` backflow 参数, 退化值为 `0.0`。

返回:
- `CompositeBackflowTerm`: 按 `bf_epsilon, bf_eta1, bf_eta2, bf_eta3` 顺序排列的 backflow。
"""
function build_restricted_composite_backflow(
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{<:Real},
    bf_epsilon::Float64,
    bf_eta1::Float64,
    bf_eta2::Float64,
    bf_eta3::Float64,
)::CompositeBackflowTerm
    return CompositeBackflowTerm([
        BackflowEpsilonTerm(
            param_name=:bf_epsilon,
            epsilon_bf=bf_epsilon,
            epsilon_mask_terms=Symbol[:eta1, :eta2, :eta3],
            source_bonds=source_bonds,
            source_amplitudes=source_amplitudes,
        ),
        BackflowEta1DoublonHoleTerm(
            param_name=:bf_eta1,
            eta1_bf=bf_eta1,
            source_bonds=source_bonds,
            source_amplitudes=source_amplitudes,
        ),
        BackflowEta2SpinExchangeTerm(
            param_name=:bf_eta2,
            eta2_bf=bf_eta2,
            source_bonds=source_bonds,
            source_amplitudes=source_amplitudes,
        ),
        BackflowEta3MixedVirtualHopTerm(
            param_name=:bf_eta3,
            eta3_bf=bf_eta3,
            source_bonds=source_bonds,
            source_amplitudes=source_amplitudes,
        ),
    ])
end

"""
用途: 根据命令行开关选择 restricted Hubbard 使用的 backflow 对象。

参数:
- `enable_backflow::Bool`: 是否启用 backflow; `false` 时返回 `NoBackflowTerm()`。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 与有向键对齐的 hopping 振幅 `t_ij`。
- `bf_epsilon::Float64`: `epsilon` backflow 参数。
- `bf_eta1::Float64`: `eta1` backflow 参数。
- `bf_eta2::Float64`: `eta2` backflow 参数。
- `bf_eta3::Float64`: `eta3` backflow 参数。

返回:
- `AbstractBackflowTerm`: 启用时为 `CompositeBackflowTerm`; 禁用时为 `NoBackflowTerm`。
"""
function build_restricted_optional_backflow(
    enable_backflow::Bool,
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{<:Real},
    bf_epsilon::Float64,
    bf_eta1::Float64,
    bf_eta2::Float64,
    bf_eta3::Float64,
)::AbstractBackflowTerm
    if !enable_backflow
        return NoBackflowTerm()
    end

    return build_restricted_composite_backflow(
        source_bonds,
        source_amplitudes,
        bf_epsilon,
        bf_eta1,
        bf_eta2,
        bf_eta3,
    )
end

"""
用途: 解析命令行布尔开关字符串。

参数:
- `raw_value::AbstractString`: 命令行输入值, 支持 `true/false`, `1/0`, `yes/no`, `on/off`。
- `option_name::AbstractString`: 选项名, 用于错误信息。

返回:
- `Bool`: 解析后的布尔值。
"""
function parse_bool_flag(raw_value::AbstractString, option_name::AbstractString)::Bool
    normalized_value = lowercase(strip(raw_value))
    if normalized_value in ("true", "t", "1", "yes", "y", "on")
        return true
    elseif normalized_value in ("false", "f", "0", "no", "n", "off")
        return false
    end

    error("Invalid value for $(option_name): $(raw_value). Expected true/false, 1/0, yes/no, or on/off.")
end

"""
用途: 解析命令行输入的固定参数字符串。

参数:
- `fixed_params_string::AbstractString`: 固定参数字符串, 格式为 `"name=value,name=value"`。

返回:
- `Dict{Symbol, Float64}`: 参数名到固定值的映射。
"""
function parse_fixed_param_string(fixed_params_string::AbstractString)::Dict{Symbol,Float64}
    fixed_param_values = Dict{Symbol,Float64}()
    stripped_input = strip(fixed_params_string)
    if isempty(stripped_input)
        return fixed_param_values
    end

    for raw_assignment in split(stripped_input, ",")
        assignment = strip(raw_assignment)
        if isempty(assignment)
            continue
        end

        pieces = split(assignment, "=")
        if length(pieces) != 2
            error("Invalid fixed parameter assignment: $(assignment). Expected format name=value.")
        end

        param_name_string = strip(pieces[1])
        param_value_string = strip(pieces[2])
        if isempty(param_name_string) || isempty(param_value_string)
            error("Invalid fixed parameter assignment: $(assignment). Expected non-empty name and value.")
        end

        param_name = Symbol(param_name_string)
        if haskey(fixed_param_values, param_name)
            error("Duplicate fixed parameter assignment for $(param_name).")
        end

        try
            fixed_param_values[param_name] = parse(Float64, param_value_string)
        catch parse_error
            error("Invalid numeric value for fixed parameter $(param_name): $(param_value_string).")
        end
    end

    return fixed_param_values
end

"""
用途: 解析命令行输入的 active 参数名列表。

参数:
- `param_names_string::AbstractString`: 参数名字符串, 格式为 `"name,name"`; 空字符串表示不显式限制 active 参数。

返回:
- `Vector{Symbol}`: 按命令行顺序排列的参数名列表。
"""
function parse_param_name_list(param_names_string::AbstractString)::Vector{Symbol}
    stripped_input = strip(param_names_string)
    if isempty(stripped_input)
        return Symbol[]
    end

    param_names = Symbol[]
    seen_param_names = Set{Symbol}()
    for raw_name in split(stripped_input, ",")
        param_name_string = strip(raw_name)
        if isempty(param_name_string)
            error("Invalid active parameter list: empty parameter name in $(param_names_string).")
        end

        param_name = Symbol(param_name_string)
        if param_name in seen_param_names
            error("Duplicate active parameter name: $(param_name).")
        end
        push!(param_names, param_name)
        push!(seen_param_names, param_name)
    end

    return param_names
end

"""
用途: 用 json 中已有的参数覆盖默认初始参数, 缺失参数保留默认值。

参数:
- `json_path::AbstractString`: 参数 json 文件路径。
- `full_param_names::Vector{Symbol}`: 完整参数名列表。
- `default_param_values::Vector{Float64}`: 与 `full_param_names` 对齐的默认初始参数。

返回:
- `Vector{Float64}`: json 覆盖后的完整初始参数向量。
"""
function build_init_params_from_json_with_defaults(
    json_path::AbstractString,
    full_param_names::Vector{Symbol},
    default_param_values::Vector{Float64},
)::Vector{Float64}
    if length(full_param_names) != length(default_param_values)
        error("full_param_names and default_param_values length mismatch.")
    end
    if !isfile(json_path)
        error("JSON file not found: $(json_path)")
    end

    raw_param_dict = JSON.parsefile(json_path)
    param_index_map = Dict(String(param_name) => param_index for (param_index, param_name) in enumerate(full_param_names))
    init_param_values = copy(default_param_values)

    for (param_name_string, param_value) in raw_param_dict
        if !haskey(param_index_map, String(param_name_string))
            continue
        end
        if !(param_value isa Number)
            error("Invalid value for key $(param_name_string) in json: $(param_value)")
        end
        init_param_values[param_index_map[String(param_name_string)]] = Float64(param_value)
    end

    return init_param_values
end

"""
用途: 检查固定参数名是否都存在于完整参数列表中。

参数:
- `full_param_names::Vector{Symbol}`: 完整参数名列表。
- `fixed_param_values::Dict{Symbol, Float64}`: 参数名到固定值的映射。

返回:
- `nothing`。
"""
function validate_fixed_param_names!(
    full_param_names::Vector{Symbol},
    fixed_param_values::Dict{Symbol,Float64},
)::Nothing
    full_param_name_set = Set(full_param_names)
    for fixed_param_name in keys(fixed_param_values)
        if !(fixed_param_name in full_param_name_set)
            error("Unknown fixed parameter: $(fixed_param_name). Available parameters: $(join(String.(full_param_names), ", ")).")
        end
    end
    return nothing
end

"""
用途: 将固定参数值覆盖到完整初始参数向量中。

参数:
- `full_param_names::Vector{Symbol}`: 完整参数名列表。
- `param_values::Vector{Float64}`: 与 `full_param_names` 对齐的完整参数值。
- `fixed_param_values::Dict{Symbol, Float64}`: 参数名到固定值的映射。

返回:
- `Vector{Float64}`: 覆盖固定值后的完整参数值。
"""
function apply_fixed_params_to_values(
    full_param_names::Vector{Symbol},
    param_values::Vector{Float64},
    fixed_param_values::Dict{Symbol,Float64},
)::Vector{Float64}
    if length(full_param_names) != length(param_values)
        error("full_param_names and param_values length mismatch.")
    end
    validate_fixed_param_names!(full_param_names, fixed_param_values)

    updated_param_values = copy(param_values)
    for (param_index, param_name) in enumerate(full_param_names)
        if haskey(fixed_param_values, param_name)
            updated_param_values[param_index] = fixed_param_values[param_name]
        end
    end
    return updated_param_values
end

"""
用途: 根据固定参数集合构造参与 SR 优化的参数下标。

参数:
- `full_param_names::Vector{Symbol}`: 完整参数名列表。
- `fixed_param_values::Dict{Symbol, Float64}`: 参数名到固定值的映射。

返回:
- `Vector{Int}`: 未固定参数在完整参数列表中的下标。
"""
function build_active_param_indices(
    full_param_names::Vector{Symbol},
    fixed_param_values::Dict{Symbol,Float64},
)::Vector{Int}
    validate_fixed_param_names!(full_param_names, fixed_param_values)
    return [
        param_index for (param_index, param_name) in enumerate(full_param_names)
        if !haskey(fixed_param_values, param_name)
    ]
end

"""
用途: 根据固定参数集合和显式 active 参数名构造参与 SR 优化的参数下标。

参数:
- `full_param_names::Vector{Symbol}`: 完整参数名列表。
- `fixed_param_values::Dict{Symbol, Float64}`: 参数名到固定值的映射。
- `active_param_names::Vector{Symbol}`: 命令行指定的 active 参数名; 为空时表示所有未固定参数。

返回:
- `Vector{Int}`: active 参数在完整参数列表中的下标, 顺序跟随完整参数列表以匹配 `compute_grad_log_psi!`。
"""
function build_active_param_indices(
    full_param_names::Vector{Symbol},
    fixed_param_values::Dict{Symbol,Float64},
    active_param_names::Vector{Symbol},
)::Vector{Int}
    if isempty(active_param_names)
        return build_active_param_indices(full_param_names, fixed_param_values)
    end

    validate_fixed_param_names!(full_param_names, fixed_param_values)
    full_param_index_map = Dict(param_name => param_index for (param_index, param_name) in enumerate(full_param_names))
    for active_param_name in active_param_names
        if !haskey(full_param_index_map, active_param_name)
            error("Unknown active parameter: $(active_param_name). Available parameters: $(join(String.(full_param_names), ", ")).")
        end
        if haskey(fixed_param_values, active_param_name)
            error("Parameter $(active_param_name) cannot be both fixed and active.")
        end
    end

    active_param_name_set = Set(active_param_names)
    return [
        param_index for (param_index, param_name) in enumerate(full_param_names)
        if param_name in active_param_name_set
    ]
end

"""
用途: 在每次构造波函数前, 将完整参数向量中的固定参数重置为命令行给定值。

参数:
- `full_param_names::Vector{Symbol}`: 完整参数名列表。
- `param_values::Vector{Float64}`: 与 `full_param_names` 对齐的完整参数值, 会被原地修改。
- `fixed_param_values::Dict{Symbol, Float64}`: 参数名到固定值的映射。

返回:
- `Vector{Float64}`: 原地修改后的 `param_values`。
"""
function enforce_fixed_params!(
    full_param_names::Vector{Symbol},
    param_values::Vector{Float64},
    fixed_param_values::Dict{Symbol,Float64},
)::Vector{Float64}
    if length(full_param_names) != length(param_values)
        error("full_param_names and param_values length mismatch.")
    end
    validate_fixed_param_names!(full_param_names, fixed_param_values)

    for (param_index, param_name) in enumerate(full_param_names)
        if haskey(fixed_param_values, param_name)
            param_values[param_index] = fixed_param_values[param_name]
        end
    end
    return param_values
end

"""
用途: 检查固定参数是否都属于 mean-field 参数集合。

参数:
- `wf_param_names::Vector{Symbol}`: 当前 ansatz 的 mean-field 参数名列表。
- `fixed_param_values::Dict{Symbol, Float64}`: 参数名到固定值的映射。

返回:
- `nothing`。若固定了非 mean-field 参数会抛出异常。
"""
function validate_fixed_mean_field_params!(
    wf_param_names::Vector{Symbol},
    fixed_param_values::Dict{Symbol,Float64},
)::Nothing
    wf_param_name_set = Set(wf_param_names)
    for fixed_param_name in keys(fixed_param_values)
        if !(fixed_param_name in wf_param_name_set)
            error(
                "Only mean-field parameters can be fixed in this SR path. " *
                "Got $(fixed_param_name), available mean-field parameters are $(join(String.(wf_param_names), ", ")).",
            )
        end
    end
    return nothing
end

"""
用途: 将 SR 优化器中的 active 参数向量合并回完整参数向量。

参数:
- `full_param_template::Vector{Float64}`: 完整参数模板, 其中固定参数已经是固定值。
- `active_param_indices::Vector{Int}`: active 参数在完整参数向量中的下标。
- `active_param_values::Vector{Float64}`: SR 优化器当前持有的 active 参数值。

返回:
- `Vector{Float64}`: 合并后的完整参数向量。
"""
function merge_active_params_into_full(
    full_param_template::Vector{Float64},
    active_param_indices::Vector{Int},
    active_param_values::Vector{Float64},
)::Vector{Float64}
    if length(active_param_indices) != length(active_param_values)
        error("active_param_indices and active_param_values length mismatch.")
    end
    full_param_values = copy(full_param_template)
    for (active_offset, param_index) in enumerate(active_param_indices)
        if param_index < 1 || param_index > length(full_param_template)
            error("active parameter index $(param_index) is outside 1:$(length(full_param_template)).")
        end
        full_param_values[param_index] = active_param_values[active_offset]
    end
    return full_param_values
end

"""
用途: 将固定参数补写进 SR 产生的最优参数 JSON。

参数:
- `json_path::AbstractString`: `extract_min_energy` 生成的 JSON 文件路径。
- `fixed_param_values::Dict{Symbol, Float64}`: 参数名到固定值的映射。

返回:
- `nothing`。
"""
function append_fixed_params_to_json!(
    json_path::AbstractString,
    fixed_param_values::Dict{Symbol,Float64},
)::Nothing
    if isempty(fixed_param_values)
        return nothing
    end
    param_dict = JSON.parsefile(json_path)
    for (param_name, param_value) in fixed_param_values
        param_dict[String(param_name)] = param_value
    end
    open(json_path, "w") do io
        JSON.print(io, param_dict)
        println(io)
    end
    return nothing
end

"""
用途: 将未参与 SR 优化的参数补写进最优参数 JSON。

参数:
- `json_path::AbstractString`: `extract_min_energy` 生成的 JSON 文件路径。
- `full_param_names::Vector{Symbol}`: 完整参数名列表。
- `full_param_values::Vector{Float64}`: 与 `full_param_names` 对齐的完整参数值模板。
- `active_param_indices::Vector{Int}`: 参与 SR 优化的参数下标。

返回:
- `nothing`。
"""
function append_inactive_params_to_json!(
    json_path::AbstractString,
    full_param_names::Vector{Symbol},
    full_param_values::Vector{Float64},
    active_param_indices::Vector{Int},
)::Nothing
    if length(full_param_names) != length(full_param_values)
        error("full_param_names and full_param_values length mismatch.")
    end

    active_param_index_set = Set(active_param_indices)
    param_dict = JSON.parsefile(json_path)
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
用途: 根据参数更新 restricted Hubbard determinant 波函数和参数导数。

参数:
- `vwf`: determinant 波函数对象。
- `param_names::Vector{Symbol}`: 完整参数名列表, 顺序为 mean-field, projector, backflow。
- `params::Vector{Float64}`: 与 `param_names` 对齐的完整参数值。
- `lx, ly, bcx, bcy, target_sz`: 晶格尺寸、边界条件和 total Sz。
- `nparams_proj::Int`: projector 参数数量。
- `nparams_backflow::Int`: backflow 参数数量。
- `Q::Float64`: stripe 波矢。
- `x0::Float64`: stripe 中心偏移。
- `active_wf_param_names::Union{Nothing, Vector{Symbol}}`: 参与求导和 SR 的 mean-field 参数名; 为 `nothing` 时使用全部 mean-field 参数。

返回:
- `nothing`。
"""
function update_ansatz!(
    vwf,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx,
    ly,
    bcx,
    bcy,
    target_sz::Int;
    nparams_proj::Int=0,
    nparams_backflow::Int=0,
    Q::Float64=0.0,
    x0::Float64=0.0,
    active_wf_param_names::Union{Nothing,Vector{Symbol}}=nothing,
)
    # 支持输入为 wf 参数 + projector 参数 + backflow 参数的拼接向量
    nparms = length(param_names)
    nparams_wf = nparms - nparams_proj - nparams_backflow
    wf_param_names = param_names[1:nparams_wf]
    wf_param_values = params[1:nparams_wf]
    projector_param_names = param_names[(nparams_wf+1):(nparams_wf+nparams_proj)]
    projector_param_values = params[(nparams_wf+1):(nparams_wf+nparams_proj)]
    backflow_param_names = param_names[(nparams_wf+nparams_proj+1):end]
    backflow_param_values = params[(nparams_wf+nparams_proj+1):end]
    derivative_wf_param_names = active_wf_param_names === nothing ? wf_param_names : active_wf_param_names
    wf_param_name_set = Set(wf_param_names)
    for derivative_param_name in derivative_wf_param_names
        if !(derivative_param_name in wf_param_name_set)
            error("Active mean-field parameter $(derivative_param_name) is not in full mean-field parameter list.")
        end
    end
    # 这里也可以把 bcx, bcy 提出来作为参数
    param_map = Dict{Symbol,Float64}(zip(wf_param_names, wf_param_values))

    chi2 = get(param_map, :chi2, 0.0)
    etax = get(param_map, :etax, 0.0)
    etay = get(param_map, :etay, 0.0)
    mu = get(param_map, :mu, 0.0)
    Delta_AF = get(param_map, :Delta_AF, 0.0)
    Delta_c = get(param_map, :Delta_c, 0.0)
    Delta_s = get(param_map, :Delta_s, 0.0)

    hubbard_params = PartonSquare.RestrictedHubbardParams(
        Lx=lx,
        Ly=ly,
        bcx=bcx,
        bcy=bcy,
        chi1=1.0,
        chi2=chi2,
        etax=etax,
        etay=etay,
        mu=mu,
        Delta_AF=Delta_AF,
        Delta_c=Delta_c,
        Delta_s=Delta_s,
        Q=Q,
        x0=x0
    )

    _, gs_U, dUt_params = PartonSquare.make_ansatz_and_derivs(hubbard_params; param_names=derivative_wf_param_names, target_sz=target_sz, Q=Q, x0=x0)

    copyto!(vwf.base_gs_U, gs_U)
    copyto!(vwf.gs_U, gs_U)
    copyto!(vwf.backflow_u, gs_U)
    copyto!(vwf.gs_U_t, permutedims(gs_U))
    dUt_matrix = zeros(Float64, size(gs_U, 2), size(gs_U, 1), length(derivative_wf_param_names))
    for (idx, name) in enumerate(derivative_wf_param_names)
        dUt_matrix[:, :, idx] = dUt_params[name]
    end
    update_vwf_params!(vwf, derivative_wf_param_names, dUt_matrix)
    if !isempty(projector_param_names)
        update_vwf_projector_params!(vwf, projector_param_names, projector_param_values)
    end
    if !isempty(backflow_param_names)
        update_vwf_backflow_params!(vwf, backflow_param_names, backflow_param_values)
    end
    init_gswf!(vwf)
end

function build_exponential_lr_func(
    lr_start::Float64,
    lr_end::Float64,
    n_steps::Int
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

function defination_observabels(lx::Int, ly::Int)::Dict{Symbol,Function}
    observables = Dict{Symbol,Function}()
    observables[:E] = local_energy
    for x in 1:lx, y in 1:ly
        i = idx(x, y, lx, ly)
        key = Symbol("Sz_$(x)_$(y)")
        observables[key] = (model, vwf) -> begin
            val = get_Sz(vwf.sampler.state[i])
            return val
        end
        key = Symbol("n_$(x)_$(y)")
        observables[key] = (model, vwf) -> begin
            st = vwf.sampler.state[i]
            n_up = (st & UP) != 0 ? 1.0 : 0.0
            n_dn = (st & DN) != 0 ? 1.0 : 0.0
            return n_up + n_dn
        end
    end
    return observables
end
function idx(x::Int, y::Int, lx::Int, ly::Int)
    return mod(x - 1, lx) * ly + mod(y - 1, ly) + 1
end

# ==============================================================================
# 4. 主程序
# ==============================================================================

function main()
    args = parse_commandline()

    session = init_mpi_session()
    rank = session.rank
    is_root = (rank == session.root)

    # ---------------------------------------------------------
    # A. 参数设定 (全部集中在这里)
    # ---------------------------------------------------------
    lx = args["Lx"]
    ly = args["Ly"]
    BCX = args["bcx"]
    BCY = args["bcy"]
    target_sz = args["target_sz"]
    doping = args["doping"]
    lambda = args["lambda"]
    stripe_center = args["stripe_center"]
    nMC = args["nMC"]
    wMC = args["wMC"]
    rMC = args["rMC"]
    dMC = args["dMC"]
    seed = args["seed"]
    n_steps = args["nSR"]
    lr = args["lr"]
    lr_end = args["lr_end"]
    if isnan(lr_end)
        lr_end = lr
    end

    t1 = args["t1"]
    t2 = args["t2"]
    U = args["U"]
    job = args["job"]
    ansatz = args["ansatz"]
    g = args["g"]
    bf_epsilon = args["bf_epsilon"]
    bf_eta1 = args["bf_eta1"]
    bf_eta2 = args["bf_eta2"]
    bf_eta3 = args["bf_eta3"]
    enable_backflow = parse_bool_flag(args["enable_backflow"], "--enable_backflow")
    init_params_json = args["init_params_json"]
    fixed_params_string = args["fixed_params"]
    active_params_string = args["active_params"]
    N_sites = lx * ly
    #要优化的参数
    if ansatz == "AFM"
        wf_param_names = [:chi2, :etax, :etay, :Delta_AF, :mu]
        wf_init_params = [args["chi2"], args["etax"], args["etay"], args["Delta_AF"], args["mu"]]
        Q = 0.0
        x0 = 0.0
    elseif ansatz == "Stripe"
        wf_param_names = [:chi2, :etax, :etay, :Delta_c, :Delta_s, :mu]
        wf_init_params = [args["chi2"], args["etax"], args["etay"], args["Delta_c"], args["Delta_s"], args["mu"]]
        Q = 2π / lambda
        if stripe_center == "site"
            x0 = 0.0
        elseif stripe_center == "bond"
            x0 = 0.5
        else
            error("Unknown stripe_center type: $stripe_center")
        end
    else
        error("Unknown ansatz type: $ansatz")
    end
    # VMC 采样参数
    meas_params = VMCParams(
        total_samples=nMC,
        warmup_steps=wMC,
        rebuild_every=rMC,
        decorr_steps=dMC,
        seed=args["seed"] + rank
    )
    # ---------------------------------------------------------

    # B. 模型与波函数初始化
    #GeneralModel定义
    bonds1 = Tuple{Int,Int}[]
    bonds2 = Tuple{Int,Int}[]
    idx(x, y) = mod(x - 1, lx) * ly + mod(y - 1, ly) + 1
    for y in 1:ly, x in 1:lx
        u = idx(x, y)
        push!(bonds1, (u, idx(x + 1, y)))
        push!(bonds1, (u, idx(x, y + 1)))
        push!(bonds2, (u, idx(x + 1, y + 1)))
        push!(bonds2, (u, idx(x - 1, y + 1)))
    end

    backflow_source_bonds = Tuple{Int,Int}[]
    backflow_source_amplitudes = Float64[]
    for (site_i, site_j) in bonds1
        push!(backflow_source_bonds, (site_i, site_j))
        push!(backflow_source_amplitudes, t1)
        push!(backflow_source_bonds, (site_j, site_i))
        push!(backflow_source_amplitudes, t1)
    end
    for (site_i, site_j) in bonds2
        push!(backflow_source_bonds, (site_i, site_j))
        push!(backflow_source_amplitudes, t2)
        push!(backflow_source_bonds, (site_j, site_i))
        push!(backflow_source_amplitudes, t2)
    end

    # Projector 定义
    projector = build_restricted_projector(lx, ly, g)
    backflow = build_restricted_optional_backflow(
        enable_backflow,
        backflow_source_bonds,
        backflow_source_amplitudes,
        bf_epsilon,
        bf_eta1,
        bf_eta2,
        bf_eta3,
    )
    proj_param_names = projector_param_names(projector)
    proj_init_params = projector_param_values(projector)
    nparams_proj = length(proj_param_names)
    backflow_param_name_list = backflow_param_names(backflow)
    backflow_init_params = backflow_param_values(backflow)
    nparams_backflow = length(backflow_param_name_list)
    # 把波函数参数和投影算符参数拼接成一个向量, 供优化器使用
    init_params = vcat(wf_init_params, proj_init_params, backflow_init_params)
    param_names = vcat(wf_param_names, proj_param_names, backflow_param_name_list)

    if !isempty(init_params_json)
        init_params = build_init_params_from_json_with_defaults(init_params_json, param_names, init_params)
        if is_root
            println("Loaded initial parameters from json: $(init_params_json)")
        end
    end
    fixed_param_values = parse_fixed_param_string(fixed_params_string)
    requested_active_param_names = parse_param_name_list(active_params_string)
    validate_fixed_mean_field_params!(wf_param_names, fixed_param_values)
    if !isempty(fixed_param_values)
        init_params = apply_fixed_params_to_values(param_names, init_params, fixed_param_values)
    end
    active_param_indices = build_active_param_indices(param_names, fixed_param_values, requested_active_param_names)
    if job == "SR" && isempty(active_param_indices)
        error("At least one parameter must remain active for SR optimization.")
    end
    uses_param_subset = length(active_param_indices) != length(param_names) || !isempty(requested_active_param_names)
    sr_param_names = uses_param_subset ? param_names[active_param_indices] : param_names
    sr_init_params = uses_param_subset ? init_params[active_param_indices] : init_params
    wf_param_name_set = Set(wf_param_names)
    projector_param_name_set = Set(proj_param_names)
    backflow_param_name_set = Set(backflow_param_name_list)
    active_wf_param_names = [
        name for name in sr_param_names
        if name in wf_param_name_set
    ]
    active_projector_param_names = [
        name for name in sr_param_names
        if name in projector_param_name_set
    ]
    active_backflow_param_names = [
        name for name in sr_param_names
        if name in backflow_param_name_set
    ]
    set_active_sr_derivative_param_names!(
        proj_param_names,
        backflow_param_name_list;
        active_projector_param_names=uses_param_subset ? active_projector_param_names : nothing,
        active_backflow_param_names=uses_param_subset ? active_backflow_param_names : nothing,
    )

    terms = OperatorTerm[]
    for (i, j) in bonds1
        push!(terms, OperatorTerm([:cdag_up, :c_up], [i, j], -t1))
        push!(terms, OperatorTerm([:cdag_up, :c_up], [j, i], -t1))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [i, j], -t1))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [j, i], -t1))
    end
    for (i, j) in bonds2
        push!(terms, OperatorTerm([:cdag_up, :c_up], [i, j], -t2))
        push!(terms, OperatorTerm([:cdag_up, :c_up], [j, i], -t2))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [i, j], -t2))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [j, i], -t2))
    end
    for i in 1:N_sites
        push!(terms, OperatorTerm([:n_up, :n_dn], [i, i], U))
    end
    ham = GeneralModel(N_sites, terms)

    nelec = Int(N_sites * (1 + doping))
    #检查target_sz的parity
    @assert (target_sz + nelec) % 2 == 0 "Wrong parity!"
    nup = (nelec + target_sz) ÷ 2
    ndn = nelec - nup
    sampler = config_Hubbard(N_sites, nup, ndn; ifPH=true)
    init_config_Hubbard!(sampler)

    vwf = vwf_det(zeros(Float64, 2 * N_sites, N_sites + target_sz), sampler; backflow=backflow)
    set_projector!(vwf, projector)
    kernel = HubbardKernel(conserve_sz=true)

    # C. 更新波函数参数
    if rank == 0
        println("Initial parameters: $init_params")
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
    update_ansatz!(vwf, param_names, init_params, lx, ly, BCX, BCY, target_sz; nparams_proj=nparams_proj, nparams_backflow=nparams_backflow, Q=Q, x0=x0)


    # D. 运行模拟
    folder = "logs"
    mkpath(folder)

    if job == "SR"
        sr_params = SRParams(vmc_params=meas_params, n_steps=n_steps, lr=lr)
        exp_lr_func = build_exponential_lr_func(lr, lr_end, n_steps)

        update_vwf_func! = (vwf, params) -> begin
            full_params = uses_param_subset ?
                          merge_active_params_into_full(init_params, active_param_indices, params) :
                          params
            derivative_wf_param_names = uses_param_subset ? active_wf_param_names : nothing
            update_ansatz!(vwf, param_names, full_params, lx, ly, BCX, BCY, target_sz; nparams_proj=nparams_proj, nparams_backflow=nparams_backflow, Q=Q, x0=x0, active_wf_param_names=derivative_wf_param_names)
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
            lr_func=exp_lr_func
        )
        if is_root
            min_energy = extract_min_energy(joinpath(folder, "sr_history.txt"))
            append_inactive_params_to_json!(
                joinpath(folder, "min_params.json"),
                param_names,
                init_params,
                active_param_indices,
            )
        end
    elseif job == "measure"
        observables = defination_observabels(lx, ly)
        # 默认不保留历史, 如需阻塞法(Binning)请在此列出观测量名称
        history_observables = [:E]
        results = run_simulation(
            ham,
            vwf,
            kernel,
            observables,
            meas_params;
            history_observables=history_observables
        )
        if is_root && results !== nothing
            means = results[:means]
            mean_dict = Dict{Symbol,Any}()
            for (key, value) in means
                if value isa Number
                    mean_dict[key] = real(value)
                else
                    mean_dict[key] = value
                end
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
                            @printf(io, "%s\t%.10f\t%.10f\t%.6f\t%.6f\n",
                                String(name), mean_val, se_val, n_eff_val, tau_val)
                        else
                            println(io, "$(String(name))\t$(mean_val)\t$(se_val)\t$(n_eff_val)\t$(tau_val)")
                        end
                    end
                end
            end

            json_file = joinpath(folder, "block_binning_mean.json")
            mean_dict_str = Dict{String,Any}()
            for (key, value) in mean_dict
                mean_dict_str[String(key)] = value
            end
            open(json_file, "w") do io
                JSON.print(io, mean_dict_str)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
