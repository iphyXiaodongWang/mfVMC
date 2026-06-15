module Utils

using LinearAlgebra
using Statistics
using JSON
using MPI

using ..MPI_VMC_Utils: init_node_mpi_session

export compute_eig_and_dU_reg1, expand_spatial_to_spinful, add_term_ij_PH, add_term_ij_nonPH, add_term_ij_pfa_pairing
export pinv_derivative
export blocking_binning
export extract_min_energy
export build_init_params_from_json
export setup_emery_dense_derivative_workspace
export fill_emery_dense_derivative_slice!
export make_column_emery_dense_tensor_shared

"""
    build_init_params_from_json(
        json_path::AbstractString,
        param_names::Vector{Symbol}
    ) -> Vector{Float64}

用途: 根据参数名顺序从 json 文件构造初始参数列表.
参数:
- json_path::AbstractString, json 文件路径.
- param_names::Vector{Symbol}, 参数名列表.
返回:
- Vector{Float64}, 按 param_names 顺序排列的参数数值.
"""
function build_init_params_from_json(
    json_path::AbstractString,
    param_names::Vector{Symbol}
)::Vector{Float64}
    param_dict = read_params_from_json(json_path)
    init_params = Float64[]
    missing_keys = String[]

    for name in param_names
        key = String(name)
        if haskey(param_dict, key)
            push!(init_params, param_dict[key])
        else
            push!(missing_keys, key)
        end
    end

    if !isempty(missing_keys)
        error("Missing parameters in json: $(join(missing_keys, ", "))")
    end

    return init_params
end
function read_params_from_json(file_path::AbstractString)::Dict{String,Float64}
    if !isfile(file_path)
        error("JSON file not found: $(file_path)")
    end

    raw_dict = JSON.parsefile(file_path)
    param_dict = Dict{String,Float64}()

    for (key, value) in raw_dict
        if !(value isa Number)
            error("Invalid value for key $(key) in json: $(value)")
        end
        param_dict[String(key)] = Float64(value)
    end

    if isempty(param_dict)
        error("No valid key-value pairs found in json: $(file_path)")
    end

    return param_dict
end
function compute_eig_and_dU_reg1(H::AbstractMatrix{T}, H_alphas; eta::Real=1e-8) where T
    H_sym = Hermitian(H)
    ε, U = eigen(H_sym)

    diff_mat = ε .- ε' .+ im * eta
    F = -real.(1.0 ./ diff_mat)
    F[diagind(F)] .= 0.0

    keys_iter = keys(H_alphas)
    dE = Dict{eltype(keys_iter),Vector{Float64}}()
    dU = Dict{eltype(keys_iter),Matrix{ComplexF64}}()

    for k in keys_iter
        dH = H_alphas[k]

        dH_MO = U' * dH * U
        dE[k] = real.(diag(dH_MO))

        # dU = sum_{m!=n} |m> <m|dH|n> / (En - Em)
        dU[k] = U * (F .* dH_MO)
    end

    return ε, U, dE, dU
end

function expand_spatial_to_spinful(U_spatial::Matrix{T}) where T
    Nlat, Norb_spatial = size(U_spatial)
    U_spin = zeros(T, 2 * Nlat, 2 * Norb_spatial)
    U_spin[1:2:end, 1:Norb_spatial] = U_spatial[:, :]
    U_spin[2:2:end, Norb_spatial+1:end] = U_spatial[:, :]
    return U_spin
end

function add_term_ij_PH(tmat, i, j, chi, eta; singlet::Bool=true)
    sign = singlet ? 1 : -1
    @assert i != j
    tmat[(i-1)*2+1, (j-1)*2+1] += chi        # Julia 1-based
    tmat[(i-1)*2+2, (j-1)*2+2] -= conj(chi) * sign
    tmat[(i-1)*2+1, (j-1)*2+2] += eta
    tmat[(j-1)*2+1, (i-1)*2+2] += eta * sign
end
function add_term_ij_nonPH(tmat, i, j, chi)
    @assert i != j
    tmat[(i-1)*2+1, (j-1)*2+1] += chi        # Julia 1-based
    tmat[(i-1)*2+2, (j-1)*2+2] += chi
end
function add_term_ij_pfa_pairing(tmat, i, j, eta; singlet::Bool=true)
    sign = singlet ? 1 : -1
    @assert i != j
    tmat[(i-1)*2+1, (j-1)*2+2] += eta
    tmat[(j-1)*2+1, (i-1)*2+2] += eta * sign
end

function pinv_derivative(A::AbstractMatrix, dA::AbstractMatrix, pinvA::AbstractMatrix)
    # 转为浮点/复数矩阵，避免类型问题
    A = Matrix{Float64}(A)
    dA = Matrix{Float64}(dA)
    pinvA = Matrix{Float64}(pinvA)
    # 伪逆
    #pinvA = pinv(A; rtol=rtol)

    # 一些常用量
    pinvAH = pinvA'        # A^{+\dagger}
    dAH = dA'        # (∂A)^\dagger

    I_left = I - pinvA * A   # I - A^+ A
    I_right = I - A * pinvA   # I - A A^+

    # 三项公式
    term1 = -pinvA * dA * pinvA
    term2 = pinvA * pinvAH * dAH * I_right
    term3 = I_left * dAH * pinvAH * pinvA

    return term1 + term2 + term3
end
squeeze1(v) = (isa(v, AbstractVector) && length(v) == 1 ? v[1] : v)
"""
阻塞法估计均值标准误，同时给出 n_eff 与 tau_int 的粗估。

参数
------
x :: AbstractVector 或 AbstractMatrix
    若为向量，视作 (T,)；若为矩阵，视作 (T, D)，T 为样本数（时间轴），D 为观测维度。
min_blocks :: Int=16
    选用“块数 ≥ min_blocks”的最后一层作为估计。

返回
------
mean::Vector/Number, se::Vector/Number, n_eff::Vector/Number, tau_int::Vector/Number, meta::Dict
"""
function blocking_binning(x::AbstractArray; min_blocks::Int=16)
    # 规范成 T×D 的矩阵
    y = ndims(x) == 1 ? reshape(float.(x), :, 1) :
        (ndims(x) == 2 ? float.(x) :
         throw(ArgumentError("x must be 1D or 2D")))
    Tlen, D = size(y)

    # 原始层：naive 方差与均值
    var0 = vec(var(y; dims=1, corrected=true))   # 长度 D
    meanv = vec(mean(y; dims=1))                  # 长度 D

    levels_se = Vector{Vector{Float64}}()
    levels_n = Int[]
    levels_b = Int[]
    k = 0

    # 至少保留 2*min_blocks 个样本才能继续阻塞
    while size(y, 1) >= 2 * min_blocks
        n = size(y, 1)
        varv = vec(var(y; dims=1, corrected=true))
        se = sqrt.(varv ./ n)
        push!(levels_se, se)
        push!(levels_n, n)
        push!(levels_b, 2^k)

        # 两两平均到下一层（若 n 为奇数，忽略最后一个）
        if isodd(n)
            n -= 1
        end
        y = 0.5 .* (y[1:2:n, :] .+ y[2:2:n, :])
        k += 1
    end

    if isempty(levels_se)
        # 数据太短：退化为 std/sqrt(T)
        se = sqrt.(var0 ./ Tlen)
        tau_int = fill(0.5, D)                # 2*tau_int = 1
        n_eff = fill(float(Tlen), D)
        meta = Dict(
            :se_levels => permutedims(se),     # 1×D
            :n_levels => [Tlen],
            :b_levels => [1],
            :n_last => Tlen,
            :b_last => 1,
            :fallback => true,
        )
        return squeeze1(meanv), squeeze1(se), squeeze1(n_eff), squeeze1(tau_int), meta
    end

    # 选用“最后一个仍满足块数 ≥ min_blocks”的层
    se = levels_se[end]
    n_last = levels_n[end]
    b_last = levels_b[end]

    var_true = se .^ 2
    var_naiv = var0 ./ Tlen
    fac = similar(var_true)
    @inbounds for i in eachindex(fac)
        fac[i] = var_naiv[i] > 0 ? var_true[i] / var_naiv[i] : NaN
    end
    tau_int = 0.5 .* fac
    n_eff = [f > 0 ? Tlen / f : 0.0 for f in fac]

    # 把每层的 se（D 向量）堆成 (L×D) 矩阵
    se_levels_mat = vcat(permutedims.(levels_se)...)

    meta = Dict(
        :se_levels => se_levels_mat,  # 大小: nlevels × D
        :n_levels => levels_n,
        :b_levels => levels_b,
        :n_last => n_last,
        :b_last => b_last,
        :fallback => false,
    )
    return squeeze1(meanv), squeeze1(se), squeeze1(n_eff), squeeze1(tau_int), meta
end

"""
阻塞法估计字典中每个观测量的均值与标准误.

参数
------
x_dict :: AbstractDict
    字典的值必须是 AbstractVector 或 AbstractMatrix, 视作单个观测量的采样序列.
min_blocks :: Int=16
    选用“块数 ≥ min_blocks”的最后一层作为估计.

返回
------
mean_dict, se_dict, n_eff_dict, tau_int_dict, meta_dict
    每个返回值都是与 x_dict 同键的字典.
"""
function blocking_binning(x_dict::AbstractDict; min_blocks::Int=16)
    mean_dict = Dict{eltype(keys(x_dict)),Any}()
    se_dict = Dict{eltype(keys(x_dict)),Any}()
    n_eff_dict = Dict{eltype(keys(x_dict)),Any}()
    tau_int_dict = Dict{eltype(keys(x_dict)),Any}()
    meta_dict = Dict{eltype(keys(x_dict)),Any}()

    for (key, value) in x_dict
        if !(value isa AbstractVector || value isa AbstractMatrix)
            error("Value for key $(key) must be AbstractVector or AbstractMatrix.")
        end
        mean_v, se_v, n_eff_v, tau_int_v, meta_v = blocking_binning(value; min_blocks=min_blocks)
        mean_dict[key] = mean_v
        se_dict[key] = se_v
        n_eff_dict[key] = n_eff_v
        tau_int_dict[key] = tau_int_v
        meta_dict[key] = meta_v
    end

    return mean_dict, se_dict, n_eff_dict, tau_int_dict, meta_dict
end

const DEFAULT_OUTPUT_NAME = "min_params.json"

"""
    parse_header_line(header_line::AbstractString) -> Vector{String}

用途: 解析 sr_history 文件的表头, 返回参数名列表.
参数:
- header_line::AbstractString, 以 "#" 开头的表头行.
返回:
- Vector{String}, 参数名列表(不包含 Step/E_mean/E_err/GradNorm).
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
参数:
- line::AbstractString, 数据行(以空白分隔).
返回:
- Vector{Float64}, 当前行的数值序列.
"""
function parse_data_line(line::AbstractString)::Vector{Float64}
    tokens = split(strip(line), r"\s+")
    return parse.(Float64, tokens)
end

"""
    read_sr_history(file_path::AbstractString) -> (param_names::Vector{String}, rows::Vector)

用途: 读取 sr_history 文件, 提取参数名与每步数据.
参数:
- file_path::AbstractString, sr_history 文件路径.
返回:
- (param_names, rows), param_names 为参数名列表, rows 为包含 step/e_mean/params 的记录数组.
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
参数:
- rows, 由 read_sr_history 返回的记录数组.
返回:
- (min_energy, min_step, min_params).
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
参数:
- param_names::Vector{String}, 参数名列表.
- params::Vector{Float64}, 参数值列表.
返回:
- Dict{String, Float64}, 参数名到数值的映射.
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

用途: 将参数字典写入 JSON 文件.
参数:
- output_path::AbstractString, 输出路径.
- param_names::Vector{String}, 参数名顺序(用于保持接口一致, 当前不参与写入).
- param_dict::Dict{String,Float64}, 参数字典.
返回:
- Nothing.
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
    extract_min_energy(input_path::AbstractString; output_path::Union{Nothing,AbstractString}=nothing) -> Nothing

用途: 读取 SR 输出文件并生成最小能量对应的参数 JSON.
参数:
- input_path::AbstractString, sr_history 文件路径.
- output_path::Union{Nothing,AbstractString}, 输出路径. 若为 nothing, 默认写到与输入同目录的 min_params.json.
返回:
- Nothing.
"""
function extract_min_energy(
    input_path::AbstractString;
    output_path::Union{Nothing,AbstractString}=nothing
)
    final_output_path = isnothing(output_path) ?
                        joinpath(dirname(input_path), DEFAULT_OUTPUT_NAME) :
                        output_path

    param_names, rows = read_sr_history(input_path)
    min_energy, min_step, min_params = find_min_energy(rows)
    param_dict = build_param_dict(param_names, min_params)
    write_json_file(final_output_path, param_names, param_dict)

    println("Min energy: $(min_energy)")
    println("Min step: $(min_step)")
    println("Saved params to: $(final_output_path)")
    return min_energy
end

"""
用途: 构造 MPI 共享内存中的 dUt 3D 张量 workspace, 同节点所有 rank 共享一份物理内存.

参数:
- session: MPISession, 全局 MPI session.
- nocc::Int: 占据轨道数.
- nspin::Int: spinful 单粒子基底维度.
- nparams::Int: mean-field 参数数量.

返回:
- 若 nparams <= 0 或单 rank 运行, 返回 nothing.
- 否则返回 NamedTuple (session, session_shm, win, shared_tensor).
  shared_tensor 维度为 (nocc, nspin, nparams), 同节点多 rank 共享物理内存.
"""
function setup_emery_dense_derivative_workspace(
    session,
    nocc::Integer,
    nspin::Integer,
    nparams::Integer,
)
    Int(nparams) <= 0 && return nothing
    session_size = hasproperty(session, :size) ? session.size : 1
    session_size <= 1 && return nothing

    session_shm = init_node_mpi_session(session)
    local_length = session_shm.rank == 0 ? Int(nocc) * Int(nspin) * Int(nparams) : 0
    win, _ = MPI.Win_allocate_shared(Ptr{Float64}, local_length, session_shm.comm)
    shared_tensor_storage = MPI.Win_shared_query(
        Array{Float64},
        (Int(nocc), Int(nspin), Int(nparams)),
        win;
        rank=0,
    )
    shared_tensor = @view shared_tensor_storage[:, :, :]
    return (
        session=session,
        session_shm=session_shm,
        win=win,
        shared_tensor=shared_tensor,
    )
end

"""
用途: 计算单个 mean-field 参数的 dU/dp 切片, 写入共享张量的一个切片.

参数:
- dest::AbstractMatrix{Float64}: 输出, 形状为 (nocc, nspin).
- params: mean-field 参数对象, 传给 build_dh_dparam 的第二个参数.
- param_name::Symbol: 参数名.
- U_full::AbstractMatrix: 完整本征矢矩阵 (nspin × nspin), 实矩阵.
- response_F::AbstractMatrix: 响应矩阵 F[mo, occ] = -1/(epsilon_mo - epsilon_occ).
- nocc::Integer: 占据轨道数.
- build_dh_dparam::Function: (params, param_name) -> Matrix{Float64}, 构造 dH/dp.

公式:
- dH = build_dh_dparam(params, param_name)
- dH_MO = U_full' * dH * U_full
- dU = U_full * (response_F .* dH_MO)
- dest .= permutedims(real.(dU[:, 1:nocc]))
"""
function fill_emery_dense_derivative_slice!(
    dest::AbstractMatrix{Float64},
    params,
    param_name::Symbol,
    U_full::AbstractMatrix,
    response_F::AbstractMatrix,
    nocc::Integer,
    build_dh_dparam::Function,
)
    nocc_int = Int(nocc)
    if size(dest, 1) != nocc_int || size(dest, 2) != size(U_full, 1)
        throw(DimensionMismatch(
            "dense derivative slice shape $(size(dest)) does not match " *
            "(nocc=$nocc_int, nspin=$(size(U_full,1)))."
        ))
    end

    dH = build_dh_dparam(params, param_name)
    dH_MO = U_full' * dH * U_full
    dU = U_full * (response_F .* dH_MO)
    dest .= permutedims(real.(dU[:, 1:nocc_int]))
    return dest
end

"""
用途: 构造 column-resolved nonPH Emery determinant 的占据轨道和参数导数,
使用 MPI 共享内存避免多 rank 重复存储.

参数:
- params: mean-field 参数对象.
- param_names::Vector{Symbol}: 需要求导的 mean-field 参数名.
- n_occupied_orbitals::Int: 占据轨道数.
- workspace: setup_emery_dense_derivative_workspace 返回的对象, 或 nothing.
- nspin::Int: spinful 单粒子基底维度.
- build_hamiltonian::Function: (params) -> Matrix{Float64}, 构造 H.
- build_dh_dparam::Function: (params, param_name) -> Matrix{Float64}, 构造 dH/dp.

返回:
- (U_occ::Matrix{Float64}, dUt_tensor::AbstractArray{Float64,3}).

实现:
- 若 workspace === nothing (单 rank), 各 rank 独立对角化并分配本地 tensor.
- 否则只有 root rank 对角化, 广播结果, 分布式填充共享 tensor.
"""
function make_column_emery_dense_tensor_shared(
    params;
    param_names::Vector{Symbol},
    n_occupied_orbitals::Int,
    workspace,
    nspin::Int,
    build_hamiltonian::Function,
    build_dh_dparam::Function,
)
    nocc = Int(n_occupied_orbitals)
    nparam = length(param_names)
    nparam > 0 || error("param_names must not be empty.")

    if workspace === nothing
        # 单 rank 路径: 不使用共享内存, 各 rank 独立分配.
        hamiltonian = build_hamiltonian(params)
        H_sym = Hermitian(hamiltonian)
        evals, U_full = eigen(H_sym)
        U_full = real.(U_full)

        response_F = _build_response_F(evals)

        dUt_tensor = zeros(Float64, nocc, nspin, nparam)
        for (param_idx, param_name) in enumerate(param_names)
            fill_emery_dense_derivative_slice!(
                @view(dUt_tensor[:, :, param_idx]),
                params,
                param_name,
                U_full,
                response_F,
                nocc,
                build_dh_dparam,
            )
        end
        return (U_occ=copy(U_full[:, 1:nocc]), dUt_tensor=dUt_tensor)
    end

    # MPI 共享内存路径.
    session = workspace.session
    world_rank = session.rank
    world_size = session.size
    world_root = session.root

    evals = nothing
    U_full = nothing
    if world_rank == world_root
        hamiltonian = build_hamiltonian(params)
        H_sym = Hermitian(hamiltonian)
        evals, U_full = eigen(H_sym)
        U_full = real.(U_full)
    end
    evals = MPI.bcast(evals, world_root, session.comm)
    U_full = MPI.bcast(U_full, world_root, session.comm)

    response_F = _build_response_F(evals)

    for param_idx in (world_rank + 1):world_size:nparam
        fill_emery_dense_derivative_slice!(
            @view(workspace.shared_tensor[:, :, param_idx]),
            params,
            param_names[param_idx],
            U_full,
            response_F,
            nocc,
            build_dh_dparam,
        )
    end
    MPI.Barrier(workspace.session_shm.comm)

    return (U_occ=copy(U_full[:, 1:nocc]), dUt_tensor=workspace.shared_tensor)
end

"""
用途: 从本征值计算响应矩阵 F.

公式:
- F[mo, occ] = -real(1 / (epsilon_mo - epsilon_occ + i*eta))  for mo != occ
- F[mo, occ] = 0  for mo == occ

参数:
- evals::AbstractVector{<:Real}: 本征值向量.
- eta::Real=1e-8: 正则化小量.

返回:
- Matrix{Float64}: 响应矩阵.
"""
function _build_response_F(evals::AbstractVector{<:Real}; eta::Real=1e-8)
    diff_mat = evals .- evals' .+ im * Float64(eta)
    F = -real.(1.0 ./ diff_mat)
    F[diagind(F)] .= 0.0
    return F
end

end
