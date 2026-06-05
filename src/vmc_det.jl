# ==============================================================================
# WorkSpace
# ==============================================================================
mutable struct R1R2WS{T}
    N::Int
    dr1::Vector{T}   # δr1 = newF1 - A[i,:]
    dr2::Vector{T}   # δr2 = newF2 - A[j,:]
    col1::Vector{T}  # col1 = Ainv' * δr1
    col2::Vector{T}  # col2 = Ainv' * δr2
    xi::Vector{T}    # xi = copy(Ainv[:, i])
    xj::Vector{T}    # xj = copy(Ainv[:, j])
    s0::Vector{T}    # 1st column of S
    s1::Vector{T}    # 2nd column of S

    has_cached_rankk_update::Bool
    cached_changed_count::Int
    cached_affected_count::Int
    cached_changed_electron_ids::Vector{Int}
    cached_changed_row_indices::Vector{Int}
    cached_affected_site_indices::Vector{Int}
    cached_sorted_electron_ids::Vector{Int}
    cached_sorted_row_indices::Vector{Int}
    cached_permutation::Vector{Int}
    cached_new_columns::Matrix{T}
    cached_c_matrix::Matrix{T}
    cached_small_k_matrix::Matrix{T}
    cached_small_k_inverse::Matrix{T}
    cached_site_block_buffer::Matrix{T}
    cached_affected_site_blocks::Matrix{T}

    backflow_chain_rule_buffer::Matrix{T}
    grad_buffer::Vector{T}
end

# ==============================================================================
# Wavefunction: vwf_det (Generic T) 
# ==============================================================================
mutable struct vwf_det{T,S}
    # -- Matrices --
    base_gs_U::Matrix{T}
    gs_U::Matrix{T}
    gs_U_t::Matrix{T}
    backflow_u::Matrix{T}

    awf_mat_t::Matrix{T}
    awf_inv::Matrix{T}
    awf_val::T

    current_ratio::T

    sampler::S
    projector::Projector.CompositeProjector
    backflow::Backflow.AbstractBackflowTerm
    backflow_debug_verify::Bool

    # -- Workspace --
    ws::R1R2WS{T}
    dUt_matrix::AbstractArray{T,3}
    param_keys::Vector{Symbol}
end

"""
用途: 构造默认的空 projector 容器。

参数:
- 无。

返回:
- `Projector.CompositeProjector`, 包含一个 `NoProjectorTerm`。
"""
function build_default_projector()
    projector = Projector.CompositeProjector([Projector.NoProjectorTerm()])
    Projector.check_projector_consistency(projector)
    return projector
end


"""
用途: 构造默认的空 backflow 对象。

参数:
- 无。

返回:
- `Backflow.NoBackflowTerm`, 表示当前 determinant 不启用 backflow。
"""
function build_default_backflow()
    return Backflow.NoBackflowTerm()
end

function vwf_det(
    U::Matrix{T},
    sampler;
    projector::Projector.CompositeProjector=build_default_projector(),
    backflow::Backflow.AbstractBackflowTerm=build_default_backflow(),
) where T
    Nlat = sampler.N_sites
    expected_rows = 2 * Nlat
    expected_cols = total_elec(sampler)

    @assert size(U, 1) == expected_rows "U rows $(size(U,1)) != 2*Nlat"
    @assert size(U, 2) >= expected_cols "U cols $(size(U,2)) < Nelec"

    dummy_ws = R1R2WS{T}(
        0,
        T[],
        T[],
        T[],
        T[],
        T[],
        T[],
        T[],
        T[],
        false,
        0,
        0,
        Int[],
        Int[],
        Int[],
        Int[],
        Int[],
        Int[],
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, 0, 0),
        T[],
    )

    nelec = expected_cols
    awf_mat_t = zeros(T, nelec, nelec)
    awf_inv = zeros(T, 1, 1)
    dUt_matrix = zeros(T, 1, 1, 1)
    param_keys = Vector{Symbol}()
    Projector.check_projector_consistency(projector)
    base_gs_u = copy(U)
    gs_u = copy(U)
    gs_u_t = permutedims(U)
    backflow_u = copy(U)

    return vwf_det{T,typeof(sampler)}(
        base_gs_u,      # base_gs_U
        gs_u,           # gs_U
        gs_u_t,         # gs_U_t
        backflow_u,     # backflow_u
        awf_mat_t,      # awf_mat_t
        awf_inv,        # awf_inv (placeholder)
        T(0),
        one(T),
        sampler,
        projector,
        backflow,
        false,
        dummy_ws,
        dUt_matrix,
        param_keys
    )
end

"""
用途: 清空当前 proposal 的 rank-k cache 有效标记与活动列数。

参数:
- `ws::R1R2WS`: determinant workspace。

返回:
- `nothing`。
"""
function reset_cached_rankk_update!(ws::R1R2WS)
    ws.has_cached_rankk_update = false
    ws.cached_changed_count = 0
    ws.cached_affected_count = 0
    return nothing
end

function ensure_ws!(v::vwf_det{T,S}) where {T,S}
    N = size(v.awf_mat_t, 1)
    n_sites = div(size(v.base_gs_U, 1), 2)
    ws = v.ws
    if ws.N != N
        ws = R1R2WS{T}(
            N,
            Vector{T}(undef, N), Vector{T}(undef, N),
            Vector{T}(undef, N), Vector{T}(undef, N),
            Vector{T}(undef, N), Vector{T}(undef, N),
            Vector{T}(undef, N), Vector{T}(undef, N),
            false,
            0,
            0,
            Vector{Int}(undef, N),
            Vector{Int}(undef, N),
            Vector{Int}(undef, n_sites),
            Vector{Int}(undef, N),
            Vector{Int}(undef, N),
            Vector{Int}(undef, N),
            Matrix{T}(undef, N, N),
            Matrix{T}(undef, N, N),
            Matrix{T}(undef, N, N),
            Matrix{T}(undef, N, N),
            Matrix{T}(undef, 2, N),
            Matrix{T}(undef, 2 * n_sites, N),
            Matrix{T}(undef, 2 * n_sites, N),
            T[]
        )
        reset_cached_rankk_update!(ws)
        v.ws = ws
    end

    if size(ws.backflow_chain_rule_buffer) != size(v.base_gs_U)
        ws.backflow_chain_rule_buffer = Matrix{T}(undef, size(v.base_gs_U))
    end

    n_params = length(v.param_keys) + Projector.projector_param_count(v.projector) + Backflow.backflow_param_count(v.backflow)
    if length(ws.grad_buffer) != n_params
        resize!(ws.grad_buffer, n_params)
    end

    return ws
end



function update_vwf_params!(vwf::vwf_det{T}, param_names::Vector{Symbol}, dUt_matrix::AbstractArray{T,3}) where T

    empty!(vwf.param_keys)

    for name in param_names
        push!(vwf.param_keys, name)
    end

    vwf.dUt_matrix = dUt_matrix
    ensure_ws!(vwf)
    return nothing
end


"""
用途: 为 determinant 波函数设置 projector 对象。

参数:
- `vwf::vwf_det`: determinant 波函数对象。
- `projector::Projector.CompositeProjector`: 新的 projector 容器。

返回:
- `nothing`。
"""
function set_projector!(vwf::vwf_det, projector::Projector.CompositeProjector)
    Projector.check_projector_consistency(projector)
    vwf.projector = projector
    ensure_ws!(vwf)
    return nothing
end


"""
用途: 为 determinant 波函数设置 backflow 对象, 并立即刷新当前有效轨道矩阵。

参数:
- `vwf::vwf_det`: determinant 波函数对象。
- `backflow::Backflow.AbstractBackflowTerm`: 新的 backflow 对象。

返回:
- `nothing`。
"""
function set_backflow!(vwf::vwf_det, backflow::Backflow.AbstractBackflowTerm)
    vwf.backflow = backflow
    refresh_backflow_orbitals!(vwf)
    ensure_ws!(vwf)
    return nothing
end


"""
用途: 更新 determinant 波函数中 projector 的参数。

参数:
- `vwf::vwf_det`: determinant 波函数对象。
- `param_names::Vector{Symbol}`: projector 参数名列表。
- `param_values::Vector{<:Real}`: 与参数名对应的参数值列表。

返回:
- `nothing`。
"""
function update_vwf_projector_params!(
    vwf::vwf_det,
    param_names::Vector{Symbol},
    param_values::Vector{<:Real},
)
    Projector.update_projector_params!(vwf.projector, param_names, param_values)
    ensure_ws!(vwf)
    return nothing
end


"""
用途: 按 projector 内部顺序更新 determinant 波函数中的 projector 参数。

参数:
- `vwf::vwf_det`: determinant 波函数对象。
- `param_values::Vector{<:Real}`: 按内部顺序排列的参数值列表。

返回:
- `nothing`。
"""
function update_vwf_projector_params!(
    vwf::vwf_det,
    param_values::Vector{<:Real},
)
    Projector.update_projector_params!(vwf.projector, param_values)
    ensure_ws!(vwf)
    return nothing
end


"""
用途: 更新 determinant 波函数中的 backflow 参数。

参数:
- `vwf::vwf_det`: determinant 波函数对象。
- `param_names::Vector{Symbol}`: backflow 参数名列表。
- `param_values::Vector{<:Real}`: 与参数名对应的参数值列表。

返回:
- `nothing`。
"""
function update_vwf_backflow_params!(
    vwf::vwf_det,
    param_names::Vector{Symbol},
    param_values::Vector{<:Real},
)
    Backflow.update_backflow_params!(vwf.backflow, param_names, param_values)
    refresh_backflow_orbitals!(vwf)
    ensure_ws!(vwf)
    return nothing
end


"""
用途: 按 backflow 内部顺序更新 determinant 波函数中的 backflow 参数。

参数:
- `vwf::vwf_det`: determinant 波函数对象。
- `param_values::Vector{<:Real}`: 按内部顺序排列的参数值列表。

返回:
- `nothing`。
"""
function update_vwf_backflow_params!(
    vwf::vwf_det,
    param_values::Vector{<:Real},
)
    Backflow.update_backflow_params!(vwf.backflow, param_values)
    refresh_backflow_orbitals!(vwf)
    ensure_ws!(vwf)
    return nothing
end


"""
用途: 获取 determinant 波函数中 projector 的参数名列表。

参数:
- `vwf::vwf_det`: determinant 波函数对象。

返回:
- `Vector{Symbol}`: projector 参数名列表。
"""
function get_vwf_projector_param_names(vwf::vwf_det)
    return Projector.projector_param_names(vwf.projector)
end


"""
用途: 获取 determinant 波函数中 projector 的参数值列表。

参数:
- `vwf::vwf_det`: determinant 波函数对象。

返回:
- `Vector{Float64}`: projector 参数值列表。
"""
function get_vwf_projector_param_values(vwf::vwf_det)
    return Projector.projector_param_values(vwf.projector)
end


"""
用途: 获取 determinant 波函数中 backflow 的参数名列表。

参数:
- `vwf::vwf_det`: determinant 波函数对象。

返回:
- `Vector{Symbol}`: backflow 参数名列表。
"""
function get_vwf_backflow_param_names(vwf::vwf_det)
    return Backflow.backflow_param_names(vwf.backflow)
end


"""
用途: 获取 determinant 波函数中 backflow 的参数值列表。

参数:
- `vwf::vwf_det`: determinant 波函数对象。

返回:
- `Vector{Float64}`: backflow 参数值列表。
"""
function get_vwf_backflow_param_values(vwf::vwf_det)
    return Backflow.backflow_param_values(vwf.backflow)
end


"""
用途: 获取 determinant 波函数的总参数名列表。

拼接顺序:
- 先返回波函数参数 `vwf.param_keys`;
- 再返回 projector 参数 `projector_param_names(vwf.projector)`。

参数:
- `vwf::vwf_det`: determinant 波函数对象。

返回:
- `Vector{Symbol}`: 总参数名列表。
"""
function get_vwf_total_param_names(vwf::vwf_det)
    wf_names = copy(vwf.param_keys)
    proj_names = Projector.projector_param_names(vwf.projector)
    backflow_names = Backflow.backflow_param_names(vwf.backflow)
    return vcat(wf_names, proj_names, backflow_names)
end


"""
用途: 根据当前采样构型刷新 determinant 使用的有效轨道矩阵。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。

返回:
- `nothing`。
"""
function refresh_backflow_orbitals!(vwf::vwf_det{T}) where {T}
    if Backflow.uses_backflow(vwf.backflow)
        refreshed_orbitals = Backflow.build_backflow_orbitals(vwf.base_gs_U, vwf.sampler.state, vwf.backflow)
        if size(vwf.backflow_u) != size(refreshed_orbitals)
            vwf.backflow_u = similar(refreshed_orbitals)
        end
        copyto!(vwf.backflow_u, refreshed_orbitals)
        copyto!(vwf.gs_U, vwf.backflow_u)
    else
        copyto!(vwf.backflow_u, vwf.base_gs_U)
        copyto!(vwf.gs_U, vwf.base_gs_U)
    end

    permutedims!(vwf.gs_U_t, vwf.gs_U, (2, 1))
    return nothing
end


"""
用途: 根据给定轨道矩阵与电子位置列表构造 Slater 方阵。

参数:
- `orbitals::AbstractMatrix{T}`: 轨道矩阵, 行对应基底索引, 列对应轨道。
- `electron_locs::Vector{Int}`: 电子所在的内部基底索引列表。

返回:
- `Matrix{T}`: 形状为 `(N_elec, N_elec)` 的 Slater 方阵。
"""
function build_slater_matrix_from_orbitals(
    orbitals::AbstractMatrix{T},
    electron_locs::Vector{Int},
) where {T}
    return Matrix{T}(orbitals[electron_locs, :])
end


"""
用途: 根据当前采样构型完整重建 determinant 的 Slater 矩阵、行列式与逆矩阵。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。

返回:
- `nothing`。
"""
function rebuild_slater_state!(vwf::vwf_det{T,S}) where {T,S}
    ss = vwf.sampler
    total_elec_count = total_elec(ss)
    refresh_backflow_orbitals!(vwf)

    if size(vwf.awf_mat_t, 1) != total_elec_count
        vwf.awf_mat_t = zeros(T, total_elec_count, total_elec_count)
    end

    for i in 1:total_elec_count
        row_in_U = ss.electron_locs[i]
        copyto!(@view(vwf.awf_mat_t[:, i]), @view(vwf.gs_U_t[:, row_in_U]))
    end

    A_physical = transpose(vwf.awf_mat_t)
    F = lu(A_physical)

    vwf.awf_val = det(F)
    vwf.awf_inv = inv(F)
    vwf.current_ratio = one(T)

    ensure_ws!(vwf)
    reset_cached_rankk_update!(vwf.ws)
    return nothing
end

function init_gswf!(vwf::vwf_det{T,S}) where {T,S}
    ss = vwf.sampler
    initialize_lists!(ss)
    rebuild_slater_state!(vwf)
    return nothing
end

function rebuild_inverse!(vwf::vwf_det)
    if Backflow.uses_backflow(vwf.backflow)
        rebuild_slater_state!(vwf)
        return nothing
    end
    vwf.awf_inv = inv(transpose(vwf.awf_mat_t))
end

@inline function rank1_update_blas!(A::Matrix{T}, alpha::T, x::Vector{T}, y::Vector{T}) where T<:Float64
    BLAS.ger!(alpha, x, y, A)
end

@inline function rank1_update_blas!(A::Matrix{T}, alpha::T, x::Vector{T}, y::Vector{T}) where T<:Complex
    y_temp = conj!(y)
    BLAS.ger!(alpha, x, y_temp, A)
end

function ratio_rank1(vwf::vwf_det{T}, k::Int, new_row_idx_U::Int) where T
    val = zero(T)
    N = size(vwf.awf_inv, 1)
    @inbounds @simd for j in 1:N
        val += vwf.gs_U_t[j, new_row_idx_U] * vwf.awf_inv[j, k]
    end
    return val
end

function update_rank1!(vwf::vwf_det{T}, k::Int, new_row_idx_U::Int, ratio::T) where T
    ws = ensure_ws!(vwf)
    A_t = vwf.awf_mat_t
    Ainv = vwf.awf_inv
    N = size(A_t, 1)

    @inbounds @simd for j in 1:N
        ws.dr1[j] = vwf.gs_U_t[j, new_row_idx_U] - A_t[j, k]
    end

    mul!(ws.col1, transpose(Ainv), ws.dr1)
    copyto!(ws.xi, @view Ainv[:, k])
    rank1_update_blas!(Ainv, -1 / ratio, ws.xi, ws.col1)

    @inbounds @simd for j in 1:N
        A_t[j, k] = vwf.gs_U_t[j, new_row_idx_U]
    end

    vwf.awf_val *= ratio
end

function ratio_rank2(vwf::vwf_det{T}, k1::Int, k2::Int, new_row1_U::Int, new_row2_U::Int) where T
    Ainv = vwf.awf_inv
    U_t = vwf.gs_U_t
    N = size(Ainv, 1)

    d11 = zero(T)
    d12 = zero(T)
    d21 = zero(T)
    d22 = zero(T)

    @inbounds @simd for j in 1:N
        u1 = U_t[j, new_row1_U]
        u2 = U_t[j, new_row2_U]
        inv_k1 = Ainv[j, k1]
        inv_k2 = Ainv[j, k2]

        d11 += u1 * inv_k1
        d12 += u1 * inv_k2
        d21 += u2 * inv_k1
        d22 += u2 * inv_k2
    end
    return d11 * d22 - d12 * d21
end

function update_rank2!(vwf::vwf_det{T}, k1::Int, k2::Int, new_row1_U::Int, new_row2_U::Int, ratio::T) where T
    ws = ensure_ws!(vwf)
    A_t = vwf.awf_mat_t
    Ainv = vwf.awf_inv
    N = size(A_t, 1)

    @inbounds @simd for j in 1:N
        ws.dr1[j] = vwf.gs_U_t[j, new_row1_U] - A_t[j, k1]
        ws.dr2[j] = vwf.gs_U_t[j, new_row2_U] - A_t[j, k2]
    end

    mul!(ws.col1, transpose(Ainv), ws.dr1)
    mul!(ws.col2, transpose(Ainv), ws.dr2)

    k_11 = 1.0 + ws.col1[k1]
    k_12 = ws.col1[k2]
    k_21 = ws.col2[k1]
    k_22 = 1.0 + ws.col2[k2]

    detK = k_11 * k_22 - k_12 * k_21
    invDet = 1.0 / detK

    i_11 = k_22 * invDet
    i_12 = -k_12 * invDet
    i_21 = -k_21 * invDet
    i_22 = k_11 * invDet

    copyto!(ws.xi, @view Ainv[:, k1])
    copyto!(ws.xj, @view Ainv[:, k2])

    @inbounds @simd for j in 1:N
        xi_val = ws.xi[j]
        xj_val = ws.xj[j]
        ws.s0[j] = i_11 * xi_val + i_21 * xj_val
        ws.s1[j] = i_12 * xi_val + i_22 * xj_val
    end

    rank1_update_blas!(Ainv, -one(T), ws.s0, ws.col1)
    rank1_update_blas!(Ainv, -one(T), ws.s1, ws.col2)

    @inbounds @simd for j in 1:N
        A_t[j, k1] = vwf.gs_U_t[j, new_row1_U]
        A_t[j, k2] = vwf.gs_U_t[j, new_row2_U]
    end

    vwf.awf_val *= ratio
end


"""
用途: 返回某个 PH 基底行在 proposal 提交后的电子编号。

参数:
- `sampler`: 当前 proposal 提交前的采样器对象。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `row_index::Int`: PH 基底中的内部行编号。

返回:
- `Int`: proposal 提交后占据该行的电子编号。若该行在 proposal 后为空, 返回 `0`。
"""
@inline function get_postproposal_electron_id(
    sampler,
    proposal::MoveProposal,
    row_index::Int,
)
    if row_index == proposal.target_map_idx_1
        return proposal.moved_electron_id_1
    elseif row_index == proposal.target_map_idx_2
        return proposal.moved_electron_id_2
    elseif row_index == proposal.source_map_idx_1 || row_index == proposal.source_map_idx_2
        return 0
    end

    return sampler.map_spin_to_id[row_index]
end


"""
用途: 收集一次 backflow proposal 对应的 determinant 局域列更新数据。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。
- `proposal::MoveProposal`: Monte Carlo proposal。

返回:
- `Tuple{Vector{Int}, Vector{Int}, Matrix{T}}`:
  `(changed_electron_ids, changed_row_indices, new_columns)`。
"""
function collect_backflow_local_column_updates(
    vwf::vwf_det{T},
    proposal::MoveProposal,
) where {T}
    backflow_term = vwf.backflow

    affected_sites = Backflow.collect_affected_site_indices(
        vwf.sampler.state,
        backflow_term,
        proposal,
    )

    if isempty(affected_sites)
        return Int[], Int[], zeros(T, size(vwf.awf_mat_t, 1), 0)
    end

    n_orb = size(vwf.awf_mat_t, 1)
    max_changed_count = 2 * length(affected_sites)
    changed_electron_ids = Vector{Int}(undef, max_changed_count)
    changed_row_indices = Vector{Int}(undef, max_changed_count)
    new_columns_unsorted = Matrix{T}(undef, n_orb, max_changed_count)
    site_block_buffer = Matrix{T}(undef, 2, size(vwf.base_gs_U, 2))
    changed_count = 0

    for site_index in affected_sites
        Backflow.fill_backflow_site_block_after_proposal!(
            site_block_buffer,
            vwf.base_gs_U,
            vwf.sampler.state,
            backflow_term,
            proposal,
            site_index,
        )
        for local_row_offset in 1:2
            row_index = 2 * (site_index - 1) + local_row_offset
            electron_id = get_postproposal_electron_id(vwf.sampler, proposal, row_index)
            if electron_id == 0
                continue
            end

            changed_count += 1
            changed_electron_ids[changed_count] = electron_id
            changed_row_indices[changed_count] = row_index
            copyto!(
                @view(new_columns_unsorted[:, changed_count]),
                @view(site_block_buffer[local_row_offset, :]),
            )
        end
    end

    if changed_count == 0
        return Int[], Int[], zeros(T, size(vwf.awf_mat_t, 1), 0)
    end

    resize!(changed_electron_ids, changed_count)
    resize!(changed_row_indices, changed_count)

    if length(unique(changed_electron_ids)) != changed_count
        error("Backflow local column update collected duplicate electron IDs: $(changed_electron_ids)")
    end

    permutation = sortperm(changed_electron_ids)
    sorted_electron_ids = changed_electron_ids[permutation]
    sorted_row_indices = changed_row_indices[permutation]
    new_columns = Matrix{T}(undef, n_orb, changed_count)
    for (column_index, old_column_index) in enumerate(permutation)
        copyto!(
            @view(new_columns[:, column_index]),
            @view(new_columns_unsorted[:, old_column_index]),
        )
    end

    return sorted_electron_ids, sorted_row_indices, new_columns
end


"""
用途: 原位计算 rank-k 局域更新所需的 `C` 与 `K` 小矩阵。

数学公式:
- `Delta = A_new^T - A_old^T`
- `C = A^{-T} * Delta`
- `K = I_k + P^T * C`, 其中 `P` 选出被替换的列编号。

参数:
- `c_matrix::AbstractMatrix{T}`: 输出 `C` 矩阵 buffer, 形状必须为 `(N, k)`。
- `small_k_matrix::AbstractMatrix{T}`: 输出 `K` 小矩阵 buffer, 形状必须为 `(k, k)`。
- `vwf::vwf_det{T}`: determinant 波函数对象。
- `changed_electron_ids::AbstractVector{Int}`: 被替换的 determinant 列编号。
- `new_columns::AbstractMatrix{T}`: proposal 后这些列的新列向量。

返回:
- `Tuple{AbstractMatrix{T}, AbstractMatrix{T}}`: `(c_matrix, small_k_matrix)`。
"""
function compute_rankk_update_factors!(
    c_matrix::AbstractMatrix{T},
    small_k_matrix::AbstractMatrix{T},
    delta_columns_buffer::AbstractMatrix{T},
    vwf::vwf_det{T},
    changed_electron_ids::AbstractVector{Int},
    new_columns::AbstractMatrix{T},
) where {T}
    if size(new_columns, 2) != length(changed_electron_ids)
        error("Rank-k update mismatch: got $(length(changed_electron_ids)) electron IDs, but $(size(new_columns, 2)) new columns.")
    end

    n_orb = size(vwf.awf_inv, 1)
    k_count = length(changed_electron_ids)
    if size(c_matrix, 1) != n_orb || size(c_matrix, 2) != k_count
        error("C matrix buffer shape mismatch: expected ($(n_orb), $(k_count)), got $(size(c_matrix)).")
    end
    if size(small_k_matrix, 1) != k_count || size(small_k_matrix, 2) != k_count
        error("Small K matrix buffer shape mismatch: expected ($(k_count), $(k_count)), got $(size(small_k_matrix)).")
    end
    if size(delta_columns_buffer, 1) != n_orb || size(delta_columns_buffer, 2) != k_count
        error("Delta buffer shape mismatch: expected ($(n_orb), $(k_count)), got $(size(delta_columns_buffer)).")
    end

    copyto!(delta_columns_buffer, new_columns)
    @inbounds for column_index in 1:k_count
        electron_id = changed_electron_ids[column_index]
        @views delta_columns_buffer[:, column_index] .-= vwf.awf_mat_t[:, electron_id]
    end
    mul!(c_matrix, transpose(vwf.awf_inv), delta_columns_buffer)

    @inbounds for column_index in 1:k_count
        for row_index in 1:k_count
            small_k_matrix[column_index, row_index] = c_matrix[changed_electron_ids[row_index], column_index]
        end
        small_k_matrix[column_index, column_index] += one(T)
    end

    return c_matrix, small_k_matrix
end


"""
用途: 计算 rank-k 局域更新所需的 `C` 与 `K` 小矩阵。

数学公式:
- `Delta = A_new^T - A_old^T`
- `C = A^{-T} * Delta`
- `K = I_k + P^T * C`, 其中 `P` 选出被替换的列编号。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。
- `changed_electron_ids::Vector{Int}`: 被替换的 determinant 列编号。
- `new_columns::AbstractMatrix{T}`: proposal 后这些列的新列向量。

返回:
- `Tuple{Matrix{T}, Matrix{T}}`: `(c_matrix, small_k_matrix)`。
"""
function compute_rankk_update_factors(
    vwf::vwf_det{T},
    changed_electron_ids::Vector{Int},
    new_columns::AbstractMatrix{T},
) where {T}
    k_count = length(changed_electron_ids)
    c_matrix = Matrix{T}(undef, size(vwf.awf_inv, 1), k_count)
    small_k_matrix = Matrix{T}(undef, k_count, k_count)
    delta_columns_buffer = Matrix{T}(undef, size(vwf.awf_inv, 1), k_count)
    return compute_rankk_update_factors!(
        c_matrix,
        small_k_matrix,
        delta_columns_buffer,
        vwf,
        changed_electron_ids,
        new_columns,
    )
end


"""
用途: 计算局域 rank-k 列替换对应的 determinant 比值。

数学公式:
- `ratio = det(K)`。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。
- `changed_electron_ids::Vector{Int}`: 被替换的 determinant 列编号。
- `new_columns::AbstractMatrix{T}`: proposal 后这些列的新列向量。

返回:
- `T`: `Psi_new / Psi_old` 的 determinant 比值。
"""
function ratio_rankk(
    vwf::vwf_det{T},
    changed_electron_ids::Vector{Int},
    new_columns::AbstractMatrix{T},
) where {T}
    if isempty(changed_electron_ids)
        return one(T)
    end

    _, small_k_matrix = compute_rankk_update_factors(vwf, changed_electron_ids, new_columns)
    return det(small_k_matrix)
end


"""
用途: 直接把当前 proposal 的 determinant 列更新写入 workspace cache。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。
- `proposal::MoveProposal`: Monte Carlo proposal。

返回:
- `Int`: 写入 cache 的活动列数。
"""
function collect_backflow_local_column_updates_into_cache!(
    vwf::vwf_det{T},
    proposal::MoveProposal,
) where {T}
    ws = ensure_ws!(vwf)
    backflow_term = vwf.backflow

    affected_sites = Backflow.collect_affected_site_indices(
        vwf.sampler.state,
        backflow_term,
        proposal,
    )

    ws.cached_affected_count = length(affected_sites)
    changed_count = 0
    site_block_buffer = ws.cached_site_block_buffer

    for (affected_offset, site_index) in enumerate(affected_sites)
        ws.cached_affected_site_indices[affected_offset] = site_index
        Backflow.fill_backflow_site_block_after_proposal!(
            site_block_buffer,
            vwf.base_gs_U,
            vwf.sampler.state,
            backflow_term,
            proposal,
            site_index,
        )
        cached_block_row = 2 * (affected_offset - 1) + 1
        copyto!(
            @view(ws.cached_affected_site_blocks[cached_block_row, :]),
            @view(site_block_buffer[1, :]),
        )
        copyto!(
            @view(ws.cached_affected_site_blocks[cached_block_row + 1, :]),
            @view(site_block_buffer[2, :]),
        )

        for local_row_offset in 1:2
            row_index = 2 * (site_index - 1) + local_row_offset
            electron_id = get_postproposal_electron_id(vwf.sampler, proposal, row_index)
            if electron_id == 0
                continue
            end

            changed_count += 1
            ws.cached_changed_electron_ids[changed_count] = electron_id
            ws.cached_changed_row_indices[changed_count] = row_index
            copyto!(
                @view(ws.cached_c_matrix[:, changed_count]),
                @view(site_block_buffer[local_row_offset, :]),
            )
        end
    end

    if changed_count == 0
        ws.cached_changed_count = 0
        return 0
    end

    permutation = @view ws.cached_permutation[1:changed_count]
    sortperm!(permutation, @view(ws.cached_changed_electron_ids[1:changed_count]))
    sorted_electron_ids = @view ws.cached_sorted_electron_ids[1:changed_count]
    sorted_row_indices = @view ws.cached_sorted_row_indices[1:changed_count]

    for (column_index, old_column_index) in enumerate(permutation)
        sorted_electron_ids[column_index] = ws.cached_changed_electron_ids[old_column_index]
        sorted_row_indices[column_index] = ws.cached_changed_row_indices[old_column_index]

        if column_index > 1 && sorted_electron_ids[column_index] == sorted_electron_ids[column_index - 1]
            error("Backflow local column update collected duplicate electron IDs: $(sorted_electron_ids[1:column_index])")
        end

        copyto!(
            @view(ws.cached_new_columns[:, column_index]),
            @view(ws.cached_c_matrix[:, old_column_index]),
        )
    end

    copyto!(@view(ws.cached_changed_electron_ids[1:changed_count]), sorted_electron_ids)
    copyto!(@view(ws.cached_changed_row_indices[1:changed_count]), sorted_row_indices)
    ws.cached_changed_count = changed_count
    return changed_count
end


"""
用途: 为当前 proposal 收集 determinant 列更新并把 rank-k 中间量写入 workspace cache。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。
- `proposal::MoveProposal`: Monte Carlo proposal。

返回:
- `T`: determinant 比值 `Psi_new / Psi_old`。
"""
function cache_backflow_rankk_update!(vwf::vwf_det{T}, proposal::MoveProposal) where {T}
    ws = ensure_ws!(vwf)
    reset_cached_rankk_update!(ws)

    changed_count = collect_backflow_local_column_updates_into_cache!(vwf, proposal)

    if changed_count == 0
        ws.has_cached_rankk_update = true
        return one(T)
    end

    changed_electron_ids = @view ws.cached_changed_electron_ids[1:changed_count]
    cached_new_columns = @view ws.cached_new_columns[:, 1:changed_count]
    cached_c_matrix = @view ws.cached_c_matrix[:, 1:changed_count]
    cached_small_k_matrix = @view ws.cached_small_k_matrix[1:changed_count, 1:changed_count]
    cached_delta_columns = @view ws.cached_small_k_inverse[:, 1:changed_count]
    compute_rankk_update_factors!(
        cached_c_matrix,
        cached_small_k_matrix,
        cached_delta_columns,
        vwf,
        changed_electron_ids,
        cached_new_columns,
    )

    ws.has_cached_rankk_update = true
    return det(cached_small_k_matrix)
end


"""
用途: 在 debug 模式下用整块 rebuild 真值校验 local ratio。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `fast_ratio::T`: local fast path 计算得到的 determinant 比值。

返回:
- `nothing`。若校验失败则抛出 error。
"""
function verify_backflow_local_ratio(
    vwf::vwf_det{T},
    proposal::MoveProposal,
    fast_ratio::T,
) where {T}
    rebuild_ratio = calc_ratio_rebuild(vwf, proposal)
    if !isapprox(fast_ratio, rebuild_ratio; atol=1e-10, rtol=1e-10)
        error("Backflow local ratio mismatch: fast=$fast_ratio rebuild=$rebuild_ratio")
    end

    return nothing
end


"""
用途: 对 determinant 状态执行局域 rank-k 列替换更新。

数学公式:
- `A_new^{-1} = A^{-1} - A^{-1} U K^{-1} V^T A^{-1}`。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。
- `changed_electron_ids::Vector{Int}`: 被替换的 determinant 列编号。
- `new_columns::AbstractMatrix{T}`: proposal 后这些列的新列向量。
- `ratio::T`: 已计算好的 determinant 比值 `det(K)`。

返回:
- `nothing`。
"""
function update_rankk!(
    vwf::vwf_det{T},
    changed_electron_ids::Vector{Int},
    new_columns::AbstractMatrix{T},
    ratio::T,
) where {T}
    if isempty(changed_electron_ids)
        return nothing
    end

    c_matrix, small_k_matrix = compute_rankk_update_factors(vwf, changed_electron_ids, new_columns)
    basis_columns = vwf.awf_inv[:, changed_electron_ids]
    small_k_inverse = inv(small_k_matrix)

    vwf.awf_inv .-= basis_columns * (small_k_inverse * transpose(c_matrix))
    vwf.awf_mat_t[:, changed_electron_ids] .= new_columns
    vwf.awf_val *= ratio
    return nothing
end


"""
用途: 使用当前 rank-k cache 中保存的 proposal 后局域 backflow 行块刷新全局轨道矩阵。

数学公式:
- 对每个缓存站点 `i`, 将 `U_b(i, sigma; x')` 写回 `backflow_u`, `gs_U` 和转置缓存 `gs_U_t`。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象, 必须已经通过 `calc_backflow_ratio_local_update` 建立 cache。

返回:
- `nothing`。
"""
function apply_cached_backflow_orbital_rows!(vwf::vwf_det{T}) where {T}
    ws = ensure_ws!(vwf)
    if !ws.has_cached_rankk_update
        error("Backflow rank-k cache is invalid. Call calc_backflow_ratio_local_update(vwf, proposal) before apply_cached_backflow_orbital_rows!(vwf).")
    end

    for affected_offset in 1:ws.cached_affected_count
        site_index = ws.cached_affected_site_indices[affected_offset]
        orbital_row = 2 * (site_index - 1) + 1
        cached_block_row = 2 * (affected_offset - 1) + 1

        copyto!(
            @view(vwf.backflow_u[orbital_row, :]),
            @view(ws.cached_affected_site_blocks[cached_block_row, :]),
        )
        copyto!(
            @view(vwf.backflow_u[orbital_row + 1, :]),
            @view(ws.cached_affected_site_blocks[cached_block_row + 1, :]),
        )
        copyto!(
            @view(vwf.gs_U[orbital_row, :]),
            @view(ws.cached_affected_site_blocks[cached_block_row, :]),
        )
        copyto!(
            @view(vwf.gs_U[orbital_row + 1, :]),
            @view(ws.cached_affected_site_blocks[cached_block_row + 1, :]),
        )
        copyto!(
            @view(vwf.gs_U_t[:, orbital_row]),
            @view(ws.cached_affected_site_blocks[cached_block_row, :]),
        )
        copyto!(
            @view(vwf.gs_U_t[:, orbital_row + 1]),
            @view(ws.cached_affected_site_blocks[cached_block_row + 1, :]),
        )
    end

    return nothing
end


"""
用途: 使用 workspace 中缓存的通用 rank-k 中间量执行 determinant 更新。

数学公式:
- `A_new^{-1} = A^{-1} - A^{-1} U K^{-1} V^T A^{-1}`。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。
- `ratio::T`: 已计算好的 determinant 比值。

返回:
- `nothing`。
"""
function update_rankk_from_cache!(vwf::vwf_det{T}, ratio::T) where {T}
    ws = ensure_ws!(vwf)
    if !ws.has_cached_rankk_update
        error("Backflow rank-k cache is invalid. Call calc_backflow_ratio_local_update(vwf, proposal) before accept_backflow_local_update!(vwf, proposal, ratio).")
    end

    changed_count = ws.cached_changed_count
    if changed_count == 0
        return nothing
    end

    changed_electron_ids = @view ws.cached_changed_electron_ids[1:changed_count]
    new_columns = @view ws.cached_new_columns[:, 1:changed_count]
    c_matrix = @view ws.cached_c_matrix[:, 1:changed_count]
    small_k_matrix = @view ws.cached_small_k_matrix[1:changed_count, 1:changed_count]
    small_k_inverse = @view ws.cached_small_k_inverse[1:changed_count, 1:changed_count]
    small_k_inverse .= inv(small_k_matrix)

    basis_columns = vwf.awf_inv[:, changed_electron_ids]
    vwf.awf_inv .-= basis_columns * (small_k_inverse * transpose(c_matrix))
    vwf.awf_mat_t[:, changed_electron_ids] .= new_columns
    vwf.awf_val *= ratio
    return nothing
end


"""
用途: 在 debug 模式下用整块 rebuild 真值校验 local accept 后的 determinant 状态。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。此时应已完成 local accept。

返回:
- `nothing`。若校验失败则抛出 error。
"""
function verify_backflow_local_accept(vwf::vwf_det{T}) where {T}
    orbitals_check = Backflow.build_backflow_orbitals(
        vwf.base_gs_U,
        vwf.sampler.state,
        vwf.backflow,
    )
    slater_check = build_slater_matrix_from_orbitals(
        orbitals_check,
        vwf.sampler.electron_locs,
    )
    awf_inv_check = inv(slater_check)
    awf_val_check = det(slater_check)

    if !isapprox(vwf.awf_val, awf_val_check; atol=1e-10, rtol=1e-10)
        error("Backflow local accept determinant mismatch: fast=$(vwf.awf_val) rebuild=$awf_val_check")
    end
    if !isapprox(vwf.awf_inv, awf_inv_check; atol=1e-10, rtol=1e-10)
        error("Backflow local accept inverse mismatch.")
    end

    return nothing
end


"""
用途: 使用局域受影响列与通用 rank-k 公式计算 backflow proposal 的比值。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。
- `proposal::MoveProposal`: Monte Carlo proposal。

返回:
- `T`: `Psi_new / Psi_old` 的 determinant 比值。
"""
function calc_backflow_ratio_local_update(vwf::vwf_det{T}, proposal::MoveProposal) where {T}
    fast_ratio = cache_backflow_rankk_update!(vwf, proposal)

    if vwf.backflow_debug_verify
        verify_backflow_local_ratio(vwf, proposal, fast_ratio)
    end

    return fast_ratio
end


"""
用途: 使用局域 rank-k 更新接受一次 backflow proposal。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `ratio::T`: 已计算好的 determinant 比值。

返回:
- `nothing`。
"""
function accept_backflow_local_update!(vwf::vwf_det{T}, proposal::MoveProposal, ratio::T) where {T}
    ws = ensure_ws!(vwf)
    if !ws.has_cached_rankk_update
        error("Backflow rank-k cache is invalid. Call calc_backflow_ratio_local_update(vwf, proposal) before accept_backflow_local_update!(vwf, proposal, ratio).")
    end

    update_rankk_from_cache!(vwf, ratio)
    commit_move!(vwf.sampler, proposal)
    apply_cached_backflow_orbital_rows!(vwf)
    reset_cached_rankk_update!(ws)

    if vwf.backflow_debug_verify
        verify_backflow_local_accept(vwf)
    end

    return nothing
end


"""
用途: 在启用 backflow 时, 通过复制 proposal 后构型并直接重建 Slater 方阵来计算比值。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。
- `proposal::MoveProposal`: Monte Carlo proposal。

返回:
- `T`: `Psi_new / Psi_old` 的 determinant 比值。
"""
function calc_ratio_rebuild(vwf::vwf_det{T}, proposal::MoveProposal) where {T}
    if proposal.site1 == 0
        return one(T)
    end

    new_sampler = copy_config(vwf.sampler)
    commit_move!(new_sampler, proposal)

    new_orbitals = Backflow.build_backflow_orbitals(vwf.base_gs_U, new_sampler.state, vwf.backflow)
    new_slater = build_slater_matrix_from_orbitals(new_orbitals, new_sampler.electron_locs)
    return det(new_slater) / vwf.awf_val
end

function find_stable_config!(vwf::vwf_det{T}, kernel::AbstractMCMCKernel, rng::AbstractRNG) where T
    ss = vwf.sampler

    # println("Searching for a numerically stable configuration...")

    max_attempts = 1000
    tol_inv = 1e-5

    for attempt in 1:max_attempts
        init_config_rand!(ss, kernel)
        refresh_backflow_orbitals!(vwf)

        # === 2. 根据新构型重建矩阵 ===
        # Sampler 已经更新了 electron_locs，直接利用它填充矩阵
        total_elec_count = total_elec(ss)
        for i in 1:total_elec_count
            # electron_locs[i] 存储的是基组索引 (2*site+spin)，对应 gs_U_t 的列
            basis_idx = ss.electron_locs[i]

            # awf_mat_t 是转置存储的 (列是电子，行是轨道)
            copyto!(@view(vwf.awf_mat_t[:, i]), @view(vwf.gs_U_t[:, basis_idx]))
        end

        # === 3. 检查数值稳定性 ===
        # 计算行列式
        F = lu(transpose(vwf.awf_mat_t), check=false)
        d = det(F)


        try
            # 计算逆矩阵
            current_inv = inv(transpose(vwf.awf_mat_t))

            # 验证逆矩阵精度
            prod_mat = current_inv * transpose(vwf.awf_mat_t)
            diff = norm(prod_mat - I)

            if diff < tol_inv
                # === 成功 ===
                vwf.awf_inv = current_inv
                vwf.awf_val = d
                vwf.current_ratio = one(T)
                ensure_ws!(vwf)
                reset_cached_rankk_update!(vwf.ws)

                # println("Stable config found at attempt $attempt. Det=$d, |A⁻¹A - I|=$diff")
                return
            end

        catch e
            # 奇异异常，继续尝试
        end
    end

    error("Failed to find a stable configuration after $max_attempts attempts. Please check your Trial Wavefunction (U).")
end


# ==============================================================================
# Generic Measurements
# ==============================================================================
function local_energy(ham, vwf::vwf_det)
    return 0
end

function compute_grad_log_psi!(vwf::vwf_det{T}) where T
    # 1. 准备 Workspace
    ws = ensure_ws!(vwf)
    ss = vwf.sampler

    A_inv = vwf.awf_inv   # Size: (N_orb, N_elec)
    # A_inv[orb, elec] -> 列优先存储，orb 变化最快

    Norb, Nelec = size(A_inv)

    # 2. 获取 Buffer (O_vec)
    O_vec = ws.grad_buffer
    fill!(O_vec, zero(T))

    # 3. 先计算波函数参数梯度部分
    wf_param_count = length(vwf.param_keys)
    has_active_backflow = Backflow.uses_backflow(vwf.backflow)
    for idx in 1:wf_param_count
        dU_t = @view vwf.dUt_matrix[:, :, idx]
        derivative_orbitals = if has_active_backflow
            # backflow 打开时 determinant 使用 U_b = B_x[U_0],
            # 因此 mean-field 参数导数必须使用 dU_b/dp = B_x[dU_0/dp]。
            Backflow.fill_backflow_chain_rule_orbitals!(
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

        # 顺序：外层电子(elec)，内层轨道(orb)
        # 优化理由：dU_t[orb, r] 和 A_inv[orb, elec] 第一维都是 orb，内存连续
        @inbounds for elec in 1:Nelec
            r = ss.electron_locs[elec]

            col_sum = zero(T)

            # SIMD 内积
            if has_active_backflow
                @simd for orb in 1:Norb
                    col_sum += A_inv[orb, elec] * derivative_orbitals[r, orb]
                end
            else
                @simd for orb in 1:Norb
                    col_sum += A_inv[orb, elec] * dU_t[orb, r]
                end
            end

            total_sum += col_sum
        end

        # 直接使用 enumerate 的索引，不再依赖计数器变量
        O_vec[idx] = total_sum
    end

    # 4. 再拼接 projector 参数梯度部分
    projector_param_count = Projector.projector_param_count(vwf.projector)
    if projector_param_count > 0
        start_idx = wf_param_count + 1
        end_idx = wf_param_count + projector_param_count
        projector_view = @view O_vec[start_idx:end_idx]
        Projector.projector_log_derivative!(projector_view, vwf.projector, ss)
    end

    # 5. 最后拼接 backflow 参数梯度部分
    backflow_pairs = Backflow.build_backflow_derivative_orbitals(vwf.base_gs_U, ss.state, vwf.backflow)
    if !isempty(backflow_pairs)
        start_idx = wf_param_count + projector_param_count + 1
        for (pair_offset, (_, derivative_orbitals)) in enumerate(backflow_pairs)
            total_sum = zero(T)

            @inbounds for elec in 1:Nelec
                row_idx = ss.electron_locs[elec]
                col_sum = zero(T)

                @simd for orb in 1:Norb
                    col_sum += A_inv[orb, elec] * derivative_orbitals[row_idx, orb]
                end

                total_sum += col_sum
            end

            O_vec[start_idx + pair_offset - 1] = total_sum
        end
    end

    # 直接返回 buffer 引用，避免 copy
    return O_vec
end
