module Model

using ..Sampler
using ..VMC

# import ..VMC: local_energy

export HeisenbergModel, HubbardModel, GeneralModel, OperatorTerm, local_energy


# ==============================================================================
# 1. Heisenberg Model
# ==============================================================================
struct HeisenbergModel
    lx::Int
    ly::Int
    Nlat::Int
    J1::Float64
    J2::Float64
    J1_bonds::Vector{Tuple{Int,Int}}
    J2_bonds::Vector{Tuple{Int,Int}}
end

function HeisenbergModel(Nlat::Int; model_params=Dict{Symbol,Any}())
    lx = get(model_params, :lx, floor(Int, sqrt(Nlat)))
    ly = get(model_params, :ly, floor(Int, Nlat / lx))
    J1 = get(model_params, :J1, 1.0)
    J2 = get(model_params, :J2, 0.0)

    bonds1 = Tuple{Int,Int}[]
    bonds2 = Tuple{Int,Int}[]
    idx(x, y) = mod(x - 1, lx) * ly + mod(y - 1, ly) + 1
    for y in 1:ly, x in 1:lx
        u = idx(x, y)
        push!(bonds1, (u, idx(x + 1, y)))
        push!(bonds1, (u, idx(x, y + 1)))
        if J2 != 0
            push!(bonds2, (u, idx(x + 1, y + 1)))
            push!(bonds2, (u, idx(x - 1, y + 1)))
        end
    end
    return HeisenbergModel(lx, ly, Nlat, J1, J2, bonds1, bonds2)
end

# --- Heisenberg Implementation ---

function energy_heisenberg_term(vwf, bonds, J)
    E = 0.0
    for (i, j) in bonds
        # 利用 VMC 中通用的 measure_SiSj (包含 SzSz 和 S+S-)
        # H = J * S_i \cdot S_j
        E += J * measure_SiSj(vwf, i, j)
    end
    return E
end

# 重载接口
function local_energy(ham::HeisenbergModel, vwf)
    E = 0.0
    if ham.J1 != 0
        E += energy_heisenberg_term(vwf, ham.J1_bonds, ham.J1)
    end
    if ham.J2 != 0
        E += energy_heisenberg_term(vwf, ham.J2_bonds, ham.J2)
    end
    return E
end


# ==============================================================================
# 2. Hubbard Model
# ==============================================================================
struct HubbardModel
    Nlat::Int
    t::Float64
    U::Float64
    hoppings::Vector{Tuple{Int,Int}} # usually NN
end

function HubbardModel(Nlat::Int; t=1.0, U=4.0, lx=nothing)
    lx = isnothing(lx) ? floor(Int, sqrt(Nlat)) : lx
    ly = floor(Int, Nlat / lx)

    bonds = Tuple{Int,Int}[]
    # idx(x, y) = mod(y-1, ly)*lx + mod(x-1, lx) + 1
    # idx(x, y) = mod(y-1, ly)*lx + mod(x-1, lx) + 1
    idx(x, y) = mod(x - 1, lx) * ly + mod(y - 1, ly) + 1
    for y in 1:ly, x in 1:lx
        u = idx(x, y)
        push!(bonds, (u, idx(x + 1, y)))
        push!(bonds, (u, idx(x, y + 1)))
    end
    return HubbardModel(Nlat, t, U, bonds)
end

# --- Hubbard Implementation ---

# 重载接口
function local_energy(ham::HubbardModel, vwf)
    ss = vwf.sampler

    # 1. Potential Energy: U * sum n_up * n_dn
    # 直接读取 Sampler 维护的双占计数，O(1) 复杂度
    E_pot = ham.U * ss.count_dbs

    # 2. Kinetic Energy: -t * sum c^dag_i c_j + h.c.
    E_kin = 0.0
    for (i, j) in ham.hoppings
        # Spin UP hoppings
        # measure_green(i, j) = <c^dag_i c_j>
        t_ij_up = measure_green(vwf, i, j, UP)
        t_ji_up = measure_green(vwf, j, i, UP) # h.c.

        # Spin DN hoppings
        t_ij_dn = measure_green(vwf, i, j, DN)
        t_ji_dn = measure_green(vwf, j, i, DN) # h.c.

        E_kin += -ham.t * real(t_ij_up + t_ji_up + t_ij_dn + t_ji_dn)
    end

    return E_kin + E_pot
end

# ==============================================================================
# 3. General Model
# ==============================================================================
struct OperatorTerm
    ops::Vector{Symbol}
    sites::Vector{Int}
    coef::Float64
end

struct GeneralModel
    Nsite::Int
    terms::Vector{OperatorTerm}
end

const DIAG_OP_TOKENS = Set{Symbol}([:Sz, :n_up, :n_dn, :n])
const NONDIAG_OP_TOKENS = Set{Symbol}([:cdag_up, :c_up, :cdag_dn, :c_dn])
const CDAG_OP_TOKENS = Set{Symbol}([:cdag_up, :cdag_dn])
const C_OP_TOKENS = Set{Symbol}([:c_up, :c_dn])
const TWO_SITE_OP_TOKENS = Set{Symbol}([:SS])

"""
    validate_operator_term(term::OperatorTerm)

校验算符项的格式与合法性。
参数:
- term::OperatorTerm, 对角项必须在左侧, 非对角项只允许一对 cdag_* 与 c_*。
  特殊项 :SS 必须作为唯一算符, 且 sites 需要提供 2 个站点。
返回:
- nothing, 若不合法会抛出 error。
"""
function validate_operator_term(term::OperatorTerm)
    if any(op -> op in TWO_SITE_OP_TOKENS, term.ops)
        if length(term.ops) != 1 || term.ops[1] != :SS
            error("Operator :SS must be the only operator in one term")
        end
        if length(term.sites) != 2
            error("Operator :SS requires exactly 2 sites, got $(length(term.sites))")
        end
        return nothing
    end

    if length(term.ops) != length(term.sites)
        error("OperatorTerm ops/sites length mismatch: $(length(term.ops)) vs $(length(term.sites))")
    end

    cdag_count = 0
    c_count = 0
    seen_nondiag = false

    for op in term.ops
        if op in DIAG_OP_TOKENS
            if seen_nondiag
                error("Diagonal operators must be on the left of non-diagonal operators")
            end
            continue
        elseif op in NONDIAG_OP_TOKENS
            seen_nondiag = true
            if op in CDAG_OP_TOKENS
                cdag_count += 1
            else
                c_count += 1
            end
        else
            error("Unknown operator token: $op")
        end
    end

    if cdag_count != c_count
        error("Non-diagonal operators must appear in cdag/c pairs, got cdag=$cdag_count, c=$c_count")
    end
    if cdag_count > 1
        error("Only one cdag/c pair is supported in one OperatorTerm")
    end

    if cdag_count == 1
        cdag_index = findfirst(op -> op in CDAG_OP_TOKENS, term.ops)
        c_index = findfirst(op -> op in C_OP_TOKENS, term.ops)
        cdag_op = term.ops[cdag_index]
        c_op = term.ops[c_index]
        if (cdag_op == :cdag_up && c_op != :c_up) || (cdag_op == :cdag_dn && c_op != :c_dn)
            error("Spin mismatch between cdag and c: $cdag_op vs $c_op")
        end
    end

    return nothing
end

"""
    find_cdag_c_pair_info(ops::Vector{Symbol}, sites::Vector{Int})

查找单对 cdag/c 的自旋、位置与顺序信息。
参数:
- ops::Vector{Symbol}, 算符 token 列表。
- sites::Vector{Int}, 与 ops 对齐的 site 列表。
返回:
- Union{Nothing,Tuple{Int8,Int,Int,Bool}}。若无非对角项返回 nothing, 否则返回 (spin, site_cdag, site_c, cdag_first)。
"""
function find_cdag_c_pair_info(ops::Vector{Symbol}, sites::Vector{Int})
    cdag_index = findfirst(op -> op in CDAG_OP_TOKENS, ops)
    c_index = findfirst(op -> op in C_OP_TOKENS, ops)
    if cdag_index === nothing || c_index === nothing
        return nothing
    end

    cdag_op = ops[cdag_index]
    spin = (cdag_op == :cdag_up) ? UP : DN
    cdag_first = cdag_index < c_index
    return (spin, sites[cdag_index], sites[c_index], cdag_first)
end

"""
    compute_diag_operator_value(op::Symbol, site::Int, ss)

计算对角算符在给定 site 的取值。
参数:
- op::Symbol, 支持 :Sz, :n_up, :n_dn, :n。
- site::Int, 站点索引(从1开始)。
- ss, 采样配置对象。
返回:
- Float64, 对角算符的数值。n 的定义为 n = n_up + n_dn。
"""
function compute_diag_operator_value(op::Symbol, site::Int, ss)
    st = ss.state[site]
    if op == :Sz
        return get_Sz(st)
    elseif op == :n_up
        return (st & UP) != 0 ? 1.0 : 0.0
    elseif op == :n_dn
        return (st & DN) != 0 ? 1.0 : 0.0
    elseif op == :n
        n_up = (st & UP) != 0 ? 1.0 : 0.0
        n_dn = (st & DN) != 0 ? 1.0 : 0.0
        return n_up + n_dn
    end

    error("Unsupported diagonal operator token: $op")
end

"""
    compute_term_energy(term::OperatorTerm, vwf)

计算单个算符项的局域能量贡献。
参数:
- term::OperatorTerm, 算符项。
- vwf, 波函数对象。
返回:
- Float64, term 的能量贡献。若包含非对角项, 使用 <c†_i c_j> 并取实部,
  并使用费米反对易关系 <c_i c†_j> = delta_ij - <c†_j c_i> 处理 c 在前的顺序。
"""
function compute_term_energy(term::OperatorTerm, vwf)
    validate_operator_term(term)

    if length(term.ops) == 1 && term.ops[1] == :SS
        site_i = term.sites[1]
        site_j = term.sites[2]
        return term.coef * measure_SiSj(vwf, site_i, site_j)
    end

    ss = vwf.sampler
    diag_factor = 1.0

    for (op, site) in zip(term.ops, term.sites)
        if op in DIAG_OP_TOKENS
            diag_factor *= compute_diag_operator_value(op, site, ss)
        end
    end

    pair = find_cdag_c_pair_info(term.ops, term.sites)
    if pair === nothing
        return term.coef * diag_factor
    end

    spin, site_cdag, site_c, cdag_first = pair
    if cdag_first
        hopping_val = measure_green(vwf, site_cdag, site_c, spin)
        return term.coef * diag_factor * real(hopping_val)
    end

    delta_val = site_cdag == site_c ? 1.0 : 0.0
    hopping_val = measure_green(vwf, site_cdag, site_c, spin)
    return term.coef * diag_factor * (delta_val - real(hopping_val))
end

"""
    local_energy(ham::GeneralModel, vwf)

计算 GeneralModel 的局域能量。
参数:
- ham::GeneralModel, 通用模型。
- vwf, 波函数对象。
返回:
- Float64, 局域能量。
"""
function local_energy(ham::GeneralModel, vwf)
    energy = 0.0
    for term in ham.terms
        energy += compute_term_energy(term, vwf)
    end
    return energy
end

end # module
