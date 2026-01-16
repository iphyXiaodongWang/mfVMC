module Model

using ..Sampler
using ..VMC 

# import ..VMC: local_energy

export HeisenbergModel, HubbardModel, local_energy


# ==============================================================================
# 1. Heisenberg Model
# ==============================================================================
struct HeisenbergModel
    lx::Int; ly::Int; Nlat::Int
    J1::Float64; J2::Float64
    J1_bonds::Vector{Tuple{Int,Int}}
    J2_bonds::Vector{Tuple{Int,Int}}
end

function HeisenbergModel(Nlat::Int; model_params=Dict{Symbol,Any}())
    lx = get(model_params, :lx, floor(Int, sqrt(Nlat)))
    ly = get(model_params, :ly, floor(Int, Nlat/lx))
    J1 = get(model_params, :J1, 1.0)
    J2 = get(model_params, :J2, 0.0)
    
    bonds1 = Tuple{Int,Int}[]; bonds2 = Tuple{Int,Int}[]
    idx(x, y) = mod(x-1, lx)*ly + mod(y-1, ly) + 1
    for y in 1:ly, x in 1:lx
        u = idx(x, y)
        push!(bonds1, (u, idx(x+1, y))); push!(bonds1, (u, idx(x, y+1)))
        if J2 != 0
            push!(bonds2, (u, idx(x+1, y+1))); push!(bonds2, (u, idx(x-1, y+1)))
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
    ly = floor(Int, Nlat/lx)
    
    bonds = Tuple{Int,Int}[]
    # idx(x, y) = mod(y-1, ly)*lx + mod(x-1, lx) + 1
    # idx(x, y) = mod(y-1, ly)*lx + mod(x-1, lx) + 1
    idx(x, y) = mod(x-1, lx)*ly + mod(y-1, ly) + 1
    for y in 1:ly, x in 1:lx
        u = idx(x, y)
        push!(bonds, (u, idx(x+1, y))); push!(bonds, (u, idx(x, y+1)))
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

end # module
