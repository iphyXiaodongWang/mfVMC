using LinearAlgebra
using Random

include("twist_Hubbard_PH.jl")

"""
用途: 构造 benchmark 使用的 spinless hopping 矩阵。

数学公式:
- `h_{ij} = -t_ij`, 其中最近邻 x/y hopping 分别为 `tx`, `ty`, 对角次近邻为 `t2`。
- 与 `build_twist_hamiltonian_terms` 一致, 每个代表 bond 加入 `i -> j` 与 `j -> i`。

参数:
- `lx, ly::Int`: 晶格尺寸。
- `tx, ty::Float64`: x/y 最近邻 hopping。
- `t2::Float64`: 对角次近邻 hopping。

返回:
- `Hermitian{Float64, Matrix{Float64}}`: 维度为 `(N_sites, N_sites)` 的 spinless hopping 矩阵。
"""
function build_twist_spinless_u0_hopping_matrix(
    lx::Int,
    ly::Int,
    tx::Float64,
    ty::Float64,
    t2::Float64,
)
    n_sites = lx * ly
    hopping_matrix = zeros(Float64, n_sites, n_sites)
    hopping_bonds = build_twist_nearest_neighbor_bonds(lx, ly)
    for (site_i, site_j) in hopping_bonds.x_bonds
        hopping_matrix[site_i, site_j] += -tx
        hopping_matrix[site_j, site_i] += -tx
    end
    for (site_i, site_j) in hopping_bonds.y_bonds
        hopping_matrix[site_i, site_j] += -ty
        hopping_matrix[site_j, site_i] += -ty
    end
    for (site_i, site_j) in hopping_bonds.diagonal_bonds
        hopping_matrix[site_i, site_j] += -t2
        hopping_matrix[site_j, site_i] += -t2
    end
    return Hermitian(hopping_matrix)
end

"""
用途: 显式构造固定 `N_up/N_down` 的 nonPH 与 PH 等价 orbitals。

数学公式:
- 先对 spinless hopping 矩阵 `h phi_a = epsilon_a phi_a` 对角化。
- nonPH electron determinant 使用 up/down block 中最低 `N_up/N_down` 个 `phi_a`。
- PH determinant 的 lower block 是 down-hole sector, 使用 down electron 未占据轨道
  `phi_{N_down+1}, ..., phi_N`, 这与 down electron determinant 通过 particle-hole
  complement identity 等价, 只差一个与构型无关的整体符号。

参数:
- `lx, ly::Int`: 晶格尺寸。
- `tx, ty::Float64`: x/y 最近邻 hopping。
- `t2::Float64`: 对角次近邻 hopping。
- `nup, ndn::Int`: 固定 up/down 电子数。

返回:
- `NamedTuple`: 包含 `spatial_eigenvalues`, `nonph_orbitals`, `ph_orbitals`, `exact_energy`。
"""
function build_twist_equivalent_u0_orbitals(
    lx::Int,
    ly::Int,
    tx::Float64,
    ty::Float64,
    t2::Float64,
    nup::Int,
    ndn::Int,
)
    n_sites = lx * ly
    spatial_eigen = eigen(build_twist_spinless_u0_hopping_matrix(lx, ly, tx, ty, t2))
    spatial_eigenvalues = spatial_eigen.values
    spatial_orbitals = spatial_eigen.vectors

    nonph_orbitals = zeros(Float64, 2 * n_sites, nup + ndn)
    if nup > 0
        nonph_orbitals[1:2:(2*n_sites), 1:nup] = spatial_orbitals[:, 1:nup]
    end
    if ndn > 0
        nonph_orbitals[2:2:(2*n_sites), (nup+1):(nup+ndn)] = spatial_orbitals[:, 1:ndn]
    end

    ph_orbitals = zeros(Float64, 2 * n_sites, n_sites + nup - ndn)
    if nup > 0
        ph_orbitals[1:2:(2*n_sites), 1:nup] = spatial_orbitals[:, 1:nup]
    end
    hole_count = n_sites - ndn
    if hole_count > 0
        ph_orbitals[2:2:(2*n_sites), (nup+1):(nup+hole_count)] =
            spatial_orbitals[:, (ndn+1):n_sites]
    end

    exact_energy = sum(spatial_eigenvalues[1:nup]) + sum(spatial_eigenvalues[1:ndn])
    return (
        spatial_eigenvalues=spatial_eigenvalues,
        nonph_orbitals=nonph_orbitals,
        ph_orbitals=ph_orbitals,
        exact_energy=exact_energy,
    )
end

"""
用途: 根据晶格、掺杂和 total Sz 计算 Hubbard benchmark 使用的粒子数。

数学公式:
- 总电子数 `N_e = N_sites * (1 + doping)`。
- `target_sz = N_up - N_down`。
- 因此 `N_up = (N_e + target_sz) / 2`, `N_down = N_e - N_up`。

参数:
- `lx, ly::Int`: 二维晶格尺寸。
- `doping::Float64`: 掺杂, 使用本项目约定 `N_e = N_sites * (1 + doping)`。
- `target_sz::Int`: 目标 total Sz, 即 `N_up - N_down`。

返回:
- `NamedTuple`: 包含 `n_sites`, `nelec`, `nup`, `ndn`。
"""
function compute_twist_benchmark_particle_numbers(
    lx::Int,
    ly::Int,
    doping::Float64,
    target_sz::Int,
)
    n_sites = lx * ly
    electron_count_float = n_sites * (1 + doping)
    nelec = round(Int, electron_count_float)
    if !isapprox(electron_count_float, nelec; atol=1.0e-8, rtol=0.0)
        error("N_sites * (1 + doping) must be an integer, got $(electron_count_float).")
    end
    if (target_sz + nelec) % 2 != 0
        error("Wrong parity between target_sz=$(target_sz) and N_e=$(nelec).")
    end
    nup = (nelec + target_sz) ÷ 2
    ndn = nelec - nup
    if nup < 0 || ndn < 0 || nup > n_sites || ndn > n_sites
        error("Invalid particle numbers: nup=$(nup), ndn=$(ndn), N_sites=$(n_sites).")
    end
    return (; n_sites=n_sites, nelec=nelec, nup=nup, ndn=ndn)
end

"""
用途: 构造 U=0 benchmark 的 physical Hubbard Hamiltonian。

数学公式:
- `H = -sum_{ij,sigma} t_ij c^dag_{i,sigma} c_{j,sigma}`。
- 本阶段 benchmark 设置 `U = 0`, 因此没有 onsite interaction。

参数:
- `lx, ly::Int`: 晶格尺寸。
- `tx, ty::Float64`: x/y 最近邻 hopping。
- `t2::Float64`: 对角次近邻 hopping。

返回:
- `GeneralModel`: physical Hamiltonian。
"""
function build_twist_u0_benchmark_model(
    lx::Int,
    ly::Int,
    tx::Float64,
    ty::Float64,
    t2::Float64,
)::GeneralModel
    term_setup = build_twist_hamiltonian_terms(lx, ly, tx, ty, t2, 0.0)
    return GeneralModel(lx * ly, term_setup.all_terms)
end

"""
用途: 构造 U=0 benchmark 的 nonPH/no-backflow determinant 波函数。

参数:
- `lx, ly::Int`: 晶格尺寸。
- `tx, ty::Float64`: x/y 最近邻 hopping, 也作为 mean-field hopping。
- `t2::Float64`: 对角次近邻 hopping。
- `nup, ndn::Int`: 物理 up/down 电子数。
- `occupied_orbitals::Matrix{Float64}`: 显式固定 Sz 的 nonPH 占据轨道矩阵。

返回:
- `vwf_det`: 已初始化 Slater 矩阵的 nonPH determinant 波函数。
"""
function build_twist_nonph_u0_benchmark_wavefunction(
    lx::Int,
    ly::Int,
    nup::Int,
    ndn::Int,
    occupied_orbitals::Matrix{Float64},
)
    n_sites = lx * ly
    nelec = nup + ndn
    sampler = config_Hubbard(n_sites, nup, ndn; ifPH=false)
    init_config_Hubbard!(sampler)

    vwf = vwf_det(zeros(Float64, 2 * n_sites, nelec), sampler; backflow=NoBackflowTerm())
    copyto!(vwf.base_gs_U, occupied_orbitals)
    copyto!(vwf.gs_U, occupied_orbitals)
    copyto!(vwf.gs_U_t, permutedims(occupied_orbitals))
    return vwf
end

"""
用途: 构造 U=0 benchmark 的 PH/no-backflow determinant 波函数。

参数:
- `lx, ly::Int`: 晶格尺寸。
- `tx, ty::Float64`: x/y 最近邻 hopping, 也作为 PH mean-field hopping。
- `t2::Float64`: 对角次近邻 hopping。
- `nup, ndn::Int`: 物理 up/down 电子数。
- `target_sz::Int`: 目标 total Sz, PH 占据轨道数为 `N_sites + target_sz`。
- `occupied_orbitals::Matrix{Float64}`: 显式固定 Sz 的 PH 占据轨道矩阵。

返回:
- `vwf_det`: 已初始化 Slater 矩阵的 PH determinant 波函数。
"""
function build_twist_ph_u0_benchmark_wavefunction(
    lx::Int,
    ly::Int,
    nup::Int,
    ndn::Int,
    target_sz::Int,
    occupied_orbitals::Matrix{Float64},
)
    n_sites = lx * ly
    n_occupied_orbitals = n_sites + target_sz
    sampler = config_Hubbard(n_sites, nup, ndn; ifPH=true)
    init_config_Hubbard!(sampler)

    vwf = vwf_det(zeros(Float64, 2 * n_sites, n_occupied_orbitals), sampler; backflow=NoBackflowTerm())
    copyto!(vwf.base_gs_U, occupied_orbitals)
    copyto!(vwf.gs_U, occupied_orbitals)
    copyto!(vwf.gs_U_t, permutedims(occupied_orbitals))
    return vwf
end

"""
用途: 根据 sampler 当前 `state` 重建 Hubbard determinant 使用的电子 row 映射。

参数:
- `sampler`: `Configuration` 或 `ConfigurationPH`。

返回:
- `nothing`。函数原地更新 `list_*`, `map_spin_to_id`, `electron_locs`。

说明:
- nonPH row set 是 up electron rows 加 down electron rows。
- PH row set 是 up electron rows 加 down-hole rows, 即没有 down electron 的 lower rows。
"""
function rebuild_twist_benchmark_sampler_maps_from_state!(sampler)::Nothing
    initialize_lists!(sampler)
    fill!(sampler.map_spin_to_id, 0)
    fill!(sampler.electron_locs, 0)

    electron_id = 0
    for site in 1:sampler.N_sites
        if has_up(sampler.state[site])
            electron_id += 1
            row_index = 2 * (site - 1) + 1
            sampler.map_spin_to_id[row_index] = electron_id
            sampler.electron_locs[electron_id] = row_index
        end
    end
    if electron_id != sampler.N_up
        error("Inconsistent up count while rebuilding benchmark sampler maps.")
    end

    for site in 1:sampler.N_sites
        state = sampler.state[site]
        use_lower_row = ifPH(sampler) ? !has_dn(state) : has_dn(state)
        if use_lower_row
            electron_id += 1
            row_index = 2 * (site - 1) + 2
            sampler.map_spin_to_id[row_index] = electron_id
            sampler.electron_locs[electron_id] = row_index
        end
    end
    if electron_id != total_elec(sampler)
        error("Inconsistent total row count while rebuilding benchmark sampler maps.")
    end
    return nothing
end

"""
用途: 为 PH/nonPH benchmark 选择一个共同的非奇异物理构型。

参数:
- `nonph_vwf`: nonPH determinant 波函数。
- `ph_vwf`: PH determinant 波函数。
- `seed::Int`: 搜索构型时使用的随机种子。
- `max_attempts::Int`: 最大尝试次数。

返回:
- `nothing`。函数会原地更新两个 sampler 的 `state/electron_locs` 和 Slater 逆矩阵。

说明:
- PH 与 nonPH 必须评估同一个电子构型, 才能比较 local energy。
- 某些小尺寸体系的节点构型会让 Slater 矩阵奇异, 因此这里用固定 seed 搜索共同非奇异构型。
"""
function initialize_matching_stable_twist_benchmark_config!(
    nonph_vwf,
    ph_vwf;
    seed::Int=20260627,
    max_attempts::Int=200,
)::Nothing
    nonph_sampler = nonph_vwf.sampler
    ph_sampler = ph_vwf.sampler
    Random.seed!(seed)
    for _ in 1:max_attempts
        init_config_Hubbard!(nonph_sampler)
        candidate_state = copy(nonph_sampler.state)
        nonph_sampler.state .= candidate_state
        ph_sampler.state .= candidate_state

        try
            rebuild_twist_benchmark_sampler_maps_from_state!(nonph_sampler)
            rebuild_twist_benchmark_sampler_maps_from_state!(ph_sampler)
            mfVMC.VMC.rebuild_slater_state!(nonph_vwf)
            mfVMC.VMC.rebuild_slater_state!(ph_vwf)
            return nothing
        catch err
            if !(err isa SingularException)
                rethrow()
            end
        end
    end
    error("Failed to find a shared non-singular benchmark configuration after $(max_attempts) attempts.")
end

"""
用途: 计算 U=0 benchmark 的 single-particle 精确能量。

数学公式:
- 对 non-interacting Slater determinant, 多体能量为最低 `N_e` 个 single-particle
  eigenvalue 的和: `E_exact = sum(epsilon[1:N_e])`。

参数:
- `lx, ly::Int`: 晶格尺寸。
- `tx, ty::Float64`: x/y 最近邻 hopping。
- `t2::Float64`: 对角次近邻 hopping。
- `nelec::Int`: 总电子数。

返回:
- `Float64`: non-interacting 精确能量。
"""
function compute_twist_nonph_u0_exact_energy(
    lx::Int,
    ly::Int,
    tx::Float64,
    ty::Float64,
    t2::Float64,
    nelec::Int,
)::Float64
    eigenvalues = eigvals(build_twist_spinless_u0_hopping_matrix(lx, ly, tx, ty, t2))
    eigenvalues = repeat(eigenvalues, inner=2)
    sort!(eigenvalues)
    return sum(eigenvalues[1:nelec])
end

"""
用途: 运行 PH/no-backflow 与 nonPH/no-backflow 的 U=0 deterministic benchmark。

参数:
- `lx, ly::Int`: 晶格尺寸。
- `tx, ty::Float64`: x/y 最近邻 hopping。
- `t2::Float64`: 对角次近邻 hopping。
- `doping::Float64`: 掺杂。
- `target_sz::Int`: 目标 total Sz。

返回:
- `NamedTuple`: 包含粒子数、PH/nonPH 占据轨道数、精确能量和两种 determinant 的 local energy。
"""
function run_twist_hubbard_ph_nonph_u0_benchmark(;
    lx::Int=2,
    ly::Int=2,
    tx::Float64=1.0,
    ty::Float64=1.0,
    t2::Float64=0.0,
    doping::Float64=0.0,
    target_sz::Int=0,
)
    numbers = compute_twist_benchmark_particle_numbers(lx, ly, doping, target_sz)
    model = build_twist_u0_benchmark_model(lx, ly, tx, ty, t2)
    orbitals = build_twist_equivalent_u0_orbitals(
        lx,
        ly,
        tx,
        ty,
        t2,
        numbers.nup,
        numbers.ndn,
    )
    nonph_vwf = build_twist_nonph_u0_benchmark_wavefunction(
        lx,
        ly,
        numbers.nup,
        numbers.ndn,
        orbitals.nonph_orbitals,
    )
    ph_vwf = build_twist_ph_u0_benchmark_wavefunction(
        lx,
        ly,
        numbers.nup,
        numbers.ndn,
        target_sz,
        orbitals.ph_orbitals,
    )
    initialize_matching_stable_twist_benchmark_config!(nonph_vwf, ph_vwf)
    exact_energy = orbitals.exact_energy
    nonph_energy = real(local_energy(model, nonph_vwf))
    ph_energy = real(local_energy(model, ph_vwf))

    return (
        n_sites=numbers.n_sites,
        nelec=numbers.nelec,
        nup=numbers.nup,
        ndn=numbers.ndn,
        nonph_occupied_orbitals=size(nonph_vwf.gs_U, 2),
        ph_occupied_orbitals=size(ph_vwf.gs_U, 2),
        exact_energy=exact_energy,
        nonph_energy=nonph_energy,
        ph_energy=ph_energy,
        nonph_error=nonph_energy - exact_energy,
        ph_error=ph_energy - exact_energy,
        ph_nonph_difference=ph_energy - nonph_energy,
    )
end

"""
用途: 打印 U=0 benchmark 的核心结果, 方便命令行人工检查。

参数:
- `result`: `run_twist_hubbard_ph_nonph_u0_benchmark` 返回的 NamedTuple。

返回:
- `nothing`。
"""
function print_twist_hubbard_ph_nonph_u0_benchmark(result)::Nothing
    println("twist Hubbard PH/nonPH U=0 benchmark")
    println("N_sites=$(result.n_sites), N_e=$(result.nelec), N_up=$(result.nup), N_down=$(result.ndn)")
    println("N_occ_nonPH=$(result.nonph_occupied_orbitals), N_occ_PH=$(result.ph_occupied_orbitals)")
    println("E_exact=$(result.exact_energy)")
    println("E_nonPH=$(result.nonph_energy), error=$(result.nonph_error)")
    println("E_PH=$(result.ph_energy), error=$(result.ph_error)")
    println("E_PH - E_nonPH=$(result.ph_nonph_difference)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_twist_hubbard_ph_nonph_u0_benchmark()
    print_twist_hubbard_ph_nonph_u0_benchmark(result)
end
