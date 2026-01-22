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
        default = 8
        "--bcx"
        help = "Boundary condition phase in X (1.0 or -1.0)"
        arg_type = Float64
        default = 0.999
        "--bcy"
        help = "Boundary condition phase in Y (1.0 or -1.0)"
        arg_type = Float64
        default = 1.001
        "--J1"
        help = "J1 coupling"
        arg_type = Float64
        default = 1.0
        "--J2"
        help = "J2 coupling"
        arg_type = Float64
        default = 1.0
        "--J3"
        help = "J3 coupling"
        arg_type = Float64
        default = 2.0
        "--Ndefect"
        help = "Number of defects"
        arg_type = Int
        default = 0
        "--defectansatz"
        help = "Defect position ansatz (FPS only)"
        arg_type = String
        default = "FPS"
        "--target_sz"
        help = "target total Sz"
        arg_type = Int
        default = 0
        "--mu"
        help = "chemical potential"
        arg_type = Float64
        default = 0.0
        "--chi1"
        help = "MF parameter chi1"
        arg_type = Float64
        default = 1.0
        "--etad1"
        help = "MF parameter etad1"
        arg_type = Float64
        default = 0.25
        "--etas1"
        help = "MF parameter etas1"
        arg_type = Float64
        default = 0.01
        "--mz"
        help = "AFM order parameter base value"
        arg_type = Float64
        default = 0.2
        "--chi2"
        help = "defect parameter chi2"
        arg_type = Float64
        default = 0.1
        "--etad2"
        help = "defect parameter etad2"
        arg_type = Float64
        default = 0.1
        "--etas2"
        help = "defect parameter etas2"
        arg_type = Float64
        default = 0.01
        "--chi3"
        help = "defect parameter chi3"
        arg_type = Float64
        default = 0.1
        "--etad3"
        help = "defect parameter etad3"
        arg_type = Float64
        default = 0.1
        "--etas3"
        help = "defect parameter etas3"
        arg_type = Float64
        default = 0.01
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
    end

    return parse_args(s)
end

"""
    xy_to_id_1based(x::Int, y::Int, lx::Int, ly::Int)

将 (x, y) 转成 1-based site index (使用周期边界).
参数:
- x::Int, x 坐标 (1-based).
- y::Int, y 坐标 (1-based).
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
返回:
- Int, 1-based site index.
"""
function xy_to_id_1based(x::Int, y::Int, lx::Int, ly::Int)::Int
    return mod1(y, ly) + (mod1(x, lx) - 1) * ly
end

"""
    defect_correction_bond(bonds::Vector{Tuple{Int,Int}}, defect_index::Vector{Int})

对 bonds 进行 defect 修正:
- 删除包含 defect 的 bond.
- 重新编号: id_new = id_old - count(defect_index < id_old).
参数:
- bonds::Vector{Tuple{Int,Int}}, 原始 1-based bonds.
- defect_index::Vector{Int}, defect 的 1-based index.
返回:
- Vector{Tuple{Int,Int}}, 修正后的 bonds (1-based).
"""
function defect_correction_bond(bonds::Vector{Tuple{Int,Int}}, defect_index::Vector{Int})
    index_set = Set(defect_index)
    new_bonds = Tuple{Int,Int}[]
    for (id0, id1) in bonds
        if id0 in index_set || id1 in index_set
            continue
        end
        id0_new = id0 - count(<(id0), defect_index)
        id1_new = id1 - count(<(id1), defect_index)
        push!(new_bonds, (id0_new, id1_new))
    end
    return new_bonds
end

"""
    build_defect_bonds(lx::Int, ly::Int, defect_positions::Vector{Tuple{Int,Int}})

构造含 defect 的 J1/J2/J3 bonds (1-based).
参数:
- lx, ly: 晶格尺寸.
- defect_positions: defect 坐标 (1-based).
返回:
- Tuple{Vector{Tuple{Int,Int}},Vector{Tuple{Int,Int}},Vector{Tuple{Int,Int}}}, 分别对应 J1/J2/J3 bonds.
"""
function build_defect_bonds(lx::Int, ly::Int, defect_positions::Vector{Tuple{Int,Int}})
    bonds_j1 = Tuple{Int,Int}[]
    bonds_j2 = Tuple{Int,Int}[]
    bonds_j3 = Tuple{Int,Int}[]

    for x in 1:lx, y in 1:ly
        id0 = xy_to_id_1based(x, y, lx, ly)
        idx = xy_to_id_1based(x + 1, y, lx, ly)
        idy = xy_to_id_1based(x, y + 1, lx, ly)
        push!(bonds_j1, (id0, idx))
        push!(bonds_j1, (id0, idy))
    end

    for (x, y) in defect_positions
        idpx = xy_to_id_1based(x + 1, y, lx, ly)
        idpy = xy_to_id_1based(x, y + 1, lx, ly)
        idmx = xy_to_id_1based(x - 1, y, lx, ly)
        idmy = xy_to_id_1based(x, y - 1, lx, ly)
        push!(bonds_j2, (idpx, idpy))
        push!(bonds_j2, (idpx, idmy))
        push!(bonds_j2, (idmx, idpy))
        push!(bonds_j2, (idmx, idmy))
        push!(bonds_j3, (idpx, idmx))
        push!(bonds_j3, (idpy, idmy))
    end

    return bonds_j1, bonds_j2, bonds_j3
end

"""
    build_defect_param_names_and_init_params(lx::Int, ly::Int, defect_positions::Vector{Tuple{Int,Int}}, args)

生成 defect ansatz 的参数列表与初始值.
参数:
- lx, ly: 晶格尺寸.
- defect_positions: defect 坐标 (1-based).
- args: ArgParse 解析结果.
返回:
- Tuple{Vector{Symbol},Vector{Float64}}。
"""
function build_defect_param_names_and_init_params(
    lx::Int,
    ly::Int,
    defect_positions::Vector{Tuple{Int,Int}},
    args
)
    param_names = Symbol[:etad1, :etas1, :mu]
    defect_set = Set(defect_positions)

    default_mz = Dict{Symbol,Float64}()
    base_mz = args["mz"]
    for x in 1:lx, y in 1:ly
        default_mz[Symbol("mz_$(x)_$(y)")] = base_mz * (-1)^(x + y)
    end

    for (x, y) in defect_positions
        xp = (mod1(x + 1, lx), y)
        yp = (x, mod1(y + 1, ly))
        xm = (mod1(x - 1, lx), y)
        ym = (x, mod1(y - 1, ly))
        around = [xp, yp, xm, ym]
        for (ax, ay) in around
            default_mz[Symbol("mz_$(ax)_$(ay)")] = 0.0
        end
    end

    for x in 1:lx, y in 1:ly
        if (x, y) ∉ defect_set
            push!(param_names, Symbol("mz_$(x)_$(y)"))
        else
            push!(param_names, Symbol("chi2_$(x)_$(y)"))
            push!(param_names, Symbol("chi3_$(x)_$(y)"))
            push!(param_names, Symbol("etas2_$(x)_$(y)"))
            push!(param_names, Symbol("etas3_$(x)_$(y)"))
            push!(param_names, Symbol("etad2_$(x)_$(y)"))
            push!(param_names, Symbol("etad3_$(x)_$(y)"))
        end
    end

    init_params = Float64[]
    for name in param_names
        name_str = String(name)
        if startswith(name_str, "mz_")
            push!(init_params, default_mz[name])
        elseif name == :etad1 || name == :etas1 || name == :mu
            push!(init_params, args[String(name)])
        else
            idx = findfirst('_', name_str)
            prefix = name_str[1:idx-1]
            push!(init_params, args[prefix])
        end
    end

    return param_names, init_params
end

"""
    update_defect_ansatz!(vwf, param_names, params, lx, ly, bcx, bcy, chi1, defect_positions, defect_index, target_sz)

根据 defect 参数更新波函数矩阵与导数.
参数:
- vwf: 波函数对象.
- param_names: 参数名列表.
- params: 参数值列表.
- lx, ly, bcx, bcy: 晶格与边界条件.
- chi1: 固定的 chi1 参数.
- defect_positions, defect_index: defect 位置与 index (1-based).
- target_sz: 目标 total Sz.
返回:
- nothing.
"""
function update_defect_ansatz!(
    vwf,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx::Int,
    ly::Int,
    bcx::Float64,
    bcy::Float64,
    chi1::Float64,
    defect_positions::Vector{Tuple{Int,Int}},
    defect_index::Vector{Int},
    target_sz::Int
)
    param_map = Dict{Symbol,Float64}(zip(param_names, params))

    mu = get(param_map, :mu, 0.0)
    etad1 = get(param_map, :etad1, 0.0)
    etas1 = get(param_map, :etas1, 0.0)

    mz = Dict{Symbol,Float64}()
    chi2 = Dict{Symbol,Float64}()
    etad2 = Dict{Symbol,Float64}()
    etas2 = Dict{Symbol,Float64}()
    chi3 = Dict{Symbol,Float64}()
    etad3 = Dict{Symbol,Float64}()
    etas3 = Dict{Symbol,Float64}()

    for (name, value) in param_map
        name_str = String(name)
        if startswith(name_str, "mz_")
            mz[name] = value
        elseif startswith(name_str, "chi2_")
            chi2[name] = value
        elseif startswith(name_str, "etad2_")
            etad2[name] = value
        elseif startswith(name_str, "etas2_")
            etas2[name] = value
        elseif startswith(name_str, "chi3_")
            chi3[name] = value
        elseif startswith(name_str, "etad3_")
            etad3[name] = value
        elseif startswith(name_str, "etas3_")
            etas3[name] = value
        elseif name == :mu || name == :etad1 || name == :etas1
            continue
        else
            error("Unknown parameter name: $name")
        end
    end

    defect_params = PartonSquare.DefectHeisenbergParams(
        Lx=lx,
        Ly=ly,
        bcx=bcx,
        bcy=bcy,
        mu=mu,
        chi1=chi1,
        etad1=etad1,
        etas1=etas1,
        mz=mz,
        chi2=chi2,
        etad2=etad2,
        etas2=etas2,
        chi3=chi3,
        etad3=etad3,
        etas3=etas3,
        defect_positions=defect_positions,
        defect_index=defect_index
    )

    _, gs_u, dut_params = PartonSquare.make_defect_ansatz_and_derivs(
        defect_params;
        param_names=param_names,
        target_sz=target_sz
    )

    copyto!(vwf.gs_U, gs_u)
    copyto!(vwf.gs_U_t, permutedims(gs_u))
    update_vwf_params!(vwf, dut_params)
    init_gswf!(vwf)
end

struct SxSq_Estimator
    N::Int
    Lx::Int
    Ly::Int
    qx::Float64
    qy::Float64
    # 预先计算坐标，避免在循环中反复计算
    coords::Vector{Tuple{Int,Int}}

    function SxSq_Estimator(Lx, Ly, qx, qy)
        N = Lx * Ly
        coords = Vector{Tuple{Int,Int}}(undef, N)
        for idx in 1:N
            # 这里的索引转换逻辑需确保与你的 Model/Sampler 一致
            # 假设 idx 1-based, 先x后y: x = (idx-1) // Ly + 1, y = mod1(idx, Ly)
            x = div(idx - 1, Ly) + 1
            y = mod1(idx, Ly)
            coords[idx] = (x, y)
        end
        new(N, Lx, Ly, qx, qy, coords)
    end
end

function (est::SxSq_Estimator)(model, vwf)
    acc = 0.0 + 0.0im

    # --- 策略选择 ---
    # 如果 N 很大 (>100)，全双重求和 O(N^2) 会非常慢，因为 measure_SxSx 本身涉及矩阵更新。
    # 我们可以采用 "随机批次参考点" (Stochastic Sampling of Reference Sites)
    # 或者如果 N 较小 (6x6=36)，直接全求和即可。

    # 这里演示全双重求和 (最稳健，适用于 Lx=6, Ly=6)

    for site_i in 1:est.N
        ri_x, ri_y = est.coords[site_i]

        for site_j in 1:est.N
            rj_x, rj_y = est.coords[site_j]

            # 1. 计算位移向量 r = rj - ri
            dx = rj_x - ri_x
            dy = rj_y - ri_y

            # 2. 计算相位 exp(-i * q * (rj - ri))
            dot_prod = est.qx * dx + est.qy * dy
            phase = cis(-dot_prod)

            # 3. 测量 <Sx_i Sx_j>
            # 注意：measure_SxSx 内部通常已经处理了 Hermitian 对称性，
            # 即它返回 real(<Si Sj>)。如果它只计算 S+ S-，则需要小心。
            val_sxsx = measure_SxSx(vwf, site_i, site_j; conserve_sz=true)

            acc += val_sxsx * phase
        end
    end

    # 4. 归一化
    # S(q) = 1/N * sum_{ij} e^{...} <Si Sj>
    return acc / est.N / est.N
end

struct SzSq_Estimator
    N::Int
    phases::Vector{ComplexF64} # 存储预计算好的 exp(-i * q * r_j)

    function SzSq_Estimator(Lx, Ly, qx, qy)
        N = Lx * Ly
        phases = Vector{ComplexF64}(undef, N)

        # 预先计算所有格点的相位因子
        # 遍历顺序必须与 Sampler/Wavefunction 中的格点索引顺序一致
        # 假设索引逻辑是: idx = (x-1)*Ly + y  (Column-major, Julia默认)
        for x in 1:Lx
            for y in 1:Ly
                idx = (x - 1) * Ly + y

                # 物理坐标 (由 1-based 转换为 0-based)
                rx = x - 1
                ry = y - 1

                # 计算相位: exp(-i * q * r)
                dot_prod = qx * rx + qy * ry
                phases[idx] = cis(-dot_prod)
            end
        end
        new(N, phases)
    end
end

function (est::SzSq_Estimator)(model, vwf)
    # magnetisation_q = sum_j Sz_j * exp(-i q r_j)
    # 这是一个复数序参数
    m_q = sum(get_Sz(s) * p for (s, p) in zip(vwf.sampler.state, est.phases))
    return abs2(m_q) / est.N / est.N
end


function total_Sz_est(model, vwf)
    return measure_total_Sz(vwf)
end

# ==============================================================================
# 3. 辅助函数
# ==============================================================================

function update_ansatz_for_phi!(vwf, params::Vector{Float64}, LX, LY, BCX, BCY)
    phi_val = params[1]
    # 这里也可以把 bcx, bcy 提出来作为参数
    flux_params = PartonSquare.U1SFluxParams(Lx=LX, Ly=LY, phi=phi_val, bcy=BCX, bcx=BCY)
    _, gs_U, dUt_params = PartonSquare.make_ansatz_and_derivs(flux_params)

    copyto!(vwf.gs_U, gs_U)
    copyto!(vwf.gs_U_t, permutedims(gs_U))
    update_vwf_params!(vwf, dUt_params)
    init_gswf!(vwf)
end

# ==============================================================================
# 4. 主程序
# ==============================================================================

