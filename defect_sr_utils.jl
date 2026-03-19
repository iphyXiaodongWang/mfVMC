using JSON

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
        "--job"
        help = "Job to be done. Can be SR and measure"
        arg_type = String
        default = "SR"
    end

    return parse_args(s)
end

"""
    build_exponential_lr_func(lr_start::Float64, lr_end::Float64, n_steps::Int) -> Function

用途: 构造 SR 学习率指数衰减函数, 并保证最后一个 step 的学习率为 lr_end.
参数:
- lr_start::Float64, 初始学习率.
- lr_end::Float64, 最后一个 step 的目标学习率.
- n_steps::Int, SR 总步数.
返回:
- Function, 形如 f(lr0, step) -> lr_step, 供 run_sr_optimization 的 lr_func 调用.
公式:
- 当 n_steps >= 2 且 lr_start > 0 时, lr_step = lr_start * gamma^(step-1),
  其中 gamma = (lr_end / lr_start)^(1 / (n_steps - 1)).
- 当 lr_end = 0 时, gamma = 0, 因此 step=1 时 lr=lr_start, step>=2 时 lr=0.
"""
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

"""
    read_params_from_json(file_path::AbstractString) -> Dict{String, Float64}

用途: 读取包含参数键值对的 json 文件, 返回参数字典.
参数:
- file_path::AbstractString, json 文件路径, 格式形如 {"param": 0.1, ...}.
返回:
- Dict{String, Float64}, 参数名到数值的映射.
"""
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
    xy_to_reduced_id_1based(
        x::Int,
        y::Int,
        lx::Int,
        ly::Int,
        defect_index::Vector{Int}
    ) -> Union{Nothing,Int}

用途: 将 full lattice 中的 1-based 坐标 `(x, y)` 转换为去除 defect 后的 reduced site index.
参数:
- x::Int, x 坐标 (1-based).
- y::Int, y 坐标 (1-based).
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
- defect_index::Vector{Int}, defect 的 1-based full index 列表.
返回:
- Union{Nothing,Int}, 若 `(x, y)` 对应位置是 defect 则返回 `nothing`, 否则返回去除 defect 后的 1-based index.
公式:
- full_id = xy_to_id_1based(x, y, lx, ly)
- reduced_id = full_id - count(id_defect < full_id)
"""
function xy_to_reduced_id_1based(
    x::Int,
    y::Int,
    lx::Int,
    ly::Int,
    defect_index::Vector{Int}
)::Union{Nothing,Int}
    full_id = xy_to_id_1based(x, y, lx, ly)
    if full_id in Set(defect_index)
        return nothing
    end
    reduced_id = full_id - count(<(full_id), defect_index)
    return reduced_id
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
    build_defect_bonds_obc(lx::Int, ly::Int, defect_positions::Vector{Tuple{Int,Int}})

构造开边界条件下含 defect 的 J1/J2/J3 bonds (1-based).
参数:
- lx, ly: 晶格尺寸.
- defect_positions: defect 坐标 (1-based).
返回:
- Tuple{Vector{Tuple{Int,Int}},Vector{Tuple{Int,Int}},Vector{Tuple{Int,Int}}}, 分别对应 J1/J2/J3 bonds.
说明:
- OBC 下不允许通过 mod1 跨越边界回卷.
- J1 仅保留晶格内部最近邻.
- J2/J3 仅在两端点均位于晶格内部时保留.
"""
function build_defect_bonds_obc(lx::Int, ly::Int, defect_positions::Vector{Tuple{Int,Int}})
    bonds_j1 = Tuple{Int,Int}[]
    bonds_j2 = Tuple{Int,Int}[]
    bonds_j3 = Tuple{Int,Int}[]

    for x in 1:lx, y in 1:ly
        id0 = xy_to_id_1based(x, y, lx, ly)

        if x < lx
            idx = xy_to_id_1based(x + 1, y, lx, ly)
            push!(bonds_j1, (id0, idx))
        end

        if y < ly
            idy = xy_to_id_1based(x, y + 1, lx, ly)
            push!(bonds_j1, (id0, idy))
        end
    end

    for (x, y) in defect_positions
        if x < lx && y < ly
            idpx = xy_to_id_1based(x + 1, y, lx, ly)
            idpy = xy_to_id_1based(x, y + 1, lx, ly)
            push!(bonds_j2, (idpx, idpy))
        end

        if x < lx && y > 1
            idpx = xy_to_id_1based(x + 1, y, lx, ly)
            idmy = xy_to_id_1based(x, y - 1, lx, ly)
            push!(bonds_j2, (idpx, idmy))
        end

        if x > 1 && y < ly
            idmx = xy_to_id_1based(x - 1, y, lx, ly)
            idpy = xy_to_id_1based(x, y + 1, lx, ly)
            push!(bonds_j2, (idmx, idpy))
        end

        if x > 1 && y > 1
            idmx = xy_to_id_1based(x - 1, y, lx, ly)
            idmy = xy_to_id_1based(x, y - 1, lx, ly)
            push!(bonds_j2, (idmx, idmy))
        end

        if x > 1 && x < lx
            idpx = xy_to_id_1based(x + 1, y, lx, ly)
            idmx = xy_to_id_1based(x - 1, y, lx, ly)
            push!(bonds_j3, (idpx, idmx))
        end

        if y > 1 && y < ly
            idpy = xy_to_id_1based(x, y + 1, lx, ly)
            idmy = xy_to_id_1based(x, y - 1, lx, ly)
            push!(bonds_j3, (idpy, idmy))
        end
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

"""
    build_reduced_to_xy_map(
        lx::Int,
        ly::Int,
        n_sites::Int,
        defect_index::Vector{Int}
    ) -> Vector{Tuple{Int,Int}}

用途: 构造 reduced index 到 full lattice 坐标的映射.
参数:
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
- n_sites::Int, 去 defect 后格点数.
- defect_index::Vector{Int}, defect 的 1-based full index 列表.
返回:
- Vector{Tuple{Int,Int}}, 长度为 n_sites, 第 i 项对应 reduced site i 的 (x, y), 均为 1-based.
"""
function build_reduced_to_xy_map(
    lx::Int,
    ly::Int,
    n_sites::Int,
    defect_index::Vector{Int}
)::Vector{Tuple{Int,Int}}
    defect_set = Set(defect_index)
    reduced_to_xy = Vector{Tuple{Int,Int}}(undef, n_sites)
    reduced_idx = 0
    for x in 1:lx, y in 1:ly
        full_id = xy_to_id_1based(x, y, lx, ly)
        if full_id in defect_set
            continue
        end
        reduced_idx += 1
        reduced_to_xy[reduced_idx] = (x, y)
    end
    if reduced_idx != n_sites
        error("n_sites mismatch: expected $(n_sites), got $(reduced_idx).")
    end
    return reduced_to_xy
end

"""
    defination_observabels(
        lx::Int,
        ly::Int,
        n_sites::Int,
        defect_index::Vector{Int}
    ) -> Dict{Symbol,Function}

用途: 构造用于 run_simulation 的观测量字典.
参数:
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
- n_sites::Int, 去 defect 后格点数.
- defect_index::Vector{Int}, defect 的 1-based full index 列表.
返回:
- Dict{Symbol,Function}, Key 为观测量名称, Value 为匿名函数 (model, vwf) -> Number.
说明:
- 能量: :E, 使用 local_energy(model, vwf).
- 每点 Sz: :Sz_i, 使用 get_Sz(vwf.sampler.state[i]).
- 关联函数: :SS_i_j (i < j), 使用 measure_SiSj(vwf, i, j).
- 派生量: :staggered_mz 与 :S_pi_pi, 在测量 :Sz_i 和 :SS_i_j 时同步累加, 避免重复计算.
- 公式: S_i·S_j = Sz_i*Sz_j + 1/2*(S+_i S-_j + S-_i S+_j).
- 公式: staggered_mz = sum_i [(-1)^(x_i+y_i) * Sz_i] / (Lx*Ly).
- 公式: S_pi_pi = [2*sum_{i<j}((-1)^((x_i-x_j)+(y_i-y_j))*<Si·Sj>) + (3/4)*Nsite] / (Lx*Ly)^2.
"""
function defination_observabels(
    lx::Int,
    ly::Int,
    n_sites::Int,
    defect_index::Vector{Int}
)::Dict{Symbol,Function}
    observables = Dict{Symbol,Function}()
    reduced_to_xy = build_reduced_to_xy_map(lx, ly, n_sites, defect_index)
    norm_lattice = Float64(lx * ly)
    norm_sq = norm_lattice^2

    # 每个 sample 内在线累加, 由 :E 观测量触发重置.
    staggered_mz_acc = Ref(0.0)
    s_pi_pi_pair_acc = Ref(0.0)
    s_pi_pi_diag_const = 0.75 * n_sites / norm_sq

    observables[:E] = (model, vwf) -> begin
        staggered_mz_acc[] = 0.0
        s_pi_pi_pair_acc[] = 0.0
        return local_energy(model, vwf)
    end

    for i in 1:n_sites
        key = Symbol("Sz_$(i)")
        x, y = reduced_to_xy[i]
        stagger_phase = (-1)^(x + y)
        stagger_coeff = stagger_phase / norm_lattice
        observables[key] = (model, vwf) -> begin
            val = get_Sz(vwf.sampler.state[i])
            staggered_mz_acc[] += stagger_coeff * real(val)
            return val
        end
    end

    for i in 1:n_sites
        x0, y0 = reduced_to_xy[i]
        for j in (i+1):n_sites
            key = Symbol("SS_$(i)_$(j)")
            x1, y1 = reduced_to_xy[j]
            phase_pi_pi = (-1)^((x0 - x1) + (y0 - y1))
            ss_coeff = 2.0 * phase_pi_pi / norm_sq
            observables[key] = (model, vwf) -> begin
                val = measure_SiSj(vwf, i, j)
                s_pi_pi_pair_acc[] += ss_coeff * real(val)
                return val
            end
        end
    end

    observables[:S_pi_pi] = (model, vwf) -> s_pi_pi_pair_acc[] + s_pi_pi_diag_const
    observables[:staggered_mz] = (model, vwf) -> staggered_mz_acc[]

    return observables
end

"""
    save_measurement_outputs(
        mean_dict::Dict{Symbol,Any},
        lx::Int,
        ly::Int,
        n_sites::Int,
        defect_index::Vector{Int},
        folder::AbstractString
    ) -> Nothing

用途: 根据测量结果保存绘图所需的 Sz/SS JSON.
参数:
- mean_dict::Dict{Symbol,Any}, 观测量均值字典, 需包含 :Sz_i 与 :SS_i_j.
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
- n_sites::Int, 去 defect 后的格点数.
- defect_index::Vector{Int}, defect 的 1-based index 列表(对应 full lattice).
- folder::AbstractString, 输出文件夹路径.
返回:
- Nothing.
说明:
- Sz.json 仅写入非 defect 格点的 key, defect 位置不输出 key.
公式: 无.
"""
function save_measurement_outputs(
    mean_dict::Dict{Symbol,Any},
    lx::Int,
    ly::Int,
    n_sites::Int,
    defect_index::Vector{Int},
    folder::AbstractString
)::Nothing
    reduced_to_xy = build_reduced_to_xy_map(lx, ly, n_sites, defect_index)

    sz_json = Dict{String,Float64}()
    # 仅写入非 defect 的格点, defect 位置不输出 key, 便于下游用缺失值识别 defect.
    for i in 1:n_sites
        x, y = reduced_to_xy[i]
        key = "mz_$(x - 1)_$(y - 1)"
        sz_key = Symbol("Sz_$(i)")
        sz_json[key] = Float64(real(mean_dict[sz_key]))
    end
    open(joinpath(folder, "Sz.json"), "w") do io
        JSON.print(io, sz_json)
    end

    ss_json = Dict{String,Float64}()
    for i in 1:(n_sites-1)
        x0, y0 = reduced_to_xy[i]
        for j in (i+1):n_sites
            x1, y1 = reduced_to_xy[j]
            key = "SS_$(x0 - 1)_$(y0 - 1)_$(x1 - 1)_$(y1 - 1)"
            ss_key = Symbol("SS_$(i)_$(j)")
            ss_json[key] = Float64(real(mean_dict[ss_key]))
        end
    end
    open(joinpath(folder, "SS_all.json"), "w") do io
        JSON.print(io, ss_json)
    end

    return nothing
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

