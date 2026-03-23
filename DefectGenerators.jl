module DefectGenerators

using Random

include("FPS.jl")

export generate_defect_positions
export generate_defect_positions_fps
export generate_defect_positions_random_sublattice_balanced
export generate_defect_positions_soft_fps
export is_sublattice_a
export list_sublattice_points

const SOFT_FPS_WEIGHT_EPSILON = 1e-12

"""
    is_sublattice_a(x::Int, y::Int) -> Bool

用途: 判断二维方格点 `(x, y)` 是否位于 A 子格.
参数:
- x::Int, x 坐标, 采用 1-based 编号.
- y::Int, y 坐标, 采用 1-based 编号.
返回:
- Bool, `true` 表示 A 子格, `false` 表示 B 子格.
公式:
- A 子格判据为 `(x + y) % 2 == 0`.
"""
function is_sublattice_a(x::Int, y::Int)::Bool
    return FPS.is_sublattice_a(x, y)
end

"""
    list_sublattice_points(lx::Int, ly::Int, want_a::Bool) -> Vector{Tuple{Int,Int}}

用途: 列出二维方格中指定子格的全部格点坐标.
参数:
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
- want_a::Bool, `true` 表示返回 A 子格, `false` 表示返回 B 子格.
返回:
- Vector{Tuple{Int,Int}}, 指定子格上的全部 `(x, y)` 坐标, 按字典序排列.
"""
function list_sublattice_points(
    lx::Int,
    ly::Int,
    want_a::Bool
)::Vector{Tuple{Int,Int}}
    return FPS.list_sublattice_points(lx, ly, want_a)
end

"""
    list_all_lattice_points(lx::Int, ly::Int) -> Vector{Tuple{Int,Int}}

用途: 列出二维方格中的全部格点坐标.
参数:
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
返回:
- Vector{Tuple{Int,Int}}, 全部 `(x, y)` 坐标, 按字典序排列.
"""
function list_all_lattice_points(
    lx::Int,
    ly::Int
)::Vector{Tuple{Int,Int}}
    if lx <= 0 || ly <= 0
        throw(ArgumentError("lx and ly must be positive."))
    end

    lattice_points = Tuple{Int,Int}[]
    sizehint!(lattice_points, lx * ly)
    for x in 1:lx
        for y in 1:ly
            push!(lattice_points, (x, y))
        end
    end
    return lattice_points
end

"""
    generate_defect_positions_fps(
        lx::Int,
        ly::Int,
        ndefect::Int;
        first_defect::Tuple{Int,Int}
    ) -> Vector{Tuple{Int,Int}}

用途: 调用现有 FPS 生成器生成 defect 位置.
参数:
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
- ndefect::Int, defect 数量.
- first_defect::Tuple{Int,Int}, FPS 的起始 defect 坐标, 采用 1-based 编号.
返回:
- Vector{Tuple{Int,Int}}, defect 坐标列表, 每项均为 1-based `(x, y)`.
"""
function generate_defect_positions_fps(
    lx::Int,
    ly::Int,
    ndefect::Int;
    first_defect::Tuple{Int,Int}=(div(lx, 3) + 1, div(ly, 3) + 1)
)::Vector{Tuple{Int,Int}}
    return FPS.generate_defect_positions_fps(
        lx,
        ly,
        ndefect;
        first_defect=first_defect
    )
end

"""
    choose_random_balanced_sublattice_counts(
        ndefect::Int,
        n_points_a::Int,
        n_points_b::Int,
        rng::AbstractRNG
    ) -> Tuple{Int,Int}

用途: 为 RANDOM 生成器决定 A/B 子格各自抽取的 defect 数量.
参数:
- ndefect::Int, 总 defect 数量.
- n_points_a::Int, A 子格可用格点数.
- n_points_b::Int, B 子格可用格点数.
- rng::AbstractRNG, 随机数生成器.
返回:
- Tuple{Int,Int}, `(count_a, count_b)`, 分别表示 A/B 子格 defect 数量.
规则:
- 若 `ndefect` 为偶数, 则 `count_a = count_b = ndefect / 2`.
- 若 `ndefect` 为奇数, 则在满足子格容量约束的前提下, 随机决定多出的 1 个 defect 落在 A 或 B 子格.
"""
function choose_random_balanced_sublattice_counts(
    ndefect::Int,
    n_points_a::Int,
    n_points_b::Int,
    rng::AbstractRNG
)::Tuple{Int,Int}
    if ndefect < 0
        throw(ArgumentError("ndefect must be non-negative."))
    end

    base_count = ndefect ÷ 2
    if iseven(ndefect)
        count_a = base_count
        count_b = base_count
        if count_a > n_points_a || count_b > n_points_b
            throw(ArgumentError("Balanced sublattice counts exceed available lattice points."))
        end
        return count_a, count_b
    end

    candidate_counts = Tuple{Int,Int}[]
    if base_count + 1 <= n_points_a && base_count <= n_points_b
        push!(candidate_counts, (base_count + 1, base_count))
    end
    if base_count <= n_points_a && base_count + 1 <= n_points_b
        push!(candidate_counts, (base_count, base_count + 1))
    end

    if isempty(candidate_counts)
        throw(ArgumentError("Cannot assign odd defect count under sublattice capacity constraints."))
    end

    selected_idx = rand(rng, 1:length(candidate_counts))
    return candidate_counts[selected_idx]
end

"""
    sample_points_without_replacement(
        candidates::Vector{Tuple{Int,Int}},
        n_pick::Int,
        rng::AbstractRNG
    ) -> Vector{Tuple{Int,Int}}

用途: 从候选格点中进行无放回随机抽样.
参数:
- candidates::Vector{Tuple{Int,Int}}, 候选格点列表.
- n_pick::Int, 需要抽取的格点数量.
- rng::AbstractRNG, 随机数生成器.
返回:
- Vector{Tuple{Int,Int}}, 长度为 `n_pick` 的随机抽样结果.
"""
function sample_points_without_replacement(
    candidates::Vector{Tuple{Int,Int}},
    n_pick::Int,
    rng::AbstractRNG
)::Vector{Tuple{Int,Int}}
    if n_pick < 0 || n_pick > length(candidates)
        throw(ArgumentError("n_pick must satisfy 0 <= n_pick <= length(candidates)."))
    end
    if n_pick == 0
        return Tuple{Int,Int}[]
    end

    permutation = randperm(rng, length(candidates))
    selected_points = Vector{Tuple{Int,Int}}(undef, n_pick)
    for idx in 1:n_pick
        selected_points[idx] = candidates[permutation[idx]]
    end
    return selected_points
end

"""
    sample_weighted_point_by_min_distance(
        candidates::Vector{Tuple{Int,Int}},
        chosen::Vector{Tuple{Int,Int}},
        used::Set{Tuple{Int,Int}},
        lx::Int,
        ly::Int,
        rng::AbstractRNG,
        alpha::Float64
    ) -> Tuple{Int,Int}

用途: 在 Soft FPS 中按最小 torus 距离平方的幂次权重随机抽样下一个 defect.
参数:
- candidates::Vector{Tuple{Int,Int}}, 全部候选格点列表.
- chosen::Vector{Tuple{Int,Int}}, 已选 defect 列表.
- used::Set{Tuple{Int,Int}}, 已选 defect 集合, 用于快速跳过已使用格点.
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
- rng::AbstractRNG, 随机数生成器.
- alpha::Float64, Soft FPS 的幂次参数, 要求 `alpha >= 0`.
返回:
- Tuple{Int,Int}, 本轮选中的 defect 坐标.
公式:
- 对每个候选点 `p`, 先计算
  `s(p) = min_{q in chosen} d_torus(p, q)^2`.
- 再按权重
  `w(p) = (s(p) + epsilon)^alpha`
  做加权随机抽样.
说明:
- 当 `alpha = 0` 时, 所有未选点的权重相同, 退化为普通均匀随机抽样.
- 当 `alpha` 增大时, 更偏向选择远离已选 defect 的格点.
"""
function sample_weighted_point_by_min_distance(
    candidates::Vector{Tuple{Int,Int}},
    chosen::Vector{Tuple{Int,Int}},
    used::Set{Tuple{Int,Int}},
    lx::Int,
    ly::Int,
    rng::AbstractRNG,
    alpha::Float64
)::Tuple{Int,Int}
    if alpha < 0
        throw(ArgumentError("alpha must be non-negative."))
    end

    available_points = Tuple{Int,Int}[]
    cumulative_weights = Float64[]
    total_weight = 0.0

    for point in candidates
        if point in used
            continue
        end

        min_distance_squared = FPS.min_distance_to_chosen(point, chosen, lx, ly)
        point_weight = alpha == 0.0 ? 1.0 : (float(min_distance_squared) + SOFT_FPS_WEIGHT_EPSILON)^alpha
        total_weight += point_weight
        push!(available_points, point)
        push!(cumulative_weights, total_weight)
    end

    if isempty(available_points)
        throw(ArgumentError("No available lattice point remains for Soft FPS sampling."))
    end

    target_weight = rand(rng) * total_weight
    selected_idx = searchsortedfirst(cumulative_weights, target_weight)
    if selected_idx > length(available_points)
        selected_idx = length(available_points)
    end
    return available_points[selected_idx]
end

"""
    generate_defect_positions_random_sublattice_balanced(
        lx::Int,
        ly::Int,
        ndefect::Int;
        seed::Int=1234
    ) -> Vector{Tuple{Int,Int}}

用途: 生成二维 RANDOM defect 分布.
参数:
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
- ndefect::Int, defect 总数量.
- seed::Int, 随机数种子. 相同 seed 应生成完全相同的 defect 配置.
返回:
- Vector{Tuple{Int,Int}}, defect 坐标列表, 每项均为 1-based `(x, y)`.
规则:
- 不加最近邻或更远距离约束.
- A/B 子格数量尽量平衡.
- 当 `ndefect` 为奇数时, 多出的 1 个 defect 随机分配到 A 或 B 子格.
- 返回前按 `(x, y)` 字典序排序, 便于复现与比对.
"""
function generate_defect_positions_random_sublattice_balanced(
    lx::Int,
    ly::Int,
    ndefect::Int;
    seed::Int=1234
)::Vector{Tuple{Int,Int}}
    @assert lx > 0 && ly > 0
    @assert ndefect >= 0 && ndefect <= lx * ly

    if ndefect == 0
        return Tuple{Int,Int}[]
    end

    rng = MersenneTwister(seed)
    points_a = list_sublattice_points(lx, ly, true)
    points_b = list_sublattice_points(lx, ly, false)
    count_a, count_b = choose_random_balanced_sublattice_counts(
        ndefect,
        length(points_a),
        length(points_b),
        rng
    )

    defect_positions = Tuple{Int,Int}[]
    append!(defect_positions, sample_points_without_replacement(points_a, count_a, rng))
    append!(defect_positions, sample_points_without_replacement(points_b, count_b, rng))
    sort!(defect_positions)
    return defect_positions
end

"""
    generate_defect_positions_soft_fps(
        lx::Int,
        ly::Int,
        ndefect::Int;
        seed::Int=1234,
        alpha::Float64=4.0
    ) -> Vector{Tuple{Int,Int}}

用途: 生成二维 SOFT_FPS defect 分布.
参数:
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
- ndefect::Int, defect 总数量.
- seed::Int, 随机数种子. 相同 seed 与相同 alpha 应生成完全相同的 defect 配置.
- alpha::Float64, Soft FPS 的幂次参数, 默认值为 `4.0`.
返回:
- Vector{Tuple{Int,Int}}, defect 坐标列表, 每项均为 1-based `(x, y)`.
规则:
- 第一个 defect 在全部 lattice site 上均匀随机选取.
- 之后每一步根据最小 torus 距离平方 `s(p)` 的幂次权重 `w(p)=s(p)^alpha` 做加权随机抽样.
- 不要求 A/B 子格平衡.
- 返回前按 `(x, y)` 字典序排序, 便于复现与比对.
"""
function generate_defect_positions_soft_fps(
    lx::Int,
    ly::Int,
    ndefect::Int;
    seed::Int=1234,
    alpha::Float64=4.0
)::Vector{Tuple{Int,Int}}
    if lx <= 0 || ly <= 0
        throw(ArgumentError("lx and ly must be positive."))
    end
    if ndefect < 0 || ndefect > lx * ly
        throw(ArgumentError("ndefect must satisfy 0 <= ndefect <= lx * ly."))
    end
    if alpha < 0
        throw(ArgumentError("alpha must be non-negative."))
    end
    if ndefect == 0
        return Tuple{Int,Int}[]
    end

    rng = MersenneTwister(seed)
    lattice_points = list_all_lattice_points(lx, ly)
    defect_positions = Tuple{Int,Int}[]

    first_index = rand(rng, 1:length(lattice_points))
    first_point = lattice_points[first_index]
    push!(defect_positions, first_point)
    used_points = Set{Tuple{Int,Int}}(defect_positions)

    for _ in 2:ndefect
        next_point = sample_weighted_point_by_min_distance(
            lattice_points,
            defect_positions,
            used_points,
            lx,
            ly,
            rng,
            alpha
        )
        push!(defect_positions, next_point)
        push!(used_points, next_point)
    end

    sort!(defect_positions)
    return defect_positions
end

"""
    generate_defect_positions(
        defect_ansatz::String,
        lx::Int,
        ly::Int,
        ndefect::Int;
        seed::Int=1234,
        alpha::Float64=4.0,
        first_defect::Union{Nothing,Tuple{Int,Int}}=nothing
    ) -> Vector{Tuple{Int,Int}}

用途: 统一调度 defect 位置生成器.
参数:
- defect_ansatz::String, 生成器名称. 当前推荐支持 `"FPS"`, `"RANDOM"` 与 `"SOFT_FPS"`, 其中 `"RANDOM_SUBLATTICE_BALANCED"` 保留为兼容别名.
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
- ndefect::Int, defect 总数量.
- seed::Int, 随机生成器使用的种子, 目前对 RANDOM 与 SOFT_FPS 生效.
- alpha::Float64, SOFT_FPS 的幂次参数, 默认值为 `4.0`.
- first_defect::Union{Nothing,Tuple{Int,Int}}, FPS 的起始 defect 坐标. 若为 `nothing`, 使用 FPS 默认值.
返回:
- Vector{Tuple{Int,Int}}, defect 坐标列表, 每项均为 1-based `(x, y)`.
"""
function generate_defect_positions(
    defect_ansatz::String,
    lx::Int,
    ly::Int,
    ndefect::Int;
    seed::Int=1234,
    alpha::Float64=4.0,
    first_defect::Union{Nothing,Tuple{Int,Int}}=nothing
)::Vector{Tuple{Int,Int}}
    normalized_ansatz = uppercase(defect_ansatz)
    if normalized_ansatz == "FPS"
        if isnothing(first_defect)
            return generate_defect_positions_fps(lx, ly, ndefect)
        end
        return generate_defect_positions_fps(lx, ly, ndefect; first_defect=first_defect)
    elseif normalized_ansatz == "RANDOM" || normalized_ansatz == "RANDOM_SUBLATTICE_BALANCED"
        return generate_defect_positions_random_sublattice_balanced(
            lx,
            ly,
            ndefect;
            seed=seed
        )
    elseif normalized_ansatz == "SOFT_FPS"
        return generate_defect_positions_soft_fps(
            lx,
            ly,
            ndefect;
            seed=seed,
            alpha=alpha
        )
    end

    throw(ArgumentError("Unsupported defect ansatz: $defect_ansatz"))
end

end # module
