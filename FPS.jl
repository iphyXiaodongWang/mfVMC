module FPS

export generate_defect_positions_fps

"""
    is_sublattice_a(x::Int, y::Int)

判断 (x, y) 是否为 A 子格.
参数:
- x::Int, x 坐标, 范围 [1, lx].
- y::Int, y 坐标, 范围 [1, ly].
返回:
- Bool, true 表示 A 子格, false 表示 B 子格.
"""
function is_sublattice_a(x::Int, y::Int)::Bool
    return (x + y) % 2 == 0
end

"""
    torus_distance_squared(p::Tuple{Int,Int}, q::Tuple{Int,Int}, lx::Int, ly::Int)

计算 2D torus 上的距离平方.
参数:
- p::Tuple{Int,Int}, 点 (x1, y1).
- q::Tuple{Int,Int}, 点 (x2, y2).
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
返回:
- Int, 距离平方.
公式:
- dx = min(|x1-x2|, lx-|x1-x2|), dy = min(|y1-y2|, ly-|y1-y2|),
  dist2 = dx^2 + dy^2.
"""
function torus_distance_squared(
    p::Tuple{Int,Int},
    q::Tuple{Int,Int},
    lx::Int,
    ly::Int
)::Int
    dx = abs(p[1] - q[1])
    dx = min(dx, lx - dx)
    dy = abs(p[2] - q[2])
    dy = min(dy, ly - dy)
    return dx * dx + dy * dy
end

"""
    list_sublattice_points(lx::Int, ly::Int, want_a::Bool)

列出指定子格的所有点.
参数:
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
- want_a::Bool, true 表示 A 子格, false 表示 B 子格.
返回:
- Vector{Tuple{Int,Int}}, 子格上的点列表, 按 (x, y) 字典序.
"""
function list_sublattice_points(lx::Int, ly::Int, want_a::Bool)::Vector{Tuple{Int,Int}}
    points = Tuple{Int,Int}[]
    for x in 1:lx
        for y in 1:ly
            if is_sublattice_a(x, y) == want_a
                push!(points, (x, y))
            end
        end
    end
    return points
end

"""
    min_distance_to_chosen(point::Tuple{Int,Int}, chosen::Vector{Tuple{Int,Int}}, lx::Int, ly::Int)

计算点到已选集合的最小 torus 距离平方.
参数:
- point::Tuple{Int,Int}, 待评估点.
- chosen::Vector{Tuple{Int,Int}}, 已选点集合.
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
返回:
- Int, 最小距离平方. 若 chosen 为空, 返回 typemax(Int).
"""
function min_distance_to_chosen(
    point::Tuple{Int,Int},
    chosen::Vector{Tuple{Int,Int}},
    lx::Int,
    ly::Int
)::Int
    if isempty(chosen)
        return typemax(Int)
    end

    min_d2 = typemax(Int)
    for q in chosen
        d2 = torus_distance_squared(point, q, lx, ly)
        if d2 < min_d2
            min_d2 = d2
            if min_d2 == 0
                break
            end
        end
    end
    return min_d2
end

"""
    select_fps_point(candidates::Vector{Tuple{Int,Int}}, chosen::Vector{Tuple{Int,Int}}, used::Set{Tuple{Int,Int}}, lx::Int, ly::Int)

在候选点中选择 FPS 最远点 (最大最小距离).
参数:
- candidates::Vector{Tuple{Int,Int}}, 候选点列表.
- chosen::Vector{Tuple{Int,Int}}, 已选点列表.
- used::Set{Tuple{Int,Int}}, 已选点集合, 用于快速跳过.
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
返回:
- Tuple{Int,Int}, 选中的点.
规则:
- 选择最小距离平方最大的点.
- 若距离相同, 按 (x, y) 字典序选择最小者, 保证确定性.
"""
function select_fps_point(
    candidates::Vector{Tuple{Int,Int}},
    chosen::Vector{Tuple{Int,Int}},
    used::Set{Tuple{Int,Int}},
    lx::Int,
    ly::Int
)::Tuple{Int,Int}
    best_point = (-1, -1)
    best_min_d2 = -1
    for p in candidates
        if p in used
            continue
        end
        min_d2 = min_distance_to_chosen(p, chosen, lx, ly)
        if min_d2 > best_min_d2
            best_min_d2 = min_d2
            best_point = p
        elseif min_d2 == best_min_d2
            if p[1] < best_point[1] || (p[1] == best_point[1] && p[2] < best_point[2])
                best_point = p
            end
        end
    end
    if best_point[1] < 0
        throw(ArgumentError("没有可用的候选点, 请检查 ndefect 与子格数量."))
    end
    return best_point
end

"""
    generate_defect_positions_fps(lx::Int, ly::Int, ndefect::Int; first_defect::Tuple{Int,Int}=(div(lx, 3), div(ly, 3)))

使用 FPS 生成缺陷位置, 并强制 A/B 子格交替.
参数:
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
- ndefect::Int, 需要的缺陷数量.
关键字参数:
- first_defect::Tuple{Int,Int}, 第一个 defect 位置 (x, y), 1-based.
返回:
- Vector{Tuple{Int,Int}}, defect_positions, 长度为 ndefect.
"""
function generate_defect_positions_fps(
    lx::Int,
    ly::Int,
    ndefect::Int;
    first_defect::Tuple{Int,Int}=(div(lx, 3) + 1, div(ly, 3) + 1)
)::Vector{Tuple{Int,Int}}
    @assert lx > 0 && ly > 0
    @assert ndefect >= 0 && ndefect <= lx * ly

    if ndefect == 0
        return Tuple{Int,Int}[]
    end

    first_x = mod1(first_defect[1], lx)
    first_y = mod1(first_defect[2], ly)
    first_point = (first_x, first_y)

    positions = Tuple{Int,Int}[]
    push!(positions, first_point)
    used = Set{Tuple{Int,Int}}(positions)

    start_is_a = is_sublattice_a(first_x, first_y)
    candidates_a = list_sublattice_points(lx, ly, true)
    candidates_b = list_sublattice_points(lx, ly, false)

    for idx in 2:ndefect
        need_is_a = isodd(idx) ? start_is_a : !start_is_a
        candidates = need_is_a ? candidates_a : candidates_b
        next_point = select_fps_point(candidates, positions, used, lx, ly)
        push!(positions, next_point)
        push!(used, next_point)
    end

    return positions
end

end
