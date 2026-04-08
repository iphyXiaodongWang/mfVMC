module Projector

export AbstractProjector, AbstractProjectorTerm
export NoProjectorTerm, GutzwillerProjectorTerm, CompositeProjector
export projector_ratio, projector_log_derivative!, projector_log_derivative
export projector_param_names, projector_param_values, projector_param_count
export update_projector_params!, check_projector_consistency


"""
用途: Projector 抽象基类。

参数:
- 无。

返回:
- 无。
"""
abstract type AbstractProjector end
abstract type AbstractProjectorTerm <: AbstractProjector end
struct NoProjectorTerm <: AbstractProjectorTerm end

mutable struct GutzwillerProjectorTerm{T<:Real} <: AbstractProjectorTerm
    param_name::Symbol
    g::T
end

function GutzwillerProjectorTerm(; param_name::Symbol=:g_gutz, g::Real=0.0)
    return GutzwillerProjectorTerm{Float64}(param_name, Float64(g))
end


mutable struct CompositeProjector <: AbstractProjector
    terms::Vector{AbstractProjectorTerm}
end
function CompositeProjector()
    return CompositeProjector(Vector{AbstractProjectorTerm}())
end


function _extract_doublon_count(sampler_state)
    if hasproperty(sampler_state, :count_dbs)
        return Float64(getproperty(sampler_state, :count_dbs))
    else
        error("sampler_state must provide property `count_dbs`.")
    end
end

function _extract_delta_doublon(proposal)
    if hasproperty(proposal, :delta_doublon)
        return Float64(getproperty(proposal, :delta_doublon))
    else
        error("proposal must provide property `delta_doublon`.")
    end
end



function projector_param_names(projector_term::NoProjectorTerm)
    return Symbol[]
end
function projector_param_names(projector_term::GutzwillerProjectorTerm)
    return Symbol[projector_term.param_name]
end
function projector_param_names(projector::CompositeProjector)
    names = Symbol[]
    for term in projector.terms
        append!(names, projector_param_names(term))
    end
    return names
end


"""
用途: 获取 term 参数值。

参数:
- `projector_term::NoProjectorTerm`。

返回:
- `Vector{Float64}`: 空列表。
"""
function projector_param_values(projector_term::NoProjectorTerm)
    return Float64[]
end
function projector_param_values(projector_term::GutzwillerProjectorTerm)
    return Float64[Float64(projector_term.g)]
end
function projector_param_values(projector::CompositeProjector)
    values = Float64[]
    for term in projector.terms
        append!(values, projector_param_values(term))
    end
    return values
end



function projector_param_count(projector::AbstractProjector)
    return length(projector_param_names(projector))
end


"""
用途: 校验 composite projector 参数名是否重复。

参数:
- `projector::CompositeProjector`。

返回:
- `nothing`。若有重复参数名会抛出异常。
"""
function check_projector_consistency(projector::CompositeProjector)
    names = projector_param_names(projector)
    seen = Set{Symbol}()
    duplicates = Symbol[]
    for name in names
        if name in seen
            push!(duplicates, name)
        else
            push!(seen, name)
        end
    end
    if !isempty(duplicates)
        duplicate_unique = unique(duplicates)
        error("Duplicate projector parameter names detected: $(duplicate_unique)")
    end
    return nothing
end


"""
用途: 设置单个 term 参数值。

参数:
- `projector_term::NoProjectorTerm`。
- `name::Symbol`: 参数名。
- `value::Real`: 参数值。

返回:
- `Bool`: 是否成功写入参数。NoProjector 固定返回 `false`。
"""
function _set_projector_param!(projector_term::NoProjectorTerm, name::Symbol, value::Real)
    return false
end
function _set_projector_param!(projector_term::GutzwillerProjectorTerm, name::Symbol, value::Real)
    if projector_term.param_name == name
        projector_term.g = Float64(value)
        return true
    end
    return false
end


"""
用途: 批量更新 composite projector 参数。

参数:
- `projector::CompositeProjector`。
- `param_names::Vector{Symbol}`: 参数名列表。
- `param_values::Vector{<:Real}`: 与参数名对齐的值列表。

返回:
- `nothing`。若参数名缺失、重复或不匹配会抛出异常。
"""
function update_projector_params!(
    projector::CompositeProjector,
    param_names::Vector{Symbol},
    param_values::Vector{<:Real},
)
    if length(param_names) != length(param_values)
        error("Length mismatch: param_names has $(length(param_names)), param_values has $(length(param_values)).")
    end

    # 防御: 输入参数名不能重复
    if length(unique(param_names)) != length(param_names)
        error("param_names contains duplicates: $(param_names)")
    end

    expected_names = projector_param_names(projector)
    expected_set = Set(expected_names)
    input_set = Set(param_names)

    if expected_set != input_set
        missing = setdiff(expected_set, input_set)
        extra = setdiff(input_set, expected_set)
        error("Projector parameter mismatch. Missing=$(collect(missing)), Extra=$(collect(extra))")
    end

    value_map = Dict{Symbol,Float64}()
    for (name, value) in zip(param_names, param_values)
        value_map[name] = Float64(value)
    end

    for term in projector.terms
        for term_name in projector_param_names(term)
            updated = _set_projector_param!(term, term_name, value_map[term_name])
            if !updated
                error("Failed to update projector parameter: $term_name")
            end
        end
    end

    return nothing
end


"""
用途: 按 projector 内部参数顺序更新参数。

参数:
- `projector::CompositeProjector`。
- `param_values::Vector{<:Real}`: 按 `projector_param_names(projector)` 顺序排列。

返回:
- `nothing`。
"""
function update_projector_params!(
    projector::CompositeProjector,
    param_values::Vector{<:Real},
)
    names = projector_param_names(projector)
    if length(names) != length(param_values)
        error("Length mismatch: expected $(length(names)) projector parameters, got $(length(param_values)).")
    end
    return update_projector_params!(projector, names, param_values)
end


"""
用途: 计算 Gutzwiller term 的 projector 比值。

数学公式:
- `ratio = exp(-g * delta_n_d)`

参数:
- `projector_term::GutzwillerProjectorTerm`。
- `sampler_state`: 采样构型对象。
- `proposal`: proposal 对象, 需要提供 `delta_doublon`。

返回:
- `Float64`: 比值 `P(C')/P(C)`。
"""
function projector_ratio(projector_term::NoProjectorTerm, sampler_state, proposal)
    return 1.0
end
function projector_ratio(projector_term::GutzwillerProjectorTerm, sampler_state, proposal)
    delta_n_d = _extract_delta_doublon(proposal)
    return exp(-Float64(projector_term.g) * delta_n_d)
end
function projector_ratio(projector::CompositeProjector, sampler_state, proposal)
    ratio_total = 1.0
    for term in projector.terms
        ratio_total *= projector_ratio(term, sampler_state, proposal)
    end
    return ratio_total
end


"""
用途: 就地写入 NoProjector 的 log-derivative。

参数:
- `buffer::AbstractVector{<:Number}`: 导数写入缓冲区。
- `projector_term::NoProjectorTerm`。
- `sampler_state`: 采样构型对象。

返回:
- `nothing`。
"""
function projector_log_derivative!(
    buffer::AbstractVector{<:Number},
    projector_term::NoProjectorTerm,
    sampler_state,
)
    if !isempty(buffer)
        fill!(buffer, 0.0)
    end
    return nothing
end
function projector_log_derivative!(
    buffer::AbstractVector{<:Number},
    projector_term::GutzwillerProjectorTerm,
    sampler_state,
)
    if length(buffer) != 1
        error("GutzwillerProjectorTerm requires buffer length = 1, got $(length(buffer)).")
    end
    n_doublon = _extract_doublon_count(sampler_state)
    buffer[1] = -n_doublon
    return nothing
end
function projector_log_derivative!(
    buffer::AbstractVector{<:Number},
    projector::CompositeProjector,
    sampler_state,
)
    expected_length = projector_param_count(projector)
    if length(buffer) != expected_length
        error("CompositeProjector requires buffer length = $expected_length, got $(length(buffer)).")
    end

    fill!(buffer, 0.0)
    cursor = 1
    for term in projector.terms
        term_count = projector_param_count(term)
        if term_count == 0
            continue
        end
        term_view = @view buffer[cursor:(cursor+term_count-1)]
        projector_log_derivative!(term_view, term, sampler_state)
        cursor += term_count
    end
    return nothing
end


"""
用途: 分配并返回 projector 的 log-derivative 向量。

参数:
- `projector::AbstractProjector`。
- `sampler_state`: 采样构型对象。

返回:
- `Vector{Float64}`: 按 `projector_param_names(projector)` 顺序排列的导数向量。
"""
function projector_log_derivative(projector::AbstractProjector, sampler_state)
    buffer = zeros(Float64, projector_param_count(projector))
    projector_log_derivative!(buffer, projector, sampler_state)
    return buffer
end


end # module
