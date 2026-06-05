module Timing

using Printf

export TimingStats, timing_reset!, timing_accumulate!, timing_report, merge_reports, @timed, ENABLE_TIMING

"""
    TimingStats

Accumulates per-label wall-clock timing (seconds) across many calls.
Thread-safe by design: each MPI rank accumulates independently, and the root
prints after MPI reduction (average across ranks).
"""
mutable struct TimingStats
    labels::Vector{String}
    total_times::Vector{Float64}
    call_counts::Vector{Int}
    min_times::Vector{Float64}
    max_times::Vector{Float64}
end

TimingStats() = TimingStats(
    String[],
    Float64[],
    Int[],
    Float64[],
    Float64[],
)

const _global_timing = TimingStats()

"""
    timing_reset!(stats=Timing._global_timing)

Reset all accumulated timing data.
"""
function timing_reset!(stats::TimingStats=_global_timing)
    empty!(stats.labels)
    empty!(stats.total_times)
    empty!(stats.call_counts)
    empty!(stats.min_times)
    empty!(stats.max_times)
end

"""
    timing_accumulate!(label::AbstractString, elapsed::Float64; stats=Timing._global_timing)

Accumulate an elapsed time under `label`.
"""
function timing_accumulate!(
    label::AbstractString,
    elapsed::Float64;
    stats::TimingStats=_global_timing,
)
    idx = findfirst(==(label), stats.labels)
    if idx === nothing
        push!(stats.labels, String(label))
        push!(stats.total_times, elapsed)
        push!(stats.call_counts, 1)
        push!(stats.min_times, elapsed)
        push!(stats.max_times, elapsed)
    else
        stats.total_times[idx] += elapsed
        stats.call_counts[idx] += 1
        if elapsed < stats.min_times[idx]
            stats.min_times[idx] = elapsed
        end
        if elapsed > stats.max_times[idx]
            stats.max_times[idx] = elapsed
        end
    end
end

"""
    @timed LABEL expr

Evaluate `expr` and accumulate its wall-clock elapsed time under `LABEL`.
Returns the value of `expr` (unchanged).  If `Timing.ENABLE_TIMING` is false
(lazy global toggle), this macro acts as a no-op.

Usage:
    val = Timing.@timed "my_label" begin
        expensive_calculation()
    end
    # or single-expression form:
    val = Timing.@timed "my_label" expensive_calculation()
"""
macro timed(label_expr, expr)
    timing_module = @__MODULE__
    quote
        if $(GlobalRef(timing_module, :ENABLE_TIMING))[]
            local _t0 = $(GlobalRef(Base, :time))()
            try
                $(esc(expr))
            finally
                $(GlobalRef(timing_module, :timing_accumulate!))(
                    $(esc(label_expr)),
                    $(GlobalRef(Base, :time))() - _t0,
                )
            end
        else
            $(esc(expr))
        end
    end
end

"""
    ENABLE_TIMING :: Ref{Bool}

Global toggle. When false, `@timed` is a no-op. Flip at runtime:
    Timing.ENABLE_TIMING[] = true   # start profiling
    Timing.ENABLE_TIMING[] = false  # stop profiling (zero overhead)
"""
const ENABLE_TIMING = Ref{Bool}(true)

"""
    timing_report(io::IO=stdout; stats=Timing._global_timing, sort_by=:total)

Print an aligned timing report. Columns:
  Label | Calls | Total(s) | Avg(ms) | Min(ms) | Max(ms) | %
"""
function timing_report(
    io::IO=stdout;
    stats::TimingStats=_global_timing,
    sort_by::Symbol=:total,
)
    if isempty(stats.labels)
        println(io, "[Timing] No data collected.")
        return nothing
    end

    grand_total = sum(stats.total_times)
    n = length(stats.labels)

    perm = if sort_by == :total
        sortperm(stats.total_times; rev=true)
    elseif sort_by == :count
        sortperm(stats.call_counts; rev=true)
    elseif sort_by == :avg
        sortperm(stats.total_times ./ stats.call_counts; rev=true)
    else
        collect(1:n)
    end

    println(io, "=" ^ 80)
    println(io, " Timing Report  (grand total = $(round(grand_total, digits=3)) s)")
    println(io, "=" ^ 80)
    @printf(io, "%-42s %8s %12s %10s %10s %10s %6s\n",
            "Label", "Calls", "Total(s)", "Avg(ms)", "Min(ms)", "Max(ms)", "%")
    println(io, "-" ^ 80)

    for i in perm
        label = stats.labels[i]
        calls = stats.call_counts[i]
        total = stats.total_times[i]
        avg = total / calls * 1000
        mn = stats.min_times[i] * 1000
        mx = stats.max_times[i] * 1000
        pct = grand_total > 0 ? total / grand_total * 100 : 0.0
        @printf(io, "%-42s %8d %12.3f %10.3f %10.3f %10.3f %5.1f%%\n",
                label, calls, total, avg, mn, mx, pct)
    end
    println(io, "-" ^ 80)
    println(io, "Grand total: $(round(grand_total, digits=3)) s over $(n) labels")
    return nothing
end

"""
    merge_reports(stats_list::Vector{TimingStats}) -> TimingStats

用途: 合并多个 rank 或多个局部 `TimingStats`, 用于生成完整 timing report。

参数:
- `stats_list::Vector{TimingStats}`: 待合并的计时统计列表。

返回:
- `TimingStats`: 合并后的统计。`total_times` 为各统计中同名 label 总时间之和,
  `call_counts` 为调用次数之和, `min_times/max_times` 为对应最小/最大单次耗时。
"""
function merge_reports(stats_list::Vector{TimingStats})::TimingStats
    merged = TimingStats()
    if isempty(stats_list)
        return merged
    end
    label_set = Set{String}()
    for s in stats_list
        for label in s.labels
            push!(label_set, label)
        end
    end
    for label in sort(collect(label_set))
        total_time = 0.0
        call_count = 0
        min_time = Inf
        max_time = 0.0
        for s in stats_list
            idx = findfirst(==(label), s.labels)
            if idx !== nothing
                total_time += s.total_times[idx]
                call_count += s.call_counts[idx]
                min_time = min(min_time, s.min_times[idx])
                max_time = max(max_time, s.max_times[idx])
            end
        end
        push!(merged.labels, label)
        push!(merged.total_times, total_time)
        push!(merged.call_counts, call_count)
        push!(merged.min_times, isfinite(min_time) ? min_time : 0.0)
        push!(merged.max_times, max_time)
    end
    return merged
end

end # module
