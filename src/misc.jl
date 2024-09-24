"""
    Loss(loss::Function)

Tracks the average model loss as `total_loss / steps`
"""
struct AverageMeasure{M} <: AbstractMetric
    name::String
    measure::M
end

name(x::AverageMeasure) = x.name

init(::AverageMeasure) = (total=0.0, n=0)

function update(x::AverageMeasure, state, ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real})
    return (total = state.total + x.measure(ŷ, y), n = state.n + 1)
end

compute(::AverageMeasure, state) = state.total / max(state.n, 1)