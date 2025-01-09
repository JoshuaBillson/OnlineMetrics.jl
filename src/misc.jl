"""
    Loss(loss::Function)

Tracks the average model loss as `total_loss / steps`
"""
mutable struct AverageMeasure{M} <: AbstractMetric
    name::String
    measure::M
    n::Int
    avg::Float64
end

AverageMeasure(measure, name::String) = AverageMeasure(name, measure, 0, 0.0)

name(x::AverageMeasure) = x.name

function step!(x::AverageMeasure, ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real})
    x.n += 1
    x.avg = (x.avg * (x.n-1) / x.n) + (x.measure(ŷ, y) / x.n)
end

value(x::AverageMeasure) = x.avg