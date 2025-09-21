"""
    AverageMeasure(measure::Function)

Tracks the average of the given measure over mini-batches.
"""
mutable struct AverageMeasure{M} <: AbstractMetric
    measure::M
    n::Int
    avg::Float64
end

AverageMeasure(measure::Function) = AverageMeasure(measure, 0, 0.0)

function step!(x::AverageMeasure, y_pred::AbstractArray{<:Real}, y_true::AbstractArray{<:Real})
    @argcheck length(y_pred) == length(y_true)
    n = x.n
    m = length(y_pred)
    x.avg = (x.avg * (n / (n + m))) + (x.measure(y_pred, y_true) * (m / (n + m)))
    x.n += m
end

value(x::AverageMeasure) = x.avg

params(x::AverageMeasure) = (;x.n)

"""
    MAE()

Measures the mean absolute error, which is defined as `|ŷ - y|`.
"""
mutable struct MAE <: AbstractMetric
    n::Int
    avg::Float64
end

MAE() = MAE(0, 0.0)

function step!(x::MAE, y_pred::AbstractArray{<:Real}, y_true::AbstractArray{<:Real})
    @argcheck length(y_pred) == length(y_true)
    x.avg = weighted_average(abs.(y_true .- y_pred), x.avg, x.n)
    x.n += length(y_pred)
end

value(x::MAE) = x.avg

params(x::MAE) = (;x.n)

"""
    MSE()

Measures the mean squared error, which is defined as ``(ŷ - y)^2``.
"""
mutable struct MSE <: AbstractMetric
    n::Int
    avg::Float64
end

MSE() = MSE(0, 0.0)

function step!(x::MSE, y_pred::AbstractArray{<:Real}, y_true::AbstractArray{<:Real})
    @argcheck length(y_pred) == length(y_true)
    x.avg = weighted_average((y_true .- y_pred) .^ 2, x.avg, x.n)
    x.n += length(y_pred)
end

value(x::MSE) = x.avg

params(x::MSE) = (;x.n)